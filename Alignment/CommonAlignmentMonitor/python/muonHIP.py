import subprocess # requires Python 2.4
import ROOT, random
from Alignment.CommonAlignmentMonitor.muonGeometry import Disk, Chamber, Layer, V, disks, chambers, layers, path

def dummyName(): return "dummy%d" % random.randint(0, 100000)

try:
  tfile = ROOT.TFile("histograms.root")
except:
  print "Warning: could not open \"histograms.root\".  Set muonHIP.tfile to the correct file."
  tfile = None

def viewer3d():
  viewer3d = subprocess.Popen(["root", "-l"], stdin=subprocess.PIPE)
  viewer3d.stdin.write(".x %s/../src/Alignment/CommonAlignmentMonitor/python/viewer3d.C+\n" % path)
  viewer3d.stdin.write("MyGeom *geom = new MyGeom()\n")

  for rawid, c in chambers.items():
    if c.barrel:
      subdetid = 1
      station = c.station
      ring = c.wheel
    else:
      subdetid = 2
      station = c.disk
      ring = c.ring
    viewer3d.stdin.write("Chamber *c%d = new Chamber(%d, %d, %d, %d, %g, %g, %g, %s, %s, %s)\n" % (rawid, rawid, subdetid, station, ring, c.loc.x, c.loc.y, c.loc.z, c.xhat.TVector3(), c.yhat.TVector3(), c.zhat.TVector3()))
    viewer3d.stdin.write("geom->add(c%d)\n" % rawid)
  viewer3d.stdin.write("geom->Draw(\"ogl\")\n")
  return viewer3d

class Selection:
  def __repr__(self):
    if self.granularity == Disk: dirname = "disk"
    elif self.granularity == Chamber: dirname = "chamber"
    elif self.granularity == Layer: dirname = "layer"
    if len(self.selection) == 1: s = ""
    else: s = "s"
    return "<Selection of %d %s%s>" % (len(self.selection), dirname, s)

  def __init__(self, func, hist, granularity, superset, tfile, iteration, merged, selection):
    self.func = func
    self.hist = hist
    self.granularity = granularity
    self.superset = superset
    self.tfile = tfile
    self.iteration = iteration
    self.merged = merged
    self.selection = selection

  def Draw(self, *args):
    self.merged.Draw(*args)
      
  def update(self, viewer3d):
    if self.granularity == Disk: superset = disks
    elif self.granularity == Chamber: superset = chambers
    elif self.granularity == Layer: superset = layers
    for c in self.selection:
      viewer3d.stdin.write("c%d.select(%d)\n" % (self.rawid, c in superset))

class Convergence(Selection):
  def __repr__(self):
    if self.granularity == Disk: dirname = "disk"
    elif self.granularity == Chamber: dirname = "chamber"
    elif self.granularity == Layer: dirname = "layer"
    if len(self.selection) == 1: s = ""
    else: s = "s"
    return "<Convergence of %d %s%s>" % (len(self.selection), dirname, s)

  def Draw(self, first=0, last=20, low=-1., high=1., base_options="", sub_options="same hist l"):
    self.merged.SetAxisRange(first-0.5, last-0.5, "X")
    self.merged.SetAxisRange(low, high, "Y")
    self.merged.Draw(base_options)
    for h in self.selection.values():
      h.Draw(sub_options)

  def profile(self, iteration, name=None, title=None, bins=100, low=-1., high=1., normalized=False, aslist=False):
    if iteration == 0:
      tree = self.tfile.Get("iter1/before")
    else:
      tree = self.tfile.Get("iter%d/after" % iteration)

    if name == None: name = dummyName()
    if title == None: title = "%s after %d iterations" % (self.param, iteration)

    th1 = ROOT.TH1F(name, title, bins, low, high)
    values = []
    for i in tree:
      for c in self.selection:
        if c.rawid == i.rawid and \
           ((c.__class__ == Disk and (i.level == 21 or i.level == 27)) or \
            (c.__class__ == Chamber and (i.level == 23 or i.level == 28)) or \
            (c.__class__ == Layer and (i.level == 24 or i.level == 29 or i.level == 1 or i.level == 2))):
          # numerical constants defined in Alignment/CommonAlignment/AlignableObjectId.h

          if normalized:
            if eval("t.%serr" % self.hist) != 0.:
              values.append(eval("t.%s/t.%s" % (self.hist, self.hist)))
              th1.Fill(values[-1])
          else:
            values.append(eval("t.%s" % self.hist))
            th1.Fill(values[-1])
          break

    if aslist: return values
    else: return th1

last = None
empty = ROOT.TH1F("empty", "No matches.", 10, 0, 1)
def select(func=(lambda c, h: True), hist="wxresid", granularity=Disk, superset=None, tfile=tfile, iteration=1):
  if superset == None:
    if granularity == Disk: superset = disks
    elif granularity == Chamber: superset = chambers
    elif granularity == Layer: superset = layers
  if granularity == Disk: dirname = "disk"
  elif granularity == Chamber: dirname = "chamber"
  elif granularity == Layer: dirname = "layer"

  tlist = ROOT.TList()
  selection = {}
  for rawid, c in superset.items():
    if (func.func_code.co_argcount == 1 and func(c)) or (func.func_code.co_argcount == 2):

      th1 = tfile.Get("iter%d/%s_%s/%s_%s_%d" % (iteration, hist, dirname, hist, dirname, rawid))

      if (func.func_code.co_argcount == 1) or (func.func_code.co_argcount == 2 and func(c, th1)):
        if th1 != None:
          tlist.Add(th1)
          selection[c] = th1

  if len(tlist) > 0:
    merged = th1.Clone()
    merged.Reset()
    merged.Merge(tlist)
    merged.SetTitle("Merged %s for selected %s" % (hist, dirname))
    merged.SetName("merged %s")
  else:
    merged = empty

  global last
  last = Selection(func, hist, granularity, superset, tfile, iteration, merged, selection)
  return last
  
def conv(func=(lambda c: True), param="x", granularity=Disk, superset=None, tfile=tfile):
  if superset == None:
    if granularity == Disk: superset = disks
    elif granularity == Chamber: superset = chambers
    elif granularity == Layer: superset = layers
  if granularity == Disk: dirname = "disk"
  elif granularity == Chamber: dirname = "chamber"
  elif granularity == Layer: dirname = "layer"

  tlist = ROOT.TList()
  selection = {}
  for rawid, c in superset.items():
    if func(c):
      th1 = tfile.Get("conv_%s_%s/conv_%s_%s_%d" % (param, dirname, param, dirname, rawid))
      if th1 != None:
        tlist.Add(th1)
        selection[c] = th1
      
  merged = ROOT.TH1F(dummyName(), "Convergence in %s" % param, 21, -0.5, 20.5)
  merged.SetStats(0)
  merged.SetAxisRange(-1., 1., "Y")
  
  global last
  last = Convergence(func, param, granularity, superset, tfile, None, merged, selection)
  return last

def selectFromText(text):
  if text == "All": return (lambda c: True)
  elif text == "DT": return (lambda c: c.barrel)

  elif text == "DT Wheel -2": return (lambda c: c.barrel and c.wheel == -2)
  elif text == "DT Wheel -1": return (lambda c: c.barrel and c.wheel == -1)
  elif text == "DT Wheel 0": return (lambda c: c.barrel and c.wheel == 0)
  elif text == "DT Wheel 1": return (lambda c: c.barrel and c.wheel == 1)
  elif text == "DT Wheel 2": return (lambda c: c.barrel and c.wheel == 2)
  elif text == "DT Station 1": return (lambda c: c.barrel and c.station == 1)
  elif text == "DT Station 2": return (lambda c: c.barrel and c.station == 2)
  elif text == "DT Station 3": return (lambda c: c.barrel and c.station == 3)
  elif text == "DT Station 4": return (lambda c: c.barrel and c.station == 4)

  elif text == "CSC": return (lambda c: not c.barrel)
  elif text == "CSC Disk -4": return (lambda c: not c.barrel and c.disk == -4)
  elif text == "CSC Disk -3": return (lambda c: not c.barrel and c.disk == -3)
  elif text == "CSC Disk -2": return (lambda c: not c.barrel and c.disk == -2)
  elif text == "CSC Disk -1": return (lambda c: not c.barrel and c.disk == -1)
  elif text == "CSC Disk 1": return (lambda c: not c.barrel and c.disk == 1)
  elif text == "CSC Disk 2": return (lambda c: not c.barrel and c.disk == 2)
  elif text == "CSC Disk 3": return (lambda c: not c.barrel and c.disk == 3)
  elif text == "CSC Disk 4": return (lambda c: not c.barrel and c.disk == 4)
  elif text == "CSC Inner Rings": return (lambda c: not c.barrel and c.ring == 1)
  elif text == "CSC Outer Rings": return (lambda c: not c.barrel and c.ring == 2)


