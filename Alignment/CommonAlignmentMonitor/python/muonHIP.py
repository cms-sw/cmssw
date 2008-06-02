import subprocess # requires Python 2.4
import ROOT, random, os
from Alignment.CommonAlignmentMonitor.muonGeometry import Disk, Chamber, Layer, V, disks, chambers, layers, path

def dummyName(): return "dummy%d" % random.randint(0, 100000)

try:
  os.stat("AlignmentMonitorMuonHIP.root")
  tfile = ROOT.TFile("AlignmentMonitorMuonHIP.root")
except:
  print "Warning: could not open \"AlignmentMonitorMuonHIP.root\".  Set muonHIP.tfile to the correct file."
  tfile = None

viewer3d = None
def view3d():
  global viewer3d
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

def getDTDisk(wheel):
  for d in disks.values():
    if d.barrel:
      if d.wheel == wheel: return d
  raise ValueError, "Not found!"

def getCSCDisk(disk):
  for d in disks.values():
    if not d.barrel:
      if d.disk == disk: return d
  raise ValueError, "Not found!"

def getDTChamber(wheel, station, sector):
  for c in chambers.values():
    if c.barrel:
      if c.wheel == wheel and c.station == station and c.sector == sector: return c
  raise ValueError, "Not found!"

def getCSCChamber(disk, ring, chamber):
  for c in chambers.values():
    if not c.barrel:
      if c.disk == disk and c.ring == ring and c.chamber == chamber: return c
  raise ValueError, "Not found!"

def getDTLayer(wheel, station, sector, superlayer):
  for l in layers.values():
    if l.barrel:
      if l.wheel == wheel and l.station == station and l.sector == sector and l.superlayer == superlayer: return l
  raise ValueError, "Not found!"

def getCSCLayer(disk, ring, chamber, layer):
  for l in layers.values():
    if not l.barrel:
      if l.disk == disk and l.ring == ring and l.chamber == chamber and l.layer == layer: return l
  raise ValueError, "Not found!"

class Selection:
  def __repr__(self):
    if self.granularity == Disk: dirname = "disk"
    elif self.granularity == Chamber: dirname = "chamber"
    elif self.granularity == Layer: dirname = "layer"
    if len(self.histograms) == 1: s = ""
    else: s = "s"
    return "<Selection of %d %s%s>" % (len(self.histograms), dirname, s)

  def __init__(self, func, hist, granularity, superset, tfile, iteration, merged, histograms, selected):
    self.func = func
    self.hist = hist
    self.granularity = granularity
    self.superset = superset
    self.tfile = tfile
    self.iteration = iteration
    self.merged = merged
    self.histograms = histograms
    self.selected = selected

  def Draw(self, *args):
    self.merged.Draw(*args)
      
  def update3d(self, viewer3d=viewer3d):
    if self.granularity == Disk: superset = disks
    elif self.granularity == Chamber: superset = chambers
    elif self.granularity == Layer: superset = layers
    for c in chambers:
      if granularity == Disk:
        test = (c.Disk in self.histograms)
      elif granularity == Chamber:
        test = (c in self.histograms)
      elif granularity == Layer:
        test = False
        for l in c.Layer:
          if l in self.histograms:
            test = True
            break

      viewer3d.stdin.write("c%d.select(%d)\n" % (c.rawid, test))

class Convergence(Selection):
  def __repr__(self):
    if self.granularity == Disk: dirname = "disk"
    elif self.granularity == Chamber: dirname = "chamber"
    elif self.granularity == Layer: dirname = "layer"
    if len(self.histograms) == 1: s = ""
    else: s = "s"
    return "<Convergence of %d %s%s>" % (len(self.histograms), dirname, s)

  def Draw(self, first=0, last=20, low=-1., high=1., base_options="", sub_options="same hist l"):
    self.merged.SetAxisRange(first-0.5, last-0.5, "X")
    self.merged.SetAxisRange(low, high, "Y")
    self.merged.Draw(base_options)
    for h in self.histograms.values():
      h.Draw(sub_options)

  def profile(self, iteration, name=None, title=None, bins=100, low=-1., high=1., normalized=False, aslist=False):
    if iteration == 0:
      tree = self.tfile.Get("iter1/before")
    else:
      tree = self.tfile.Get("iter%d/after" % iteration)

    if name == None: name = dummyName()
    if title == None: title = "%s after %d iterations" % (self.hist, iteration)

    th1 = ROOT.TH1F(name, title, bins, low, high)
    values = []
    for i in tree:
      for c in self.histograms:
        if c.rawid == i.rawid and \
           ((c.__class__ == Disk and (i.level == 21 or i.level == 27)) or \
            (c.__class__ == Chamber and (i.level == 23 or i.level == 28)) or \
            (c.__class__ == Layer and (i.level == 24 or i.level == 29 or i.level == 1 or i.level == 2))):
          # numerical constants defined in Alignment/CommonAlignment/AlignableObjectId.h

          if normalized:
            if eval("i.%serr" % self.hist) != 0.:
              values.append(eval("i.%s/i.%s" % (self.hist, self.hist)))
              th1.Fill(values[-1])
          else:
            values.append(eval("i.%s" % self.hist))
            th1.Fill(values[-1])
          break

    global lastprofile
    lastprofile = th1
    if aslist: return values
    else: return lastprofile

last = None
lastprofile = None
empty = ROOT.TH1F("empty", "No matches.", 10, 0, 1)
def select(func=(lambda c: True), hist="wxresid", granularity=Chamber, superset=None, tfile=tfile, iteration=1):
  if superset == None:
    if granularity == Disk: superset = disks
    elif granularity == Chamber: superset = chambers
    elif granularity == Layer: superset = layers
  if granularity == Disk: dirname = "disk"
  elif granularity == Chamber: dirname = "chamber"
  elif granularity == Layer: dirname = "layer"

  tlist = ROOT.TList()
  histograms = {}
  selected = {}
  for rawid, c in superset.items():
    if (func.func_code.co_argcount == 1 and func(c)) or (func.func_code.co_argcount == 2):

      th1 = tfile.Get("iter%d/%s_%s/%s_%s_%d" % (iteration, hist, dirname, hist, dirname, rawid))

      if (func.func_code.co_argcount == 1) or (func.func_code.co_argcount == 2 and th1 != None and func(c, th1)):
        if th1 != None:
          tlist.Add(th1)
          histograms[c] = th1
          selected[c.rawid] = c

  if len(tlist) > 0:
    merged = th1.Clone()
    merged.Reset()
    merged.Merge(tlist)
    merged.SetTitle("Merged %s for selected %ss" % (hist, dirname))
    merged.SetName("merged %s" % hist)
  else:
    merged = empty

  global last
  last = Selection(func, hist, granularity, superset, tfile, iteration, merged, histograms, selected)
  return last
  
def conv(func=(lambda c: True), hist="x", granularity=Chamber, superset=None, tfile=tfile):
  if superset == None:
    if granularity == Disk: superset = disks
    elif granularity == Chamber: superset = chambers
    elif granularity == Layer: superset = layers
  if granularity == Disk: dirname = "disk"
  elif granularity == Chamber: dirname = "chamber"
  elif granularity == Layer: dirname = "layer"

  tlist = ROOT.TList()
  histograms = {}
  selected = {}
  bins = 21
  for rawid, c in superset.items():
    if (func.func_code.co_argcount == 1 and func(c)) or (func.func_code.co_argcount == 2):

      th1 = tfile.Get("conv_%s_%s/conv_%s_%s_%d" % (hist, dirname, hist, dirname, rawid))

      if (func.func_code.co_argcount == 1) or (func.func_code.co_argcount == 2 and th1 != None and func(c, th1)):
        if th1 != None:
          tlist.Add(th1)
          histograms[c] = th1
          selected[c.rawid] = c
          if th1.GetNbinsX() > bins: bins = th1.GetNbinsX()
      
  merged = ROOT.TH1F(dummyName(), "Convergence in %s" % hist, bins, -0.5, bins + 0.5)
  merged.SetStats(0)
  merged.SetAxisRange(-1., 1., "Y")
  
  global last
  last = Convergence(func, hist, granularity, superset, tfile, None, merged, histograms, selected)
  return last

textToFunc = {"All": (lambda c: True), \
              "DT": (lambda c: c.barrel), \
              "DT Wheel -2": (lambda c: c.barrel and c.wheel == -2), \
              "DT Wheel -1": (lambda c: c.barrel and c.wheel == -1), \
              "DT Wheel 0": (lambda c: c.barrel and c.wheel == 0), \
              "DT Wheel 1": (lambda c: c.barrel and c.wheel == 1), \
              "DT Wheel 2": (lambda c: c.barrel and c.wheel == 2), \
              "DT Station 1": (lambda c: c.barrel and c.station == 1), \
              "DT Station 2": (lambda c: c.barrel and c.station == 2), \
              "DT Station 3": (lambda c: c.barrel and c.station == 3), \
              "DT Station 4": (lambda c: c.barrel and c.station == 4), \
              "CSC": (lambda c: not c.barrel), \
              "CSC Disk -4": (lambda c: not c.barrel and c.disk == -4), \
              "CSC Disk -3": (lambda c: not c.barrel and c.disk == -3), \
              "CSC Disk -2": (lambda c: not c.barrel and c.disk == -2), \
              "CSC Disk -1": (lambda c: not c.barrel and c.disk == -1), \
              "CSC Disk 1": (lambda c: not c.barrel and c.disk == 1), \
              "CSC Disk 2": (lambda c: not c.barrel and c.disk == 2), \
              "CSC Disk 3": (lambda c: not c.barrel and c.disk == 3), \
              "CSC Disk 4": (lambda c: not c.barrel and c.disk == 4), \
              "CSC Inner Ring": (lambda c: not c.barrel and (abs(c.disk) == 1 and c.ring in (1, 2, 4)) or (abs(c.disk) > 1 and c.ring == 1)), \
              "CSC Outer Ring": (lambda c: not c.barrel and (abs(c.disk) == 1 and c.ring == 3) or (abs(c.disk) > 1 and c.ring > 1))}
