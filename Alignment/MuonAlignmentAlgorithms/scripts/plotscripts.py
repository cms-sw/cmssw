import ROOT, array, os, re, random
from math import *
import time
import pickle

# python 2.6 has json modue; <2.6 could use simplejson
try:
  import json
except ImportError:
  import simplejson as json

# sign conventions and some dimensions
from signConventions import *

# common muon types structures
from mutypes import *

CPP_LOADED = False

# containers for test results for map plots
MAP_RESULTS_SAWTOOTH = {}
MAP_RESULTS_FITSIN = {}
MAP_RESULTS_BINS = {}

# general container for all test results
TEST_RESULTS = {}

#############################################################
# Convenience functions

def wheelm2only(dt, wheel, station, sector): return dt == "DT" and wheel == -2
def wheelm1only(dt, wheel, station, sector): return dt == "DT" and wheel == -1
def wheel0only(dt, wheel, station, sector): return dt == "DT" and wheel == 0
def wheelp1only(dt, wheel, station, sector): return dt == "DT" and wheel == 1
def wheelp2only(dt, wheel, station, sector): return dt == "DT" and wheel == 2

def wheelLetter(wheel):
  if   wheel == -2: return "A"
  elif wheel == -1: return "B"
  elif wheel ==  0: return "C"
  elif wheel == +1: return "D"
  elif wheel == +2: return "E"
  else: raise Exception

def wheelNumber(wheell):
  if   wheell == "A": return -2
  elif wheell == "B": return -1
  elif wheell == "C": return 0
  elif wheell == "D": return 1
  elif wheell == "E": return 2
  else: raise Exception

def mean(xlist):
  s, n = 0., 0.
  for x in xlist:
    s += x
    n += 1.
  return s/n

def rms(xlist):
  s2, n = 0., 0.
  for x in xlist:
    s2 += x**2
    n += 1.
  return sqrt(s2/n)

def stdev(xlist):
  s, s2, n = 0., 0., 0.
  for x in xlist:
    s += x
    s2 += x**2
    n += 1.
  return sqrt(s2/n - (s/n)**2)

def wmean(xlist):
  s, w = 0., 0.
  for x, e in xlist:
    if e > 0.:
      wi = 1./e**2
      s += x*wi
      w += wi
  return s/w, sqrt(1./w)

#############################################################

tdrStyle = None
def setTDRStyle():
  global tdrStyle
  tdrStyle = ROOT.TStyle("tdrStyle","Style for P-TDR")
# For the canvas:
  tdrStyle.SetCanvasBorderMode(0)
  tdrStyle.SetCanvasColor(ROOT.kWhite)
  tdrStyle.SetCanvasDefH(600) #Height of canvas
  tdrStyle.SetCanvasDefW(600) #Width of canvas
  tdrStyle.SetCanvasDefX(0)   #POsition on screen
  tdrStyle.SetCanvasDefY(0)

# For the Pad:
  tdrStyle.SetPadBorderMode(0)
  # tdrStyle.SetPadBorderSize(Width_t size = 1)
  tdrStyle.SetPadColor(ROOT.kWhite)
  tdrStyle.SetPadGridX(False)
  tdrStyle.SetPadGridY(False)
  tdrStyle.SetGridColor(0)
  tdrStyle.SetGridStyle(3)
  tdrStyle.SetGridWidth(1)

# For the frame:
  tdrStyle.SetFrameBorderMode(0)
  tdrStyle.SetFrameBorderSize(1)
  tdrStyle.SetFrameFillColor(0)
  tdrStyle.SetFrameFillStyle(0)
  tdrStyle.SetFrameLineColor(1)
  tdrStyle.SetFrameLineStyle(1)
  tdrStyle.SetFrameLineWidth(1)

# For the histo:
  # tdrStyle.SetHistFillColor(1)
  # tdrStyle.SetHistFillStyle(0)
  tdrStyle.SetHistLineColor(1)
  tdrStyle.SetHistLineStyle(0)
  tdrStyle.SetHistLineWidth(1)
  # tdrStyle.SetLegoInnerR(Float_t rad = 0.5)
  # tdrStyle.SetNumberContours(Int_t number = 20)

  tdrStyle.SetEndErrorSize(2)
#  tdrStyle.SetErrorMarker(20)
  tdrStyle.SetErrorX(0.)

  tdrStyle.SetMarkerStyle(20)

#For the fit/function:
  tdrStyle.SetOptFit(1)
  tdrStyle.SetFitFormat("5.4g")
  tdrStyle.SetFuncColor(2)
  tdrStyle.SetFuncStyle(1)
  tdrStyle.SetFuncWidth(1)

#For the date:
  tdrStyle.SetOptDate(0)
  # tdrStyle.SetDateX(Float_t x = 0.01)
  # tdrStyle.SetDateY(Float_t y = 0.01)

# For the statistics box:
  tdrStyle.SetOptFile(0)
  tdrStyle.SetOptStat(0) # To display the mean and RMS:   SetOptStat("mr")
  tdrStyle.SetStatColor(ROOT.kWhite)
  tdrStyle.SetStatFont(42)
  tdrStyle.SetStatFontSize(0.025)
  tdrStyle.SetStatTextColor(1)
  tdrStyle.SetStatFormat("6.4g")
  tdrStyle.SetStatBorderSize(1)
  tdrStyle.SetStatH(0.1)
  tdrStyle.SetStatW(0.15)
  # tdrStyle.SetStatStyle(Style_t style = 1001)
  # tdrStyle.SetStatX(Float_t x = 0)
  # tdrStyle.SetStatY(Float_t y = 0)

# Margins:
  tdrStyle.SetPadTopMargin(0.05)
  tdrStyle.SetPadBottomMargin(0.13)
  tdrStyle.SetPadLeftMargin(0.13)
  tdrStyle.SetPadRightMargin(0.05)

# For the Global title:
  tdrStyle.SetOptTitle(0)
  tdrStyle.SetTitleFont(42)
  tdrStyle.SetTitleColor(1)
  tdrStyle.SetTitleTextColor(1)
  tdrStyle.SetTitleFillColor(10)
  tdrStyle.SetTitleFontSize(0.05)
  # tdrStyle.SetTitleH(0) # Set the height of the title box
  # tdrStyle.SetTitleW(0) # Set the width of the title box
  # tdrStyle.SetTitleX(0) # Set the position of the title box
  # tdrStyle.SetTitleY(0.985) # Set the position of the title box
  # tdrStyle.SetTitleStyle(Style_t style = 1001)
  # tdrStyle.SetTitleBorderSize(2)

# For the axis titles:
  tdrStyle.SetTitleColor(1, "XYZ")
  tdrStyle.SetTitleFont(42, "XYZ")
  tdrStyle.SetTitleSize(0.06, "XYZ")
  # tdrStyle.SetTitleXSize(Float_t size = 0.02) # Another way to set the size?
  # tdrStyle.SetTitleYSize(Float_t size = 0.02)
  tdrStyle.SetTitleXOffset(0.9)
  tdrStyle.SetTitleYOffset(1.05)
  # tdrStyle.SetTitleOffset(1.1, "Y") # Another way to set the Offset

# For the axis labels:
  tdrStyle.SetLabelColor(1, "XYZ")
  tdrStyle.SetLabelFont(42, "XYZ")
  tdrStyle.SetLabelOffset(0.007, "XYZ")
  tdrStyle.SetLabelSize(0.05, "XYZ")

# For the axis:
  tdrStyle.SetAxisColor(1, "XYZ")
  tdrStyle.SetStripDecimals(True)
  tdrStyle.SetTickLength(0.03, "XYZ")
  tdrStyle.SetNdivisions(510, "XYZ")
  tdrStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
  tdrStyle.SetPadTickY(1)

# Change for log plots:
  tdrStyle.SetOptLogx(0)
  tdrStyle.SetOptLogy(0)
  tdrStyle.SetOptLogz(0)

# Postscript options:
  tdrStyle.SetPaperSize(20.,20.)
  # tdrStyle.SetLineScalePS(Float_t scale = 3)
  # tdrStyle.SetLineStyleString(Int_t i, const char* text)
  # tdrStyle.SetHeaderPS(const char* header)
  # tdrStyle.SetTitlePS(const char* pstitle)

  # tdrStyle.SetBarOffset(Float_t baroff = 0.5)
  # tdrStyle.SetBarWidth(Float_t barwidth = 0.5)
  # tdrStyle.SetPaintTextFormat(const char* format = "g")
  # tdrStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
  # tdrStyle.SetTimeOffset(Double_t toffset)
  # tdrStyle.SetHistMinimumZero(True)

  tdrStyle.cd()

setTDRStyle()

def set_palette(name=None, ncontours=999):
    """Set a color palette from a given RGB list
    stops, red, green and blue should all be lists of the same length
    see set_decent_colors for an example"""

    if name == "halfgray":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = map(lambda x: 1. - (1.-x)/2., [1.00, 0.84, 0.61, 0.34, 0.00])
        green = map(lambda x: 1. - (1.-x)/2., [1.00, 0.84, 0.61, 0.34, 0.00])
        blue  = map(lambda x: 1. - (1.-x)/2., [1.00, 0.84, 0.61, 0.34, 0.00])
    elif name == "gray":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [1.00, 0.84, 0.61, 0.34, 0.00]
        green = [1.00, 0.84, 0.61, 0.34, 0.00]
        blue  = [1.00, 0.84, 0.61, 0.34, 0.00]
    elif name == "blues":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [1.00, 0.84, 0.61, 0.34, 0.00]
        green = [1.00, 0.84, 0.61, 0.34, 0.00]
        blue  = [1.00, 1.00, 1.00, 1.00, 1.00]
    elif name == "reds":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [1.00, 1.00, 1.00, 1.00, 1.00]
        green = [1.00, 0.84, 0.61, 0.34, 0.00]
        blue  = [1.00, 0.84, 0.61, 0.34, 0.00]
    elif name == "antigray":
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [1.00, 0.84, 0.61, 0.34, 0.00]
        green = [1.00, 0.84, 0.61, 0.34, 0.00]
        blue  = [1.00, 0.84, 0.61, 0.34, 0.00]
        red.reverse()
        green.reverse()
        blue.reverse()
    elif name == "fire":
        stops = [0.00, 0.20, 0.80, 1.00]
        red   = [1.00, 1.00, 1.00, 0.50]
        green = [1.00, 1.00, 0.00, 0.00]
        blue  = [0.20, 0.00, 0.00, 0.00]
    elif name == "antifire":
        stops = [0.00, 0.20, 0.80, 1.00]
        red   = [0.50, 1.00, 1.00, 1.00]
        green = [0.00, 0.00, 1.00, 1.00]
        blue  = [0.00, 0.00, 0.00, 0.20]
    else:
        # default palette, looks cool
        stops = [0.00, 0.34, 0.61, 0.84, 1.00]
        red   = [0.00, 0.00, 0.87, 1.00, 0.51]
        green = [0.00, 0.81, 1.00, 0.20, 0.00]
        blue  = [0.51, 1.00, 0.12, 0.00, 0.00]

    s = array.array('d', stops)
    r = array.array('d', red)
    g = array.array('d', green)
    b = array.array('d', blue)

    npoints = len(s)
    ROOT.TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    ROOT.gStyle.SetNumberContours(ncontours)

set_palette()

######################################################################################################
## sector phi edges in: me11 me12 me13 me14 me21 me22 me31 me32 me41 me42 mb1 mb2 mb3 mb4
## index:               0    1    2    3    4    5    6    7    8    9    10  11  12  13

#phiedgesCSC36 = [pi/180.*(-175. + 10.*i) for i in range(36)]
#phiedgesCSC18 = [pi/180.*(-175. + 20.*i) for i in range(18)]
phiedgesCSC36 = [pi/180.*(-5. + 10.*i) for i in range(36)]
phiedgesCSC18 = [pi/180.*(-5. + 20.*i) for i in range(18)]
phiedges = [
   phiedgesCSC36,
   phiedgesCSC36,
   phiedgesCSC36,
   phiedgesCSC36,
   phiedgesCSC18,
   phiedgesCSC36,
   phiedgesCSC18,
   phiedgesCSC36,
   phiedgesCSC18,
   phiedgesCSC36,
   [0.35228048120123945, 0.87587781482541827, 1.3994776462193192, 1.923076807996136, 2.4466741416203148, 2.970273973014216,
    -2.7893121723885534, -2.2657148387643748, -1.7421150073704739, -1.2185158455936571, -0.69491851196947851, -0.17131868057557731],
   [0.22000706229660855, 0.74360690430428489, 1.267204926935573, 1.7908033890915052, 2.3144032310991816, 2.8380012537304697,
    -2.9215855912931841, -2.3979857492855081, -1.8743877266542202, -1.3507892644982882, -0.82718942249061178, -0.30359139985932365],
   [0.29751957124275596, 0.82111826253905784, 1.3447162969496083, 1.8683158980376524, 2.3919145893339548, 2.915512623744505,
    -2.844073082347037, -2.3204743910507353, -1.7968763566401849, -1.2732767555521407, -0.74967806425583894, -0.22608002984528835],
   [3.0136655290752188, -2.7530905195097337, -2.2922883025568734, -1.9222915077192773, -1.5707963267948966, -1.2193011458705159,
    -0.84930435103291968, -0.38850213408005951, 0.127927124514574, 0.65152597487624719, 1.1322596819239259, 1.5707963267948966, 
    2.0093329716658674, 2.4900666787135459]]

def phiedges2c():
  lines = []
  for ed in phiedges[:]:
    ed.sort()
    #print ed
    ed.extend([999 for n in range(0,37-len(ed))])
    lines.append('{' + ', '.join(map(str, ed)) + '}')
  #print lines
  res = ', '.join(lines)
  ff = open("phiedges_export.h",mode="w")
  print>>ff,'double phiedges[14][37] = {' + res + '};'
  ff.close()

class SawTeethFunction:
  def __init__(self, name):
    self.name = name
    self.edges = (phiedges[stationIndex(name)])[:]
    self.ed = sorted(self.edges)
    # add some padding to the end
    self.ed.append(pi+1.)
    self.n = len(self.edges)
  def __call__(self, xx, par):
    # wrap x in the most negative phi sector into positive phi
    x = xx[0]
    if x < self.ed[0]: x += 2*pi
    # locate sector
    for i in range(0,self.n):
      if x <= self.ed[i]: continue
      if x > self.ed[i+1]: continue
      return par[i*2] + par[i*2+1]*(x - self.ed[i])
    return 0
  def pp(self):
    print self.name, self.n
    print self.edges
    print self.ed


def stationIndex(name):
  if ("MB" in name or "ME" in name):
    # assume the name is ID
    pa = idToPostalAddress(name)
    if pa is None: return None
    if pa[0]=="CSC":
      if pa[2]==1 and pa[3]==1: return 0
      if pa[2]==1 and pa[3]==2: return 1
      if pa[2]==1 and pa[3]==3: return 2
      if pa[2]==1 and pa[3]==4: return 3
      if pa[2]==2 and pa[3]==1: return 4
      if pa[2]==2 and pa[3]==2: return 5
      if pa[2]==3 and pa[3]==1: return 6
      if pa[2]==3 and pa[3]==2: return 7
      if pa[2]==4 and pa[3]==1: return 8
      if pa[2]==4 and pa[3]==2: return 9
    if pa[0]=="DT":
      if pa[2]==1: return 10
      if pa[2]==2: return 11
      if pa[2]==3: return 12
      if pa[2]==4: return 13
  else:
    if ("mem11" in name or "mep11" in name): return 0
    if ("mem12" in name or "mep12" in name): return 1
    if ("mem13" in name or "mep13" in name): return 2
    if ("mem14" in name or "mep14" in name): return 3
    if ("mem21" in name or "mep21" in name): return 4
    if ("mem22" in name or "mep22" in name): return 5
    if ("mem31" in name or "mep31" in name): return 6
    if ("mem32" in name or "mep32" in name): return 7
    if ("mem41" in name or "mep41" in name): return 8
    if ("mem42" in name or "mep42" in name): return 9
    if ("st1" in name): return 10
    if ("st2" in name): return 11
    if ("st3" in name): return 12
    if ("st4" in name): return 13



def philines(name, window, abscissa):
    global philine_tlines, philine_labels
    philine_tlines = []
    edges = phiedges[stationIndex(name)]
    #print name, len(edges)
    for phi in edges:
        if abscissa is None or abscissa[0] < phi < abscissa[1]:
            philine_tlines.append(ROOT.TLine(phi, -window, phi, window))
            philine_tlines[-1].SetLineStyle(2)
            philine_tlines[-1].Draw()
    if "st" in name: # DT labels
        philine_labels = []
        edges = sorted(edges[:])
        if "st4" in name:
            labels = [" 7", " 8", " 9", "14", "10", "11", "12", " 1", " 2", " 3", "13", " 4", " 5", " 6"]
        else: 
            labels = [" 8", " 9", "10", "11", "12", " 1", " 2", " 3", " 4", " 5", " 6"]
            edges = edges[1:]
        for phi, label in zip(edges, labels):
            littlebit = 0.
            if label in (" 7", " 9", "14", "10", "11"): littlebit = 0.05
            philine_labels.append(ROOT.TText(phi-0.35+littlebit, -0.9*window, label))
            philine_labels[-1].Draw()
        philine_labels.append(ROOT.TText(-2.9, -0.75*window, "Sector:"))
        philine_labels[-1].Draw()
    if "CSC" in name: # DT labels
        philine_labels = []
        edges = sorted(edges[:])
        labels = [" 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                  "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36"]
        #else: 
        #    labels = [" 8", " 9", "10", "11", "12", " 1", " 2", " 3", " 4", " 5", " 6"]
        #    edges = edges[1:]
        for phi, label in zip(edges, labels):
            littlebit = 0.
            #if label in (" 7", " 9", "14", "10", "11"): littlebit = 0.05
            philine_labels.append(ROOT.TText(phi+littlebit, -0.9*window, label))
            philine_labels[-1].SetTextFont(42)
            philine_labels[-1].SetTextSize(0.028)
            philine_labels[-1].Draw()
        philine_labels.append(ROOT.TText(0, -0.78*window, "Chamber:"))
        philine_labels[-1].SetTextSize(0.035)
        philine_labels[-1].Draw()

def zlines(window, abscissa):
    global zline_tlines
    zline_tlines = []
    for z in -401.625, -133.875, 133.875, 401.625:
        if abscissa is None or abscissa[0] < z < abscissa[1]:
            zline_tlines.append(ROOT.TLine(z, -window, z, window))
            zline_tlines[-1].SetLineStyle(2)
            zline_tlines[-1].Draw()
    zline_labels = []
    zline_labels.append(ROOT.TText(-550, -0.9*window, "-2"))
    zline_labels.append(ROOT.TText(-300, -0.9*window, "-1"))
    zline_labels.append(ROOT.TText(-10, -0.9*window, "0"))
    zline_labels.append(ROOT.TText(250, -0.9*window, "+1"))
    zline_labels.append(ROOT.TText(500, -0.9*window, "+2"))
    for z in zline_labels: z.Draw()
    zline_labels.append(ROOT.TText(-600, -0.75*window, "Wheel:")); zline_labels[-1].Draw()

def rlines(disk, window, abscissa):
    global rline_tlines
    rline_tlines = []
    if disk == 1: rl = [150., 270., 480.]
    else: rl = [350.]
    for r in rl:
        if abscissa is None or abscissa[0] < r < abscissa[1]:
            rline_tlines.append(ROOT.TLine(r, -window, r, window))
            rline_tlines[-1].SetLineStyle(2)
            rline_tlines[-1].Draw()

######################################################################################################

def getReportByPostalAddress(postal_address, report):
  for r in report:
    if postal_address == r.postal_address:
      return r
  return None


######################################################################################################

def DBMC(database, reports, window=10., windows=None, selection=None, phi=False, 
         color=ROOT.kBlue-8, style=1, bins=50, normalized=False, getvalues=False, name="", canvas=None, reportdiff=False, inlog=True):
    return DBdiff(database, None, reports, None, window, windows, selection, phi, color, style, bins, normalized, getvalues,
                  name, canvas, reportdiff, inlog)


def DBdiff(database1, database2, reports1, reports2, 
           window=10., windows=None, selection=None, phi=False, color=ROOT.kBlue-8,
           style=1, bins=50, normalized=False, getvalues=False, name="tmp", canvas=None, reportdiff=False, inlog=False ):

    tdrStyle.SetOptStat("emrou")
    tdrStyle.SetStatW(0.40)

    wnd = [window]*6
    if windows is not None:
      i=0
      for w in windows:
        wnd[i] = windows[i]
        i+=1
    
    global hx, hy, hz, hphix, hphiy, hphiz
    
    if phi:
        hx = ROOT.TH1F("%s_phi" % name, "", bins, -wnd[0], wnd[0])
    else:
        hx = ROOT.TH1F("%s_x" % name, "", bins, -wnd[0], wnd[0])
    hy = ROOT.TH1F("%s_y" % name, "", bins, -wnd[1], wnd[1])
    hz = ROOT.TH1F("%s_z" % name, "", bins, -wnd[2], wnd[2])
    hphix = ROOT.TH1F("%s_phix" % name, "", bins, -wnd[3], wnd[3])
    hphiy = ROOT.TH1F("%s_phiy" % name, "", bins, -wnd[4], wnd[4])
    hphiz = ROOT.TH1F("%s_phiz" % name, "", bins, -wnd[5], wnd[5])
        
    for r1 in reports1:
        if selection is None or (selection.__code__.co_argcount == len(r1.postal_address) and selection(*r1.postal_address)):
            if reports2 is None:
                r2 = Report(r1.chamberId, r1.postal_address, r1.name)
                r2.add_parameters(ValErr(0., 0., 0.), ValErr(0., 0., 0.), ValErr(0., 0., 0.), 
                    ValErr(0., 0., 0.), ValErr(0., 0., 0.), ValErr(0., 0., 0.), 0., 0., 0., 0.)
            else:
                r2 = getReportByPostalAddress(r1.postal_address, reports2)
                if r2 is None: continue

            found = False
            if r1.postal_address[0] == "DT":
                if r1.postal_address[1:] in database1.dt:
                    found = True
                    db1 = database1.dt[r1.postal_address[1:]]
                    if database2 is None:
                        db2 = DTAlignable()
                        db2.x = db2.y = db2.z = db2.phix = db2.phiy = db2.phiz = 0.
                        db2.xx = db2.xy = db2.xz = db2.yx = db2.yy = db2.yz = db2.zx = db2.zy = db2.zz = 0.
                    else:
                        db2 = database2.dt[r1.postal_address[1:]]
                
            else:
                # skip ME1/a
                if r1.postal_address[2]==1 and r1.postal_address[3]==4: continue
                if r1.postal_address[1:] in database1.csc:
                    found = True
                    db1 = database1.csc[r1.postal_address[1:]]
                    if database2 is None:
                        db2 = CSCAlignable()
                        db2.x = db2.y = db2.z = db2.phix = db2.phiy = db2.phiz = 0.
                        db2.xx = db2.xy = db2.xz = db2.yx = db2.yy = db2.yz = db2.zx = db2.zy = db2.zz = 0.
                    else:
                        db2 = database2.csc[r1.postal_address[1:]]

            if found and r1.status == "PASS" and r2.status == "PASS":
                if r1.deltax is not None and r2.deltax is not None and r1.deltax.error is not None and \
                   r2.deltax.error is not None and (r1.deltax.error**2 + r2.deltax.error**2) > 0.:
                    delta = db1.x - db2.x
                    if reportdiff: delta -= r1.deltax.value
                    if normalized:
                        fill = delta/sqrt(r1.deltax.error**2 + r2.deltax.error**2) * signConventions[r1.postal_address][0]
                    else:
                        if phi:
                            fill = delta/signConventions[r1.postal_address][3] * 1000. * signConventions[r1.postal_address][0]
                        else:
                            fill = delta * 10. * signConventions[r1.postal_address][0]
                    hx.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["x"].append((fill, 10. * sqrt(r1.deltax.error**2 + r2.deltax.error**2)))

                if r1.deltay is not None and r2.deltay is not None and r1.deltay.error is not None and \
                   r2.deltay.error is not None and (r1.deltay.error**2 + r2.deltay.error**2) > 0.:
                    delta = db1.y - db2.y
                    if reportdiff: delta -= r1.deltay.value
                    if normalized:
                        fill = delta/sqrt(r1.deltay.error**2 + r2.deltay.error**2) * signConventions[r1.postal_address][1]
                    else:
                        fill = delta * 10. * signConventions[r1.postal_address][1]
                    hy.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["y"].append((fill, 10. * sqrt(r1.deltay.error**2 + r2.deltay.error**2)))

                if r1.deltaz is not None and r2.deltaz is not None and r1.deltaz.error is not None and \
                   r2.deltaz.error is not None and (r1.deltaz.error**2 + r2.deltaz.error**2) > 0.:
                    delta = db1.z - db2.z
                    if reportdiff: delta -= r1.deltaz.value
                    if normalized:
                        fill = delta/sqrt(r1.deltaz.error**2 + r2.deltaz.error**2) * signConventions[r1.postal_address][2]
                    else:
                        fill = delta * 10. * signConventions[r1.postal_address][2]
                    hz.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["z"].append((fill, 10. * sqrt(r1.deltaz.error**2 + r2.deltaz.error**2)))

                if r1.deltaphix is not None and r2.deltaphix is not None and r1.deltaphix.error is not None and \
                   r2.deltaphix.error is not None and (r1.deltaphix.error**2 + r2.deltaphix.error**2) > 0.:
                    delta = db1.phix - db2.phix
                    if reportdiff: delta -= r1.deltaphix.value
                    if normalized:
                        fill = delta/sqrt(r1.deltaphix.error**2 + r2.deltaphix.error**2)
                    else:
                        fill = delta * 1000.
                    hphix.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["phix"].append((fill, 10. * sqrt(r1.deltaphix.error**2 + r2.deltaphix.error**2)))

                if r1.deltaphiy is not None and r2.deltaphiy is not None and r1.deltaphiy.error is not None and \
                   r2.deltaphiy.error is not None and (r1.deltaphiy.error**2 + r2.deltaphiy.error**2) > 0.:
                    delta = db1.phiy - db2.phiy
                    if reportdiff: 
                      delta -= r1.deltaphiy.value
                      if abs(delta)>0.02/1000: print r1.postal_address, 1000*delta, "=", 1000*db1.phiy - 1000*db2.phiy, "-", 1000*r1.deltaphiy.value, "... ",1000*db1.phiy , 1000*db2.phiy
                    if normalized:
                        fill = delta/sqrt(r1.deltaphiy.error**2 + r2.deltaphiy.error**2)
                    else:
                        fill = delta * 1000.
                    hphiy.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["phiy"].append((fill, 10. * sqrt(r1.deltaphiy.error**2 + r2.deltaphiy.error**2)))

                if r1.deltaphiz is not None and r2.deltaphiz is not None and r1.deltaphiz.error is not None and \
                   r2.deltaphiz.error is not None and (r1.deltaphiz.error**2 + r2.deltaphiz.error**2) > 0.:
                    delta = db1.phiz - db2.phiz
                    if reportdiff: delta -= r1.deltaphiz.value
                    if normalized:
                        fill = delta/sqrt(r1.deltaphiz.error**2 + r2.deltaphiz.error**2)
                    else:
                        fill = delta * 1000.
                    hphiz.Fill(fill)
                    if getvalues not in (False, None):
                        getvalues["phiz"].append((fill, 10. * sqrt(r1.deltaphiz.error**2 + r2.deltaphiz.error**2)))

    if not normalized:
        if phi:
            hx.SetXTitle("#delta_{#phi} position (mrad)")
        else:
            hx.SetXTitle("#delta_{x'} (mm)")
        hy.SetXTitle("#delta_{y'} (mm)")
        hz.SetXTitle("#delta_{z'} (mm)")
        hphix.SetXTitle("#delta_{#phi_{x}} (mrad)")
        hphiy.SetXTitle("#delta_{#phi_{y}} (mrad)")
        hphiz.SetXTitle("#delta_{#phi_{z}} (mrad)")
        if reportdiff:
          if phi:
              hx.SetXTitle("#delta_{#phi}(XML) - #delta_{#phi}(report) position (mrad)")
          else:
              hx.SetXTitle("#delta_{x'}(XML) - #delta_{x'}(report) (mm)")
          hy.SetXTitle("#delta_{y'}(XML) - #delta_{y'}(report) (mm)")
          hz.SetXTitle("#delta_{z'}(XML) - #delta_{z'}(report) (mm)")
          hphix.SetXTitle("#delta_{#phi_{x}}(XML) - #delta_{#phi_{x}}(report) (mrad)")
          hphiy.SetXTitle("#delta_{#phi_{y}}(XML) - #delta_{#phi_{y}}(report) (mrad)")
          hphiz.SetXTitle("#delta_{#phi_{z}}(XML) - #delta_{#phi_{z}}(report) (mrad)")          
    else:
        if phi:
            hx.SetXTitle("#delta_{#phi}/#sigma_{#phi} position")
        else:
            hx.SetXTitle("#delta_{x'}/#sigma_{x'}")
        hy.SetXTitle("#delta_{y'}/#sigma_{y'}")
        hz.SetXTitle("#delta_{z'}/#sigma_{z'}")
        hphix.SetXTitle("#delta_{#phi_{x}}/#sigma_{#phi_{x}}")
        hphiy.SetXTitle("#delta_{#phi_{y}}/#sigma_{#phi_{y}}")
        hphiz.SetXTitle("#delta_{#phi_{z}}/#sigma_{#phi_{z}}")

    for h in hx, hy, hz, hphix, hphiy, hphiz:
        h.GetXaxis().CenterTitle()
        h.GetYaxis().CenterTitle()
        h.SetFillColor(color)
        h.SetLineStyle(style)

    if canvas is not None: c = canvas
    else: c = c1
    
    if normalized:
        fx = ROOT.TF1("fx", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hx.GetEntries()*2.*window/bins), -window, window)
        fy = ROOT.TF1("fy", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hy.GetEntries()*2.*window/bins), -window, window)
        fz = ROOT.TF1("fz", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hz.GetEntries()*2.*window/bins), -window, window)
        fphix = ROOT.TF1("fphix", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hphix.GetEntries()*2.*window/bins), -window, window)
        fphiy = ROOT.TF1("fphiy", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hphiy.GetEntries()*2.*window/bins), -window, window)
        fphiz = ROOT.TF1("fphiz", "%g * exp(-x**2/2.)/sqrt(2.*3.1415926)" % (hphiz.GetEntries()*2.*window/bins), -window, window)
        for f in fx, fy, fz, fphix, fphiy, fphiz:
            f.SetLineWidth(2)
            f.SetLineColor(ROOT.kBlue)
        for h, f in (hx, fx), (hy, fy), (hz, fz), (hphix, fphix), (hphiy, fphiy), (hphiz, fphiz):
            h.SetAxisRange(0, 1.1*max(h.GetMaximum(), f.GetMaximum()), "Y")
        
        c.Clear()
        c.Divide(3, 2)
        c.GetPad(1).cd(); hx.Draw(); fx.Draw("same")
        c.GetPad(2).cd(); hy.Draw(); fy.Draw("same")
        c.GetPad(3).cd(); hz.Draw(); fz.Draw("same")
        c.GetPad(4).cd(); hphix.Draw(); fphix.Draw("same")
        c.GetPad(5).cd(); hphiy.Draw(); fphiy.Draw("same")
        c.GetPad(6).cd(); hphiz.Draw(); fphiz.Draw("same")
        return hx, hy, hz, hphix, hphiy, hphiz, fx, fy, fz, fphix, fphiy, fphiz
    else:
        nvar = 6

        c.Clear()
        if nvar == 4: c.Divide(2, 2)
        if nvar == 6: c.Divide(3, 2)
        c.GetPad(1).cd(); hx.Draw()
        c.GetPad(2).cd(); hy.Draw()
        if nvar == 4:
          c.GetPad(3).cd(); hphiy.Draw()
          c.GetPad(4).cd(); hphiz.Draw()
        if nvar == 6:
          c.GetPad(3).cd(); hz.Draw()
          c.GetPad(4).cd(); hphix.Draw()
          c.GetPad(5).cd(); hphiy.Draw()
          c.GetPad(6).cd(); hphiz.Draw()

        if inlog:
          if hx.GetEntries()>0:    c.GetPad(1).SetLogy(1)
          if hy.GetEntries()>0:    c.GetPad(2).SetLogy(1)
          if nvar == 4:
            if hphiy.GetEntries()>0: c.GetPad(3).SetLogy(1)
            if hphiz.GetEntries()>0: c.GetPad(4).SetLogy(1)
          if nvar == 6:
            if hz.GetEntries()>0:    c.GetPad(3).SetLogy(1)
            if hphix.GetEntries()>0: c.GetPad(4).SetLogy(1)
            if hphiy.GetEntries()>0: c.GetPad(5).SetLogy(1)
            if hphiz.GetEntries()>0: c.GetPad(6).SetLogy(1)

        return hx, hy, hz, hphix, hphiy, hphiz



def DBMCVersus(quantity, versus, database, reports, window=10., selection=None, color=ROOT.kBlack):
    return DBdiffVersus(quantity, versus, database, None, reports, None, window, selection, color)

def DBdiffVersus(quantity, versus, database1, database2, reports1, reports2, windwselection=None, color=ROOT.kBlack):
    tdrStyle.SetOptStat("")

    domain = []
    values = []
    errors = []
        
    for r1 in reports1:
        if selection is None or (selection.__code__.co_argcount == len(r1.postal_address) and selection(*r1.postal_address)):
            if reports2 is None:
                r2 = Report(r1.chamberId, r1.postal_address, r1.name)
                r2.add_parameters(ValErr(0., 0., 0.), ValErr(0., 0., 0.), ValErr(0., 0., 0.), 
                                  ValErr(0., 0., 0.), ValErr(0., 0., 0.), ValErr(0., 0., 0.), 0., 0., 0.)
            else:
                found = False
                for r2 in reports2:
                    if r1.postal_address == r2.postal_address:
                        found = True
                        break
                if not found: continue

            found = False
            if r1.postal_address[0] == "DT":
                if r1.postal_address[1:] in database1.dt:
                    found = True
                    db1 = database1.dt[r1.postal_address[1:]]
                    if database2 is None:
                        db2 = DTAlignable()
                        db2.x = db2.y = db2.z = db2.phix = db2.phiy = db2.phiz = 0.
                        db2.xx = db2.xy = db2.xz = db2.yx = db2.yy = db2.yz = db2.zx = db2.zy = db2.zz = 0.
                    else:
                        db2 = database2.dt[r1.postal_address[1:]]
            else:
                if r1.postal_address[1:] in database1.csc:
                    found = True
                    db1 = database1.csc[r1.postal_address[1:]]
                    if database2 is None:
                        db2 = CSCAlignable()
                        db2.x = db2.y = db2.z = db2.phix = db2.phiy = db2.phiz = 0.
                        db2.xx = db2.xy = db2.xz = db2.yx = db2.yy = db2.yz = db2.zx = db2.zy = db2.zz = 0.
                    else:
                        db2 = database2.csc[r1.postal_address[1:]]

            if found and r1.status == "PASS" and r2.status == "PASS":
                okay = False

                if quantity == "phi":
                    if r1.deltax is not None and r2.deltax is not None and r1.deltax.error is not None and \
                       r2.deltax.error is not None and (r1.deltax.error**2 + r2.deltax.error**2) > 0.:
                        okay = True
                        values.append((db1.x - db2.x)/
                                      signConventions[r1.postal_address][3] * 1000. * signConventions[r1.postal_address][0])
                        errors.append((r1.deltax.error**2 + r2.deltax.error**2)/
                                      signConventions[r1.postal_address][3] * 1000. * signConventions[r1.postal_address][0])

                elif quantity == "x":
                    if r1.deltax is not None and r2.deltax is not None and r1.deltax.error is not None and \
                       r2.deltax.error is not None and (r1.deltax.error**2 + r2.deltax.error**2) > 0.:
                        okay = True
                        values.append((db1.x - db2.x) * 10. * signConventions[r1.postal_address][0])
                        errors.append((r1.deltax.error**2 + r2.deltax.error**2) * 10. * signConventions[r1.postal_address][0])

                elif quantity == "y":
                    if r1.deltay is not None and r2.deltay is not None and r1.deltay.error is not None and \
                       r2.deltay.error is not None and (r1.deltay.error**2 + r2.deltay.error**2) > 0.:
                        okay = True
                        values.append((db1.y - db2.y) * 10. * signConventions[r1.postal_address][1])
                        errors.append((r1.deltay.error**2 + r2.deltay.error**2) * 10. * signConventions[r1.postal_address][1])

                elif quantity == "z":
                    if r1.deltaz is not None and r2.deltaz is not None and r1.deltaz.error is not None and \
                       r2.deltaz.error is not None and (r1.deltaz.error**2 + r2.deltaz.error**2) > 0.:
                        okay = True
                        values.append((db1.z - db2.z) * 10. * signConventions[r1.postal_address][2])
                        errors.append((r1.deltaz.error**2 + r2.deltaz.error**2) * 10. * signConventions[r1.postal_address][2])

                elif quantity == "phix":
                    if r1.deltaphix is not None and r2.deltaphix is not None and r1.deltaphix.error is not None and \
                       r2.deltaphix.error is not None and (r1.deltaphix.error**2 + r2.deltaphix.error**2) > 0.:
                        okay = True
                        values.append((db1.phix - db2.phix) * 1000.)
                        errors.append((r1.deltaphix.error**2 + r2.deltaphix.error**2) * 1000.)

                elif quantity == "phiy":
                    if r1.deltaphiy is not None and r2.deltaphiy is not None and r1.deltaphiy.error is not None and \
                       r2.deltaphiy.error is not None and (r1.deltaphiy.error**2 + r2.deltaphiy.error**2) > 0.:
                        okay = True
                        values.append((db1.phiy - db2.phiy) * 1000.)
                        errors.append((r1.deltaphiy.error**2 + r2.deltaphiy.error**2) * 1000.)

                elif quantity == "phiz":
                    if r1.deltaphiz is not None and r2.deltaphiz is not None and r1.deltaphiz.error is not None and \
                       r2.deltaphiz.error is not None and (r1.deltaphiz.error**2 + r2.deltaphiz.error**2) > 0.:
                        okay = True
                        values.append((db1.phiz - db2.phiz) * 1000.)
                        errors.append((r1.deltaphiz.error**2 + r2.deltaphiz.error**2) * 1000.)

                else: raise Exception

                if okay:
                    if versus == "r": domain.append(signConventions[r1.postal_address][3])
                    elif versus == "phi": domain.append(signConventions[r1.postal_address][4])
                    elif versus == "z": domain.append(signConventions[r1.postal_address][5])
                    else: raise Exception

    if versus == "r":
        bkgndhist = ROOT.TH1F("bkgndhist", "", 100, 0., 800.)
        bkgndhist.SetXTitle("R (cm)")
    elif versus == "phi":
        bkgndhist = ROOT.TH1F("bkgndhist", "", 100, -pi, pi)
        bkgndhist.SetXTitle("#phi (rad)")
    elif versus == "z":
        bkgndhist = ROOT.TH1F("bkgndhist", "", 100, -1100., 1100.)
        bkgndhist.SetXTitle("z (cm)")
    bkgndhist.GetXaxis().CenterTitle()

    bkgndhist.SetAxisRange(-window, window, "Y")
    if quantity == "phi": bkgndhist.SetYTitle("#delta_{#phi} position (mrad)")
    elif quantity == "x": bkgndhist.SetYTitle("#delta_{x'} (mm)")
    elif quantity == "y": bkgndhist.SetYTitle("#delta_{y'} (mm)")
    elif quantity == "z": bkgndhist.SetYTitle("#delta_{z'} (mm)")
    elif quantity == "phix": bkgndhist.SetYTitle("#delta_{#phi_{x}} (mrad)")
    elif quantity == "phiy": bkgndhist.SetYTitle("#delta_{#phi_{y}} (mrad)")
    elif quantity == "phiz": bkgndhist.SetYTitle("#delta_{#phi_{z}} (mrad)")
    else: raise Exception
    bkgndhist.GetYaxis().CenterTitle()

    if len(domain) == 0:
        tgraph = ROOT.TGraphErrors(0)
    else:
        tgraph = ROOT.TGraphErrors(len(domain), array.array("d", domain), array.array("d", values), 
                                                array.array("d", [0.]*len(domain)), array.array("d", errors))
    tgraph.SetMarkerColor(color)
    tgraph.SetLineColor(color)

    bkgndhist.Draw()
    if tgraph.GetN() > 0: tgraph.Draw("p")
    return bkgndhist, tgraph, domain, values, errors

######################################################################################################

def idToPostalAddress(id):
  # only len==9 ids can correspond to valid postal address
  if len(id)!=9: return None
  if id[0:2]=="MB":
    #print id
    pa = ("DT", int(id[2:4]), int(id[5]), int(id[7:9]))
    #print pa
    if pa[1]<-2 or pa[1]>2: return None
    if pa[2]>4: return None
    if pa[3]<1 or pa[3]>14 or (pa[3]==4 and pa[3]>12): return None
    return pa
  elif id[0:2]=="ME":
    if id[2]=="+": ec=1
    elif id[2]=="-": ec=2
    else: return None
    pa = ("CSC", ec, int(id[3]), int(id[5]), int(id[7:9]))
    if pa[2]<1 or pa[2]>4: return None
    if pa[3]<1 or pa[3]>4 or (pa[2]>1 and pa[3]>2): return None
    if pa[4]<1 or pa[4]>36 or (pa[2]>1 and pa[3]==1 and pa[4]>18): return None
    return pa
  else: return None


def postalAddressToId(postal_address):
  if postal_address[0] == "DT":
    wheel, station, sector = postal_address[1:]
    w = "%+d"%wheel
    if w=="+0": w = "-0"
    return "MB%s/%d/%02d" % (w, station, sector)
  elif postal_address[0] == "CSC":
    endcap, station, ring, chamber = postal_address[1:]
    if endcap != 1: station = -1 * abs(station)
    return "ME%+d/%d/%02d" % (station, ring, chamber)


def nameToId(name):
  if name[0:2] == "MB":
    wh = name[4]
    if   wh == "A": w = "-2"
    elif wh == "B": w = "-1"
    elif wh == "C": w = "-0"
    elif wh == "D": w = "+1"
    elif wh == "E": w = "+2"
    else: return ""
    station = name[7]
    sector = name[11:13]
    return "MB%s/%s/%s" % (w, station, sector)
  elif name[0:2] == "ME":
    if name[2]=="p": endcap = "+"
    elif name[2]=="m": endcap = "-"
    else: return ""
    station = name[3]
    ring = name[4]
    chamber = name[6:8]
    return "ME%s%s/%s/%s" % (endcap, station, ring, chamber)
  return None


def availableCellsDT(reports):
  dts = []
  # DT wheels
  for iwheel in DT_TYPES:
    if iwheel[1]=="ALL": continue
    dts.append(iwheel[0])
  # DT wheel & station
  for wheel in DT_TYPES:
    if wheel[1]=="ALL": continue
    for station in wheel[2]:
      dts.append(wheel[0]+'/'+station[1])
  # DT station & sector 
  for wheel in DT_TYPES:
    if wheel[1]!="ALL": continue
    for station in wheel[2]:
      for sector in range(1,station[2]+1):
        ssector = "%02d" % sector
        dts.append(wheel[0]+'/'+station[1]+'/'+ssector)
  # DT station & ALL sectors 
  for wheel in DT_TYPES:
    if wheel[1]!="ALL": continue
    for station in wheel[2]:
        dts.append(wheel[0]+'/'+station[1])
  # DT chambers
  for wheel in DT_TYPES:
    if wheel[1]=="ALL": continue
    for station in wheel[2]:
      for sector in range(1,station[2]+1):
        ssector = "%02d" % sector
        label = "MBwh%sst%ssec%s" % (wheelLetter(int(wheel[1])),station[1],ssector)
        if len(reports)==0:
          # no reports case: do not include chambers 
          #dts.append(wheel[0]+'/'+station[1]+'/'+ssector)
          continue
        found = False
        for r in reports:
          if r.name == label:
            found = True
            break
        if not found: continue
        if r.status == "TOOFEWHITS" and r.posNum+r.negNum==0: continue
        if r.status == "NOFIT": continue
        dts.append(wheel[0]+'/'+station[1]+'/'+ssector)
  return dts


def availableCellsCSC(reports):
  cscs = []
  # CSC station
  for endcap in CSC_TYPES:
    for station in endcap[2]:
      cscs.append("%s%s" % (endcap[0], station[1]))
  # CSC station & ring 
  for endcap in CSC_TYPES:
    for station in endcap[2]:
      for ring in station[2]:
        if ring[1]=="ALL": continue
        #label = "CSCvsphi_me%s%s%s" % (endcap[1], station[1], ring[1])
        cscs.append("%s%s/%s" % (endcap[0], station[1],ring[1]))
  # CSC station and chamber
  for endcap in CSC_TYPES:
    for station in endcap[2]:
      for ring in station[2]:
        if ring[1]!="ALL": continue
        for chamber in range(1,ring[2]+1):
          #label = "CSCvsr_me%s%sch%02d" % (endcap[1], station[1], chamber)
          cscs.append("%s%s/ALL/%02d" % (endcap[0], station[1],chamber))
  # CSC station and ALL chambers
  for endcap in CSC_TYPES:
    for station in endcap[2]:
      for ring in station[2]:
        if ring[1]!="ALL": continue
        #label = "CSCvsr_me%s%schALL" % (endcap[1], station[1])
        cscs.append("%s%s/ALL" % (endcap[0], station[1]))
  # CSC chambers
  for endcap in CSC_TYPES:
    for station in endcap[2]:
      for ring in station[2]:
        if ring[1]=="ALL": continue
        for chamber in range(1,ring[2]+1):
          # exclude non instrumented ME4/2 
          if station[1]=="4" and ring[1]=="2":
            if endcap[1]=="m": continue
            if chamber<9 or chamber>13: continue
          schamber = "%02d" % chamber
          label = "ME%s%s%s_%s" % (endcap[1], station[1], ring[1], schamber)
          if len(reports)==0:
            # no reports case: do not include chambers 
            #cscs.append(endcap[0]+station[1]+'/'+ring[1]+'/'+schamber)
            continue
          found = False
          for r in reports:
            if r.name == label:
              found = True
              break
          if not found: continue
          if r.status == "TOOFEWHITS" and r.posNum+r.negNum==0: continue
          if r.status == "NOFIT": continue
          cscs.append(endcap[0]+station[1]+'/'+ring[1]+'/'+schamber)
  return cscs


DQM_SEVERITY = [
  {"idx":0, "name": "NONE", "color": "lightgreen", "hex":"#90EE90"},
  {"idx":1, "name": "LOWSTAT05", "color": "lightgreen", "hex":"#96D953"},
  {"idx":2, "name": "LOWSTAT075", "color": "lightgreen", "hex":"#94E26F"},
  {"idx":3, "name": "LOWSTAT1", "color": "yellowgreen", "hex":"#9ACD32"},
  {"idx":4, "name": "LOWSTAT", "color": "yellow", "hex":"#FFFF00"},
  {"idx":5, "name": "TOLERABLE", "color": "lightpink", "hex":"#FFB6C1"},
  {"idx":6, "name": "SEVERE", "color": "orange", "hex":"#FFA500"},
  {"idx":7, "name": "CRITICAL", "color": "red", "hex":"#FF0000"}];


def addToTestResults(c,res):
  if len(res)>0:
    if c in TEST_RESULTS: TEST_RESULTS[c].extend(res)
    else: TEST_RESULTS[c] = res


def testEntry(testID,scope,descr,severity):
  s = 0
  for sev in DQM_SEVERITY:
    if sev["name"]==severity: s = sev["idx"]
  return {"testID":testID,"scope":scope,"descr":descr,"severity":s}


def testZeroWithin5Sigma(x):
  if abs(x[1])==0.: return 0.
  pull = abs(x[0])/abs(x[1])
  if pull <= 5: return 0.
  else: return pull


def testDeltaWithin5Sigma(x,sx):
  n = len(x)
  res = []
  dr = []
  #print x
  #print sx
  for i in range(1,n+1):
    x1 = x[i-1]
    sx1 = sx[i-1]
    x2 = x[0]
    sx2 = sx[0]
    if i<n: 
      x2 = x[i]
      sx2 = sx[i]
    sig1 = sqrt( (sx1[0]-sx1[1])**2 + x1[1]**2 )
    sig2 = sqrt( (sx2[0]-sx2[1])**2 + x2[1]**2 )
    df = abs(x1[0]-x2[0]) - 3*( sig1 + sig2 )
    #df = abs(sx1[1]-sx2[0]) - 5*(abs(x1[1]) + abs(x2[1]))
    #print i, df, '= abs(',sx1[1],'-',sx2[0],')-5*(abs(',x1[1],')+abs(',x2[1],'))'
    dr.append(df)
    if df > 0: res.append(i)
  #print dr
  #print res
  return res


def doTestsForReport(cells,reports):
  for c in cells:
    # can a cell be converted to a chamber postal address?
    postal_address = idToPostalAddress(c)
    if not postal_address: continue
    
    # is this chamber in _report?
    found = False
    for r in reports:
      if r.postal_address == postal_address:
        found = True
        break
    if not found: continue

    # chamber's tests result
    res = []
    scope = postal_address[0]
    
    # noting could be done if fitting fails
    if r.status == "FAIL" or r.status == "MINUITFAIL":
      res.append(testEntry("FAILURE",scope,r.status+" failure","CRITICAL"))
      addToTestResults(c,res)
      continue

    # noting could be done if TOOFEWHITS
    nseg = r.posNum + r.negNum
    if r.status == "TOOFEWHITS" and nseg>0:
      res.append(testEntry("LOW_STAT",scope,"low stat, #segments=%d"%nseg,"LOWSTAT"))
      addToTestResults(c,res)
      continue

    # set shades of light green according to sidma(dx)
    sdx = 10.*r.deltax.error
    if sdx>0.5:
      if sdx<0.75: res.append(testEntry("LOW_STAT_DDX05",scope,"low stat, delta(dx)=%f #segments=%d" % (sdx,nseg),"LOWSTAT05"))
      elif sdx<1.: res.append(testEntry("LOW_STAT_DDX075",scope,"low stat, delta(dx)=%f #segments=%d" % (sdx,nseg),"LOWSTAT075"))
      else: res.append(testEntry("LOW_STAT_DDX1",scope,"low stat, delta(dx)=%f #segments=%d" % (sdx,nseg),"LOWSTAT1"))

    # check chi2
    if r.redchi2 > 20.: #2.5:
      res.append(testEntry("BIG_CHI2",scope,"chi2=%f>20" % r.redchi2,"TOLERABLE"))

    # check medians
    medx, meddx = 10.*r.median_x, 1000.*r.median_dxdz
    #medy, meddy = 10.*r.median_y, 1000.*r.median_dydz
    if medx>2:  res.append(testEntry("BIG_MED_X",scope,"median dx=%f>2 mm"%medx,"SEVERE"))
    #if medy>3: res.append(testEntry("BIG_MED_Y",scope,"median dy=%f>3 mm"%medy,"SEVERE"))
    if meddx>2: res.append(testEntry("BIG_MED_DXDZ",scope,"median d(dx/dz)=%f>2 mrad"%meddx,"SEVERE"))
    #if meddy>3: res.append(testEntry("BIG_MED_DYDZ",scope,"median d(dy/dz)=%f>3 mrad"%meddy,"SEVERE"))

    # check residuals far from zero
    isDTst4 = False
    if postal_address[0] == "DT" and postal_address[2]==4: isDTst4 = True
    dx, dy, dpy, dpz = 10.*r.deltax.value, 0., 1000.*r.deltaphiy.value, 1000.*r.deltaphiz.value
    if not isDTst4: dy = 10.*r.deltay.value
    if dx>0.2:   res.append(testEntry("BIG_LAST_ITR_DX",scope,"dx=%f>0.2 mm"%dx,"CRITICAL"))
    if dy>0.2:   res.append(testEntry("BIG_LAST_ITR_DY",scope,"dy=%f>0.2 mm"%dy,"CRITICAL"))
    if dpy>0.2:   res.append(testEntry("BIG_LAST_ITR_DPHIY",scope,"dphiy=%f>0.2 mrad"%dpy,"CRITICAL"))
    if dpz>0.2:   res.append(testEntry("BIG_LAST_ITR_DPHIZ",scope,"dphiz=%f>0.2 mrad"%dpz,"CRITICAL"))
    #if ddx>0.03: res.append(testEntry("BIG_DX",scope,"dphix=%f>0.03 mrad"%ddx,"CRITICAL"))

    addToTestResults(c,res)


def doTestsForMapPlots(cells):
  for c in cells:
    res = []
    
    scope = "zzz"
    if c[0:2]=="MB": scope = "DT"
    if c[0:2]=="ME": scope = "CSC"
    if scope == "zzz":
      print "strange cell ID: ", c
      return None
    
    if c in MAP_RESULTS_FITSIN:
      t = MAP_RESULTS_FITSIN[c]
      t_a = testZeroWithin5Sigma(t['a'])
      t_s = testZeroWithin5Sigma(t['sin'])
      t_c = testZeroWithin5Sigma(t['cos'])
      if t_a+t_s+t_c >0:
        descr = "map fitsin 5 sigma away from 0; pulls : a=%.2f sin=%.2f, cos=%.2f" % (t_a,t_s,t_c)
        res.append(testEntry("MAP_FITSIN",scope,descr,"SEVERE"))
    
    if c in MAP_RESULTS_SAWTOOTH:
      t = MAP_RESULTS_SAWTOOTH[c]
      
      t_a = testDeltaWithin5Sigma(t['a'],t['da'])
      if len(t_a)>0:
        descr = "map discontinuities: %s" % ",".join(map(str,t_a))
        res.append(testEntry("MAP_DISCONTIN",scope,descr,"SEVERE"))

      t_b = map(testZeroWithin5Sigma, t['b'])
      t_bi = []
      for i in range(0,len(t_b)):
        if t_b[i]>0: t_bi.append(i+1)
      if len(t_bi)>0:
        descr = "map sawteeth: %s" % ",".join(map(str,t_bi))
        res.append(testEntry("MAP_SAWTEETH",scope,descr,"TOLERABLE"))

    addToTestResults(c,res)


def saveTestResultsMap(run_name):
  if len(MAP_RESULTS_SAWTOOTH)+len(MAP_RESULTS_FITSIN)==0: return None
  ff = open("tmp_test_results_map__%s.pkl" % run_name, "wb")
  pickle.dump(MAP_RESULTS_SAWTOOTH, ff)
  pickle.dump(MAP_RESULTS_FITSIN, ff)
  ff.close()


def loadTestResultsMap(run_name):
  print "tmp_test_results_map__%s.pkl" % run_name, os.access("tmp_test_results_map__%s.pkl" % run_name,os.F_OK)
  if not os.access("tmp_test_results_map__%s.pkl" % run_name,os.F_OK): return None
  global MAP_RESULTS_FITSIN, MAP_RESULTS_SAWTOOTH
  ff = open("tmp_test_results_map__%s.pkl" % run_name, "rb")
  MAP_RESULTS_SAWTOOTH = pickle.load(ff)
  MAP_RESULTS_FITSIN = pickle.load(ff)
  ff.close()
  #execfile("tmp_test_results_map__%s.py" % run_name)
  #print 'asasas', MAP_RESULTS_FITSIN
  return True

def writeDQMReport(fname_dqm, run_name):
  tests = []
  for c in TEST_RESULTS:
    tests.append({"objID":c, "name":c, "list":TEST_RESULTS[c]})
  lt = time.localtime(time.time())
  lts = "%04d-%02d-%02d %02d:%02d:%02d %s" % (lt[0], lt[1], lt[2], lt[3], lt[4], lt[5], time.tzname[1])
  dqm_report = {"run":run_name, "genDate": lts, "report":tests}
  ff = open(fname_dqm,mode="w")
  print >>ff, "var DQM_REPORT = "
  json.dump(dqm_report,ff)
  #print >>ff, "];"
  ff.close()


def doTests(reports, pic_ids, fname_base, fname_dqm, run_name):
  # find available baseline
  dts = []
  cscs = []
  if len(reports)>0:
    dts  = availableCellsDT(reports)
    cscs = availableCellsCSC(reports)
  elif len(pic_ids)>0:
    dts  = [id for id in pic_ids if 'MB' in id]
    cscs = [id for id in pic_ids if 'ME' in id]
  mulist = ['Run: '+run_name,['ALL',['MU']],['DT',dts],['CSC',cscs]]
  ff = open(fname_base,mode="w")
  print >>ff, "var MU_LIST = ["
  json.dump(mulist,ff)
  print >>ff, "];"
  ff.close()
  
  doTestsForReport(dts,reports)
  doTestsForReport(cscs,reports)
  
  loadTestResultsMap(run_name)
  doTestsForMapPlots(dts)
  doTestsForMapPlots(cscs)
  
  writeDQMReport(fname_dqm, run_name)


######################################################################################################

def plotmedians(reports1, reports2, selection=None, binsx=100, windowx=5., ceilingx=None, binsy=100, windowy=5., 
                ceilingy=None, binsdxdz=100, windowdxdz=5., ceilingdxdz=None, binsdydz=100, windowdydz=5., ceilingdydz=None, 
                r1text=" before", r2text=" after", which="median"):
    tdrStyle.SetOptStat("emrou")
    tdrStyle.SetStatW(0.40)
    tdrStyle.SetStatFontSize(0.05)

    global hmediandxdz_after, hmediandxdz_before, hmediandxdz_beforecopy, \
           hmediandydz_after, hmediandydz_before, hmediandydz_beforecopy, \
           hmedianx_after, hmedianx_before, hmedianx_beforecopy, \
           hmediany_after, hmediany_before, hmediany_beforecopy, tlegend 

    hmedianx_before = ROOT.TH1F("hmedianx_before", "", binsx, -windowx, windowx)
    hmediany_before = ROOT.TH1F("hmediany_before", "", binsy, -windowy, windowy)
    hmediandxdz_before = ROOT.TH1F("hmediandxdz_before", "", binsdxdz, -windowdxdz, windowdxdz)
    hmediandydz_before = ROOT.TH1F("hmediandydz_before", "", binsdydz, -windowdydz, windowdydz)
    hmedianx_after = ROOT.TH1F("hmedianx_after", "", binsx, -windowx, windowx)
    hmediany_after = ROOT.TH1F("hmediany_after", "", binsy, -windowy, windowy)
    hmediandxdz_after = ROOT.TH1F("hmediandxdz_after", "", binsdxdz, -windowdxdz, windowdxdz)
    hmediandydz_after = ROOT.TH1F("hmediandydz_after", "", binsdydz, -windowdydz, windowdydz)

    if which == "median":
        whichx = whichy = whichdxdz = whichdydz = "median"
    elif which == "bigmean":
        whichx = "mean30"
        whichy = "mean30"
        whichdxdz = "mean20"
        whichdydz = "mean50"
    elif which == "mean":
        whichx = "mean15"
        whichy = "mean15"
        whichdxdz = "mean10"
        whichdydz = "mean25"
    elif which == "bigwmean":
        whichx = "wmean30"
        whichy = "wmean30"
        whichdxdz = "wmean20"
        whichdydz = "wmean50"
    elif which == "wmean":
        whichx = "wmean15"
        whichy = "wmean15"
        whichdxdz = "wmean10"
        whichdydz = "wmean25"
    elif which == "bigstdev":
        whichx = "stdev30"
        whichy = "stdev30"
        whichdxdz = "stdev20"
        whichdydz = "stdev50"
    elif which == "stdev":
        whichx = "stdev15"
        whichy = "stdev15"
        whichdxdz = "stdev10"
        whichdydz = "stdev25"
    else:
        raise Exception(which + " not recognized")

    for r1 in reports1:
        if selection is None or (selection.__code__.co_argcount == len(r1.postal_address) and selection(*r1.postal_address)):
            found = False
            for r2 in reports2:
                if r1.postal_address == r2.postal_address:
                    found = True
                    break
            if not found: continue

            #skip ME1/1a
            if r1.postal_address[0]=='CSC':
              if r1.postal_address[2]==1 and r1.postal_address[3]==4: continue

            if r1.status == "PASS" and r2.status == "PASS":
                hmedianx_before.Fill(10.*eval("r1.%s_x" % whichx))
                hmediandxdz_before.Fill(1000.*eval("r1.%s_dxdz" % whichdxdz))
                hmedianx_after.Fill(10.*eval("r2.%s_x" % whichx))
                hmediandxdz_after.Fill(1000.*eval("r2.%s_dxdz" % whichdxdz))

                if r1.median_y is not None:
                    hmediany_before.Fill(10.*eval("r1.%s_y" % whichy))
                    hmediandydz_before.Fill(1000.*eval("r1.%s_dydz" % whichdydz))
                    hmediany_after.Fill(10.*eval("r2.%s_y" % whichy))
                    hmediandydz_after.Fill(1000.*eval("r2.%s_dydz" % whichdydz))

    hmedianx_beforecopy = hmedianx_before.Clone()
    hmediany_beforecopy = hmediany_before.Clone()
    hmediandxdz_beforecopy = hmediandxdz_before.Clone()
    hmediandydz_beforecopy = hmediandydz_before.Clone()
    hmedianx_beforecopy.SetLineStyle(2)
    hmediany_beforecopy.SetLineStyle(2)
    hmediandxdz_beforecopy.SetLineStyle(2)
    hmediandydz_beforecopy.SetLineStyle(2)

    hmedianx_before.SetFillColor(ROOT.kMagenta+2)
    hmediany_before.SetFillColor(ROOT.kMagenta+2)
    hmediandxdz_before.SetFillColor(ROOT.kMagenta+2)
    hmediandydz_before.SetFillColor(ROOT.kMagenta+2)
    hmedianx_after.SetFillColor(ROOT.kYellow)
    hmediany_after.SetFillColor(ROOT.kYellow)
    hmediandxdz_after.SetFillColor(ROOT.kYellow)
    hmediandydz_after.SetFillColor(ROOT.kYellow)

    hmedianx_after.SetXTitle("median(#Deltax) (mm)")
    hmediany_after.SetXTitle("median(#Deltay) (mm)")
    hmediandxdz_after.SetXTitle("median(#Deltadx/dz) (mrad)")
    hmediandydz_after.SetXTitle("median(#Deltadydz) (mrad)")
    hmedianx_after.GetXaxis().CenterTitle()
    hmediany_after.GetXaxis().CenterTitle()
    hmediandxdz_after.GetXaxis().CenterTitle()
    hmediandydz_after.GetXaxis().CenterTitle()

    if ceilingx is not None: hmedianx_after.SetAxisRange(0., ceilingx, "Y")
    if ceilingy is not None: hmediany_after.SetAxisRange(0., ceilingy, "Y")
    if ceilingdxdz is not None: hmediandxdz_after.SetAxisRange(0., ceilingdxdz, "Y")
    if ceilingdydz is not None: hmediandydz_after.SetAxisRange(0., ceilingdydz, "Y")

    c1.Clear()
    c1.Divide(2, 2)

    c1.GetPad(1).cd()
    hmedianx_after.Draw()
    hmedianx_before.Draw("same")
    hmedianx_after.Draw("same")
    hmedianx_beforecopy.Draw("same")
    hmedianx_after.Draw("axissame")

    tlegend = ROOT.TLegend(0.17, 0.75-0.05, 0.45+0.05, 0.9)
    tlegend.SetFillColor(ROOT.kWhite)
    tlegend.SetBorderSize(0)
    tlegend.AddEntry(hmedianx_after, r2text, "f")
    tlegend.AddEntry(hmedianx_before, r1text, "f")
    tlegend.Draw()

    c1.GetPad(2).cd()
    hmediandxdz_after.Draw()
    hmediandxdz_before.Draw("same")
    hmediandxdz_after.Draw("same")
    hmediandxdz_beforecopy.Draw("same")
    hmediandxdz_after.Draw("axissame")

    c1.GetPad(3).cd()
    hmediany_after.Draw()
    hmediany_before.Draw("same")
    hmediany_after.Draw("same")
    hmediany_beforecopy.Draw("same")
    hmediany_after.Draw("axissame")

    c1.GetPad(4).cd()
    hmediandydz_after.Draw()
    hmediandydz_before.Draw("same")
    hmediandydz_after.Draw("same")
    hmediandydz_beforecopy.Draw("same")
    hmediandydz_after.Draw("axissame")

    return hmediandxdz_after, hmediandxdz_before, hmediandxdz_beforecopy, \
           hmediandydz_after, hmediandydz_before, hmediandydz_beforecopy, \
           hmedianx_after,    hmedianx_before, hmedianx_beforecopy, \
           hmediany_after,    hmediany_before, hmediany_beforecopy, tlegend 


######################################################################################################

def createPeaksProfile(the2d, rebin=1):
  htmp = ROOT.gROOT.FindObject(the2d.GetName()+"_peaks")
  if htmp != None: htmp.Delete()

  hpeaks = the2d.ProjectionX(the2d.GetName()+"_peaks")
  hpeaks.Reset()
  hpeaks.Rebin(rebin)
  bad_fit_bins = []
  for i in xrange(0, int(the2d.GetNbinsX()), rebin):
    tmp = the2d.ProjectionY("tmp", i+1, i + rebin)
    nn = tmp.GetEntries()

    drange = tmp.GetRMS()
    drange = 2.*drange
    fgaus = ROOT.TF1("fgaus","gaus", tmp.GetMean() - drange, tmp.GetMean() + drange)
    fgaus.SetParameter(0,nn)
    fgaus.SetParameter(1,tmp.GetMean())
    fgaus.SetParameter(2,tmp.GetRMS())
    #print "  ", i, nn, tmp.GetMean() , drange, "[", tmp.GetMean() - drange, tmp.GetMean() + drange, ']'

    fitOk = False
    if nn > 10:     # good to fit
      fr = tmp.Fit("fgaus","RNSQ")
      #print "       ", fgaus.GetParameter(1), " +- ", fgaus.GetParError(1), "   fitres = " , fr.Status() , fr.CovMatrixStatus()
      hpeaks.SetBinContent(i/rebin+1, fgaus.GetParameter(1))
      hpeaks.SetBinError(i/rebin+1, fgaus.GetParError(1))
      if fr.Status()==0 and fr.CovMatrixStatus()==3 : fitOk = True
    if not fitOk:
      bad_fit_bins.append(i/rebin+1)
      if nn > 1. and tmp.GetRMS() > 0: # use mean
        hpeaks.SetBinContent(i/rebin+1, tmp.GetMean())
        hpeaks.SetBinError(i/rebin+1, ROOT.TMath.StudentQuantile(0.841345,nn) * tmp.GetRMS() / sqrt(nn))
      else:
        hpeaks.SetBinContent(i/rebin+1, 0.)
        hpeaks.SetBinError(i/rebin+1, 0.)
  if len(bad_fit_bins): print "createPeaksProfile bad fit bins: ", bad_fit_bins
  return hpeaks


######################################################################################################

def mapplot(tfiles, name, param, mode="from2d", window=10., abscissa=None, title="", 
            widebins=False, fitsine=False, fitline=False, reset_palette=False, fitsawteeth=False, fitpeaks=False, peaksbins=1, fixfitpars={}, **args):
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(0)
    tdrStyle.SetOptStat(0)
    #tdrStyle.SetOptStat("emrou")
    tdrStyle.SetOptFit(0)
    tdrStyle.SetTitleFontSize(0.05)
    tdrStyle.SetPadRightMargin(0.1) # to see the pallete labels on the left
    
    c1.Clear()
    c1.ResetAttPad()
    
    if reset_palette: set_palette("blues")
    global hist, hist2d, hist2dweight, tline1, tline2, tline3
    
    if fitsine or fitsawteeth:
        id = mapNameToId(name)
        if not id:
            print "bad id for ", name
            raise Exception
    
    hdir = "AlignmentMonitorMuonSystemMap1D/iter1/"
    hpref= "%s_%s" % (name, param)
    hhh  = hdir+hpref
    
    combine_all = False
    if "ALL" in name  and  ("CSCvsr" in name  or  "DTvsz" in name): combine_all = True
    
    add1d =  ("vsphi" in name) and (param == "x")

    if "h2d" in args:
      hist2d = args["h2d"].Clone(hpref+"_2d_")
      if "CSC" in name  and  add1d: hist1d = args["h1d"].Clone(hpref+"_1d_")

    elif combine_all:
      nch = 12
      if  "DT" in name  and  name[6:9]=='st4': nch = 14
      if  "CSC" in name: nch = 36
      chambers = ["%02d" % ch for ch in range (2,nch+1)]

      ch_hhh = hhh.replace('ALL','01')
      ch_hpref = hpref.replace('ALL','01')
      hist2d = tfiles[0].Get(ch_hhh+"_2d").Clone(ch_hpref+"_2d_")
      if "CSC" in name  and  add1d: hist1d = tfiles[0].Get(ch_hhh+"_1d").Clone(ch_hpref+"_1d_")
      
      for ch in chambers:
        ch_hhh = hhh.replace('ALL',ch)
        ch_hpref = hpref.replace('ALL',ch)
        hist2d.Add(tfiles[0].Get(ch_hhh+"_2d"))
        if "CSC" in name  and  add1d: hist1d.Add(tfiles[0].Get(ch_hhh+"_1d"))
        for tfile in tfiles[1:]:
          hist2d.Add(tfile.Get(ch_hhh+"_2d"))
          if "CSC" in name   and  add1d: hist1d.Add(tfile.Get(ch_hhh+"_1d"))
    
    else:
      hist2d = tfiles[0].Get(hhh+"_2d").Clone(hpref+"_2d_")
      if "CSC" in name  and  add1d: hist1d = tfiles[0].Get(hhh+"_1d").Clone(hpref+"_1d_")
      for tfile in tfiles[1:]:
        hist2d.Add(tfile.Get(hhh+"_2d"))
        if "CSC" in name   and  add1d: hist1d.Add(tfile.Get(hhh+"_1d"))

    
    if mode == "from2d":
        the2d = hist2d
        
        hist = the2d.ProjectionX()
        hist.Reset()
        
        skip = 1
        if widebins:
            hist.Rebin(10)
            skip = 10

        #f = ROOT.TF1("g", "gaus", -40., 40)
        for i in xrange(0, int(the2d.GetNbinsX()), skip):
            tmp = the2d.ProjectionY("tmp", i+1, i + skip)
            if tmp.GetEntries() > 1:
                #tmp.Fit("g","LNq")
                hist.SetBinContent(i/skip+1, tmp.GetMean())
                hist.SetBinError(i/skip+1, ROOT.TMath.StudentQuantile(0.841345,tmp.GetEntries()) * tmp.GetRMS() / sqrt(tmp.GetEntries()))
                #hist.SetBinError(i/skip+1, tmp.GetRMS() / sqrt(tmp.GetEntries()))
                #hist.SetBinError(i/skip+1, f.GetParameter(2))
            else:
                #hist.SetBinContent(i/skip+1, 2000.)
                #hist.SetBinError(i/skip+1, 1000.)
                hist.SetBinContent(i/skip+1, 0.)
                hist.SetBinError(i/skip+1, 0.)

        hpeaks = createPeaksProfile(the2d, peaksbins)

    else:
        raise Exception

    hist.SetAxisRange(-window, window, "Y")
    if abscissa is not None: hist.SetAxisRange(abscissa[0], abscissa[1], "X")
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.75)
    hist.GetXaxis().CenterTitle()
    hist.GetYaxis().CenterTitle()
    hist.GetYaxis().SetTitleOffset(0.75)
    hist.GetXaxis().SetTitleSize(0.05)
    hist.GetYaxis().SetTitleSize(0.05)
    hist.SetTitle(title)
    if "vsphi" in name: hist.SetXTitle("Global #phi position (rad)")
    elif "vsz" in name: hist.SetXTitle("Global z position (cm)")
    elif "vsr" in name: hist.SetXTitle("Global R position (cm)")
    if "DT" in name:
        if param == "x": hist.SetYTitle("x' residual (mm)")
        if param == "dxdz": hist.SetYTitle("dx'/dz residual (mrad)")
        if param == "y": hist.SetYTitle("y' residual (mm)")
        if param == "dydz": hist.SetYTitle("dy'/dz residual (mrad)")
    if "CSC" in name:
        if param == "x": hist.SetYTitle("r#phi residual (mm)")
        if param == "dxdz": hist.SetYTitle("d(r#phi)/dz residual (mrad)")
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    hist.Draw()
    hist2d.Draw("colzsame")
    if widebins: hist.Draw("samee1")
    else: hist.Draw("same")

    hpeaks.SetMarkerStyle(20)
    hpeaks.SetMarkerSize(0.9)
    hpeaks.SetMarkerColor(ROOT.kRed)
    hpeaks.SetLineColor(ROOT.kRed)
    hpeaks.SetLineWidth(2)
    #if fitpeaks: hpeaks.Draw("same")
    hpeaks.Draw("same")

    if fitsine and "vsphi" in name:
        global fitsine_const, fitsine_sin, fitsine_cos, fitsine_chi2, fitsine_ndf
        if 'CSC' in name:
          f = ROOT.TF1("f", "[0] + [1]*sin(x) + [2]*cos(x)", -pi/180.*5., pi*(2.-5./180.))
        else:
          f = ROOT.TF1("f", "[0] + [1]*sin(x) + [2]*cos(x)", -pi, pi)
        f.SetLineColor(ROOT.kRed)
        f.SetLineWidth(2)
        if len(fixfitpars)>0:
          for fpar in fixfitpars.keys():
            f.FixParameter(fpar, fixfitpars[fpar])
        #hist.Fit(f,"N")
        if fitpeaks: hpeaks.Fit(f,"NQ")
        else: hist.Fit(f,"NEQ")
        if len(fixfitpars)>0:
          for fpar in fixfitpars.keys():
            f.ReleaseParameter(fpar)
        fitsine_const = f.GetParameter(0), f.GetParError(0)
        fitsine_sin = f.GetParameter(1), f.GetParError(1)
        fitsine_cos = f.GetParameter(2), f.GetParError(2)
        fitsine_chi2 = f.GetChisquare()
        fitsine_ndf = f.GetNDF()
        global MAP_RESULTS_FITSIN
        # 'phi' coefficienct will be updated further for CSC
        MAP_RESULTS_FITSIN[id] = {'a':fitsine_const, 'phi':fitsine_const, 'sin': fitsine_sin, 'cos': fitsine_cos, 'chi2': fitsine_chi2, 'ndf': fitsine_ndf}
        f.Draw("same")
        global fitsine_ttext, fitsine_etext
        text_xposition = -1.
        if 'CSC' in name: text_xposition = 2.
        fitsine_ttext = ROOT.TLatex(text_xposition, 0.8*window, 
                "%+.3f %+.3f sin#phi %+.3f cos#phi" % (fitsine_const[0], fitsine_sin[0], fitsine_cos[0]))
        fitsine_ttext.SetTextColor(ROOT.kRed)
        fitsine_ttext.SetTextSize(0.05)
        fitsine_ttext.Draw()
        fitsine_etext = ROOT.TLatex(text_xposition, 0.70*window, 
                " #pm%.3f    #pm%.3f            #pm%.3f" % (fitsine_const[1], fitsine_sin[1], fitsine_cos[1]))
        fitsine_etext.SetTextColor(ROOT.kRed)
        fitsine_etext.SetTextSize(0.045)
        fitsine_etext.Draw()

        # additional estimate of phiz ring rotation from 1d distribution
        if 'CSC' in name and add1d:
          # zero-order rough fit to obtain the fitting range:
          f0 = ROOT.TF1("f0", "gaus", hist1d.GetBinLowEdge(1), -hist1d.GetBinLowEdge(1))
          fit = hist1d.Fit(f0,"NRQ")
          rangea, rangeb = hist1d.GetMean() - hist1d.GetRMS(), hist1d.GetMean() + hist1d.GetRMS()
          if fit==0: rangea, rangeb = f0.GetParameter(1) - f0.GetParameter(2), f0.GetParameter(1) + f0.GetParameter(2)
          #print rangea, rangeb
          
          # second fit for finding the peak:
          f1 = ROOT.TF1("f1", "gaus", rangea, rangeb)
          fit = hist1d.Fit(f1,"NRQ")
          nn = hist1d.GetEntries()
          dphiz, ephiz = 0, 0
          if nn>0:  dphiz, ephiz = hist1d.GetMean(), ROOT.TMath.StudentQuantile(0.841345,nn) * hist1d.GetRMS() / sqrt(nn)
          if fit==0: dphiz, ephiz = f1.GetParameter(1), f1.GetParError(1)
          #print dphiz, ephiz
          MAP_RESULTS_FITSIN[id]['phi'] = (dphiz, ephiz)
          
          global ttex_sine_, ttex_sine, ttex_1d_, ttex_1d
          postal_address = idToPostalAddress(id+'/01')
          ttex_sine_ = ROOT.TLatex(0, 0.8*window,"#Delta#phi_{z}^{sine} (mrad):")
          ttex_sine_.SetTextColor(ROOT.kGreen+2); ttex_sine_.SetTextSize(0.04); ttex_sine_.Draw()
          ttex_sine = ROOT.TLatex(0, 0.7*window,"   %+.3f#pm%.3f" %
                                  (-100*fitsine_const[0]/signConventions[postal_address][3], 
                                   100*fitsine_const[1]/signConventions[postal_address][3]))
          ttex_sine.SetTextColor(ROOT.kGreen+2); ttex_sine.SetTextSize(0.04); ttex_sine.Draw()
          ttex_1d_ = ROOT.TLatex(0, 0.6*window,"#Delta#phi_{z}^{phi} (mrad):")
          ttex_1d_.SetTextColor(ROOT.kGreen+2); ttex_1d_.SetTextSize(0.04); ttex_1d_.Draw()
          ttex_1d = ROOT.TLatex(0, 0.5*window,"   %+.3f#pm%.3f" % (-dphiz, ephiz))
          ttex_1d.SetTextColor(ROOT.kGreen+2); ttex_1d.SetTextSize(0.04); ttex_1d.Draw()
          ROOT.gPad.Update()

    if fitline:
        f = ROOT.TF1("f", "[0] + [1]*x", -1000., 1000.)
        hist2d.Fit(f, "q")
        hist2d.GetFunction("f").SetLineColor(ROOT.kRed)
        global fitline_const, fitline_linear, fitline_chi2, fitline_ndf
        fitline_const = hist2d.GetFunction("f").GetParameter(0), hist2d.GetFunction("f").GetParError(0)
        fitline_linear = hist2d.GetFunction("f").GetParameter(1), hist2d.GetFunction("f").GetParError(1)
        fitline_chi2 = hist2d.GetFunction("f").GetChisquare()
        fitline_ndf = hist2d.GetFunction("f").GetNDF()
        hist2d.GetFunction("f").Draw("same")
        global fitline_ttext
        if "vsz" in name: which = "Z"
        elif "vsr" in name: which = "R"
        fitline_ttext = ROOT.TText(hist.GetBinCenter(hist.GetNbinsX()/4), 
                0.8*window, "%.3g %+.3g %s" % (fitline_const[0], fitline_linear[0], which))
        fitline_ttext.SetTextColor(ROOT.kRed)
        fitline_ttext.Draw()

    ROOT.gPad.RedrawAxis()

    if "vsphi" in name: 
        if not widebins: philines(name, window, abscissa)
        if abscissa is None:
          if 'CSC' in name:
            tline1 = ROOT.TLine(-pi/180.*5., 0, pi*(2.-5./180.), 0); tline1.Draw()
            tline2 = ROOT.TLine(-pi/180.*5., -window, pi*(2.-5./180.), -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(-pi/180.*5., window, pi*(2.-5./180.), window); tline3.Draw()
          else:
            tline1 = ROOT.TLine(-pi, 0, pi, 0); tline1.Draw()
            tline2 = ROOT.TLine(-pi, -window, pi, -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(-pi, window, pi, window); tline3.Draw()
        else:
            tline1 = ROOT.TLine(abscissa[0], 0, abscissa[1], 0); tline1.Draw()
            tline2 = ROOT.TLine(abscissa[0], -window, abscissa[1], -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(abscissa[0], window, abscissa[1], window); tline3.Draw()
    elif "vsz" in name:
        if not widebins: zlines(window, abscissa)
        if abscissa is None:
            tline1 = ROOT.TLine(-660, 0, 660, 0); tline1.Draw()
            tline2 = ROOT.TLine(-660, -window, 660, -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(-660, window, 660, window); tline3.Draw()
        else:
            tline1 = ROOT.TLine(abscissa[0], 0, abscissa[1], 0); tline1.Draw()
            tline2 = ROOT.TLine(abscissa[0], -window, abscissa[1], -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(abscissa[0], window, abscissa[1], window); tline3.Draw()
    elif "vsr" in name:
        if "mem1" in name or "mep1" in name and not widebins: rlines(1, window, abscissa)
        if "mem2" in name or "mep2" in name and not widebins: rlines(2, window, abscissa)
        if "mem3" in name or "mep3" in name and not widebins: rlines(3, window, abscissa)
        if "mem4" in name or "mep4" in name and not widebins: rlines(4, window, abscissa)
        if abscissa is None:
            tline1 = ROOT.TLine(100, 0, 700, 0); tline1.Draw()
            tline2 = ROOT.TLine(100, -window, 700, -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(100, window, 700, window); tline3.Draw()
        else:
            tline1 = ROOT.TLine(abscissa[0], 0, abscissa[1], 0); tline1.Draw()
            tline2 = ROOT.TLine(abscissa[0], -window, abscissa[1], -window); tline2.SetLineWidth(2); tline2.Draw()
            tline3 = ROOT.TLine(abscissa[0], window, abscissa[1], window); tline3.Draw()

    if "vsphi" in name and fitsawteeth:
        global CPP_LOADED
        if not CPP_LOADED:
            phiedges2c()
            ROOT.gROOT.ProcessLine(".L phiedges_fitfunctions.C++")
            CPP_LOADED = True
        fn={0: ROOT.fitf0,
            1: ROOT.fitf2,
            2: ROOT.fitf2,
            3: ROOT.fitf3,
            4: ROOT.fitf4,
            5: ROOT.fitf5,
            6: ROOT.fitf6,
            7: ROOT.fitf7,
            8: ROOT.fitf8,
            9: ROOT.fitf9,
            10: ROOT.fitf10,
            11: ROOT.fitf11,
            12: ROOT.fitf12,
            13: ROOT.fitf13
        } [stationIndex(name)]
        fn.SetNpx(5000)
        fn.SetLineColor(ROOT.kYellow)
        hist.Fit(fn,"N")
        fn.Draw("same")

        # get properly arranged phi edges
        edges = (phiedges[stationIndex(name)])[:]
        ed = sorted(edges)
        # add some padding to the end
        ed.append(pi+abs(ed[0]))

        global sawtooth_a, sawtooth_b
        sawtooth_a = []
        sawtooth_da = []
        sawtooth_b = []
        for pr in range(0,fn.GetNpar(),2):
            sawtooth_a.append( (fn.GetParameter(pr), fn.GetParError(pr)) )
            sawtooth_b.append( (fn.GetParameter(pr+1), fn.GetParError(pr+1)) )
            sawtooth_da.append( (fn.Eval(ed[pr/2]+0.01), fn.Eval(ed[pr/2+1]-0.01)) )
        global MAP_RESULTS_SAWTOOTH
        MAP_RESULTS_SAWTOOTH[id] = {'a': sawtooth_a, 'da': sawtooth_da, 'b': sawtooth_b, 'chi2': fn.GetChisquare(), 'ndf': fn.GetNDF()}

    # fill number of contributiong bins
    
    
    #ROOT.SetOwnership(hist,False)
    ROOT.SetOwnership(hist2d,False)
    ROOT.SetOwnership(hist,False)
    ROOT.SetOwnership(tline1,False)
    ROOT.SetOwnership(tline2,False)
    ROOT.SetOwnership(tline3,False)
    return hist


def mapNameToId(name):
  if "DT" in name:
    wh = "-ALL"
    if name.find('wh')>1: wh = name[name.find('wh')+2]
    if   wh == "A": w = "-2"
    elif wh == "B": w = "-1"
    elif wh == "C": w = "-0"
    elif wh == "D": w = "+1"
    elif wh == "E": w = "+2"
    elif wh == "-ALL": w = "-ALL"
    else: return None
    station=''
    if wh == "-ALL": 
        if name.find('sec')<0: return None
        station = name[name.find('sec')-1]
        sector = ''
        sector = name[name.find('sec')+3:name.find('sec')+5]
        return "MB%s/%s/%s" % (w, station, sector)
    if name.find('st')>1: station = name[name.find('st')+2]
    else: return None
    return "MB%s/%s" % (w, station)
  elif "CSC" in name:
    p = name.find('me')
    if p<0: return None
    if name[p+2]=="p": endcap = "+"
    elif name[p+2]=="m": endcap = "-"
    else: return None
    station = name[p+3]
    pch = name.find('ch')
    if pch<0:
        ring = name[p+4]
        return "ME%s%s/%s" % (endcap, station, ring)
    ring = 'ALL'
    chamber = name[pch+2:pch+4]
    return "ME%s%s/%s/%s" % (endcap, station, ring, chamber)
  return None


##################################################################################
# "param" may be one of "deltax" (Delta x position residuals), 
#      "deltadxdz" (Delta (dx/dz) angular residuals), 
#      "curverr" (Delta x * d(Delta q/pT)/d(Delta x) = Delta q/pT in the absence of misalignment)

def curvatureplot(tfiles, name, param, mode="from2d", window=15., widebins=False, title="", fitgauss=False, fitconst=False, fitline=False, fitpeaks=True, reset_palette=False):
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(0)
    tdrStyle.SetOptStat(0)
    tdrStyle.SetOptFit(0)
    tdrStyle.SetTitleFontSize(0.05)

    c1.Clear()
    if reset_palette: set_palette("blues")
    global hist, histCOPY, hist2d, tline1, tline2, tline3, tline4, tline5

    hdir = "AlignmentMonitorMuonVsCurvature/iter1/"

    if name not in ("all", "top", "bottom"):
        hsuffix = "_%s_%s" % (name, param)
        prof = tfiles[0].Get(hdir+"tprofile"+hsuffix).Clone("tprofile_"+hsuffix)
        hist2d = tfiles[0].Get(hdir+"th2f"+hsuffix).Clone("th2f_"+hsuffix)
        for tfile in tfiles[1:]:
            prof.Add(tfile.Get(hdir+"tprofile"+hsuffix))
            hist2d.Add(tfile.Get(hdir+"th2f"+hsuffix))
    else:
        prof = None
        hist2d = None
        for wheel in "m2", "m1", "z", "p1", "p2":
            if name == "all": sectors = "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"
            elif name == "top": sectors = "01", "02", "03", "04", "05", "06"
            elif name == "bottom": sectors = "07", "08", "09", "10", "11", "12"
            else: raise Exception

            for sector in sectors:
                hsuffix = "_%s_%s" % ("wheel%s_sector%s" % (wheel, sector), param)
                for tfile in tfiles:
                    if prof is None:
                        prof = tfiles[0].Get(hdir+"tprofile"+hsuffix).Clone("tprofile_"+hsuffix)
                        hist2d = tfiles[0].Get(hdir+"th2f"+hsuffix).Clone("tprofile_"+hsuffix)
                    else:
                        prof.Add(tfile.Get(hdir+"tprofile"+hsuffix))
                        hist2d.Add(tfile.Get(hdir+"th2f"+hsuffix))

    hist = ROOT.TH1F("hist", "", prof.GetNbinsX(), prof.GetBinLowEdge(1), -prof.GetBinLowEdge(1))
    for i in xrange(1, prof.GetNbinsX()+1):
        hist.SetBinContent(i, prof.GetBinContent(i))
        hist.SetBinError(i, prof.GetBinError(i))

    if mode == "plain":
        hist = prof
    elif mode == "from2d":
        skip = 1
        if widebins:
            hist.Rebin(5)
            skip = 5
        htmp = ROOT.gROOT.FindObject("tmp")
        if htmp != None: htmp.Delete()

        for i in xrange(0, int(prof.GetNbinsX()), skip):
            tmp = hist2d.ProjectionY("tmp", i+1, i + skip)
            if tmp.GetEntries() > 1:
                hist.SetBinContent(i/skip+1, tmp.GetMean())
                hist.SetBinError(i/skip+1, ROOT.TMath.StudentQuantile(0.841345,tmp.GetEntries()) * tmp.GetRMS() / sqrt(tmp.GetEntries()))
                #hist.SetBinError(i/skip+1, tmp.GetRMS() / sqrt(tmp.GetEntries()))
            else:
                #hist.SetBinContent(i/skip+1, 2000.)
                #hist.SetBinError(i/skip+1, 1000.)
                hist.SetBinContent(i/skip+1, 0.)
                hist.SetBinError(i/skip+1, 0.)

        hpeaks = createPeaksProfile(hist2d, skip)

    else:
        raise Exception


    if fitgauss:
        f = ROOT.TF1("f", "[0] + [1]*exp(-x**2/2/0.01**2)", hist.GetBinLowEdge(1), -hist.GetBinLowEdge(1))
        f.SetParameters(0, 0., 0.01)
        if fitpeaks: hpeaks.Fit(f, "q")
        else: hist.Fit(f, "q")
        f.SetLineColor(ROOT.kRed)
        global fitgauss_diff, fitgauss_chi2, fitgauss_ndf
#         fitter = ROOT.TVirtualFitter.GetFitter()
#         fitgauss_diff = f.GetParameter(0) - f.GetParameter(1), \
#                         sqrt(f.GetParError(0)**2 + f.GetParError(1)**2 + 2.*fitter.GetCovarianceMatrixElement(0, 1))
        fitgauss_diff = f.GetParameter(1), f.GetParError(1)
        fitgauss_chi2 = f.GetChisquare()
        fitgauss_ndf = f.GetNDF()

    global fitline_intercept, fitline_slope
    if fitconst:
        f = ROOT.TF1("f", "[0]", hist.GetBinLowEdge(1), -hist.GetBinLowEdge(1))
        if fitpeaks: hpeaks.Fit(f, "q")
        else: hist.Fit(f, "q")
        f.SetLineColor(ROOT.kRed)
        fitline_intercept = f.GetParameter(0), f.GetParError(0)

    if fitline:
        f = ROOT.TF1("f", "[0] + [1]*x", hist.GetBinLowEdge(1), -hist.GetBinLowEdge(1))
        if fitpeaks: hpeaks.Fit(f, "qNE")
        else: hist.Fit(f, "qNE")
        f.SetLineColor(ROOT.kRed)
        global f2, f3
        f2 = ROOT.TF1("2", "[0] + [1]*x", hist.GetBinLowEdge(1), -hist.GetBinLowEdge(1))
        f3 = ROOT.TF1("2", "[0] + [1]*x", hist.GetBinLowEdge(1), -hist.GetBinLowEdge(1))
        f2.SetParameters(f.GetParameter(0), f.GetParameter(1) + f.GetParError(1))
        f3.SetParameters(f.GetParameter(0), f.GetParameter(1) - f.GetParError(1))
        f2.SetLineColor(ROOT.kRed)
        f3.SetLineColor(ROOT.kRed)
        f2.SetLineStyle(2)
        f3.SetLineStyle(2)
        fitline_intercept = f.GetParameter(0), f.GetParError(0)
        fitline_slope = f.GetParameter(1), f.GetParError(1)

    hist2d.SetAxisRange(-window, window, "Y")
    hist2d.SetMarkerStyle(20)
    hist2d.SetMarkerSize(0.75)
    hist2d.GetXaxis().CenterTitle()
    hist2d.GetYaxis().CenterTitle()
    if param == "curverr":
        hist2d.GetYaxis().SetTitleOffset(1.35)
    else:
        hist2d.GetYaxis().SetTitleOffset(0.75)
    hist2d.GetXaxis().SetTitleOffset(1.2)
    hist2d.GetXaxis().SetTitleSize(0.05)
    hist2d.GetYaxis().SetTitleSize(0.05)
    hist2d.SetTitle(title)
    if param == "pterr": hist2d.SetXTitle("qp_{T} (GeV/c)")
    else: hist2d.SetXTitle("q/p_{T} (c/GeV)")
    if param == "deltax": hist2d.SetYTitle("#Deltax' (mm)")
    if param == "deltadxdz": hist2d.SetYTitle("#Deltadx'/dz (mrad)")
    if param == "pterr": hist2d.SetYTitle("#Deltap_{T}/p_{T} (%)")
    if param == "curverr": hist2d.SetYTitle("#Deltaq/p_{T} (c/GeV)")
    hist2d.Draw("colz")
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    hist.Draw("same")
    #histCOPY = hist.Clone()
    #histCOPY.SetXTitle("")
    #histCOPY.SetYTitle("")

    #if widebins:
    #    histCOPY.Draw("samee1")
    #    histCOPY.Draw("sameaxis")
    #else:
    #    histCOPY.Draw("same")
    #    histCOPY.Draw("sameaxis")

    if fitline:
        f.Draw("same")
        #f2.Draw("same")
        #f3.Draw("same")

    hpeaks.SetMarkerStyle(20)
    hpeaks.SetMarkerSize(0.9)
    hpeaks.SetMarkerColor(ROOT.kRed)
    hpeaks.SetLineColor(ROOT.kRed)
    hpeaks.SetLineWidth(2)
    #if fitpeaks: hpeaks.Draw("same")
    hpeaks.Draw("same")

    #tline1 = ROOT.TLine(hist.GetBinLowEdge(1), -window, hist.GetBinLowEdge(1), window)
    #tline2 = ROOT.TLine(hist.GetBinLowEdge(1), window, -hist.GetBinLowEdge(1), window)
    #tline3 = ROOT.TLine(-hist.GetBinLowEdge(1), window, -hist.GetBinLowEdge(1), -window)
    #tline4 = ROOT.TLine(-hist.GetBinLowEdge(1), -window, hist.GetBinLowEdge(1), -window)
    tline5 = ROOT.TLine(-hist.GetBinLowEdge(1), 0., hist.GetBinLowEdge(1), 0.)
    tline5.Draw()
    #for t in tline1, tline2, tline3, tline4, tline5: t.Draw()


def curvatureDTsummary(tfiles, window=15., pdgSfactor=False):
    global h, gm2, gm1, gz, gp1, gp2, tlegend

    set_palette("blues")
    phis = {-2: [], -1: [], 0: [], 1: [], 2: []}
    diffs = {-2: [], -1: [], 0: [], 1: [], 2: []}
    differrs = {-2: [], -1: [], 0: [], 1: [], 2: []}
    for wheelstr, wheel in ("m2", "-2"), ("m1", "-1"), ("z", "0"), ("p1", "+1"), ("p2", "+2"):
        for sector in "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12":
            curvatureplot(tfiles, "wheel%s_sector%s" % (wheelstr, sector), "deltax", 
                    title="Wheel %s, sector %s" % (wheel, sector), fitgauss=True, reset_palette=False)
            if fitgauss_diff[1] < window:
                uncertainty = fitgauss_diff[1]
                if pdgSfactor and (fitgauss_chi2/fitgauss_ndf) > 1.: uncertainty *= sqrt(fitgauss_chi2/fitgauss_ndf)

                phis[int(wheel)].append(signConventions["DT", int(wheel), 1, int(sector)][4])
                diffs[int(wheel)].append(fitgauss_diff[0])
                differrs[int(wheel)].append(uncertainty)

    h = ROOT.TH1F("h", "", 1, -pi, pi)
    h.SetAxisRange(-window, window, "Y")
    h.SetXTitle("#phi (rad)")
    h.SetYTitle("#Deltax(p_{T} #rightarrow #infty) - #Deltax(p_{T} #rightarrow 0) (mm)")
    h.GetXaxis().CenterTitle()
    h.GetYaxis().CenterTitle()

    gm2 = ROOT.TGraphErrors(len(phis[-2]), array.array("d", phis[-2]), array.array("d", diffs[-2]), 
            array.array("d", [0.]*len(phis[-2])), array.array("d", differrs[-2]))
    gm1 = ROOT.TGraphErrors(len(phis[-1]), array.array("d", phis[-1]), array.array("d", diffs[-1]), 
            array.array("d", [0.]*len(phis[-1])), array.array("d", differrs[-1]))
    gz = ROOT.TGraphErrors(len(phis[0]), array.array("d", phis[0]), array.array("d", diffs[0]), 
            array.array("d", [0.]*len(phis[0])), array.array("d", differrs[0]))
    gp1 = ROOT.TGraphErrors(len(phis[1]), array.array("d", phis[1]), array.array("d", diffs[1]), 
            array.array("d", [0.]*len(phis[1])), array.array("d", differrs[1]))
    gp2 = ROOT.TGraphErrors(len(phis[2]), array.array("d", phis[2]), array.array("d", diffs[2]), 
            array.array("d", [0.]*len(phis[2])), array.array("d", differrs[2]))

    gm2.SetMarkerStyle(21); gm2.SetMarkerColor(ROOT.kRed); gm2.SetLineColor(ROOT.kRed)
    gm1.SetMarkerStyle(22); gm1.SetMarkerColor(ROOT.kBlue); gm1.SetLineColor(ROOT.kBlue)
    gz.SetMarkerStyle(3); gz.SetMarkerColor(ROOT.kBlack); gz.SetLineColor(ROOT.kBlack)
    gp1.SetMarkerStyle(26); gp1.SetMarkerColor(ROOT.kBlue); gp1.SetLineColor(ROOT.kBlue)
    gp2.SetMarkerStyle(25); gp2.SetMarkerColor(ROOT.kRed); gp2.SetLineColor(ROOT.kRed)

    h.Draw()
    tlegend = ROOT.TLegend(0.25, 0.2, 0.85, 0.5)
    tlegend.SetFillColor(ROOT.kWhite)
    tlegend.SetBorderSize(0)
    tlegend.AddEntry(gm2, "Wheel -2", "p")
    tlegend.AddEntry(gm1, "Wheel -1", "p")
    tlegend.AddEntry(gz, "Wheel 0", "p")
    tlegend.AddEntry(gp1, "Wheel +1", "p")
    tlegend.AddEntry(gp2, "Wheel +2", "p")
    tlegend.Draw()

    gm2.Draw("p")
    gm1.Draw("p")
    gz.Draw("p")
    gp1.Draw("p")
    gp2.Draw("p")


def getname(r):
    if r.postal_address[0] == "DT":
        wheel, station, sector = r.postal_address[1:]
        return "DT wheel %d, station %d, sector %d" % (wheel, station, sector)
    elif r.postal_address[0] == "CSC":
        endcap, station, ring, chamber = r.postal_address[1:]
        if endcap != 1: station = -1 * abs(station)
        return "CSC ME%d/%d chamber %d" % (station, ring, chamber)

ddt=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
def clearDDT():
    for i in range(0,15):
        ddt[i]=0.

def printDeltaTs():
    n = 0
    for t in ddt:
        if n==0 or n==7 or n==15: print "%d calls" % t
        else: print "%d : %0.3f ms" % (n,t*1000.0)
        n += 1

def bellcurves(tfile, reports, name, twobin=True, suppressblue=False):
    t1 = time.time()
    ddt[0] += 1
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.1)
    tdrStyle.SetOptStat(0)
    tdrStyle.SetHistMinimumZero()

    c1.Clear()
    c1.ResetAttPad()

    found = False
    for r in reports:
        if r.name == name:
            found = True
            break
    if not found: raise Exception("Not a valid name")
    if r.status == "FAIL":
        #raise Exception, "Fit failed"
        print "Fit failed"
        c1.Clear()
        return
    
    Pos = "Pos"; Neg = "Neg"
    if not twobin:
        Pos = ""; Neg = ""

    pdirPos = "MuonAlignmentFromReference/%s%s" % (name, Pos)
    pdirNeg = "MuonAlignmentFromReference/%s%s" % (name, Neg)

    t2 = time.time()
    ddt[1] = 1./ddt[0]*((ddt[0]-1)*ddt[1] + t2-t1)

    chamber_x = tfile.Get(pdirPos+"_x")
    chamber_x_fit = tfile.Get(pdirPos+"_x_fit")
    chamber_y = tfile.Get(pdirPos+"_y")
    chamber_y_fit = tfile.Get(pdirPos+"_y_fit")
    chamber_dxdz = tfile.Get(pdirPos+"_dxdz")
    chamber_dxdz_fit = tfile.Get(pdirPos+"_dxdz_fit")
    chamber_dydz = tfile.Get(pdirPos+"_dydz")
    chamber_dydz_fit = tfile.Get(pdirPos+"_dydz_fit")
    chamber_alphax = tfile.Get(pdirPos+"_alphax")
    chamber_alphax_fit = tfile.Get(pdirPos+"_alphax_fit")
    chamber_alphay = tfile.Get(pdirPos+"_alphay")
    chamber_alphay_fit = tfile.Get(pdirPos+"_alphay_fit")
    if twobin:
      chamber_x_fit2 = tfile.Get(pdirNeg+"_x_fit")
      chamber_y_fit2 = tfile.Get(pdirNeg+"_y_fit")
      chamber_dxdz_fit2 = tfile.Get(pdirNeg+"_dxdz_fit")
      chamber_dydz_fit2 = tfile.Get(pdirNeg+"_dydz_fit")
      chamber_alphax_fit2 = tfile.Get(pdirNeg+"_alphax_fit")
      chamber_alphay_fit2 = tfile.Get(pdirNeg+"_alphay_fit")

    if not chamber_x:
        chamber_x = tfile.Get(pdirPos+"_residual")
        chamber_x_fit = tfile.Get(pdirPos+"_residual_fit")
        chamber_dxdz = tfile.Get(pdirPos+"_resslope")
        chamber_dxdz_fit = tfile.Get(pdirPos+"_resslope_fit")
        chamber_alphax = tfile.Get(pdirPos+"_alpha")
        chamber_alphax_fit = tfile.Get(pdirPos+"_alpha_fit")
        if twobin:
          chamber_x_fit2 = tfile.Get(pdirNeg+"_residual_fit")
          chamber_dxdz_fit2 = tfile.Get(pdirNeg+"_resslope_fit")
          chamber_alphax_fit2 = tfile.Get(pdirNeg+"_alpha_fit")

    if not chamber_x:
        print "Can't find neither "+pdirPos+"_x  nor "+pdirPos+"_residual"
        return

    t3 = time.time()
    ddt[2] = 1./ddt[0]*((ddt[0]-1)*ddt[2] + t3-t2)

    ####
    chamber_x.SetAxisRange(-50., 50., "X")
    if chamber_x.GetRMS()>15: chamber_x.SetAxisRange(-75., 75., "X")
    chamber_dxdz.SetAxisRange(-30., 30., "X")
    chamber_alphax.SetAxisRange(-50., 50., "X")
    if not not chamber_y:
        chamber_y.SetAxisRange(-75., 75., "X")
        chamber_dydz.SetAxisRange(-120., 120., "X")
        chamber_alphay.SetAxisRange(-120., 120., "X")
        chamber_alphay.SetAxisRange(-75., 75., "Y")
    ####

    chamber_x.SetXTitle("Local x residual (mm)")
    chamber_dxdz.SetXTitle("Local dx/dz residual (mrad)")
    chamber_alphax.SetXTitle("Local dx/dz residual (mrad)")
    chamber_alphax.SetYTitle("Local x residual (mm)")
    if not not chamber_y:
        chamber_y.SetXTitle("Local y residual (mm)")
        chamber_dydz.SetXTitle("Local dy/dz residual (mrad)")
        chamber_alphay.SetXTitle("Local dy/dz residual (mrad)")
        chamber_alphay.SetYTitle("Local y residual (mm)")
    if name[0:2] == "ME":
        chamber_x.SetXTitle("Local r#phi residual (mm)")
        chamber_dxdz.SetXTitle("Local d(r#phi)/dz residual (mrad)")
        chamber_alphax.SetXTitle("Local d(r#phi)/dz residual (mrad)")
        chamber_alphax.SetYTitle("Local r#phi residual (mm)")

    t4 = time.time()
    ddt[3] = 1./ddt[0]*((ddt[0]-1)*ddt[3] + t4-t3)

    for h in chamber_x, chamber_dxdz, chamber_alphax, chamber_alphax, \
             chamber_y, chamber_dydz, chamber_alphay, chamber_alphay:
        if not not h:
            h.GetXaxis().CenterTitle()
            h.GetYaxis().CenterTitle()
            h.GetXaxis().SetLabelSize(0.05)
            h.GetYaxis().SetLabelSize(0.05)
            h.GetXaxis().SetTitleSize(0.07)
            h.GetYaxis().SetTitleSize(0.07)
            h.GetXaxis().SetTitleOffset(0.9)
            h.GetYaxis().SetTitleOffset(0.9)

    if twobin:
      for f in chamber_x_fit2, chamber_y_fit2, chamber_dxdz_fit2, chamber_dydz_fit2, \
               chamber_alphax_fit2, chamber_alphay_fit2:
          if not not f:
               f.SetLineColor(4)
    if not twobin:
        suppressblue = True

    t5 = time.time()
    ddt[4] = 1./ddt[0]*((ddt[0]-1)*ddt[4] + t5-t4)

    global l1, l2, l3, l4
    if not not chamber_y:
        c1.Clear()
        c1.Divide(3, 2)
        chamber_x.SetTitle(getname(r))

        c1.GetPad(1).cd()
        chamber_x.Draw()
        if not suppressblue: chamber_x_fit2.Draw("same")
        chamber_x_fit.Draw("same")
        l1 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_x_fit.GetParameter(1), chamber_x_fit.GetParError(1),
                         chamber_x_fit.GetParameter(2), chamber_x_fit.GetParError(2)))
        l1.Draw()

        c1.GetPad(2).cd()
        chamber_dxdz.Draw()
        if not suppressblue: chamber_dxdz_fit2.Draw("same")
        chamber_dxdz_fit.Draw("same")
        l2 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_dxdz_fit.GetParameter(1), chamber_dxdz_fit.GetParError(1),
                         chamber_dxdz_fit.GetParameter(2), chamber_dxdz_fit.GetParError(2)))
        l2.Draw()
        
        c1.GetPad(3).cd()
        chamber_alphax.Draw("col")
        if not suppressblue: chamber_alphax_fit2.Draw("same")
        chamber_alphax_fit.Draw("same")
        
        c1.GetPad(4).cd()
        chamber_y.Draw()
        if not suppressblue: chamber_y_fit2.Draw("same")
        chamber_y_fit.Draw("same")
        l3 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_y_fit.GetParameter(1), chamber_y_fit.GetParError(1),
                         chamber_y_fit.GetParameter(2), chamber_y_fit.GetParError(2)))
        l3.Draw()
        
        c1.GetPad(5).cd()
        chamber_dydz.Draw()
        if not suppressblue: chamber_dydz_fit2.Draw("same")
        chamber_dydz_fit.Draw("same")
        l4 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_dydz_fit.GetParameter(1), chamber_dydz_fit.GetParError(1),
                         chamber_dydz_fit.GetParameter(2), chamber_dydz_fit.GetParError(2)))
        l4.Draw()

        for lb in l1,l2,l3,l4:
          lb.SetNDC(1)
          lb.SetTextColor(ROOT.kRed)
        
        c1.GetPad(6).cd()
        chamber_alphay.Draw("col")
        if not suppressblue: chamber_alphay_fit2.Draw("same")
        chamber_alphay_fit.Draw("same")

    else:
        c1.Clear()
        c1.Divide(3, 1)
        chamber_x.SetTitle(getname(r))

        c1.GetPad(1).cd()
        chamber_x.Draw()
        if not suppressblue: chamber_x_fit2.Draw("same")
        chamber_x_fit.Draw("same")
        l1 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_x_fit.GetParameter(1), chamber_x_fit.GetParError(1),
                         chamber_x_fit.GetParameter(2), chamber_x_fit.GetParError(2)))
        l1.Draw()
        
        c1.GetPad(2).cd()
        chamber_dxdz.Draw()
        if not suppressblue: chamber_dxdz_fit2.Draw("same")
        chamber_dxdz_fit.Draw("same")
        l2 = ROOT.TLatex(0.67,0.8,"#splitline{#mu: %0.2f#pm%0.2f}{#sigma: %0.1f#pm%0.1f}" % (
                         chamber_dxdz_fit.GetParameter(1), chamber_dxdz_fit.GetParError(1),
                         chamber_dxdz_fit.GetParameter(2), chamber_dxdz_fit.GetParError(2)))
        l2.Draw()
        
        c1.GetPad(3).cd()
        chamber_alphax.Draw("col")
        if not suppressblue: chamber_alphax_fit2.Draw("same")
        chamber_alphax_fit.Draw("same")

        for lb in l1,l2:
          lb.SetNDC(1)
          lb.SetTextColor(ROOT.kRed)

    t6 = time.time()
    ddt[5] = 1./ddt[0]*((ddt[0]-1)*ddt[5] + t6-t5)
    ddt[6] = 1./ddt[0]*((ddt[0]-1)*ddt[6] + t6-t1)


def polynomials(tfile, reports, name, twobin=True, suppressblue=False):
    t1 = time.time()
    ddt[7] += 1
    global label1, label2, label3, label4, label5, label6, label7, label8, label9
    plotDirectory = "MuonAlignmentFromReference"
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.1)
    tdrStyle.SetOptStat(0)

    c1.Clear()
    c1.ResetAttPad()

    found = False
    for r in reports:
        if r.name == name:
            found = True
            break
    if not found: raise Exception("Not a valid name")

    if r.status == "FAIL":
        #raise Exception, "Fit failed"
        print "Fit failed"
        c1.Clear()
        return

    Pos = "Pos"; Neg = "Neg"
    if not twobin:
        Pos = ""; Neg = ""

    pdirPos = "MuonAlignmentFromReference/%s%s" % (name, Pos)
    pdirNeg = "MuonAlignmentFromReference/%s%s" % (name, Neg)

    global chamber_x_trackx, chamber_x_trackx_fit, chamber_y_trackx, chamber_y_trackx_fit, \
        chamber_dxdz_trackx, chamber_dxdz_trackx_fit, chamber_dydz_trackx, chamber_dydz_trackx_fit, \
        chamber_x_trackx_fit2, chamber_y_trackx_fit2, chamber_dxdz_trackx_fit2, chamber_dydz_trackx_fit2
    global chamber_x_tracky, chamber_x_tracky_fit, chamber_y_tracky, chamber_y_tracky_fit, \
        chamber_dxdz_tracky, chamber_dxdz_tracky_fit, chamber_dydz_tracky, chamber_dydz_tracky_fit, \
        chamber_x_tracky_fit2, chamber_y_tracky_fit2, chamber_dxdz_tracky_fit2, chamber_dydz_tracky_fit2
    global chamber_x_trackdxdz, chamber_x_trackdxdz_fit, chamber_y_trackdxdz, chamber_y_trackdxdz_fit, \
        chamber_dxdz_trackdxdz, chamber_dxdz_trackdxdz_fit, chamber_dydz_trackdxdz, chamber_dydz_trackdxdz_fit, \
        chamber_x_trackdxdz_fit2, chamber_y_trackdxdz_fit2, chamber_dxdz_trackdxdz_fit2, chamber_dydz_trackdxdz_fit2
    global chamber_x_trackdydz, chamber_x_trackdydz_fit, chamber_y_trackdydz, chamber_y_trackdydz_fit, \
        chamber_dxdz_trackdydz, chamber_dxdz_trackdydz_fit, chamber_dydz_trackdydz, chamber_dydz_trackdydz_fit, \
        chamber_x_trackdydz_fit2, chamber_y_trackdydz_fit2, chamber_dxdz_trackdydz_fit2, chamber_dydz_trackdydz_fit2

    chamber_x_trackx = tfile.Get(pdirPos+"_x_trackx")
    chamber_x_trackx_fit = tfile.Get(pdirPos+"_x_trackx_fitline")
    chamber_y_trackx = tfile.Get(pdirPos+"_y_trackx")
    chamber_y_trackx_fit = tfile.Get(pdirPos+"_y_trackx_fitline")
    chamber_dxdz_trackx = tfile.Get(pdirPos+"_dxdz_trackx")
    chamber_dxdz_trackx_fit = tfile.Get(pdirPos+"_dxdz_trackx_fitline")
    chamber_dydz_trackx = tfile.Get(pdirPos+"_dydz_trackx")
    chamber_dydz_trackx_fit = tfile.Get(pdirPos+"_dydz_trackx_fitline")
    chamber_x_trackx_fit2 = tfile.Get(pdirNeg+"_x_trackx_fitline")
    chamber_y_trackx_fit2 = tfile.Get(pdirNeg+"_y_trackx_fitline")
    chamber_dxdz_trackx_fit2 = tfile.Get(pdirNeg+"_dxdz_trackx_fitline")
    chamber_dydz_trackx_fit2 = tfile.Get(pdirNeg+"_dydz_trackx_fitline")

    chamber_x_tracky = tfile.Get(pdirPos+"_x_tracky")
    chamber_x_tracky_fit = tfile.Get(pdirPos+"_x_tracky_fitline")
    chamber_y_tracky = tfile.Get(pdirPos+"_y_tracky")
    chamber_y_tracky_fit = tfile.Get(pdirPos+"_y_tracky_fitline")
    chamber_dxdz_tracky = tfile.Get(pdirPos+"_dxdz_tracky")
    chamber_dxdz_tracky_fit = tfile.Get(pdirPos+"_dxdz_tracky_fitline")
    chamber_dydz_tracky = tfile.Get(pdirPos+"_dydz_tracky")
    chamber_dydz_tracky_fit = tfile.Get(pdirPos+"_dydz_tracky_fitline")
    chamber_x_tracky_fit2 = tfile.Get(pdirNeg+"_x_tracky_fitline")
    chamber_y_tracky_fit2 = tfile.Get(pdirNeg+"_y_tracky_fitline")
    chamber_dxdz_tracky_fit2 = tfile.Get(pdirNeg+"_dxdz_tracky_fitline")
    chamber_dydz_tracky_fit2 = tfile.Get(pdirNeg+"_dydz_tracky_fitline")

    chamber_x_trackdxdz = tfile.Get(pdirPos+"_x_trackdxdz")
    chamber_x_trackdxdz_fit = tfile.Get(pdirPos+"_x_trackdxdz_fitline")
    chamber_y_trackdxdz = tfile.Get(pdirPos+"_y_trackdxdz")
    chamber_y_trackdxdz_fit = tfile.Get(pdirPos+"_y_trackdxdz_fitline")
    chamber_dxdz_trackdxdz = tfile.Get(pdirPos+"_dxdz_trackdxdz")
    chamber_dxdz_trackdxdz_fit = tfile.Get(pdirPos+"_dxdz_trackdxdz_fitline")
    chamber_dydz_trackdxdz = tfile.Get(pdirPos+"_dydz_trackdxdz")
    chamber_dydz_trackdxdz_fit = tfile.Get(pdirPos+"_dydz_trackdxdz_fitline")
    chamber_x_trackdxdz_fit2 = tfile.Get(pdirNeg+"_x_trackdxdz_fitline")
    chamber_y_trackdxdz_fit2 = tfile.Get(pdirNeg+"_y_trackdxdz_fitline")
    chamber_dxdz_trackdxdz_fit2 = tfile.Get(pdirNeg+"_dxdz_trackdxdz_fitline")
    chamber_dydz_trackdxdz_fit2 = tfile.Get(pdirNeg+"_dydz_trackdxdz_fitline")

    chamber_x_trackdydz = tfile.Get(pdirPos+"_x_trackdydz")
    chamber_x_trackdydz_fit = tfile.Get(pdirPos+"_x_trackdydz_fitline")
    chamber_y_trackdydz = tfile.Get(pdirPos+"_y_trackdydz")
    chamber_y_trackdydz_fit = tfile.Get(pdirPos+"_y_trackdydz_fitline")
    chamber_dxdz_trackdydz = tfile.Get(pdirPos+"_dxdz_trackdydz")
    chamber_dxdz_trackdydz_fit = tfile.Get(pdirPos+"_dxdz_trackdydz_fitline")
    chamber_dydz_trackdydz = tfile.Get(pdirPos+"_dydz_trackdydz")
    chamber_dydz_trackdydz_fit = tfile.Get(pdirPos+"_dydz_trackdydz_fitline")
    chamber_x_trackdydz_fit2 = tfile.Get(pdirNeg+"_x_trackdydz_fitline")
    chamber_y_trackdydz_fit2 = tfile.Get(pdirNeg+"_y_trackdydz_fitline")
    chamber_dxdz_trackdydz_fit2 = tfile.Get(pdirNeg+"_dxdz_trackdydz_fitline")
    chamber_dydz_trackdydz_fit2 = tfile.Get(pdirNeg+"_dydz_trackdydz_fitline")

    if not chamber_x_trackx:
        chamber_x_trackx = tfile.Get(pdirPos+"_residual_trackx")
        chamber_x_trackx_fit = tfile.Get(pdirPos+"_residual_trackx_fitline")
        chamber_dxdz_trackx = tfile.Get(pdirPos+"_resslope_trackx")
        chamber_dxdz_trackx_fit = tfile.Get(pdirPos+"_resslope_trackx_fitline")
        chamber_x_trackx_fit2 = tfile.Get(pdirNeg+"_residual_trackx_fitline")
        chamber_dxdz_trackx_fit2 = tfile.Get(pdirNeg+"_resslope_trackx_fitline")

        chamber_x_tracky = tfile.Get(pdirPos+"_residual_tracky")
        chamber_x_tracky_fit = tfile.Get(pdirPos+"_residual_tracky_fitline")
        chamber_dxdz_tracky = tfile.Get(pdirPos+"_resslope_tracky")
        chamber_dxdz_tracky_fit = tfile.Get(pdirPos+"_resslope_tracky_fitline")
        chamber_x_tracky_fit2 = tfile.Get(pdirNeg+"_residual_tracky_fitline")
        chamber_dxdz_tracky_fit2 = tfile.Get(pdirNeg+"_resslope_tracky_fitline")

        chamber_x_trackdxdz = tfile.Get(pdirPos+"_residual_trackdxdz")
        chamber_x_trackdxdz_fit = tfile.Get(pdirPos+"_residual_trackdxdz_fitline")
        chamber_dxdz_trackdxdz = tfile.Get(pdirPos+"_resslope_trackdxdz")
        chamber_dxdz_trackdxdz_fit = tfile.Get(pdirPos+"_resslope_trackdxdz_fitline")
        chamber_x_trackdxdz_fit2 = tfile.Get(pdirNeg+"_residual_trackdxdz_fitline")
        chamber_dxdz_trackdxdz_fit2 = tfile.Get(pdirNeg+"_resslope_trackdxdz_fitline")

        chamber_x_trackdydz = tfile.Get(pdirPos+"_residual_trackdydz")
        chamber_x_trackdydz_fit = tfile.Get(pdirPos+"_residual_trackdydz_fitline")
        chamber_dxdz_trackdydz = tfile.Get(pdirPos+"_resslope_trackdydz")
        chamber_dxdz_trackdydz_fit = tfile.Get(pdirPos+"_resslope_trackdydz_fitline")
        chamber_x_trackdydz_fit2 = tfile.Get(pdirNeg+"_residual_trackdydz_fitline")
        chamber_dxdz_trackdydz_fit2 = tfile.Get(pdirNeg+"_resslope_trackdydz_fitline")

    if not chamber_x_trackx:
        print "Can't find neither "+pdirPos+"_residual  nor "+pdirPos+"_residual_trackx"
        return

    chamber_x_trackx = chamber_x_trackx.Clone()
    chamber_dxdz_trackx = chamber_dxdz_trackx.Clone()
    chamber_x_tracky = chamber_x_tracky.Clone()
    chamber_dxdz_tracky = chamber_dxdz_tracky.Clone()
    chamber_x_trackdxdz = chamber_x_trackdxdz.Clone()
    chamber_dxdz_trackdxdz = chamber_dxdz_trackdxdz.Clone()
    chamber_x_trackdydz = chamber_x_trackdydz.Clone()
    chamber_dxdz_trackdydz = chamber_dxdz_trackdydz.Clone()

    if not not chamber_y_trackx:
        chamber_y_trackx = chamber_y_trackx.Clone()
        chamber_dydz_trackx = chamber_dydz_trackx.Clone()
        chamber_y_tracky = chamber_y_tracky.Clone()
        chamber_dydz_tracky = chamber_dydz_tracky.Clone()
        chamber_y_trackdxdz = chamber_y_trackdxdz.Clone()
        chamber_dydz_trackdxdz = chamber_dydz_trackdxdz.Clone()
        chamber_y_trackdydz = chamber_y_trackdydz.Clone()
        chamber_dydz_trackdydz = chamber_dydz_trackdydz.Clone()

    if not not chamber_y_trackx:
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_x_trackx")); chamber_x_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dxdz_trackx")); chamber_dxdz_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_x_tracky")); chamber_x_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dxdz_tracky")); chamber_dxdz_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_x_trackdxdz")); chamber_x_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dxdz_trackdxdz")); chamber_dxdz_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_x_trackdydz")); chamber_x_trackdydz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dxdz_trackdydz")); chamber_dxdz_trackdydz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_y_trackx")); chamber_y_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dydz_trackx")); chamber_dydz_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_y_tracky")); chamber_y_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dydz_tracky")); chamber_dydz_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_y_trackdxdz")); chamber_y_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dydz_trackdxdz")); chamber_dydz_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_y_trackdydz")); chamber_y_trackdydz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_dydz_trackdydz")); chamber_dydz_trackdydz.Merge(tlist)
    else:
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_residual_trackx")); chamber_x_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_resslope_trackx")); chamber_dxdz_trackx.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_residual_tracky")); chamber_x_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_resslope_tracky")); chamber_dxdz_tracky.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_residual_trackdxdz")); chamber_x_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_resslope_trackdxdz")); chamber_dxdz_trackdxdz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_residual_trackdydz")); chamber_x_trackdydz.Merge(tlist)
        tlist = ROOT.TList(); tlist.Add(tfile.Get(pdirNeg+"_resslope_trackdydz")); chamber_dxdz_trackdydz.Merge(tlist)

    rr1=10.
    rr2=10.
    chamber_x_trackx.SetAxisRange(-rr1, rr1, "Y")
    chamber_dxdz_trackx.SetAxisRange(-rr2, rr2, "Y")
    chamber_x_tracky.SetAxisRange(-rr1, rr1, "Y")
    chamber_dxdz_tracky.SetAxisRange(-rr2, rr2, "Y")
    chamber_x_trackdxdz.SetAxisRange(-rr1, rr1, "Y")
    chamber_dxdz_trackdxdz.SetAxisRange(-rr2, rr2, "Y")
    chamber_x_trackdydz.SetAxisRange(-rr1, rr1, "Y")
    chamber_dxdz_trackdydz.SetAxisRange(-rr2, rr2, "Y")

    rr3=10.
    if not not chamber_y_trackx:
        chamber_y_trackx.SetAxisRange(-rr3, rr3, "Y")
        chamber_dydz_trackx.SetAxisRange(-rr3, rr3, "Y")
        chamber_y_tracky.SetAxisRange(-rr3, rr3, "Y")
        chamber_dydz_tracky.SetAxisRange(-rr3, rr3, "Y")
        chamber_y_trackdxdz.SetAxisRange(-rr3, rr3, "Y")
        chamber_dydz_trackdxdz.SetAxisRange(-rr3, rr3, "Y")
        chamber_y_trackdydz.SetAxisRange(-rr3, rr3, "Y")
        chamber_dydz_trackdydz.SetAxisRange(-rr3, rr3, "Y")

    for h in chamber_x_trackx, chamber_y_trackx, chamber_dxdz_trackx, chamber_dydz_trackx, \
             chamber_x_tracky, chamber_y_tracky, chamber_dxdz_tracky, chamber_dydz_tracky, \
             chamber_x_trackdxdz, chamber_y_trackdxdz, chamber_dxdz_trackdxdz, chamber_dydz_trackdxdz, \
             chamber_x_trackdydz, chamber_y_trackdydz, chamber_dxdz_trackdydz, chamber_dydz_trackdydz:
        if not not h:
            h.SetMarkerStyle(20)
            h.SetMarkerSize(0.5)
            h.GetXaxis().SetLabelSize(0.12)
            h.GetYaxis().SetLabelSize(0.12)
            h.GetXaxis().SetNdivisions(505)
            h.GetYaxis().SetNdivisions(505)
            h.GetXaxis().SetLabelOffset(0.03)
            h.GetYaxis().SetLabelOffset(0.03)

    trackdxdz_minimum, trackdxdz_maximum = None, None
    for h in chamber_x_trackdxdz, chamber_y_trackdxdz, chamber_dxdz_trackdxdz, chamber_dydz_trackdxdz:
        if not not h:
            for i in xrange(1, h.GetNbinsX()+1):
                if h.GetBinError(i) > 0.01 and h.GetBinContent(i) - h.GetBinError(i) < 10. and \
                   h.GetBinContent(i) + h.GetBinError(i) > -10.:
                    if not trackdxdz_minimum or trackdxdz_minimum > h.GetBinCenter(i): 
                        trackdxdz_minimum = h.GetBinCenter(i)
                    if trackdxdz_maximum < h.GetBinCenter(i): 
                        trackdxdz_maximum = h.GetBinCenter(i)
    if not not trackdxdz_minimum and not not trackdxdz_maximum:
        for h in chamber_x_trackdxdz, chamber_y_trackdxdz, chamber_dxdz_trackdxdz, chamber_dydz_trackdxdz:
            if not not h:
                h.SetAxisRange(trackdxdz_minimum, trackdxdz_maximum, "X")

    trackdydz_minimum, trackdydz_maximum = None, None
    for h in chamber_x_trackdydz, chamber_y_trackdydz, chamber_dxdz_trackdydz, chamber_dydz_trackdydz:
        if not not h:
            for i in xrange(1, h.GetNbinsX()+1):
                if h.GetBinError(i) > 0.01 and h.GetBinContent(i) - h.GetBinError(i) < 10. and \
                   h.GetBinContent(i) + h.GetBinError(i) > -10.:
                    if not trackdydz_minimum or trackdydz_minimum > h.GetBinCenter(i): 
                        trackdydz_minimum = h.GetBinCenter(i)
                    if trackdydz_maximum < h.GetBinCenter(i): 
                        trackdydz_maximum = h.GetBinCenter(i)
    if not not trackdydz_minimum and not not trackdydz_maximum:
        for h in chamber_x_trackdydz, chamber_y_trackdydz, chamber_dxdz_trackdydz, chamber_dydz_trackdydz:
            if not not h:
                h.SetAxisRange(trackdydz_minimum, trackdydz_maximum, "X")

    for f in chamber_x_trackx_fit2, chamber_y_trackx_fit2, chamber_dxdz_trackx_fit2, chamber_dydz_trackx_fit2, \
             chamber_x_tracky_fit2, chamber_y_tracky_fit2, chamber_dxdz_tracky_fit2, chamber_dydz_tracky_fit2, \
             chamber_x_trackdxdz_fit2, chamber_y_trackdxdz_fit2, chamber_dxdz_trackdxdz_fit2, chamber_dydz_trackdxdz_fit2, \
             chamber_x_trackdydz_fit2, chamber_y_trackdydz_fit2, chamber_dxdz_trackdydz_fit2, chamber_dydz_trackdydz_fit2:
        if not not f:
            f.SetLineColor(4)

    if not not chamber_y_trackx:
        c1.Clear()
        #c1.Divide(5, 5, 1e-5, 1e-5)
        pads = [None]
        pads.append(ROOT.TPad("p1" ,"",0.00,0.78,0.07,1.00,0,0,0))
        pads.append(ROOT.TPad("p2" ,"",0.07,0.78,0.34,1.00,0,0,0))
        pads.append(ROOT.TPad("p3" ,"",0.34,0.78,0.56,1.00,0,0,0))
        pads.append(ROOT.TPad("p4" ,"",0.56,0.78,0.78,1.00,0,0,0))
        pads.append(ROOT.TPad("p5" ,"",0.78,0.78,1.00,1.00,0,0,0))
        pads.append(ROOT.TPad("p6" ,"",0.00,0.56,0.07,0.78,0,0,0))
        pads.append(ROOT.TPad("p7" ,"",0.07,0.56,0.34,0.78,0,0,0))
        pads.append(ROOT.TPad("p8" ,"",0.34,0.56,0.56,0.78,0,0,0))
        pads.append(ROOT.TPad("p9" ,"",0.56,0.56,0.78,0.78,0,0,0))
        pads.append(ROOT.TPad("p10","",0.78,0.56,1.00,0.78,0,0,0))
        pads.append(ROOT.TPad("p11","",0.00,0.34,0.07,0.56,0,0,0))
        pads.append(ROOT.TPad("p12","",0.07,0.34,0.34,0.56,0,0,0))
        pads.append(ROOT.TPad("p13","",0.34,0.34,0.56,0.56,0,0,0))
        pads.append(ROOT.TPad("p14","",0.56,0.34,0.78,0.56,0,0,0))
        pads.append(ROOT.TPad("p15","",0.78,0.34,1.00,0.56,0,0,0))
        pads.append(ROOT.TPad("p16","",0.00,0.07,0.07,0.34,0,0,0))
        pads.append(ROOT.TPad("p17","",0.07,0.07,0.34,0.34,0,0,0))
        pads.append(ROOT.TPad("p18","",0.34,0.07,0.56,0.34,0,0,0))
        pads.append(ROOT.TPad("p19","",0.56,0.07,0.78,0.34,0,0,0))
        pads.append(ROOT.TPad("p20","",0.78,0.07,1.00,0.34,0,0,0))
        pads.append(ROOT.TPad("p21","",0.00,0.00,0.07,0.07,0,0,0))
        pads.append(ROOT.TPad("p22","",0.07,0.00,0.34,0.07,0,0,0))
        pads.append(ROOT.TPad("p23","",0.34,0.00,0.56,0.07,0,0,0))
        pads.append(ROOT.TPad("p24","",0.56,0.00,0.78,0.07,0,0,0))
        pads.append(ROOT.TPad("p25","",0.78,0.00,1.00,0.07,0,0,0))
        for p in pads:
          if not not p:
            p.Draw()
            ROOT.SetOwnership(p,False)

        label1 = ROOT.TPaveLabel(0, 0, 1, 1, "x residuals (mm)","")
        label2 = ROOT.TPaveLabel(0, 0, 1, 1, "y residuals (mm)","")
        label3 = ROOT.TPaveLabel(0, 0, 1, 1, "dx/dz residuals (mrad)","")
        label4 = ROOT.TPaveLabel(0, 0, 1, 1, "dy/dz residuals (mrad)","")
        label5 = ROOT.TPaveLabel(0, 0, 1, 1, "x position (cm)","")
        label6 = ROOT.TPaveLabel(0, 0, 1, 1, "y position (cm)","")
        label7 = ROOT.TPaveLabel(0, 0, 1, 1, "dx/dz angle (rad)","")
        label8 = ROOT.TPaveLabel(0, 0, 1, 1, "dy/dz angle (rad)","")
        label9 = ROOT.TPaveLabel(0, 0.85, 1, 1, getname(r),"NDC")

        for l in label1, label2, label3, label4, label5, label6, label7, label8, label9:
            l.SetBorderSize(0)
            l.SetFillColor(ROOT.kWhite)

        for l in label1, label2, label3, label4:
            l.SetTextAngle(90)
            l.SetTextSize(0.09)
        
        #label9.SetTextAngle(30)
        label9.SetTextSize(0.59)

        pads[1].cd(); label1.Draw()
        pads[6].cd(); label2.Draw()
        pads[11].cd(); label3.Draw()
        pads[16].cd(); label4.Draw()
        pads[22].cd(); label5.Draw()
        pads[23].cd(); label6.Draw()
        pads[24].cd(); label7.Draw()
        pads[25].cd(); label8.Draw()

        pads[2].SetRightMargin(1e-5)
        pads[2].SetBottomMargin(1e-5)
        pads[2].SetLeftMargin(0.17)
        pads[3].SetLeftMargin(1e-5)
        pads[3].SetRightMargin(1e-5)
        pads[3].SetBottomMargin(1e-5)
        pads[4].SetLeftMargin(1e-5)
        pads[4].SetRightMargin(1e-5)
        pads[4].SetBottomMargin(1e-5)
        pads[5].SetLeftMargin(1e-5)
        pads[5].SetBottomMargin(1e-5)

        pads[7].SetRightMargin(1e-5)
        pads[7].SetBottomMargin(1e-5)
        pads[7].SetTopMargin(1e-5)
        pads[7].SetLeftMargin(0.17)
        pads[8].SetLeftMargin(1e-5)
        pads[8].SetRightMargin(1e-5)
        pads[8].SetBottomMargin(1e-5)
        pads[8].SetTopMargin(1e-5)
        pads[9].SetLeftMargin(1e-5)
        pads[9].SetRightMargin(1e-5)
        pads[9].SetBottomMargin(1e-5)
        pads[9].SetTopMargin(1e-5)
        pads[10].SetLeftMargin(1e-5)
        pads[10].SetBottomMargin(1e-5)
        pads[10].SetTopMargin(1e-5)

        pads[12].SetRightMargin(1e-5)
        pads[12].SetBottomMargin(1e-5)
        pads[12].SetTopMargin(1e-5)
        pads[12].SetLeftMargin(0.17)
        pads[13].SetLeftMargin(1e-5)
        pads[13].SetRightMargin(1e-5)
        pads[13].SetBottomMargin(1e-5)
        pads[13].SetTopMargin(1e-5)
        pads[14].SetLeftMargin(1e-5)
        pads[14].SetRightMargin(1e-5)
        pads[14].SetBottomMargin(1e-5)
        pads[14].SetTopMargin(1e-5)
        pads[15].SetLeftMargin(1e-5)
        pads[15].SetBottomMargin(1e-5)
        pads[15].SetTopMargin(1e-5)

        pads[17].SetRightMargin(1e-5)
        pads[17].SetTopMargin(1e-5)
        pads[17].SetLeftMargin(0.17)
        pads[18].SetLeftMargin(1e-5)
        pads[18].SetRightMargin(1e-5)
        pads[18].SetTopMargin(1e-5)
        pads[19].SetLeftMargin(1e-5)
        pads[19].SetRightMargin(1e-5)
        pads[19].SetTopMargin(1e-5)
        pads[20].SetLeftMargin(1e-5)
        pads[20].SetTopMargin(1e-5)
        
        chamber_x_trackx.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_tracky.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdxdz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdydz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_trackx.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_tracky.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_trackdxdz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_trackdydz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_y_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackx.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_tracky.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdxdz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdydz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)
        
        # chamber_dydz_trackx
        chamber_dydz_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dydz_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dydz_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)

        pads[2].cd()
        chamber_x_trackx.Draw("e1")
        if not suppressblue: chamber_x_trackx_fit2.Draw("samel")
        chamber_x_trackx_fit.Draw("samel")
        #label99 = ROOT.TPaveLabel(0, 0.8, 1, 1, getname(r),"NDC")
        print getname(r)
        #label99 = ROOT.TPaveLabel(0, 0.8, 1, 1, "aaa","NDC")
        label9.Draw()
        #pads[2].Modified()
        
        pads[3].cd()
        chamber_x_tracky.Draw("e1")
        if not suppressblue: chamber_x_tracky_fit2.Draw("samel")
        chamber_x_tracky_fit.Draw("samel")
        
        pads[4].cd()
        chamber_x_trackdxdz.Draw("e1")
        if not suppressblue: chamber_x_trackdxdz_fit2.Draw("samel")
        chamber_x_trackdxdz_fit.Draw("samel")
        
        pads[5].cd()
        chamber_x_trackdydz.Draw("e1")
        if not suppressblue: chamber_x_trackdydz_fit2.Draw("samel")
        chamber_x_trackdydz_fit.Draw("samel")
        
        pads[7].cd()
        chamber_y_trackx.Draw("e1")
        if not suppressblue: chamber_y_trackx_fit2.Draw("samel")
        chamber_y_trackx_fit.Draw("samel")
        
        pads[8].cd()
        chamber_y_tracky.Draw("e1")
        if not suppressblue: chamber_y_tracky_fit2.Draw("samel")
        chamber_y_tracky_fit.Draw("samel")
        
        pads[9].cd()
        chamber_y_trackdxdz.Draw("e1")
        if not suppressblue: chamber_y_trackdxdz_fit2.Draw("samel")
        chamber_y_trackdxdz_fit.Draw("samel")
        
        pads[10].cd()
        chamber_y_trackdydz.Draw("e1")
        if not suppressblue: chamber_y_trackdydz_fit2.Draw("samel")
        chamber_y_trackdydz_fit.Draw("samel")
        
        pads[12].cd()
        chamber_dxdz_trackx.Draw("e1")
        if not suppressblue: chamber_dxdz_trackx_fit2.Draw("samel")
        chamber_dxdz_trackx_fit.Draw("samel")
        
        pads[13].cd()
        chamber_dxdz_tracky.Draw("e1")
        if not suppressblue: chamber_dxdz_tracky_fit2.Draw("samel")
        chamber_dxdz_tracky_fit.Draw("samel")
        
        pads[14].cd()
        chamber_dxdz_trackdxdz.Draw("e1")
        if not suppressblue: chamber_dxdz_trackdxdz_fit2.Draw("samel")
        chamber_dxdz_trackdxdz_fit.Draw("samel")
        
        pads[15].cd()
        chamber_dxdz_trackdydz.Draw("e1")
        if not suppressblue: chamber_dxdz_trackdydz_fit2.Draw("samel")
        chamber_dxdz_trackdydz_fit.Draw("samel")
        
        pads[17].cd()
        chamber_dydz_trackx.Draw("e1")
        if not suppressblue: chamber_dydz_trackx_fit2.Draw("samel")
        chamber_dydz_trackx_fit.Draw("samel")
        
        pads[18].cd()
        chamber_dydz_tracky.Draw("e1")
        if not suppressblue: chamber_dydz_tracky_fit2.Draw("samel")
        chamber_dydz_tracky_fit.Draw("samel")
        
        pads[19].cd()
        chamber_dydz_trackdxdz.Draw("e1")
        if not suppressblue: chamber_dydz_trackdxdz_fit2.Draw("samel")
        chamber_dydz_trackdxdz_fit.Draw("samel")
        
        pads[20].cd()
        chamber_dydz_trackdydz.Draw("e1")
        if not suppressblue: chamber_dydz_trackdydz_fit2.Draw("samel")
        chamber_dydz_trackdydz_fit.Draw("samel")

    else:
        c1.Clear()
        #c1.Divide(5, 3, 1e-5, 1e-5)
        pads = [None]
        pads.append(ROOT.TPad("p1" ,"",0.00,0.55,0.07,1.00,0,0,0))
        pads.append(ROOT.TPad("p2" ,"",0.07,0.55,0.34,1.00,0,0,0))
        pads.append(ROOT.TPad("p3" ,"",0.34,0.55,0.56,1.00,0,0,0))
        pads.append(ROOT.TPad("p4" ,"",0.56,0.55,0.78,1.00,0,0,0))
        pads.append(ROOT.TPad("p5" ,"",0.78,0.55,1.00,1.00,0,0,0))
        pads.append(ROOT.TPad("p6" ,"",0.00,0.1,0.07,0.55,0,0,0))
        pads.append(ROOT.TPad("p7" ,"",0.07,0.1,0.34,0.55,0,0,0))
        pads.append(ROOT.TPad("p8" ,"",0.34,0.1,0.56,0.55,0,0,0))
        pads.append(ROOT.TPad("p9" ,"",0.56,0.1,0.78,0.55,0,0,0))
        pads.append(ROOT.TPad("p10","",0.78,0.1,1.00,0.55,0,0,0))
        pads.append(ROOT.TPad("p11","",0.00,0.,0.07,0.1,0,0,0))
        pads.append(ROOT.TPad("p12","",0.07,0.,0.34,0.1,0,0,0))
        pads.append(ROOT.TPad("p13","",0.34,0.,0.56,0.1,0,0,0))
        pads.append(ROOT.TPad("p14","",0.56,0.,0.78,0.1,0,0,0))
        pads.append(ROOT.TPad("p15","",0.78,0.,1.00,0.1,0,0,0))
        for p in pads:
          if not not p:
            p.Draw()
            ROOT.SetOwnership(p,False)

        label1 = ROOT.TPaveLabel(0, 0, 1, 1, "x residuals (mm)")
        label2 = ROOT.TPaveLabel(0, 0, 1, 1, "dx/dz residuals (mrad)")
        label3 = ROOT.TPaveLabel(0, 0.3, 1, 1, "x position (cm)")
        label4 = ROOT.TPaveLabel(0, 0.3, 1, 1, "y position (cm)")
        label5 = ROOT.TPaveLabel(0, 0.3, 1, 1, "dx/dz angle (rad)")
        label6 = ROOT.TPaveLabel(0, 0.3, 1, 1, "dy/dz angle (rad)")
        label9 = ROOT.TPaveLabel(0, 0.85, 1, 1, getname(r),"NDC")

        if name[0:2] == "ME":
            label1 = ROOT.TPaveLabel(0, 0, 1, 1, "r#phi residuals (mm)")
            label2 = ROOT.TPaveLabel(0, 0, 1, 1, "d(r#phi)/dz residuals (mrad)")

        for l in label1, label2, label3, label4, label5, label6, label9:
            l.SetBorderSize(0)
            l.SetFillColor(ROOT.kWhite)

        for l in label1, label2:
            l.SetTextAngle(90)
            l.SetTextSize(0.09)

        #label9.SetTextAngle(30)
        label9.SetTextSize(0.29)

        pads[1].cd(); label1.Draw()
        pads[6].cd(); label2.Draw()
        pads[12].cd(); label3.Draw()
        pads[13].cd(); label4.Draw()
        pads[14].cd(); label5.Draw()
        pads[15].cd(); label6.Draw()
        #pads[11].cd(); label9.Draw()

        pads[2].SetRightMargin(1e-5)
        pads[2].SetBottomMargin(1e-5)
        pads[3].SetLeftMargin(1e-5)
        pads[3].SetRightMargin(1e-5)
        pads[3].SetBottomMargin(1e-5)
        pads[4].SetLeftMargin(1e-5)
        pads[4].SetRightMargin(1e-5)
        pads[4].SetBottomMargin(1e-5)
        pads[5].SetLeftMargin(1e-5)
        pads[5].SetBottomMargin(1e-5)

        pads[7].SetRightMargin(1e-5)
        pads[7].SetTopMargin(1e-5)
        pads[8].SetLeftMargin(1e-5)
        pads[8].SetRightMargin(1e-5)
        pads[8].SetTopMargin(1e-5)
        pads[9].SetLeftMargin(1e-5)
        pads[9].SetRightMargin(1e-5)
        pads[9].SetTopMargin(1e-5)
        pads[10].SetLeftMargin(1e-5)
        pads[10].SetTopMargin(1e-5)

        chamber_x_trackx.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_tracky.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdxdz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdydz.GetXaxis().SetLabelColor(ROOT.kWhite)
        chamber_x_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)
        # chamber_dxdz_trackx
        chamber_dxdz_tracky.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdxdz.GetYaxis().SetLabelColor(ROOT.kWhite)
        chamber_dxdz_trackdydz.GetYaxis().SetLabelColor(ROOT.kWhite)

        pads[2].cd()
        chamber_x_trackx.Draw("e1")
        if not suppressblue: chamber_x_trackx_fit2.Draw("samel")
        chamber_x_trackx_fit.Draw("samel")
        label9.Draw()
        
        pads[3].cd()
        chamber_x_tracky.Draw("e1")
        if not suppressblue: chamber_x_tracky_fit2.Draw("samel")
        chamber_x_tracky_fit.Draw("samel")
        
        pads[4].cd()
        chamber_x_trackdxdz.Draw("e1")
        if not suppressblue: chamber_x_trackdxdz_fit2.Draw("samel")
        chamber_x_trackdxdz_fit.Draw("samel")
        
        pads[5].cd()
        chamber_x_trackdydz.Draw("e1")
        if not suppressblue: chamber_x_trackdydz_fit2.Draw("samel")
        chamber_x_trackdydz_fit.Draw("samel")
        
        pads[7].cd()
        chamber_dxdz_trackx.Draw("e1")
        if not suppressblue: chamber_dxdz_trackx_fit2.Draw("samel")
        chamber_dxdz_trackx_fit.Draw("samel")
        
        pads[8].cd()
        chamber_dxdz_tracky.Draw("e1")
        if not suppressblue: chamber_dxdz_tracky_fit2.Draw("samel")
        chamber_dxdz_tracky_fit.Draw("samel")
        
        pads[9].cd()
        chamber_dxdz_trackdxdz.Draw("e1")
        if not suppressblue: chamber_dxdz_trackdxdz_fit2.Draw("samel")
        chamber_dxdz_trackdxdz_fit.Draw("samel")
        
        pads[10].cd()
        chamber_dxdz_trackdydz.Draw("e1")
        if not suppressblue: chamber_dxdz_trackdydz_fit2.Draw("samel")
        chamber_dxdz_trackdydz_fit.Draw("samel")

    tn = time.time()
    ddt[8] = 1./ddt[7]*((ddt[7]-1)*ddt[8] + tn-t1)

##################################################################################

def segdiff(tfiles, component, pair, **args):
    tdrStyle.SetOptFit(1)
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.05)
    tdrStyle.SetStatW(0.2)
    tdrStyle.SetStatY(0.9)
    tdrStyle.SetStatFontSize(0.06)

    if component[0:2] == "dt":
        wheel = args["wheel"]
        wheelletter = wheelLetter(wheel)
        sector = args["sector"]
        profname = "%s_%s_%02d_%s" % (component, wheelletter, sector, str(pair))
        posname = "pos" + profname
        negname = "neg" + profname
        #print profname

        station1 = int(str(pair)[0])
        station2 = int(str(pair)[1])
        phi1 = signConventions["DT", wheel, station1, sector][4]
        phi2 = signConventions["DT", wheel, station2, sector][4]
        if abs(phi1 - phi2) > 1.:
            if phi1 > phi2: phi1 -= 2.*pi
            else: phi1 += 2.*pi
        phi = (phi1 + phi2) / 2.
        while (phi < -pi): phi += 2.*pi
        while (phi > pi): phi -= 2.*pi

    elif component[0:3] == "csc":
        endcap = args["endcap"]
        if endcap=="m":
            endcapnum=2
            endcapsign="-"
        elif endcap=="p":
            endcapnum=1
            endcapsign="+"
        else: raise Exception
        
        ring = args["ring"]
        if ring>2 or ring<1: raise Exception
        station1 = int(str(pair)[0])
        station2 = int(str(pair)[1])
        if   ring==1: ringname="inner"
        elif ring==2: ringname="outer"
        else: raise Exception
        
        chamber = args["chamber"]
        if (ring==1 and chamber>18) or (ring==2 and chamber>36): raise Exception
        
        profname = "csc%s_%s_%s_%02d_%s" % (ringname,component[4:], endcap, chamber, str(pair))
        posname = "pos" + profname
        negname = "neg" + profname
        #print profname

        station1 = int(str(pair)[0])
        station2 = int(str(pair)[1])
        phi1 = signConventions["CSC", endcapnum, station1, ring, chamber][4]
        phi2 = signConventions["CSC", endcapnum, station2, ring, chamber][4]
        if abs(phi1 - phi2) > 1.:
            if phi1 > phi2: phi1 -= 2.*pi
            else: phi1 += 2.*pi
        phi = (phi1 + phi2) / 2.
        while (phi < -pi*5./180.): phi += 2.*pi
        while (phi > pi*(2.-5./180.)): phi -= 2.*pi
    
    else: raise Exception

    if "window" in args: window = args["window"]
    else: window = 5.

    global tmpprof, tmppos, tmpneg
    pdir = "AlignmentMonitorSegmentDifferences/iter1/"
    tmpprof = tfiles[0].Get(pdir + profname).Clone()
    tmpprof.SetMarkerStyle(8)
    tmppos = tfiles[0].Get(pdir + posname).Clone()
    tmpneg = tfiles[0].Get(pdir + negname).Clone()
    for tfile in tfiles[1:]:
        tmpprof.Add(tfile.Get(pdir + profname))
        tmppos.Add(tfile.Get(pdir + posname))
        tmpneg.Add(tfile.Get(pdir + negname))

    for i in xrange(1, tmpprof.GetNbinsX()+1):
        if tmpprof.GetBinError(i) < 1e-5:
            tmpprof.SetBinError(i, 100.)
    tmpprof.SetAxisRange(-window, window, "Y")

    f = ROOT.TF1("p1", "[0] + [1]*x", tmpprof.GetBinLowEdge(1), -tmpprof.GetBinLowEdge(1))
    f.SetParameters((tmppos.GetMean() + tmpneg.GetMean())/2., 0.)

    tmpprof.SetXTitle("q/p_{T} (c/GeV)")
    if component == "dt13_resid":
        tmpprof.SetYTitle("#Deltax^{local} (mm)")
        tmppos.SetXTitle("#Deltax^{local} (mm)")
        tmpneg.SetXTitle("#Deltax^{local} (mm)")
        f.SetParNames("#Deltax^{local}_{0}", "Slope")
    if component == "dt13_slope":
        tmpprof.SetYTitle("#Deltadx/dz^{local} (mrad)")
        tmppos.SetXTitle("#Deltadx/dz^{local} (mrad)")
        tmpneg.SetXTitle("#Deltadx/dz^{local} (mrad)")
        f.SetParNames("#Deltadx/dz^{local}_{0}", "Slope")
    if component == "dt2_resid":
        tmpprof.SetYTitle("#Deltay^{local} (mm)")
        tmppos.SetXTitle("#Deltay^{local} (mm)")
        tmpneg.SetXTitle("#Deltay^{local} (mm)")
        f.SetParNames("#Deltay^{local}_{0}", "Slope")
    if component == "dt2_slope":
        tmpprof.SetYTitle("#Deltady/dz^{local} (mrad)")
        tmppos.SetXTitle("#Deltady/dz^{local} (mrad)")
        tmpneg.SetXTitle("#Deltady/dz^{local} (mrad)")
        f.SetParNames("#Deltady/dz^{local}_{0}", "Slope")
    if component == "csc_resid":
        tmpprof.SetXTitle("q/p_{z} (c/GeV)")
        tmpprof.SetYTitle("#Delta(r#phi)^{local} (mm)")
        tmppos.SetXTitle("#Delta(r#phi)^{local} (mm)")
        tmpneg.SetXTitle("#Delta(r#phi)^{local} (mm)")
        f.SetParNames("#Delta(r#phi)^{local}_{0}", "Slope")
    if component == "csc_slope":
        tmpprof.SetXTitle("q/p_{z} (c/GeV)")
        tmpprof.SetYTitle("#Deltad(r#phi)/dz^{local} (mrad)")
        tmppos.SetXTitle("#Deltad(r#phi)/dz^{local} (mrad)")
        tmpneg.SetXTitle("#Deltad(r#phi)/dz^{local} (mrad)")
        f.SetParNames("#Deltad(r#phi)/dz^{local}_{0}", "Slope")
    
    tmpprof.GetXaxis().CenterTitle()
    tmpprof.GetYaxis().CenterTitle()
    tmppos.GetXaxis().CenterTitle()
    tmpneg.GetXaxis().CenterTitle()
    if component[0:2] == "dt":
        tmpprof.SetTitle("MB%d - MB%d, wheel %d, sector %02d" % (station1, station2, int(wheel), int(sector)))
    elif component[0:3] == "csc":
        tmpprof.SetTitle("ME%d - ME%d, for ME%s%d/%d/%d" % (station1, station2, endcapsign, station2, ring, chamber))
    else: raise Exception

    tmppos.SetTitle("Positive muons")
    tmpneg.SetTitle("Negative muons")

    c1.Clear()
    c1.Divide(2, 1)
    c1.GetPad(1).cd()
    fit1 = tmpprof.Fit("p1", "qS")
    tmpprof.Draw("e1")
    c1.GetPad(2).cd()
    c1.GetPad(2).Divide(1, 2)
    c1.GetPad(2).GetPad(1).cd()
    tmppos.Draw()
    f = ROOT.TF1("gausR", "[0]*exp(-(x - [1])**2 / 2. / [2]**2) / sqrt(2.*3.1415926) / [2]", 
                 tmppos.GetMean() - tmppos.GetRMS(), tmppos.GetMean() + tmppos.GetRMS())
    f.SetParameters(tmppos.GetEntries() * ((10. - -10.)/100.), tmppos.GetMean(), tmppos.GetRMS())
    f.SetParNames("Constant", "Mean", "Sigma")
    fit2 = tmppos.Fit("gausR", "qRS")
    c1.GetPad(2).GetPad(2).cd()
    tmpneg.Draw()
    f = ROOT.TF1("gausR", "[0]*exp(-(x - [1])**2 / 2. / [2]**2) / sqrt(2.*3.1415926) / [2]", 
                 tmpneg.GetMean() - tmpneg.GetRMS(), tmpneg.GetMean() + tmpneg.GetRMS())
    f.SetParameters(tmpneg.GetEntries() * ((10. - -10.)/100.), tmpneg.GetMean(), tmpneg.GetRMS())
    f.SetParNames("Constant", "Mean", "Sigma")
    fit3 = tmpneg.Fit("gausR", "qRS")

    fit1ok = fit1.Status()==0 and fit1.CovMatrixStatus()==3
    fit2ok = fit2.Status()==0 and fit2.CovMatrixStatus()==3
    fit3ok = fit3.Status()==0 and fit3.CovMatrixStatus()==3

    fitresult1 = None, None
    if fit1ok:
        fitresult1 = tmpprof.GetFunction("p1").GetParameter(0), tmpprof.GetFunction("p1").GetParError(0)
    fitresult2 = None, None
    if fit2ok and fit3ok:
        fitresult2 = (tmppos.GetFunction("gausR").GetParameter(1) + tmpneg.GetFunction("gausR").GetParameter(1)) / 2., \
                     sqrt(tmppos.GetFunction("gausR").GetParError(1)**2 + tmpneg.GetFunction("gausR").GetParError(1)**2) / 2.
    return phi, fitresult1[0], fitresult1[1], fitresult2[0], fitresult2[1], fit1ok, fit2ok, fit3ok



##################################################################################

def segdiff_xalign(tfiles, component, **args):
    tdrStyle.SetOptFit(1)
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.05)
    tdrStyle.SetStatW(0.2)
    tdrStyle.SetStatY(0.9)
    tdrStyle.SetStatFontSize(0.06)
    
    if component[0:4] == "x_dt":
        wheel = int(args["wheel"])
        if int(wheel)<0:
          wheell = "m%d" % abs(wheel)
          endcapsign="-"
        else:
          wheell = "p%d" % abs(wheel)
          endcapsign="+"
        station_dt = component[4]
        station_csc_1 = args["cscstations"][0]
        if station_csc_1=='1':  ring_1 = 3
        else:                   ring_1 = 2
        sector = args["sector"]
        profname = "%s%s_W%sS%02d" % (component, station_csc_1, wheell, sector)
        posname_1 = "pos_" + profname
        negname_1 = "neg_" + profname
        if len(args["cscstations"]) > 1:
          station_csc_2 = args["cscstations"][1]
          if station_csc_2=='1':  ring_2 = 3
          else:                   ring_2 = 2
          profname = "%s%s_W%sS%02d" % (component, station_csc_2, wheell, sector)
          posname_2 = "pos_" + profname
          negname_2 = "neg_" + profname

        phi = signConventions["DT", wheel, int(station_dt), sector][4]
        while (phi < -pi): phi += 2.*pi
        while (phi > pi): phi -= 2.*pi
        
    else: raise Exception

    if "window" in args: window = args["window"]
    else: window = 5.

    global tmppos, tmpneg, tmppos_2, tmpneg_2
    pdir = "AlignmentMonitorSegmentDifferences/iter1/"
    tmppos = tfiles[0].Get(pdir + posname_1).Clone()
    tmpneg = tfiles[0].Get(pdir + negname_1).Clone()
    if len(args["cscstations"]) > 1:
      tmppos_2 = tfiles[0].Get(pdir + posname_2).Clone()
      tmpneg_2 = tfiles[0].Get(pdir + negname_2).Clone()
    tmpneg.Rebin(2); tmppos.Rebin(2)
    for tfile in tfiles[1:]:
      tmppos.Add(tfile.Get(pdir + posname_1))
      tmpneg.Add(tfile.Get(pdir + negname_1))
      if len(args["cscstations"]) > 1:
        tmppos_2.Add(tfile.Get(pdir + posname_2))
        tmpneg_2.Add(tfile.Get(pdir + negname_2))
      tmpneg_2.Rebin(2); tmppos_2.Rebin(2)

    result = {}
    result["fit_ok"] = False
    result["phi"] = phi
    ntot = tmppos.GetEntries() + tmpneg.GetEntries()
    if ntot == 0:
      return result

    tmppos.SetXTitle("#Deltax^{loc}_{MB} - r_{DT}/r_{CSC}#times#Deltax^{loc}_{ME} (mm)")
    tmpneg.SetXTitle("#Deltax^{loc}_{MB} - r_{DT}/r_{CSC}#times#Deltax^{loc}_{ME} (mm)")
    title1 =  "MB(W%+d St%s Sec%d) - ME%s%s/%d" % (wheel, station_dt, int(sector), endcapsign, station_csc_1, ring_1)
    tmppos.SetTitle("Positive #mu:  %s" % title1);
    tmpneg.SetTitle("Negative #mu:  %s" % title1);
    tmppos.GetXaxis().CenterTitle()
    tmpneg.GetXaxis().CenterTitle()
    if len(args["cscstations"]) > 1:
      tmppos.SetXTitle("#Deltax^{loc}_{DT} - r_{DT}/r_{CSC}#times#Deltax^{loc}_{CSC} (mm)")
      tmpneg.SetXTitle("#Deltax^{loc}_{DT} - r_{DT}/r_{CSC}#times#Deltax^{loc}_{CSC} (mm)")
      title2 =  "MB(W%+d St%s Sec%d) - ME%s%s/%d" % (wheel, station_dt, int(sector), endcapsign, station_csc_2, ring_2)
      tmppos_2.SetTitle("Positive #mu:  %s" % title2);
      tmpneg_2.SetTitle("Negative #mu:  %s" % title2);
      tmppos_2.GetXaxis().CenterTitle()
      tmpneg_2.GetXaxis().CenterTitle()

    c1.Clear()
    c1.Divide(2, 2)

    c1.GetPad(1).cd()
    tmppos.Draw()
    fpos = ROOT.TF1("gpos", "gaus", tmppos.GetMean() - tmppos.GetRMS(), tmppos.GetMean() + tmppos.GetRMS())
    fpos.SetParameters(tmppos.GetEntries() * 2.5, tmppos.GetMean(), tmppos.GetRMS())
    fit_pos = tmppos.Fit("gpos", "qRS")

    c1.GetPad(3).cd()
    tmpneg.Draw()
    fneg = ROOT.TF1("gneg", "gaus", tmpneg.GetMean() - tmpneg.GetRMS(), tmpneg.GetMean() + tmpneg.GetRMS())
    fneg.SetParameters(tmpneg.GetEntries() * 2.5, tmpneg.GetMean(), tmpneg.GetRMS())
    fit_neg = tmpneg.Fit("gneg", "qRS")
    
    result["fit_ok"] = (fit_pos.Status()==0 and fit_pos.CovMatrixStatus()==3 and fit_neg.Status()==0 and fit_neg.CovMatrixStatus()==3)
    result["fit_peak"] = (fpos.GetParameter(1)*tmppos.GetEntries() + fneg.GetParameter(1)*tmpneg.GetEntries()) / ntot
    result["fit_peak_error"] = sqrt( (fpos.GetParError(1)*tmppos.GetEntries())**2 + (fneg.GetParError(1)*tmpneg.GetEntries())**2) / ntot
    
    if len(args["cscstations"]) > 1:
      c1.GetPad(2).cd()
      tmppos_2.Draw()
      fpos_2 = ROOT.TF1("gpos2", "gaus", tmppos_2.GetMean() - tmppos_2.GetRMS(), tmppos_2.GetMean() + tmppos_2.GetRMS())
      fpos_2.SetParameters(tmppos_2.GetEntries() * 2.5, tmppos_2.GetMean(), tmppos_2.GetRMS())
      fit_pos_2 = tmppos_2.Fit("gpos2", "qRS")

      c1.GetPad(4).cd()
      tmpneg_2.Draw()
      fneg_2 = ROOT.TF1("gneg2", "gaus", tmpneg_2.GetMean() - tmpneg_2.GetRMS(), tmpneg_2.GetMean() + tmpneg_2.GetRMS())
      fneg_2.SetParameters(tmpneg_2.GetEntries() * 2.5, tmpneg_2.GetMean(), tmpneg_2.GetRMS())
      fit_neg_2 = tmpneg_2.Fit("gneg2", "qRS")

      result["fit_ok_2"] = (fit_pos_2.Status()==0 and fit_pos_2.CovMatrixStatus()==3 and fit_neg_2.Status()==0 and fit_neg_2.CovMatrixStatus()==3)
      ntot = tmppos_2.GetEntries() + tmpneg_2.GetEntries()
      result["fit_peak_2"] = (fpos_2.GetParameter(1)*tmppos_2.GetEntries() + fneg_2.GetParameter(1)*tmpneg_2.GetEntries()) / ntot
      result["fit_peak_error_2"] = sqrt( (fpos_2.GetParError(1)*tmppos_2.GetEntries())**2 + (fneg_2.GetParError(1)*tmpneg_2.GetEntries())**2) / ntot

    return result

##################################################################################

def segdiffvsphi_xalign(tfiles, wheel, window=10.):
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.05)

    global htemp, gtemp_12, gtemp_21, gtemp_11, tlegend
    htemp = ROOT.TH1F("htemp", "", 1, -pi, pi)
    gtemp_11_phi, gtemp_11_val, gtemp_11_err = [], [], []
    gtemp_12_phi, gtemp_12_val, gtemp_12_err = [], [], []
    gtemp_21_phi, gtemp_21_val, gtemp_21_err = [], [], []
    for sector in xrange(1, 12+1):
      #print "sect", sector
      r1 = segdiff_xalign(tfiles, "x_dt1_csc", wheel=wheel, sector=sector, cscstations = "12")
      r2 = segdiff_xalign(tfiles, "x_dt2_csc", wheel=wheel, sector=sector, cscstations = "1")
      
      if r1["fit_ok"]:
        gtemp_11_phi.append(r1["phi"])
        gtemp_11_val.append(r1["fit_peak"])
        gtemp_11_err.append(r1["fit_peak_error"])
      if r1["fit_ok_2"]:
        gtemp_12_phi.append(r1["phi"])
        gtemp_12_val.append(r1["fit_peak_2"])
        gtemp_12_err.append(r1["fit_peak_error_2"])
      if r2["fit_ok"]:
        gtemp_21_phi.append(r2["phi"])
        gtemp_21_val.append(r2["fit_peak"])
        gtemp_21_err.append(r2["fit_peak_error"])

    #print "len(gtemp_11_phi) ",len(gtemp_11_phi)
    #print "len(gtemp_12_phi) ",len(gtemp_12_phi)
    #print "len(gtemp_21_phi) ",len(gtemp_21_phi)
    if len(gtemp_11_phi) > 0:
        gtemp_11 = ROOT.TGraphErrors(len(gtemp_11_phi), array.array("d", gtemp_11_phi), array.array("d", gtemp_11_val), 
                                     array.array("d", [0.] * len(gtemp_11_phi)), array.array("d", gtemp_11_err))
    if len(gtemp_12_phi) > 0:
        gtemp_12 = ROOT.TGraphErrors(len(gtemp_12_phi), array.array("d", gtemp_12_phi), array.array("d", gtemp_12_val), 
                                     array.array("d", [0.] * len(gtemp_12_phi)), array.array("d", gtemp_12_err))
    if len(gtemp_11_phi) > 0:
        gtemp_21 = ROOT.TGraphErrors(len(gtemp_21_phi), array.array("d", gtemp_21_phi), array.array("d", gtemp_21_val), 
                                     array.array("d", [0.] * len(gtemp_21_phi)), array.array("d", gtemp_21_err))

    if len(gtemp_11_phi) > 0:
        gtemp_11.SetMarkerStyle(20);  gtemp_11.SetMarkerSize(1.5);  
        gtemp_11.SetMarkerColor(ROOT.kRed);  gtemp_11.SetLineColor(ROOT.kRed)
    if len(gtemp_12_phi) > 0:
        gtemp_12.SetMarkerStyle(22);  gtemp_12.SetMarkerSize(1.);  
        gtemp_12.SetMarkerColor(ROOT.kGreen+2);  gtemp_12.SetLineColor(ROOT.kGreen+2)
    if len(gtemp_21_phi) > 0:
        gtemp_21.SetMarkerStyle(21);  gtemp_21.SetMarkerSize(1.5);  
        gtemp_21.SetMarkerColor(ROOT.kBlue);  gtemp_21.SetLineColor(ROOT.kBlue)
    
    htemp.SetTitle("Wheel %+d" % wheel)
    htemp.SetAxisRange(-window, window, "Y")
    htemp.SetXTitle("#phi of MB")
    htemp.SetYTitle("#Deltax^{loc}_{DT} - r_{DT}/r_{CSC}#times#Deltax^{loc}_{CSC} (mm)")
    htemp.GetXaxis().CenterTitle()
    htemp.GetYaxis().CenterTitle()
    htemp.GetYaxis().SetTitleOffset(0.75)

    c1.Clear()
    htemp.Draw()
    if len(gtemp_12_phi) > 0:
        gtemp_12.Draw("p")
    if len(gtemp_21_phi) > 0:
        gtemp_21.Draw("p")
    if len(gtemp_11_phi) > 0:
        gtemp_11.Draw("p")

    tlegend = ROOT.TLegend(0.59, 0.75, 0.99, 0.92)
    tlegend.SetBorderSize(0)
    tlegend.SetFillColor(ROOT.kWhite)
    if len(gtemp_11_phi) > 0:
        tlegend.AddEntry(gtemp_11, "MB1 - ME1/3 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_11_val), stdev(gtemp_11_val)), "pl")
    if len(gtemp_21_phi) > 0:
        tlegend.AddEntry(gtemp_21, "MB2 - ME1/3 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_21_val), stdev(gtemp_21_val)), "pl")
    if len(gtemp_12_phi) > 0:
        tlegend.AddEntry(gtemp_12, "MB1 - ME2/2 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_12_val), stdev(gtemp_12_val)), "pl")
    #if len(gtemp_12_phi) > 0:
    #    tlegend.AddEntry(gtemp_12, "total mean: %4.2f, total RMS: %4.2f" % \
    #                               (mean(gtemp_11_val + gtemp_12_val + gtemp_21_val), 
    #                               stdev(gtemp_11_val + gtemp_12_val + gtemp_21_val)), "")
    tlegend.Draw()

    f_11 = ROOT.TF1("f11", "[0] + [1]*sin(x) + [2]*cos(x)", -pi, pi)
    f_11.SetLineColor(ROOT.kRed)
    f_11.SetLineWidth(2)
    f_21 = ROOT.TF1("f21", "[0] + [1]*sin(x) + [2]*cos(x)", -pi, pi)
    f_21.SetLineColor(ROOT.kBlue)
    f_21.SetLineWidth(2)
    if len(gtemp_11_phi) > 0:
      gtemp_11.Fit(f_11,"")
    if len(gtemp_21_phi) > 0:
      gtemp_21.Fit(f_21,"")
    
    global f_txt,f_11_txt, f_21_txt 
    f_txt = ROOT.TLatex(-2.9, -0.7*window, "ME1/3 ring corrections equivalents:")
    f_txt.SetTextSize(0.028)
    f_txt.Draw()
    if len(gtemp_11_phi) > 0:
      rdt = signConventions[("DT", 2, 1, 1)][3]*10
      f_11_txt = ROOT.TLatex(-2.9, -0.8*window, "#Deltax=%.2f#pm%.2f mm   #Deltay=%.2f#pm%.2f mm   #Delta#phi_{z}=%.2f#pm%.2f mrad" % (
                             -f_11.GetParameter(1), f_11.GetParError(1), f_11.GetParameter(2), f_11.GetParError(2), -f_11.GetParameter(0)/rdt*1000, f_11.GetParError(0)/rdt*1000))
      f_11_txt.SetTextSize(0.028)
      f_11_txt.SetTextColor(ROOT.kRed)
      f_11_txt.Draw()
    if len(gtemp_11_phi) > 0:
      rdt = signConventions[("DT", 2, 2, 1)][3]*10
      f_21_txt = ROOT.TLatex(-2.9, -0.9*window, "#Deltax=%.2f#pm%.2f mm   #Deltay=%.2f#pm%.2f mm   #Delta#phi_{z}=%.2f#pm%.2f mrad" % (
                             -f_21.GetParameter(1), f_21.GetParError(1), f_21.GetParameter(2), f_21.GetParError(2), -f_21.GetParameter(0)/rdt*1000, f_21.GetParError(0)/rdt*1000))
      f_21_txt.SetTextSize(0.028)
      f_21_txt.SetTextColor(ROOT.kBlue)
      f_21_txt.Draw()

##################################################################################

def segdiffvsphi(tfiles, reports, component, wheel, window=5., excludesectors=()):
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.05)

    global htemp, gtemp_12, gtemp2_12, gtemp_23, gtemp2_23, gtemp_34, gtemp2_34, tlegend
    htemp = ROOT.TH1F("htemp", "", 1, -pi, pi)
    gtemp_12_phi, gtemp_12_val, gtemp_12_err, gtemp_12_val2, gtemp_12_err2 = [], [], [], [], []
    gtemp_23_phi, gtemp_23_val, gtemp_23_err, gtemp_23_val2, gtemp_23_err2 = [], [], [], [], []
    gtemp_34_phi, gtemp_34_val, gtemp_34_err, gtemp_34_val2, gtemp_34_err2 = [], [], [], [], []
    for sector in xrange(1, 12+1):
        #print "sect", sector
        r1_found, r2_found, r3_found, r4_found = False, False, False, False
        for r1 in reports:
            if r1.postal_address == ("DT", wheel, 1, sector):
                r1_found = True
                break
        for r2 in reports:
            if r2.postal_address == ("DT", wheel, 2, sector):
                r2_found = True
                break
        for r3 in reports:
            if r3.postal_address == ("DT", wheel, 3, sector):
                r3_found = True
                break
        for r4 in reports:
            if r4.postal_address == ("DT", wheel, 4, sector):
                r4_found = True
                break
        #print "rfounds: ", r1_found, r2_found, r3_found, r4_found
        
        if sector not in excludesectors:
            if r1_found and r2_found and r1.status == "PASS" and r2.status == "PASS":
                phi, val, err, val2, err2, fit1, fit2, fit3 = segdiff(tfiles, component, 12, wheel=wheel, sector=sector)
                #print "segdif 12", phi, val, err, val2, err2, fit1, fit2, fit3
                if fit1 and fit2 and fit3:
                    gtemp_12_phi.append(phi)
                    gtemp_12_val.append(val)
                    gtemp_12_err.append(err)
                    gtemp_12_val2.append(val2)
                    gtemp_12_err2.append(err2)
            if r2_found and r3_found and r2.status == "PASS" and r3.status == "PASS":
                phi, val, err, val2, err2, fit1, fit2, fit3 = segdiff(tfiles, component, 23, wheel=wheel, sector=sector)
                #print "segdif 23", phi, val, err, val2, err2, fit1, fit2, fit3
                if fit1 and fit2 and fit3:
                    gtemp_23_phi.append(phi)
                    gtemp_23_val.append(val)
                    gtemp_23_err.append(err)
                    gtemp_23_val2.append(val2)
                    gtemp_23_err2.append(err2)
            if component[:4] == "dt13":
                if r3_found and r4_found and r3.status == "PASS" and r4.status == "PASS":
                    phi, val, err, val2, err2, fit1, fit2, fit3 = segdiff(tfiles, component, 34, wheel=wheel, sector=sector)
                    #print "segdif 34", phi, val, err, val2, err2, fit1, fit2, fit3
                    if fit1 and fit2 and fit3:
                        gtemp_34_phi.append(phi)
                        gtemp_34_val.append(val)
                        gtemp_34_err.append(err)
                        gtemp_34_val2.append(val2)
                        gtemp_34_err2.append(err2)

    #print "len(gtemp_12_phi) ", len(gtemp_12_phi)
    #print "len(gtemp_23_phi) ",len(gtemp_23_phi)
    #print "len(gtemp_34_phi) ",len(gtemp_34_phi)
    if len(gtemp_12_phi) > 0:
        gtemp_12 = ROOT.TGraphErrors(len(gtemp_12_phi), array.array("d", gtemp_12_phi), array.array("d", gtemp_12_val), 
                                     array.array("d", [0.] * len(gtemp_12_phi)), array.array("d", gtemp_12_err))
        gtemp2_12 = ROOT.TGraphErrors(len(gtemp_12_phi), array.array("d", gtemp_12_phi), array.array("d", gtemp_12_val2), 
                                      array.array("d", [0.] * len(gtemp_12_phi)), array.array("d", gtemp_12_err2))
    if len(gtemp_23_phi) > 0:
        gtemp_23 = ROOT.TGraphErrors(len(gtemp_23_phi), array.array("d", gtemp_23_phi), array.array("d", gtemp_23_val), 
                                     array.array("d", [0.] * len(gtemp_23_phi)), array.array("d", gtemp_23_err))
        gtemp2_23 = ROOT.TGraphErrors(len(gtemp_23_phi), array.array("d", gtemp_23_phi), array.array("d", gtemp_23_val2), 
                                      array.array("d", [0.] * len(gtemp_23_phi)), array.array("d", gtemp_23_err2))
    if len(gtemp_34_phi) > 0:
        gtemp_34 = ROOT.TGraphErrors(len(gtemp_34_phi), array.array("d", gtemp_34_phi), array.array("d", gtemp_34_val), 
                                     array.array("d", [0.] * len(gtemp_34_phi)), array.array("d", gtemp_34_err))
        gtemp2_34 = ROOT.TGraphErrors(len(gtemp_34_phi), array.array("d", gtemp_34_phi), array.array("d", gtemp_34_val2), 
                                      array.array("d", [0.] * len(gtemp_34_phi)), array.array("d", gtemp_34_err2))

    if len(gtemp_12_phi) > 0:
        gtemp_12.SetMarkerStyle(20);  gtemp_12.SetMarkerSize(1.);  
        gtemp_12.SetMarkerColor(ROOT.kBlue);  gtemp_12.SetLineColor(ROOT.kBlue)
        gtemp2_12.SetMarkerStyle(24); gtemp2_12.SetMarkerSize(1.); 
        gtemp2_12.SetMarkerColor(ROOT.kBlue); gtemp2_12.SetLineColor(ROOT.kBlue)
    if len(gtemp_23_phi) > 0:
        gtemp_23.SetMarkerStyle(21);  gtemp_23.SetMarkerSize(1.);  
        gtemp_23.SetMarkerColor(ROOT.kRed);   gtemp_23.SetLineColor(ROOT.kRed)
        gtemp2_23.SetMarkerStyle(25); gtemp2_23.SetMarkerSize(1.); 
        gtemp2_23.SetMarkerColor(ROOT.kRed);  gtemp2_23.SetLineColor(ROOT.kRed)
    if len(gtemp_34_phi) > 0 and component[:4] == "dt13":
        gtemp_34.SetMarkerStyle(22);  gtemp_34.SetMarkerSize(1.25);  
        gtemp_34.SetMarkerColor(ROOT.kGreen+2);  gtemp_34.SetLineColor(ROOT.kGreen+2)
        gtemp2_34.SetMarkerStyle(26); gtemp2_34.SetMarkerSize(1.25); 
        gtemp2_34.SetMarkerColor(ROOT.kGreen+2); gtemp2_34.SetLineColor(ROOT.kGreen+2)

    if wheel == 0: htemp.SetTitle("Wheel %d" % wheel)
    else: htemp.SetTitle("Wheel %+d" % wheel)
    htemp.SetAxisRange(-window, window, "Y")
    htemp.SetXTitle("Average #phi of pair (rad)")
    if component == "dt13_resid": htemp.SetYTitle("#Deltax^{local} (mm)")
    if component == "dt13_slope": htemp.SetYTitle("#Deltadx/dz^{local} (mrad)")
    if component == "dt2_resid": htemp.SetYTitle("#Deltay^{local} (mm)")
    if component == "dt2_slope": htemp.SetYTitle("#Deltady/dz^{local} (mrad)")
    htemp.GetXaxis().CenterTitle()
    htemp.GetYaxis().CenterTitle()
    htemp.GetYaxis().SetTitleOffset(0.75)

    c1.Clear()
    htemp.Draw()
    if len(gtemp_12_phi) > 0:
        gtemp_12.Draw("p")
        gtemp2_12.Draw("p")
    if len(gtemp_23_phi) > 0:
        gtemp_23.Draw("p")
        gtemp2_23.Draw("p")
    if len(gtemp_34_phi) > 0:
        gtemp_34.Draw("p")
        gtemp2_34.Draw("p")

    tlegend = ROOT.TLegend(0.5, 0.72, 0.9, 0.92)
    tlegend.SetBorderSize(0)
    tlegend.SetFillColor(ROOT.kWhite)
    if len(gtemp_12_phi) > 0:
        tlegend.AddEntry(gtemp_12, "MB1 - MB2 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_12_val), stdev(gtemp_12_val)), "pl")
    if len(gtemp_23_phi) > 0:
        tlegend.AddEntry(gtemp_23, "MB2 - MB3 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_23_val), stdev(gtemp_23_val)), "pl")
    if len(gtemp_34_phi) > 0:
        tlegend.AddEntry(gtemp_34, "MB3 - MB4 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_34_val), stdev(gtemp_34_val)), "pl")
    if len(gtemp_12_phi) > 0:
        tlegend.AddEntry(gtemp_12, "total mean: %4.2f, total RMS: %4.2f" % \
                                   (mean(gtemp_12_val + gtemp_23_val + gtemp_34_val), 
                                   stdev(gtemp_12_val + gtemp_23_val + gtemp_34_val)), "")
    tlegend.Draw()


##################################################################################

def segdiffvsphicsc(tfiles, component, pair, window=5., **args):
    tdrStyle.SetOptTitle(1)
    tdrStyle.SetTitleBorderSize(1)
    tdrStyle.SetTitleFontSize(0.05)

    if not component[0:3] == "csc": Exception
    
    endcap = args["endcap"]
    if endcap=="m":
      endcapnum=2
      endcapsign="-"
    elif endcap=="p":
      endcapnum=1
      endcapsign="+"
    else: raise Exception
    
    station1 = int(str(pair)[0])
    station2 = int(str(pair)[1])
    if not station2-station1==1: raise Exception
    
    rings = [1,2]
    if station2==4: rings = [1]
    
    
    global htemp, gtemp_1, gtemp2_1, gtemp_2, gtemp2_2, tlegend
    htemp = ROOT.TH1F("htemp", "", 1, -pi*5./180., pi*(2.-5./180.))
    gtemp_1_phi, gtemp_1_val, gtemp_1_err, gtemp_1_val2, gtemp_1_err2 = [], [], [], [], []
    gtemp_2_phi, gtemp_2_val, gtemp_2_err, gtemp_2_val2, gtemp_2_err2 = [], [], [], [], []
    
    for ring in rings:
      chambers = xrange(1,37)
      if ring == 1: chambers = xrange(1,19)
      
      for chamber in chambers:
        phi, val, err, val2, err2, fit1, fit2, fit3 = segdiff(tfiles, component, pair, endcap=endcap, ring=ring, chamber=chamber)
        if fit1 and fit2 and fit3:
          if ring==1:
            gtemp_1_phi.append(phi)
            gtemp_1_val.append(val)
            gtemp_1_err.append(err)
            gtemp_1_val2.append(val2)
            gtemp_1_err2.append(err2)
          if ring==2:
            gtemp_2_phi.append(phi)
            gtemp_2_val.append(val)
            gtemp_2_err.append(err)
            gtemp_2_val2.append(val2)
            gtemp_2_err2.append(err2)

    #print "len(gtemp_12_phi) ", len(gtemp_12_phi)
    #print "len(gtemp_23_phi) ",len(gtemp_23_phi)
    #print "len(gtemp_34_phi) ",len(gtemp_34_phi)
    if len(gtemp_1_phi) > 0:
        gtemp_1 = ROOT.TGraphErrors(len(gtemp_1_phi), array.array("d", gtemp_1_phi), array.array("d", gtemp_1_val), 
                                     array.array("d", [0.] * len(gtemp_1_phi)), array.array("d", gtemp_1_err))
        gtemp2_1 = ROOT.TGraphErrors(len(gtemp_1_phi), array.array("d", gtemp_1_phi), array.array("d", gtemp_1_val2), 
                                      array.array("d", [0.] * len(gtemp_1_phi)), array.array("d", gtemp_1_err2))
    if len(gtemp_2_phi) > 0:
        gtemp_2 = ROOT.TGraphErrors(len(gtemp_2_phi), array.array("d", gtemp_2_phi), array.array("d", gtemp_2_val), 
                                     array.array("d", [0.] * len(gtemp_2_phi)), array.array("d", gtemp_2_err))
        gtemp2_2 = ROOT.TGraphErrors(len(gtemp_2_phi), array.array("d", gtemp_2_phi), array.array("d", gtemp_2_val2), 
                                      array.array("d", [0.] * len(gtemp_2_phi)), array.array("d", gtemp_2_err2))

    if len(gtemp_1_phi) > 0:
        gtemp_1.SetMarkerStyle(20);  gtemp_1.SetMarkerSize(1.);  
        gtemp_1.SetMarkerColor(ROOT.kBlue);  gtemp_1.SetLineColor(ROOT.kBlue)
        gtemp2_1.SetMarkerStyle(24); gtemp2_1.SetMarkerSize(1.); 
        gtemp2_1.SetMarkerColor(ROOT.kBlue); gtemp2_1.SetLineColor(ROOT.kBlue)
    if len(gtemp_2_phi) > 0:
        gtemp_2.SetMarkerStyle(21);  gtemp_2.SetMarkerSize(1.);  
        gtemp_2.SetMarkerColor(ROOT.kRed);  gtemp_2.SetLineColor(ROOT.kRed)
        gtemp2_2.SetMarkerStyle(25); gtemp2_2.SetMarkerSize(1.); 
        gtemp2_2.SetMarkerColor(ROOT.kRed); gtemp2_2.SetLineColor(ROOT.kRed)

    htemp.SetTitle("ME%s%d - ME%s%d" % (endcapsign,station2,endcapsign,station1))
    htemp.SetAxisRange(-window, window, "Y")
    htemp.SetXTitle("Average #phi of pair (rad)")
    if component == "csc_resid": htemp.SetYTitle("#Delta(r#phi)^{local} (mm)")
    if component == "csc_slope": htemp.SetYTitle("#Deltad(r#phi)/dz^{local} (mrad)")
    htemp.GetXaxis().CenterTitle()
    htemp.GetYaxis().CenterTitle()
    htemp.GetYaxis().SetTitleOffset(0.75)

    c1.Clear()
    htemp.Draw()
    if len(gtemp_1_phi) > 0:
        gtemp_1.Draw("p")
        gtemp2_1.Draw("p")
    if len(gtemp_2_phi) > 0:
        gtemp_2.Draw("p")
        gtemp2_2.Draw("p")

    tlegend = ROOT.TLegend(0.5, 0.72, 0.9, 0.92)
    tlegend.SetBorderSize(0)
    tlegend.SetFillColor(ROOT.kWhite)
    if len(gtemp_1_phi) > 0:
        tlegend.AddEntry(gtemp_1, "ring 1 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_1_val), stdev(gtemp_1_val)), "pl")
    if len(gtemp_2_phi) > 0:
        tlegend.AddEntry(gtemp_2, "ring 2 (mean: %4.2f, RMS: %4.2f)" % (mean(gtemp_2_val), stdev(gtemp_2_val)), "pl")
    #if len(gtemp_12_phi) > 0:
    #    tlegend.AddEntry(gtemp_12, "total mean: %4.2f, total RMS: %4.2f" % \
    #                               (mean(gtemp_12_val + gtemp_23_val + gtemp_34_val), 
    #                               stdev(gtemp_12_val + gtemp_23_val + gtemp_34_val)), "")
    tlegend.Draw()



##################################################################################
# makes a scatterplot of corrections coming either from reports (if xml geometries are None)
# or from geometryX and geometryY (WRT the common initial geometry0)

def corrections2D(reportsX=None, reportsY=None, geometry0=None, geometryX=None, geometryY=None, 
                  window=25., selection=None, name="tmp", canvas=None, pre_title_x=None, pre_title_y=None,
                  which="110011"):

  tdrStyle.SetOptStat(0)
  tdrStyle.SetStatW(0.40)

  # determine what are we plotting: report vs report  or  xml vs xml
  mode = None
  check_reports = False
  if reportsX is not None  and  reportsY is not None: 
    mode = "reports"
    check_reports = True
  if geometry0 is not None  and  geometryX is not None  and  geometryY is not None: 
    mode = "xmls"
  if mode is None:
    print "Either couple of reports or three geometries have to be given as input. Exiting..."
    return

  # setup ranges with the maximum [-window,window] that later will be optimized to [-wnd_adaptive,wnd_adaptive]
  wnd = [window]*6
  wnd_adaptive = [.1]*6
  
  global hx, hy, hz, hphix, hphiy, hphiz
  bins=2000
  hx    = ROOT.TH2F("%s_x" % name, "", bins, -wnd[0], wnd[0], bins, -wnd[0], wnd[0])
  hy    = ROOT.TH2F("%s_y" % name, "", bins, -wnd[1], wnd[1], bins, -wnd[1], wnd[1])
  hz    = ROOT.TH2F("%s_z" % name, "", bins, -wnd[2], wnd[2], bins, -wnd[2], wnd[2])
  hphix = ROOT.TH2F("%s_phix" % name, "", bins, -wnd[3], wnd[3], bins, -wnd[3], wnd[3])
  hphiy = ROOT.TH2F("%s_phiy" % name, "", bins, -wnd[4], wnd[4], bins, -wnd[4], wnd[4])
  hphiz = ROOT.TH2F("%s_phiz" % name, "", bins, -wnd[5], wnd[5], bins, -wnd[5], wnd[5])
  hhh = [hx, hy, hz, hphix, hphiy, hphiz]
  
  # initialize PCA objects
  global pca_x, pca_y, pca_z, pca_phix, pca_phiy, pca_phiz
  pca_x = ROOT.TPrincipal(2,"D")
  pca_y = ROOT.TPrincipal(2,"D")
  pca_z = ROOT.TPrincipal(2,"D")
  pca_phix = ROOT.TPrincipal(2,"D")
  pca_phiy = ROOT.TPrincipal(2,"D")
  pca_phiz = ROOT.TPrincipal(2,"D")
  pcas = [pca_x, pca_y, pca_z, pca_phix, pca_phiy, pca_phiz]
  
  # arrays to later fill graphs with
  ax=[]; ay=[]; az=[]; aphix=[]; aphiy=[]; aphiz=[]
  aaa = [ax, ay, az, aphix, aphiy, aphiz]
  
  # list of postal addresses
  postal_addresses = []
  
  # if reports are given, use them to fill addresses and do extra checks
  if check_reports:
    for r1 in reportsX:
      # skip ME1/a
      if r1.postal_address[0]=='CSC'  and  r1.postal_address[2]==1 and r1.postal_address[3]==4: continue
      if selection is None or (selection.__code__.co_argcount == len(r1.postal_address) and selection(*r1.postal_address)):
        r2 = getReportByPostalAddress(r1.postal_address, reportsY)
        if r2 is None: 
          print "bad r2 in ",r1.postal_address
          continue
      
        if r1.status != "PASS" or r2.status != "PASS":
          print "bad status", r1.postal_address, r1.status, r2.status
          continue
      postal_addresses.append(r1.postal_address)
  # otherwise, use chamber addresses from xmls
  else:
    for key in geometry0.dt.keys():
      if len(key)==3  and  key in geometryX.dt  and  key in geometryY.dt:
        postal_addresses.append( tuple(['DT'] + list(key)) )
    for key in geometry0.csc.keys():
      # skip ME1/a
      if key[2]==1 and key[3]==4: continue
      if len(key)==4  and  key in geometryX.csc  and  key in geometryY.csc:
        postal_addresses.append( tuple(['CSC'] + list(key)) )

  # fill the values
  for addr in postal_addresses:

    # checks the selection function
    if not (selection is None or (selection.__code__.co_argcount == len(addr) and selection(*addr)) ): continue

    factors = [10. * signConventions[addr][0], 10. * signConventions[addr][1], 10. * signConventions[addr][2],
               1000., 1000., 1000. ]

    if check_reports:
      rX = getReportByPostalAddress(addr, reportsX)
      rY = getReportByPostalAddress(addr, reportsY)
      deltasX = [rX.deltax, rX.deltay, rX.deltaz, rX.deltaphix, rX.deltaphiy, rX.deltaphiz]
      deltasY = [rY.deltax, rY.deltay, rY.deltaz, rY.deltaphix, rY.deltaphiy, rY.deltaphiz]
      
    if mode == "reports":

      checks = map( lambda d1, d2: d1 is not None  and  d2 is not None  and  d1.error is not None   \
                                   and  d2.error is not None and (d1.error**2 + d2.error**2) > 0. , \
                    deltasX, deltasY)

      for i in range(len(checks)):
        if not checks[i]: continue
        fillX = deltasX[i].value * factors[i]
        fillY = deltasY[i].value * factors[i]
        aaa[i].append([fillX,fillY])
        pcas[i].AddRow(array.array('d',[fillX,fillY]))
        mx = max(abs(fillX), abs(fillY))
        if mx > wnd_adaptive[i]: wnd_adaptive[i] = mx
        
    if mode == "xmls":

      db0 = dbX = dbY = None
      if addr[0] == "DT":
        db0, dbX, dbY  = geometry0.dt[addr[1:]], geometryX.dt[addr[1:]], geometryY.dt[addr[1:]]
      if addr[0] == 'CSC':
        db0, dbX, dbY  = geometry0.csc[addr[1:]], geometryX.csc[addr[1:]], geometryY.csc[addr[1:]]

      checks = [True]*6
      if check_reports:
        checks = map( lambda d1, d2: d1 is not None  and  d2 is not None ,  deltasX, deltasY)

      gdeltas0 = [db0.x, db0.y, db0.z, db0.phix, db0.phiy, db0.phiz]
      gdeltasX = [dbX.x, dbX.y, dbX.z, dbX.phix, dbX.phiy, dbX.phiz]
      gdeltasY = [dbY.x, dbY.y, dbY.z, dbY.phix, dbY.phiy, dbY.phiz]

      for i in range(len(checks)):
        if not checks[i]: continue
        fillX = (gdeltasX[i] - gdeltas0[i]) * factors[i]
        fillY = (gdeltasY[i] - gdeltas0[i]) * factors[i]
        aaa[i].append([fillX,fillY])
        pcas[i].AddRow(array.array('d',[fillX,fillY]))
        mx = max(abs(fillX), abs(fillY))
        if mx > wnd_adaptive[i]: wnd_adaptive[i] = mx
        #if addr[0] == 'CSC' and i==1 and (abs(fillX)>0.01 or abs(fillY)>0.01): print addr, ": hugeCSC i=%d dx=%.03g dy=%.03g"%(i,fillX,fillY)
        #if addr[0] == 'CSC' and i==2 and (abs(fillX)>0.02 or abs(fillY)>0.02): print addr, ": hugeCSC i=%d dx=%.03g dy=%.03g"%(i,fillX,fillY)
        #if addr[0] == 'CSC' and i==3 and (abs(fillX)>0.05 or abs(fillY)>0.05): print addr, ": hugeCSC i=%d dx=%.03g dy=%.03g"%(i,fillX,fillY)

  if mode == "xmls":  
    if pre_title_x is None: pre_title_x = "geometry 1 "
    if pre_title_y is None: pre_title_y = "geometry 2 "
  if mode == "reports":  
    if pre_title_x is None: pre_title_x = "iteration's "
    if pre_title_y is None: pre_title_y = "other iteration's "
  tmptitles =  ["#Deltax (mm)", "#Deltay (mm)", "#Deltaz (mm)", 
                "#Delta#phi_{x} (mrad)", "#Delta#phi_{y} (mrad)", "#Delta#phi_{z} (mrad)"]
  htitles = []
  for t in tmptitles: htitles.append([pre_title_x + t, pre_title_y + t])
  
  if canvas is not None: c = canvas
  else: c = c1
  c.Clear()
  ndraw = which.count('1')
  if ndraw > 4: c.Divide(3, 2)
  elif ndraw > 2: c.Divide(2, 2)
  elif ndraw > 1: c.Divide(2, 1)

  global lines, graphs, texs
  lines = [];  graphs = []; texs = []
  
  ipad = 0
  for i in range(6):
    
    # decode 'which' binary mask
    if ( int(which,2) & (1<<i) ) == 0: continue
    
    ipad += 1
    c.GetPad(ipad).cd()
    c.GetPad(ipad).SetGridx(1)
    c.GetPad(ipad).SetGridy(1)
    
    wn = 1.08 * wnd_adaptive[i]
    hhh[i].GetXaxis().SetRangeUser(-wn, wn)
    hhh[i].GetYaxis().SetRangeUser(-wn, wn)
    hhh[i].SetXTitle(htitles[i][0])
    hhh[i].SetYTitle(htitles[i][1])
    hhh[i].GetXaxis().CenterTitle()
    hhh[i].GetYaxis().CenterTitle()
    hhh[i].Draw()
    
    if len(aaa[i]) == 0: continue

    a1, a2 = map( lambda x: array.array('d',x),  list(zip(*aaa[i])) )
    g = ROOT.TGraph(len(a1), a1, a2)
    g.SetMarkerStyle(5)
    g.SetMarkerSize(0.3)
    g.SetMarkerColor(ROOT.kBlue)
    graphs.append(g)

    pcas[i].MakePrincipals()
    #pcas[i].Print()
    #pcas[i].MakeHistograms()    
    b = pcas[i].GetEigenVectors()(1,0) / pcas[i].GetEigenVectors()(0,0)
    a = pcas[i].GetMeanValues()[1] - b * pcas[i].GetMeanValues()[0]
    #print a, b, "   ", pcas[i].GetEigenValues()[0], pcas[i].GetEigenValues()[1]
    
    cov = pcas[i].GetCovarianceMatrix()
    r = cov(0,1)/sqrt(cov(1,1)*cov(0,0))
    print "r, RMSx, RMSy =", r, g.GetRMS(1), g.GetRMS(2)
    texrms = ROOT.TLatex(0.17,0.87, "RMS x,y = %.02g, %.02g" % (g.GetRMS(1),g.GetRMS(2)))
    texr = ROOT.TLatex(0.17,0.80, "r = %.02g" % r)
    for t in texr, texrms:
      t.SetNDC(1)
      t.SetTextColor(ROOT.kBlue)
      t.SetTextSize(0.053)
      t.Draw()
      texs.append(t)
    
    g.Draw("p")

    if not isnan(b):
      wn = wnd_adaptive[i]
      line = ROOT.TLine(-wn, a - b*wn, wn, a + b*wn)
      line.SetLineColor(ROOT.kRed)
      line.Draw()
      lines.append(line)

  #return hx, hy, hphiy, hphiz, pca_x, pca_y, pca_phiy, pca_phiz
  return aaa
