#!/usr/bin/env python3 

from __future__ import print_function
import ROOT
import sys
import getopt
from ROOT import *

from copy import deepcopy

ROOT.gROOT.SetBatch()        # don't pop up canvases
# ROOT.gROOT.SetStyle('Plain') # white background

#####################################################
### GLOBAL VARS
#
rocsInCol = 2
rocsInRow = 8;
rocXLen = 1.0 / rocsInRow

maxRocIdx = rocsInCol * rocsInRow - 1 # 0...15

onlineMaxLadder = [6, 14, 22, 32]
onlineMaxBlade = [11, 17]

maxOnlineModule = 4
maxOnlineDisk = 3
#
useNumberAsPartName = False  # does it accept numbers or mO, pI etc.
inputFileName = "input.dat" #default file name
hRes, vRes = 1920, 1080
useFileSuffix = False
colorCoded = False
pixelAlive = False
### GLOBAL VARS

class HistogramManager:
  def __init__(self):
    self.barrelHists = []
    self.forwardHists = []
    
    # barrel histograms
    i = 1
    for maxLadder in onlineMaxLadder:
      nBinsX = (maxOnlineModule * 2 + 1) * rocsInRow
      nBinsY = (maxLadder * 2 + 1) * rocsInCol

      name = "PXBarrel_Layer" + str(i);
      title = "PXBarrel_Layer" + str(i);

      histObj = ROOT.TH2F(name, title, nBinsX, -(maxOnlineModule + 0.5), maxOnlineModule + 0.5, nBinsY, -(maxLadder + 0.5), maxLadder + 0.5)
      histObj.GetXaxis().SetTitle("OnlineSignedModule");
      histObj.GetYaxis().SetTitle("OnlineSignedLadder");
      histObj.SetOption("COLZ")
      histObj.SetStats(0)
      histObj.SetTitle("OnlineSignedModule")

      self.barrelHists.append(deepcopy(histObj))
      i = i + 1

    # forward histograms
    i = 1
    for maxBlade in onlineMaxBlade:
      nBinsX = (maxOnlineDisk * 2 + 1) * rocsInRow
      nBinsY = (maxBlade * 2 + 1) * 2 * rocsInCol

      name = "PXForward_Ring" + str(i);
      title = "PXForward_Ring" + str(i);

      histObj = ROOT.TH2F(name, title, nBinsX, -(maxOnlineDisk + 0.5), maxOnlineDisk + 0.5, nBinsY, -(maxBlade + 0.5), maxBlade + 0.5)
      histObj.GetXaxis().SetTitle("OnlineSignedDisk");
      histObj.GetYaxis().SetTitle("OnlineSignedBladePanel");
      histObj.SetOption("COLZ")
      histObj.SetStats(0)
      

      self.forwardHists.append(deepcopy(histObj))
      i = i + 1

  def fillHistograms(self, barrelObjs, forwardObjs):
    for b in barrelObjs:
      coord = b.GetXYCoords()
      histRef = self.barrelHists[b.layer - 1]
      
      binNum = histRef.FindBin(coord[0], coord[1])
      if colorCoded:
        binContent = TranslateReasonStringBPix(b.reason)
      elif pixelAlive:
        binContent = float(b.reason)
      else:
        binContent = b.roc + 1

      histRef.SetBinContent(binNum, binContent)
      # self.barrelHists[b.layer - 1].Fill(coord[0], coord[1], b.roc + 1) 

    for f in forwardObjs:
      coord = f.GetXYCoords()
      histRef = self.forwardHists[f.ring - 1]
      
      binNum = histRef.FindBin(coord[0], coord[1])

      if colorCoded:
        binContent = TranslateReasonStringFPix(f.reason)
      else:
        binContent = f.roc + 1
      histRef.SetBinContent(binNum, binContent)      

      # self.forwardHists[f.ring - 1].Fill(coord[0], coord[1], f.roc + 1)
  
  def drawLine(self, lineObj, x1, x2, y1, y2, width=2, style=1, color=1):  
    lineObj.SetLineWidth(width)
    lineObj.SetLineStyle(style)
    lineObj.SetLineColor(color)

    lineObj.DrawLine(x1, y1, x2, y2)

  def drawRectangle(self, lineObj, x1, x2, y1, y2, width=2, style=1, color=1):
    self.drawLine(lineObj, x1, x2, y2, y2, width, style, color)
    self.drawLine(lineObj, x2, x2, y2, y1, width, style, color)
    self.drawLine(lineObj, x2, x1, y1, y1, width, style, color)
    self.drawLine(lineObj, x1, x1, y1, y2, width, style, color)

  def prettifyCanvas(self, hist):
    nBinsX = hist.GetXaxis().GetNbins()
    xMin = hist.GetXaxis().GetXmin()
    xMax = hist.GetXaxis().GetXmax()
    nBinsY = hist.GetYaxis().GetNbins()
    yMin = hist.GetYaxis().GetXmin()
    yMax = hist.GetYaxis().GetXmax()

    xLen = int(xMax)
    yLen = int(yMax)

    name = hist.GetName()[0:3]
    isBarrel = True if name != "PXF" else False
    print((name, isBarrel))

    xBaseStep = 1
    xRange = (nBinsX - 1) // (rocsInRow * 2) + 1
    yBaseStep = (yMax - yMin) / nBinsY
    yRange = (nBinsY - 1) // (2) + 1
    if not isBarrel:
      yBaseStep = yBaseStep * 2
      yRange = yRange // 2

    # horizontal 
    x1 = xMin
    x2 = xMin + xLen
    y1 = yMin
    y2 = yMin

    lineObj = ROOT.TLine()
    lineObj.SetBit(ROOT.kCanDelete)

    for i in range(yRange):
      w = 1 if i % 2 else 2
      self.drawLine(lineObj, x1, x2, y1, y2, w)
      self.drawLine(lineObj, x1, x2, -y1, -y2, w)
      self.drawLine(lineObj, -x1, -x2, -y1, -y2,w )
      self.drawLine(lineObj, -x1, -x2, y1, y2, w)

      y1 = y1 + yBaseStep
      y2 = y2 + yBaseStep

    # vertical
    x1 = xMin
    x2 = xMin
    y1 = yMin
    y2 = yMin + yLen

    for i in range(xRange):
      self.drawLine(lineObj, x1, x2, y1, y2, style = 9)
      self.drawLine(lineObj, x1, x2, -y1, -y2, style = 9)
      self.drawLine(lineObj, -x1, -x2, -y1, -y2, style = 9)
      self.drawLine(lineObj, -x1, -x2, y1, y2, style = 9)

      x1 = x1 + xBaseStep
      x2 = x2 + xBaseStep

    # mark zero ROC
    zeroModuleHeight = yBaseStep if isBarrel else yBaseStep * 0.5 # because there are two panels height of roc is smaller

    yRange = int(yMax) if isBarrel else int(yMax) * 2
    
    x1_base = 0                         + xMin
    x2_base = xBaseStep / float(rocsInRow) + xMin
    y1_base = zeroModuleHeight          + yMin
    y2_base = 2 * zeroModuleHeight      + yMin

    for i in range(yRange):
      y1 = y1_base + i * (zeroModuleHeight * 2) - (zeroModuleHeight if i % 2 else 0)
      y2 = y2_base + i * (zeroModuleHeight * 2) - (zeroModuleHeight if i % 2 else 0)
      
      #negative ladders/blades
      for j in range(int(xMax)):
        x1 = x1_base + j * xBaseStep
        x2 = x2_base + j * xBaseStep
        if yMax == 6.5 and x1 <0:
          y1 = y1_base + i * (zeroModuleHeight * 2) + (zeroModuleHeight if i % 2 else 0)
          y2 = y2_base + i * (zeroModuleHeight * 2) + (zeroModuleHeight if i % 2 else 0)
          self.drawRectangle(lineObj,(xBaseStep+x1-(x2-x1)),(xBaseStep+x2-(x2-x1)), y1+(y1-y2), y2+(y1-y2), color=8)
          x1, x2 = -x1, -x2
          yPosChange = -zeroModuleHeight if i % 2 else zeroModuleHeight
          self.drawRectangle(lineObj, x1, x2, y1 - yPosChange-2*(zeroModuleHeight if i % 2 else 0), y2 - yPosChange-2*(zeroModuleHeight if i % 2 else 0), color=8)
        else:
          self.drawRectangle(lineObj, x1, x2, y1, y2, color=8)

          x1, x2 = -x1, -x2
          yPosChange = -zeroModuleHeight if i % 2 else zeroModuleHeight
          self.drawRectangle(lineObj, x1, x2, y1 - yPosChange, y2 - yPosChange, color=8)


      # positive ladders/blades
      y1 = y1 - yMin + yBaseStep
      y2 = y2 - yMin + yBaseStep

      for j in range(int(xMax)):
        x1 = x1_base + j * xBaseStep
        x2 = x2_base + j * xBaseStep

        if yMax== 6.5 and x1 <0:
          self.drawRectangle(lineObj, xBaseStep+x1-(x2-x1), xBaseStep+x2-(x2-x1), y1+(y1-y2), y2+(y1-y2), color=8)
          x1, x2 = -x1, -x2
          self.drawRectangle(lineObj, x1, x2, y1 - yPosChange- 2*(zeroModuleHeight if i % 2 else 0), y2 - yPosChange-2*(zeroModuleHeight if i % 2 else 0), color=8)
        else:
          self.drawRectangle(lineObj, x1, x2, y1, y2, color=8)
          x1, x2 = -x1, -x2
          self.drawRectangle(lineObj, x1, x2, y1 - yPosChange, y2 - yPosChange, color=8)

#      hist.GetZaxis().SetRangeUser(-0.5,15.5)

  def saveHistograms(self):
    for hists in [self.barrelHists, self.forwardHists]:
      for hist in hists:
        # if hist.GetEntries():
        c1 = ROOT.TCanvas(hist.GetName(), hist.GetName(), hRes, vRes)
        if colorCoded:
          hist.GetZaxis().SetRangeUser(0,5)
          ROOT.gStyle.SetPalette(55)
        elif pixelAlive:
          hist.GetZaxis().SetRangeUser(0,4160)
          ROOT.gStyle.SetPalette(70)
        hist.Draw()

        txt = TLatex()
        txt.SetNDC()
        txt.SetTextFont(1)
        txt.SetTextColor(1)
        txt.SetTextAlign(22)
        txt.SetTextAngle(0)
        txt.SetTextSize(0.05)
        txt.DrawLatex(0.5, 0.95, hist.GetName())

        xMin = hist.GetXaxis().GetXmin()

        yMin = hist.GetYaxis().GetXmin()

        box1 = TBox(xMin*1.1,yMin*1.25,xMin*1,yMin*1.15);
        box1.SetFillColor(kRed+3)
        box1.Draw()
        txt.SetTextSize(0.035)
        txt.DrawLatex(0.25, 0.077, "Dead At Beginning")


        box2 = TBox(xMin*0.45,yMin*1.25,xMin*0.35,yMin*1.15);
        box2.SetFillColor(kAzure+2)
        box2.Draw()
        txt.SetTextSize(0.035)
        txt.DrawLatex(0.47, 0.077, "Dead At End")


        self.prettifyCanvas(hist)
        colorString = ""
        if colorCoded:
          colorString = "_coded"
        elif pixelAlive:
          colorString = "_pixelalive"
        if useFileSuffix:
          c1.Print(hist.GetName() + colorString + "_" + inputFileName[:-4] + ".png")
        else:
          c1.Print(hist.GetName() + colorString + ".png")
            
#####################################################

class Barrel:
  def __init__ (self, part, sector, layer, ladder, module, roc, reason="unknown"):
    self.part = part
    self.sector = sector
    self.layer = layer
    self.ladder = ladder
    self.module = module
    self.roc = roc
    self.isCoverted = False
    self.reason = reason
  def __str__(self):
    return str([self.part, self.sector, self.layer, self.ladder, self.module])
  def convertParts(self):
    if not self.isCoverted:
      self.ladder = -self.ladder if self.part % 2 else self.ladder
      self.module = -self.module if self.part <= 2 else self.module
      isConverted = True
  def GetXYCoords(self):

    xBase = -0.625 + ((maxRocIdx - self.roc if self.roc >= rocsInRow else self.roc) + 1) * rocXLen

    flipY = False
    if self.module < 0:
      if self.ladder < 0:
        if abs(self.ladder) % 2:
          flipY = True
      else:
        if self.ladder % 2 == 0:
          flipY = True
    else:
      if self.ladder < 0:
        if abs(self.ladder) % 2 == 0:
          flipY = True
      else:
        if self.ladder % 2:
          flipY = True

    tmpRoc = maxRocIdx - self.roc if flipY else self.roc;

    yBase = -0.5 * (tmpRoc // rocsInRow)

    x = self.module + (xBase if self.module < 0 else -xBase - rocXLen)
    y = self.ladder + yBase   
            
    #print("roc=%d\t: (%f;%f)"%(self.roc, x, y))
    
    return x, y    

class Forward:
  def __init__ (self, part, disk, blade, panel, ring, roc, reason="unknown"):
    self.part = part
    self.disk = disk
    self.blade = blade
    self.panel = panel
    self.ring = ring
    self.roc = roc
    self.reason = reason
    self.isCoverted = False
  def __str__(self):
    return str([self.part, self.disk, self.blade, self.panel, self.ring])
  def convertParts(self):
    if not self.isCoverted:
      self.blade = -self.blade if self.part % 2 else self.blade
      self.disk = -self.disk if self.part <= 2 else self.disk 
      self.isCoverted = True
  def GetXYCoords(self):

    xBase = -0.625 + ((maxRocIdx - self.roc if self.roc >= rocsInRow else self.roc) + 1) * rocXLen

    x = self.disk + (xBase if self.disk < 0 else -xBase - rocXLen)
    
    flipY = (self.panel == 2 if self.disk < 0 else self.panel == 1)

    tmpRoc = maxRocIdx - self.roc if flipY else self.roc;

    yBase = -0.25 - 0.25 * (tmpRoc // rocsInRow) + 0.5 * (self.panel - 1)

    y = self.blade + yBase

    # print("roc=%d\t: (%f;%f)"%(self.roc, x, y))
    return x, y

def TranslatePartString(thePartStr):
  if thePartStr == "mO":
    return 1
  elif thePartStr == "mI":
    return 2
  elif thePartStr == "pO":
    return 3
  elif thePartStr == "pI":
    return 4
  else:
    print("Unrecognized part <%s>, the script is likely to crash..." % (thePartStr))

def TranslateReasonStringBPix(theReasonStr):
  if theReasonStr == "unknown":
    return 1
  elif theReasonStr == "notprogrammable":
    return 1
  elif theReasonStr == "vcthr":
    return 2
  elif theReasonStr == "pixelalive":
    return 2
  elif theReasonStr == "iana":
    return 2
  elif theReasonStr == "calib":
    return 2
  elif theReasonStr== "fedphases":
    return 4
  elif theReasonStr == "tbmdelay":
    return 1
  elif theReasonStr == "power":
    return 5
  else:
    return 1
    print("Unrecognized part <%s>, the script is likely to crash..." % (theReasonStr))

def TranslateReasonStringFPix(theReasonStr):
  if theReasonStr == "flaky":
    return 1
  elif theReasonStr == "power":   #check github for the real reason
    return 5
  elif theReasonStr == "tbmdelay":  #
    return 1
  elif theReasonStr == "unknown":
    return 2
  else:
    return 2
    print("Unrecognized part <%s>, the script is likely to crash..." % (theReasonStr))

    
def GetOnlineBarrelCharacteristics(detElements, roc, reason="unknown"):
  onlinePart = int(detElements[1][1:]) if useNumberAsPartName else TranslatePartString(detElements[1][1:])
  onlineSector = int(detElements[2][3:])
  onlineLayer = int(detElements[3][3:])
  
  if detElements[4][-1] == "H" or detElements[4][-1] == "F":
    onlineLadder = int(detElements[4][3:-1])
  else:
    onlineLadder = int(detElements[4][3:])
    
  onlineModule = int(detElements[5][3:])

  return Barrel(*[onlinePart, onlineSector, onlineLayer, onlineLadder, onlineModule, roc, reason])

def GetOnlineForwardCharacteristics(detElements, roc, reason="unknown"):
  onlinePart = int(detElements[1][1:]) if useNumberAsPartName else TranslatePartString(detElements[1][1:])
  onlineDisk = int(detElements[2][1:])
  onlineBlade = int(detElements[3][3:])
  onlinePanel = int(detElements[4][3:])
  onlineRing = int(detElements[5][3:])

  return Forward(*[onlinePart, onlineDisk, onlineBlade, onlinePanel, onlineRing, roc, reason])


def GetAffectedRocs(rocString):

    rocString = str(rocString)
    iComma=rocString.find(",")
    listOfRocs = []
    
    if iComma!=-1:
        listOfRocs.extend(GetAffectedRocs(rocString[0:iComma]))
        listOfRocs.extend(GetAffectedRocs(rocString[iComma+1:len(rocString)]))
    else:
        iHyphen=rocString.find("-")
        if iHyphen!=-1:
            start=int(rocString[0:iHyphen])
            end=int(rocString[iHyphen+1:len(rocString)])+1
            listOfRocs.extend(range(start,end))
        else:
            return [int(rocString)]
        
    return listOfRocs
    


#####################################################

histMan = HistogramManager()
barrelObjs = []
forwardObjs = []

if len(sys.argv) > 1:
  inputFileName = sys.argv[1]
  print(inputFileName)
  
  if len(sys.argv) > 2:
    opts, args = getopt.getopt(sys.argv[2:], "bscp", ["help", "output="])
    for o, a in opts:
      if o == "-b":
        useNumberAsPartName = False
      if o == "-s":
        useFileSuffix = True
      if o == "-c":
        colorCoded = True
      if o == "-p":
        pixelAlive = True

i = 1
with open (inputFileName, "r") as inputFile:

  for item in inputFile:
#    print("Processing record #%d" % (i))
    
    inputs = item.split(" ")

    if len(inputs) >= 2: # but take only first 2 elements (ignore others like '\n')

      detElements = inputs[0].split("_")
      if detElements[3]=='LYR1' and (detElements[1]=='BmI' or detElements[1]=='BmO'):

        rocs = []
        roc = GetAffectedRocs(inputs[1])
        for roc_rotate in roc:
          if int(str(roc_rotate)) <= 7:
            rocs.append(int(str(roc_rotate))+8)
          elif int(str(roc_rotate)) >= 8:
            rocs.append(int(str(roc_rotate)) -8)
      else:
        rocs = GetAffectedRocs(inputs[1])
#      rocs = GetAffectedRocs(inputs[1]) #int(inputs[1]) #- 1 #shifts 0..16 rocNum to 0..15

      if len(inputs) == 3:
        reason = str(inputs[2]).lower().strip()
      else:
        reason="unknown"

      for roc in rocs:
        if detElements[0][0] == "B":
          barrelObj = GetOnlineBarrelCharacteristics(detElements, roc, reason)
          #print(barrelObj)
          barrelObj.convertParts()
          # print(barrelObj)
          barrelObjs.append(barrelObj)
        elif detElements[0][0] == "F":
          forwardObj = GetOnlineForwardCharacteristics(detElements, roc, reason)
          # print(forwardObj)
          forwardObj.convertParts()
          # print(forwardObj)
          forwardObjs.append(forwardObj)
        else:
          print("Not recognized part type")
          
        i = i + 1

histMan.fillHistograms(barrelObjs, forwardObjs)
histMan.saveHistograms()  
