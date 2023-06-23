#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
from ROOT import *
from copy import deepcopy
from array import array

gROOT.SetBatch()        # don't pop up canvases

#Find Data files

def getFileInPath(rfile):
   import os 
   for dir in os.environ['CMSSW_SEARCH_PATH'].split(":"):
     if os.path.exists(os.path.join(dir,rfile)): return os.path.join(dir,rfile)                                                                          
   return None


# Default values
inputFileName = "DQM_V0013_R000292154__StreamExpressCosmics__Commissioning2017-Express-v1__DQMIO.root"
limitsFileName = "limits.dat"
outputDirectoryName = "OUT/"
minMaxFileName = "minmax.out"
#detIDsFileName = "DATA/detids.dat"
detIDsFileName = getFileInPath('DQM/SiStripMonitorClient/data/detids.dat')
#default one
baseRootDirs = ["DQMData/Run 292154/PixelPhase1/Run summary/Phase1_MechanicalView"
                ,"DQMData/Run 292154/PixelPhase1/Run summary/Tracks"
                ]
                
                    
maxPxBarrel = 4
maxPxForward = 3
barrelLadderShift = [0, 14, 44, 90]

forwardDiskXShift = [25, 75, 125]
forwardDiskYShift = 45; # to make +DISK on top in the 'strip-like' layout

plotWidth, plotHeight = 3000, 2000
extremeBinsNum = 20

limits = ["num_digis 0.01 90 1 0",
          "num_clusters 0.01 25 1 0",
          "Trechitsize_y 0.01 10 0 0",
          "Trechitsize_x 0.01 10 0 0",
          "Tresidual_y 0.0000001 0.004 0 1",
          "Tresidual_x 0.0000001 0.004 0 1",
          "Tcharge 2000 80000 0 0",
          "Thitefficiency 0.95 1 0 0",
          #"Tmissing 0.01 500 0 0",
          "Tnum_clusters_ontrack 0.01 15 1 0",
          "Tsize 0.01 15 0 0",
          #"Tvalid 0.01 90 0 0",
          "adc 0.01 256 0 0",
          "charge 2000 80000 0 0",
          "size 0.01 15 0 0",]

class TH2PolyOfflineMaps:
  
  ###
  # LOTS OF CODE BORROWED FROM: PYTHONBINREADER, PIXELTRACKERMAP
  ###
  
  ############################################################################
  
  def __TraverseDirTree(self, dir):
    
    try:
      currPath = (dir.GetPath().split(":/"))[1]
    except:
      print("Exception raised: Path not found in the input file")
      return
  
    for obj in dir.GetListOfKeys():
      if not obj.IsFolder():
        if obj.ReadObjectAny(TClass.GetClass("TH2")):
          th2 = deepcopy(obj.ReadObj())
          name = th2.GetName()
          if 6 < th2.GetNbinsX() < 10 and name.find("per") != -1 and name.find("Lumisection") == -1: #take only module lvl plots
            print(''.join([dir.GetPath(), '/', name]))
            
            # fix when there are plots starting with the same strings in different directories
            prefix = ""
            for i in self.dirs:
              if currPath.startswith(i):
                prefix = self.dirsAliases[i]
                break
            # print(currPath, prefix)
            th2.SetName(prefix + th2.GetName())
            self.listOfNumHistograms.append(th2)
      else:
        self.__TraverseDirTree(obj.ReadObj())     
  
  def __GetPartStr(self, isXlowerThanZero, isYlowerThanZero):
    if isXlowerThanZero and isYlowerThanZero:
      return "mO"
    if isXlowerThanZero and isYlowerThanZero == False:
      return "mI"
    if isXlowerThanZero == False and isYlowerThanZero:
      return "pO"
    if isXlowerThanZero == False and isYlowerThanZero == False:
      return "pI"
      
  def __GetBarrelSector(self, layer, signedLadder, signedModule): #adapted from PixelBarrelName
    theLadder = abs(signedLadder)
    theModule = abs(signedModule)
    
    sector = 0
    
    if layer == 1:
    
      if theLadder == 1:
        if theModule >= 2:
          return 1
        else:
          return 2
      if theLadder == 2:
        if theModule >= 3:
          return 2
        else:
          return 3
      if theLadder == 3:
        if theModule >= 4:
          return 3
        else:
          return 4
      if theLadder == 4:
        if theModule >= 2:
          return 5
        else:
          return 6
      if theLadder == 5:
        if theModule >= 3:
          return 6
        else:
          return 7
      if theLadder == 6:
        if theModule >= 4:
          return 7
        else:
          return 8
    # here is used simplified form of assignment, see source file for reference
    elif layer == 2:
      i = theLadder // 5
      sector = i * 3
      shortLadder = theLadder - 5 * i
      for i in range(0, shortLadder, 2):
        sector = sector + 1
      return sector
    elif layer == 3:
      sector = 1
      for i in range(2, theLadder, 3):
        if (i + 1) % 3 == 0:
          sector = sector + 1
      return sector
    elif layer == 4:
      sector = (theLadder + 3) // 4
      return sector
  
  def __BuildOnlineBarrelName(self, signedModule, signedLadder, layer): #in Phase1 it is assumed that there are only full modules
    thePart = self.__GetPartStr(signedModule < 0, signedLadder < 0)
    theSector = str(self.__GetBarrelSector(layer, signedLadder, signedModule))
    return "BPix_B" + thePart + "_SEC" + theSector + "_LYR" + str(layer) + "_LDR" + str(abs(signedLadder)) + "F_MOD" + str(abs(signedModule))
  
  def __BuildOnlineDiskName(self, signedDisk, signedBlade, panel, ring):
    thePart = self.__GetPartStr(signedDisk < 0, signedBlade < 0)
    return "FPix_B" + thePart + "_D" + str(abs(signedDisk)) + "_BLD" + str(abs(signedBlade)) + "_PNL" + str(panel) + "_RNG" + str(ring) 
  
  def __GroupHistograms(self):
    currentGroupName = ""
    groupOfHists = []
    self.groupedHistograms = []
    
    ##### GROUP ALL LAYERS/RINGS HAVING SIMILAR INFORMATION
    for obj in self.listOfNumHistograms:  
      objName = obj.GetName()
      objNameSplit = objName.split("_")
      objNameCollected = ''.join(objNameSplit[0:-1])
      if objNameCollected != currentGroupName:
        if len(groupOfHists):
          self.groupedHistograms.append(groupOfHists)
          groupOfHists = []
          
        currentGroupName = objNameCollected
      groupOfHists.append(obj)
    self.groupedHistograms.append(groupOfHists) #the last group
    
  def __AddNamedBins(self, geoFile, tX, tY, sX, sY, applyModuleRotation = False):

    for line in geoFile:
      lineSpl = line.strip().split("\"")
      #New TH2Poly bin ID = full_name_(detId)
      detId = str(lineSpl[0].split(" ")[1])+"_("+str(lineSpl[0].split(" ")[0])+")"
      vertices = lineSpl[1]
      xy = vertices.split(" ")
      x, y = array('d'), array('d')
      verNum = 1
      for coord in xy:
        coordSpl = coord.split(",")
        if applyModuleRotation:
          x.append(-(float(coordSpl[0]) * sX + tX))
          y.append((float(coordSpl[1]) * sY + tY))
        else:
          x.append(float(coordSpl[0]) * sX + tX)
          y.append(float(coordSpl[1]) * sY + tY)
        verNum = verNum + 1
      #close polygon
      x.append(x[0])
      y.append(y[0])
      
      if applyModuleRotation:
        bin = TGraph(verNum, y, x)
      else:
        bin = TGraph(verNum, x, y)
      # bin = TGraph(verNum, y, x) # rotation by 90 deg (so that it had the same layout as for the strips)
      bin.SetName(detId)
      
      self.__BaseTrackerMap.AddBin(bin)
    
  def __CreateTrackerBaseMap(self):
  
    self.__BaseTrackerMap = TH2Poly("Summary", "", -10, 160, -70, 70)
    # self.__BaseTrackerMap = TH2Poly("Summary", "Tracker Map", 0, 0, 0, 0)
    self.__BaseTrackerMap.SetFloat(1)
    self.__BaseTrackerMap.GetXaxis().SetTitle("")
    self.__BaseTrackerMap.GetYaxis().SetTitle("")
    self.__BaseTrackerMap.SetOption("COLZ L")
    self.__BaseTrackerMap.SetStats(0)
  
    # BARREL FIRST
    for i in range(maxPxBarrel):
      with open(self.geometryFilenames[i], "r") as geoFile:
        currBarrelTranslateX = 0
        currBarrelTranslateY = barrelLadderShift[i]
        
        self.__AddNamedBins(geoFile, currBarrelTranslateX, currBarrelTranslateY, 1, 1, True)
      
      # break # debug only 1st layer
      
    # MINUS FORWARD
    for i in range(-maxPxForward, 0):
      with open(self.geometryFilenames[maxPxBarrel + maxPxForward + i], "r") as geoFile:
        currForwardTranslateX = forwardDiskXShift[-i - 1]
        currForwardTranslateY = -forwardDiskYShift
        
        self.__AddNamedBins(geoFile, currForwardTranslateX, currForwardTranslateY, 1, 1)
        
    # PLUS FORWARD
    for i in range(maxPxForward):
      with open(self.geometryFilenames[maxPxBarrel + maxPxForward + i], "r") as geoFile:
        currForwardTranslateX = forwardDiskXShift[i]
        currForwardTranslateY = forwardDiskYShift
        
        self.__AddNamedBins(geoFile, currForwardTranslateX, currForwardTranslateY, 1, 1)
   
    # self.__BaseTrackerMap.Fill("305139728", 2)
    
    print("Base Tracker Map: constructed")
    
  ############################################################################
  def __init__(self, inputDQMName, outputDirName, minMaxFileName, limits,  modDicName, runNumber, dirs, dirsAliases):
#  def __init__(self, inputDQMName, outputDirName, minMaxFileName, limitsFileName, modDicName, runNumber, dirs, dirsAliases):
    self.inputFileName = inputDQMName
    self.outputDirName = outputDirName
    self.minMaxFileName = minMaxFileName
#    self.limitsFileName = limitsFileName
    self.detIDsFileName = modDicName
    self.limits = limits
    
    self.runNumber = runNumber
    self.dirs = dirs
    self.dirsAliases = dirsAliases
    
    self.inputFile = TFile(self.inputFileName)
    self.listOfNumHistograms = []
    self.availableNames = []
    
    self.maxLadderToLayer = {6:1, 14:2, 22:3, 32:4}
    self.maxBladeToRing = {11:1, 17:2}
    
    self.geometryFilenames = []
    for i in range(maxPxBarrel):
       self.geometryFilenames.append(getFileInPath("DQM/SiStripMonitorClient/data/Geometry/vertices_barrel_" + str(i + 1))) 
#      self.geometryFilenames.append("DATA/Geometry/vertices_barrel_" + str(i + 1))
    for i in range(-maxPxForward, maxPxForward + 1):
      if i == 0:
        continue #there is no 0 disk
      self.geometryFilenames.append(getFileInPath("DQM/SiStripMonitorClient/data/Geometry/vertices_forward_" + str(i)))
#      self.geometryFilenames.append("DATA/Geometry/vertices_forward_" + str(i))
    
    self.internalData = {}
    
    if self.inputFile.IsOpen():
      print("%s opened successfully!" % (self.inputFileName))
      #Get all neeeded histograms
      for dir in self.dirs:
        self.__TraverseDirTree(self.inputFile.Get(dir))
      # print("Histograms to read %d" % (len(self.listOfNumHistograms)))
      
      self.detDict = {}
      
      with open(self.detIDsFileName, "r") as detIDs:  # create dictionary online -> rawid
        for entry in detIDs:
          items = entry.replace("\n", " ").split(" ")
          self.detDict.update({items[1] : int(items[0])})
          # init internal data structure
          self.internalData.update({int(items[0]) : {}})
          
      self.rawToOnlineDict = dict((v,k) for k,v in self.detDict.items())    
      
      self.__GroupHistograms()
      
      self.__CreateTrackerBaseMap()
      
    else:
      print("Unable to open file %s" % (self.inputFileName))
      
    ### CREATE LIMITS DICTIONARY
    
    self.limitsDic = {}
    for y in limits:

      lineSpl = y.strip().split(" ")

      if len(lineSpl) < 5:
        continue
        
      currName = lineSpl[0]
      zMin = float(lineSpl[1])
      zMax = float(lineSpl[2])
      isLog = False if lineSpl[3] == "0" else True
      isAbs = False if lineSpl[4] == "0" else True

      self.limitsDic.update({currName : {"zMin" : zMin, "zMax" : zMax, "isLog" : isLog, "isAbs" : isAbs}})
 #     print limitsDic

  def ReadHistograms(self):
    if self.inputFile.IsOpen():
      for group in self.groupedHistograms:
        # name = ''.join(group[0].GetName().split("_")[0:-1])
        if len(group) == 0:
          return
        print(group[0].GetName())
        name = ''.join(group[0].GetName().split("_per_")[0])
        self.availableNames.append(name)
        # print(name)
        for obj in group:
          nbinsX = obj.GetNbinsX()
          nbinsY = obj.GetNbinsY()
          
          if nbinsX == 9: # BARREL
            maxX = nbinsX // 2
            maxY = nbinsY // 2
            
            for x in range(-maxX, maxX + 1):
              if x == 0:
                continue
              for y in range(-maxY, maxY + 1, 1):
                if y == 0:
                  continue
                onlineName = self.__BuildOnlineBarrelName(x, y, self.maxLadderToLayer[maxY])
                self.internalData[self.detDict[onlineName]].update({name : obj.GetBinContent(x + maxX + 1, y + maxY + 1)})         
                
          elif nbinsX == 7: # FORWARD
            maxX = nbinsX // 2
            maxY = nbinsY // 4
                  
            for x in range(-maxX, maxX + 1):
              if x == 0:
                continue
              for y in range(-maxY, maxY + 1):
                if int(y) == 0:
                  continue
                for panel in range(1, 3):
                  onlineName = self.__BuildOnlineDiskName(x, y, panel, self.maxBladeToRing[maxY])
                  self.internalData[self.detDict[onlineName]].update({name : obj.GetBinContent(x + maxX + 1, (y + maxY) * 2 + (3-panel))})  
          else:
            print("Unrecognized plot")
      else:
        print("Histograms saved to internal data structure")
        
  def DumpData(self):
    for key in self.internalData:
      print("#"*20)
      print(key)
      module = self.internalData[key]
      for d in module:
        print((d, module[d]))
    
    print(len(self.internalData))
    
    for i in self.availableNames:
      print(i)
    print(len(self.availableNames))
      
  def PrintTrackerMaps(self):
    monitoredValues = []
    gStyle.SetPalette(1)
    for key in self.internalData:
      monitoredValues = self.internalData[key].keys()
      # print(monitoredValues)
      break
    
    if os.path.exists(self.outputDirName) == False: # check whether directory exists
      os.system("mkdir " + self.outputDirName)
    
    with open(self.outputDirName + self.minMaxFileName, "w") as minMaxFile:
    
      for mv in monitoredValues:
        currentHist = deepcopy(self.__BaseTrackerMap)
        # currentHist.SetTitle("Run " + self.runNumber + ": Tracker Map for " + mv) // to make it compatible between ROOT v.
        histoTitle = "Run " + self.runNumber + ": Tracker Map for " + mv
          
        applyLogScale = False
        applyAbsValue = False
        if mv in self.limitsDic:
          limitsElem = self.limitsDic[mv]
          
          print(mv + " found in limits dictionary - applying custom limits...")
          
          currentHist.SetMinimum(limitsElem["zMin"])
          currentHist.SetMaximum(limitsElem["zMax"])
          applyLogScale = limitsElem["isLog"]
          applyAbsValue = limitsElem["isAbs"]

        listOfVals = []
        onlineName = ""
        nameId = ""

        for detId in self.internalData:
          val = (self.internalData[detId])[mv]
          onlineName = self.rawToOnlineDict[detId]
          listOfVals.append([val, detId, onlineName])
          #New TH2Poly bin ID = full_name_(detId)
          nameId = str(onlineName)+"_("+str(detId)+")"

          if applyAbsValue:
             currentHist.Fill(str(nameId), abs(val))
          else:
             currentHist.Fill(str(nameId), val)
          
        listOfVals = sorted(listOfVals, key = lambda item: item[0])
        
        minMaxFile.write("\n" + mv + "\n\n")
        
        minMaxFile.write("MIN:\n")
        for i in range(extremeBinsNum):
          minMaxFile.write("\t" + str(listOfVals[i][1]) + " " + str(listOfVals[i][2]) + " " + str(listOfVals[i][0]) + "\n")
        
        minMaxFile.write("MAX:\n")
        for i in range(extremeBinsNum):
          minMaxFile.write("\t" + str(listOfVals[-i - 1][1]) + " " + str(listOfVals[-i - 1][2]) + " " + str(listOfVals[-i - 1][0]) + "\n")
        
        #Canvas name = MyT for both Pixel and Strip Tracker Maps
        c1 = TCanvas("MyT", "MyT", plotWidth , plotHeight)
        
        if applyLogScale:
          c1.SetLogz()

        currentHist.Draw("AC COLZ L")        

        gPad.Update()
        palette = currentHist.FindObject("palette");
        palette.SetX1NDC(0.89);
        palette.SetX2NDC(0.91);
        palette.SetLabelSize(0.05);
        gPad.Update()

        ### IMPORTANT - REALTIVE POSITIONING IS MESSY IN CURRENT VERION OF PYROOT
        ### IT CAN CHANGE FROM VERSION TO VERSION, SO YOU HAVE TO ADJUST IT FOR YOUR NEEDS
        ### !!!!!!!!!!!!!
              
        # draw axes (z, phi -> BARREL; x, y -> FORWARD)
        ###################################################
        
        ### z arrow
        arrow = TArrow(0.05, 27.0, 0.05, -30.0, 0.02, "|>")
        arrow.SetLineWidth(4)
        arrow.Draw()
        ### phi arrow
        phiArrow = TArrow(0.0, 27.0, 30.0, 27.0, 0.02, "|>")
        phiArrow.SetLineWidth(4)
        phiArrow.Draw()
        ### x arror
        xArrow = TArrow(25.0, 44.5, 50.0, 44.5, 0.02, "|>")
        xArrow.SetLineWidth(4)
        xArrow.Draw()
        ### y arror
        yArrow = TArrow(25.0, 44.5, 25.0, 69.5, 0.02, "|>")
        yArrow.SetLineWidth(4)
        yArrow.Draw()

        ###################################################
        
        # add some captions        
        txt = TLatex()
        txt.SetNDC()
        txt.SetTextFont(1)
        txt.SetTextColor(1)
        txt.SetTextAlign(22)
        txt.SetTextAngle(0)
        
        # draw new-style title
        txt.SetTextSize(0.05)
        txt.DrawLatex(0.5, 0.95, histoTitle)
        
        txt.SetTextSize(0.03)
        
        txt.DrawLatex(0.5, 0.125, "-DISK")
        txt.DrawLatex(0.5, 0.075, "NUMBER ->")
        txt.DrawLatex(0.5, 0.875, "+DISK")
        
        txt.DrawLatex(0.17, 0.35, "+z")
        txt.DrawLatexNDC(0.36, 0.685, "+phi") # WAY TO FORCE IT TO DRAW LATEX CORRECTLY NOT FOUND ('#' DOESN'T WORK)
        txt.DrawLatex(0.38, 0.73, "+x")
        txt.DrawLatex(0.26, 0.875, "+y")
        
        txt.SetTextAngle(90)
        txt.DrawLatex(0.17, 0.5, "BARREL")
  
        #save to the png
        c1.Print(self.outputDirName + mv + ".png")

        #Clean canvas, change settings, save as root
        c1.Clear()
        c1.SetLogz(False)
        currentHist.GetZaxis().UnZoom()
        currentHist.SetLineColor(kBlack)
        currentHist.Draw("AL COLZ")
        currentHist.GetXaxis().SetRangeUser(-10,155)

        palette.SetX1NDC(0.92);
        palette.SetX2NDC(0.94);
        palette.SetY1NDC(0.02);
        palette.SetY2NDC(0.91);
        gPad.SetRightMargin(0.08);
        gPad.SetLeftMargin(0.01);
        gPad.SetTopMargin(0.09);
        gPad.SetBottomMargin(0.02);
        gPad.Update()

        zarrow = TArrow(0, 27, 0, -30, 0.02, "|>")
        zarrow.SetLineWidth(3)
        zarrow.Draw()
        phiArrow.SetLineWidth(3)
        phiArrow.Draw()
        xArrow.SetLineWidth(3)
        xArrow.Draw()
        yArrow.SetLineWidth(3)
        yArrow.Draw()

        txt.Clear()
        txt.SetTextAngle(0)
        txt.SetTextSize(0.05)
        PixelTitle = "Run " + self.runNumber + ": Pixel " + mv
        txt.DrawLatex(0.5, 0.95, PixelTitle)

        txt.SetTextSize(0.04)
        txt.SetNDC(False)
        txt.DrawLatex(75, -65, "-DISK")
        txt.DrawLatex(75, 65, "+DISK")
        txt.DrawLatex(50, -60, "NUMBER ->")

        txt.DrawLatex(-5, -20, "+z")
        txt.DrawLatex(35, 30, "+phi")
        txt.DrawLatex(55, 45, "+x")
        txt.DrawLatex(30, 65, "+y")

        txt.SetTextAngle(90)
        txt.DrawLatex(-5, 0, "BARREL")

        c1.SaveAs(self.outputDirName + mv + ".root")
        c1.Close()

  def __del__(self):
    if self.inputFile :
      if self.inputFile.IsOpen():
        self.inputFile.Close()
      
#--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--
for i in range(1, len(sys.argv), 1):
  if i == 1:
    inputFileName = sys.argv[i]
  elif i == 2:
    plotWidth = int(sys.argv[i])
  elif i == 3:
    plotHeight = int(sys.argv[i])
#  elif i == 4:
#    limitsFileName = sys.argv[i]
#  elif i == 5:
  elif i == 4:
    detIDsFileName = sys.argv[i]

deductedRunNumber = inputFileName.split("_R000")[1][0:6]
print(deductedRunNumber)

baseRootDirs = ["DQMData/Run " + deductedRunNumber + "/PixelPhase1/Run summary/Phase1_MechanicalView"    #maybe read it from the input file???
                ,"DQMData/Run " + deductedRunNumber + "/PixelPhase1/Run summary/Tracks"
                ]
                
baseRootDirsAliases = {baseRootDirs[0]:""
                    , baseRootDirs[1]:"T"
                    }

readerObj = TH2PolyOfflineMaps(inputFileName, outputDirectoryName, minMaxFileName, limits, detIDsFileName, deductedRunNumber, baseRootDirs, baseRootDirsAliases)   
#readerObj = TH2PolyOfflineMaps(inputFileName, outputDirectoryName, minMaxFileName, limitsFileName, detIDsFileName, deductedRunNumber, baseRootDirs, baseRootDirsAliases)  
readerObj.ReadHistograms()
# readerObj.DumpData()
readerObj.PrintTrackerMaps()
