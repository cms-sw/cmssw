#!/usr/bin/env python3

from __future__ import print_function
import sys
from ROOT import *
from array import array
from copy import deepcopy

gROOT.SetBatch()        # don't pop up canvases

#get data files

def getFileInPath(rfile):
   import os
   for dir in os.environ['CMSSW_SEARCH_PATH'].split(":"):
     if os.path.exists(os.path.join(dir,rfile)): return os.path.join(dir,rfile)
   return None

# Default values
inputFileName = "DQM.root"
outputFileName = "DQMTree.root"
#detIDsFileName = "DATA/detids.dat"
detIDsFileName = getFileInPath('DQM/SiStripMonitorClient/data/detids.dat')

class ModuleLvlValuesReader:

  ############################################################################
  
  def __TraverseDirTree(self, dir):
  
    currPath = (dir.GetPath().split(":/"))[1]
  
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
       
  def __CreateDummyStructAsStr(self, dicData):
    str = "struct MyStruct{"
    str = str + "Int_t key;"
    leafStr = "key/I"
    for k in dicData:
      str = str + "Float_t " + k + ";"
      leafStr = leafStr + ":" + k + "/F"
    str = str + "};"
    return str, leafStr
    
  ############################################################################

  def __init__(self, inputDQMName, outputDQMName, modDicName):
    self.inputFileName = inputDQMName
    self.outputFileName = outputDQMName
    self.detIDsFileName = modDicName

    index = self.inputFileName.find('R000')
    runNumber = self.inputFileName[index+4:index+10]

    self.dirs = ['DQMData/Run %s/PixelPhase1/Run summary/Phase1_MechanicalView' % (runNumber),
                 'DQMData/Run %s/PixelPhase1/Run summary/Tracks' % (runNumber)]
    self.dirsAliases = {self.dirs[0]:"", self.dirs[1]: "T"}
                
    self.inputFile = TFile(self.inputFileName)
    self.listOfNumHistograms = []
    self.availableNames = []

    self.maxLadderToLayer = {6:1, 14:2, 22:3, 32:4}
    self.maxBladeToRing = {11:1, 17:2}
    
    self.internalData = {}
    
    if self.inputFile.IsOpen():
      print("%s opened successfully!" % (self.inputFileName))
      #Get all neeeded histograms
      for dir in self.dirs:
        self.__TraverseDirTree(self.inputFile.Get(dir))
      print("Histograms to read %d" % (len(self.listOfNumHistograms)))
      
      self.detDict = {}
      
      with open(self.detIDsFileName, "r") as detIDs:  # create dictionary online -> rawid
        for entry in detIDs:
          items = entry.replace("\n", " ").split(" ")
          self.detDict.update({items[1] : int(items[0])})
          # init internal data structure
          self.internalData.update({int(items[0]) : {}})
          
      self.__GroupHistograms()
      
    else:
      print("Unable to open file %s" % (self.inputFileName))
  
  def ReadHistograms(self):
    if self.inputFile.IsOpen():
      for group in self.groupedHistograms:
        # name = ''.join(group[0].GetName().split("_")[0:-1])
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
                  self.internalData[self.detDict[onlineName]].update({name : obj.GetBinContent(x + maxX + 1, (y + maxY) * 2 + panel)})  
          else:
            print("Unrecognized plot")
      else:
        print("Histograms saved to internal data structure")
  def CreateTree(self): # too much complication lvl, use CreateTree2
    if len(self.internalData):
      # <- TOTAL WORKAROUND: START -> #
      # SEE: http://wlav.web.cern.ch/wlav/pyroot/tpytree.html
      
      leafStr = ""
      for key in self.internalData:
        s, leafStr = self.__CreateDummyStructAsStr(self.internalData[key])
        print(s)
        print(leafStr)
        gROOT.ProcessLine(s)
        break #all modules are assumed to have the same set of measured parameters
      
      from ROOT import MyStruct
      ms = MyStruct()     
      # <- TOTAL WORKAROUND: END -> #
      
      self.outputFile = TFile(self.outputFileName, "recreate")
      tree = TTree("tree", "readData")
      tree.Branch("b", ms, leafStr)
      
      tree.SetBranchAddress("b", tree)
      
      for key in self.internalData:    
        setattr(ms, "key", key)
        for d in self.internalData[key]:
          setattr(ms, d, (self.internalData[key])[d])
        tree.Fill()
        
        # break
      tree.Write()
      self.outputFile.Close()
      
      print("Data saved as TTree object")
      
  def CreateTree2(self): # use for TkCommissioner
    if len(self.internalData):
    
      self.outputFile = TFile(self.outputFileName, "recreate")
      tree = TTree("tree", "readData")
      
      key = array("i", [0])
      tree.Branch("detid", key, "detid/i")
      
      dat = {}
      for k in self.internalData:
        for d in self.internalData[k]:
          dat.update({d : array("f", [0])})
          tree.Branch(d, dat[d], d + "/F")
        break
        
      for k in self.internalData:
        key[0] = k
        for d in self.internalData[k]:
          (dat[d])[0] = (self.internalData[k])[d]
        tree.Fill()
          
      tree.Write()
      self.outputFile.Close()
      
      print("Data saved as TTree object")
      
  def DumpData(self):
    for key in self.internalData:
      print("#"*20)
      print(key)
      module = self.internalData[key]
      for d in module:
        print((d, module[d]))
        
    for i in self.availableNames:
      print(i)
    print(len(self.availableNames))
      
  def __del__(self):
    if self.inputFile.IsOpen():
      self.inputFile.Close()
    
#--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--
for i in range(1, len(sys.argv), 1):
  if i == 1:
    inputFileName = sys.argv[i]
  elif i == 2:
#    detIDsFileName = sys.argv[i]
#  elif i == 3:
    outputFileName = sys.argv[i]

#readerObj = ModuleLvlValuesReader(inputFileName, outputFileName)
readerObj = ModuleLvlValuesReader(inputFileName, outputFileName, detIDsFileName)
readerObj.ReadHistograms()
readerObj.CreateTree2()
# readerObj.DumpData()
