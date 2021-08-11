#!/usr/bin/env python3

from __future__ import print_function
import sys
import math
from ROOT import *
from copy import deepcopy
from scipy import signal

gROOT.SetBatch()        # don't pop up canvases

class InefficientDeadROCs:
  ############################################################################
  
  def __TraverseDirTree(self, dir):
  
    for obj in dir.GetListOfKeys():
      if not obj.IsFolder():
        if obj.ReadObjectAny(TClass.GetClass("TH2")):
          th1 = deepcopy(obj.ReadObj())
          name = th1.GetName()
          if name.startswith(self.lookForStr): #take only module lvl plots
            # print(''.join([dir.GetPath(), '/', name]))
            newName = name.split(self.lookForStr)[1]
            th1.SetName(newName)
            
            # used to sort outputs by disk/layer
            layer = 0
            # print(newName)
            if newName.startswith("B"):
              layer = "B" + ((newName.split("_LYR"))[1])[0]
            else:
              layer = ((newName.split("_D"))[1])[0]
              if newName.startswith("FPix_Bm"):
                layer = "-" + layer
              layer = "F" + layer
            
            if layer in self.dicOfModuleHistograms:
              self.dicOfModuleHistograms[layer].append(th1)
            else:
              self.dicOfModuleHistograms.update({layer : [th1]})        
      else:
        self.__TraverseDirTree(obj.ReadObj())
        
  def __init__(self, inputDQMName, outputFileName, noiseOutputFileName, dirs):
  
    self.inputFileName = inputDQMName
    self.outputFileName = outputFileName
    self.noiseOutputFileName = noiseOutputFileName
    self.dirs = dirs
    
    self.lookForStr = "digi_occupancy_per_col_per_row_"
    
    self.rocMaxCol = 52
    self.rocMaxRow = 80
    self.rocsInRow = 8
    self.rocsInCol = 2
    
    self.inputFile = TFile(self.inputFileName)
    self.dicOfModuleHistograms = {}
    
    ### THRESHOLDS SECTION
    self.pixelNoisynessTh = 6
    self.rocOccupancyTh = 200
    
    self.barrelNoisyColumnTh = 1.35
    self.barrelNoisyColumnTh2 = 4.5
    self.endcapNoisyColumnTh = 1.5
    
    self.barrelInefficientDColTh = 8#2.5
    self.endcapInefficientDColTh = 30#8
    
    ### ###################
    
    if self.inputFile.IsOpen():
      print("%s opened successfully!" % (self.inputFileName))
      #Get all neeeded histograms
      for dir in self.dirs:
        self.__TraverseDirTree(self.inputFile.Get(dir))
      print("Histograms to read: %d" % (len(self.dicOfModuleHistograms)))
      
      self.detDict = {}
      
    else:
      print("Unable to open file %s" % (self.inputFileName))
      
  def __lmsExp(self, data, xMin, xMax):
    meanOfX = (xMax + xMin) * 0.5
    meanOfY = sum( [math.log(data[i]) for i in range(len(data))] ) / len(data)
    
    D = 0
    for i in range(xMin, xMax + 1):
      D = D + (i - meanOfX)**2
      # print(D)

    a = 0
    for i in range(len(data)):
      a = a + math.log(data[i]) * (xMin + i - meanOfX)
    a = a/D
    
    lnb = meanOfY - a * meanOfX
    
    return a, math.exp(lnb)
    
  def __lmsLin(self, data, xMin, xMax):
    meanOfX = (xMax + xMin) * 0.5
    meanOfY = sum(data) / len(data)
    
    D = 0
    for i in range(xMin, xMax + 1):
      D = D + (i - meanOfX)**2

    a = 0
    for i in range(len(data)):
      a = a + data[i] * (xMin + i - meanOfX)
    a = a/D
    
    b = meanOfY - a * meanOfX
    
    return a, b, D
    
  def __customMedianFilter(self, array, radius = 2):
    # contrary to scipy implementation it provides adaptive kernel size instead of copying data on boundaries
    filtered = [0 for i in range(len(array))]
    currArray = []
    for i in range(len(array)):
      if i - radius < 0:
        currArray = array[0 : i + radius + 1]
      elif i + radius + 1 >= len(array):
        currArray = array[i - radius : ]
      
      currArray.sort()
      filtered[i] = currArray[len(currArray) // 2]
      
    return filtered
    
  def __getROCData(self, hist, startPixel, endPixel, row, repeatFilter = 3, filterKernelSize = 5):
    pixelArr = []
    columnsWithSuspiciouslyNoisyPixels = []
    
    for x in range(startPixel, endPixel):
      
      columnPixels = [hist.GetBinContent(x, y + 1) for y in range(row * self.rocMaxRow, (row + 1) * self.rocMaxRow)]
      columnSum = sum(columnPixels)
      
      columnMean = columnSum / len(columnPixels)
      for i in range(len(columnPixels)):
        if columnPixels[i] > self.pixelNoisynessTh * columnMean:
          # col = (startPixel % self.rocMaxCol) + 1
          
          columnsWithSuspiciouslyNoisyPixels.append(x)
          
          # print("WARNING:\t %s : %dx%d:%d may contain NOISY PIXELS instead of NOISY COLUMNS" % (hist.GetName(), col, row + 1, startPixel + i))
          break
      
      pixelArr.append(columnSum)  
      
    if len(pixelArr) == 0:
      return None, None, None                                             # ROC down
    
    medFiltRes, sciPyMedFiltRes = deepcopy(pixelArr), deepcopy(pixelArr)
    for i in range(repeatFilter):
      sciPyMedFiltRes = signal.medfilt(sciPyMedFiltRes, filterKernelSize) # 5 is obligatory to filter doublets!!!
      medFiltRes = self.__customMedianFilter(medFiltRes, filterKernelSize // 2)
      
    return pixelArr, medFiltRes, columnsWithSuspiciouslyNoisyPixels, sciPyMedFiltRes
    
  def __getPixelArrWithRemovedDrops(self, pixelArr, medFiltRes):
    return [ (pixelArr[i] if pixelArr[i] > medFiltRes[i] else medFiltRes[i]) if 0 < i < len(pixelArr) - 1 else min(medFiltRes) for i in range(len(pixelArr))]
    
  def __normalizeArray(self, pixelArr):
    c_min, c_max = min(pixelArr), max(pixelArr)
    if c_min != c_max:
      c_diff_inv = 1.0 / (c_max - c_min)
      return [ (pixelArr[i] - c_min) * c_diff_inv for i in range(len(pixelArr))]
    return [0 for i in range(len(pixelArr))]
    
  def __setNormalizedArrayZeroInThePoint(self, pixelArr, pt):
    c_diff_inv = 1.0 / (1.0 - pt)
    
    return [ (pixelArr[i] - pt) * c_diff_inv for i in range(len(pixelArr))]
    
  def __determineBarrelNoise(self, noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, maxMed, val, pos, rocCol, rocRow):
    noisyROC = False;
    if meanOfPixels < self.rocOccupancyTh:
      #print("Very low mean occupancy: %f in %s in (col, row) (%d, %d)...\tSkipping noisy ROC calculation" % (meanOfPixels, histName, rocCol, rocRow) )
      noisyROC = True
    else:
      th = self.barrelNoisyColumnTh * maxMed
      if val > th:
        if pos not in columnsWithSuspiciouslyNoisyPixels:
          rocNum, xCoordInROC = self.__convertCoordinatesFromHistToROCSpace(histName, pos, rocRow)
          noiseFile.write("%s\t(x, row)->[rocNum, xRoc]\t(%d, %d)->[%d, %d];\t{VAL, TH}\t{%f, %f}\n" % (histName, pos, rocRow+1, rocNum, xCoordInROC, val, th))
          
          return 1, noisyROC
        # else:
          # print("WARNING: rejecting %s (x, row) (%d, %d) as being affected by a few noisy pixel(s)" % (histName, pos, rocRow+1))
          
    return 0, noisyROC
    
  def __determineBarrelNoise2(self, noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, normMeanOfPixels, normVal, pos, rocCol, rocRow):
    noisyROC = False;
    if meanOfPixels < self.rocOccupancyTh:
      #print("Very low mean occupancy: %f in %s in (col, row) (%d, %d)...\tSkipping noisy ROC calculation" % (meanOfPixels, histName, rocCol, rocRow) )
      noisyROC = True
    else:
      th = self.barrelNoisyColumnTh2 * normMeanOfPixels
      if normVal > th:
        if pos not in columnsWithSuspiciouslyNoisyPixels:
          rocNum, xCoordInROC = self.__convertCoordinatesFromHistToROCSpace(histName, pos, rocRow)
          noiseFile.write("%s\t(x, row)->[rocNum, xRoc]\t(%d, %d)->[%d, %d];\t{NORMVAL, TH}\t{%f, %f}\n" % (histName, pos, rocRow+1, rocNum, xCoordInROC, normVal, th))
          
          return 1, noisyROC
        # else:
          # print("WARNING: rejecting %s (x, row) (%d, %d) as being affected by a few noisy pixel(s)" % (histName, pos, rocRow+1))
          
    return 0, noisyROC
        
  def __determineEndcapNoise(self, noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, linVal, val, pos, rocCol, rocRow):
    noisyROC = False;
    if meanOfPixels < self.rocOccupancyTh:
      # print("Very low mean occupancy: %f in %s in (col, row) (%d, %d)...\tSkipping noisy ROC calculation" % (meanOfPixels, histName, rocCol, rocRow) )
      noisyROC = True
    
    else:
      th = self.endcapNoisyColumnTh * linVal
      if val > th:
        if pos not in columnsWithSuspiciouslyNoisyPixels:
          rocNum, xCoordInROC = self.__convertCoordinatesFromHistToROCSpace(histName, pos, rocRow)
          noiseFile.write("%s\t(x, row)->[rocNum, xRoc]\t(%d, %d)->[%d, %d];\t{VAL, TH}\t{%f, %f}\n" % (histName, pos, rocRow+1, rocNum, xCoordInROC, val, th))
          
          return 1, noisyROC
        # else:
          # print("WARNING: rejecting %s (x, row) (%d, %d) as being affected by a few noisy pixel(s)" % (histName, pos, rocRow+1))
          
    return 0, noisyROC
    
  def __convertCoordinatesFromHistToROCSpace(self, histName, histXpos, histRow):
    tempXROC = (histXpos / self.rocMaxCol) # 0,...,7
    tempYROC = histRow
    
    tempXCoordInROC = histXpos % self.rocMaxCol
    
    realXROC, realYROC = tempXROC, tempYROC
    xCoordInROC = tempXCoordInROC
    
    rocNum = 0
    
    if histName.find("BPix_Bp") != -1: #zero ROC is in top left corner
      realYROC = 1 - tempYROC
      if realYROC == 1:
        rocNum = 15 - realXROC
        xCoordInROC = self.rocMaxCol - 1 - xCoordInROC
      else:
        rocNum = realXROC
    else: # zero ROC is in bottom right corner
      realXROC = 7 - tempXROC
      if realYROC == 1:
        rocNum = 15 - realXROC
      else:
        rocNum = realXROC
        xCoordInROC = self.rocMaxCol - 1 - xCoordInROC
        
    return rocNum, xCoordInROC
    
    
  def __determineBarrelDColInefficiencyAndNoise(self, medFiltRes, histName, pixelArr, pixelArrWithoutDrops, startPixel, rocCol, rocRow, outputFile, columnsWithSuspiciouslyNoisyPixels, noiseFile):
    meanOfPixels = sum(medFiltRes) / len(medFiltRes)
    maxMed = max(medFiltRes)
    minMed = min(medFiltRes) 

    normMeanOfPixels = sum(pixelArrWithoutDrops) / len(pixelArrWithoutDrops)
    # print( meanOfPixels, maxMed, minMed )
    
    doubleDeadCols = 0
    noisyColsNum = 0
    noisyROC = 0
    
    # for x in range(startPixel, endPixel, 1):
    for i in range(1, len(pixelArr) - 2):
      # print(i , i + 1)
      bin1valDiff = minMed - pixelArr[i + 0]#hist.GetBinContent(x+0)
      bin2valDiff = minMed - pixelArr[i + 1]
      # WE ONLY WANT A SET OF TWO COLUMNS SO ADJACENT COLUMNS HAVE TO BE NORMAL
      bin0valDiff = minMed - pixelArr[i - 1]
      bin3valDiff = minMed - pixelArr[i + 2]
      
      # currentDoubleBinThreshold = minMed / math.sqrt(meanOfPixels) * self.barrelInefficientDColTh # error in bin entry grows as sqrt(N)
      currentDoubleBinThreshold = math.sqrt(meanOfPixels) * self.barrelInefficientDColTh # error in bin entry grows as sqrt(N)
                        
      if bin1valDiff > currentDoubleBinThreshold and bin2valDiff > currentDoubleBinThreshold and not bin3valDiff > currentDoubleBinThreshold and not bin0valDiff > currentDoubleBinThreshold:

        doubleColInRoc = ((i + startPixel) % (self.rocMaxCol)) // 2 + 1
        doubleDeadCols = doubleDeadCols + 1
        
        # outputFile.write("%s,\tX: %d-%d\tROC COLUMN: %d\tROC ROW: %d\tDOUBLE COL IN ROC: %d\tTH: %f\tMIN IN ROC: %f\tBINVAL: %f\n" % (histName, startPixel + (i + 0), startPixel + (i + 1), rocCol, rocRow, doubleColInRoc, currentDoubleBinThreshold, minMed, pixelArr[i]))
        rocNum, xCoordInROC = self.__convertCoordinatesFromHistToROCSpace(histName, startPixel + i, rocRow)
        outputFile.write("%s\t(x, row)->[rocNum, doubleXPixelColInROC]\t(%d, %d)->[%d, %d];\t{MIN - VAL, TH}\t{%f, %f}\n" % (histName, startPixel + i, rocRow + 1, rocNum, xCoordInROC / 2, bin1valDiff, currentDoubleBinThreshold))

      # HANDLE NOISY PIXELS
      if noisyROC == True:  #don't go inside if noisyness was determined already
        continue
      
      # res = self.__determineBarrelNoise(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, maxMed, pixelArr[i], startPixel + i, rocCol, rocRow)
      # noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]
      # if i == len(pixelArr) - 3: #  CHECK NOISYNESS IN THE RIGHTMOST INNER COL
        # res = self.__determineBarrelNoise(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, maxMed, pixelArr[i + 1], startPixel + i + 1, rocCol, rocRow)
        # noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]      
      
      # NORMALIZED MEAN NOISE DETERMINATION METHOD
      res = self.__determineBarrelNoise2(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, normMeanOfPixels, pixelArrWithoutDrops[i], startPixel + i, rocCol, rocRow)
      noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]
      if i == len(pixelArr) - 3: #  CHECK NOISYNESS IN THE RIGHTMOST INNER COL
        res = self.__determineBarrelNoise2(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, normMeanOfPixels, pixelArrWithoutDrops[i + 1], startPixel + i + 1, rocCol, rocRow)
        noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]
        
        
        
    return doubleDeadCols, noisyColsNum
  
  def __determineEndcapDColInefficiencyAndNoise(self, medFiltRes, histName, pixelArr, startPixel, rocCol, rocRow, outputFile, columnsWithSuspiciouslyNoisyPixels, noiseFile):
    doubleDeadCols = 0
    noisyColsNum = 0
    noisyROC = 0
    
    useLin = True
    # <D> might be used for high noise ROC recognition
    a, b, D = self.__lmsLin(medFiltRes, startPixel, len(medFiltRes) + startPixel)
                  
    meanOfPixels = sum(medFiltRes) / len(medFiltRes)
    
    # for x in range(startPixel, endPixel, 1):
    for i in range(1, len(pixelArr) - 2):
      
      if useLin == True:
        linVal1 = a * (i + startPixel + 0) + b
        linVal2 = a * (i + startPixel + 1) + b
        
        linVal0 = a * (i + startPixel - 1) + b
        linVal3 = a * (i + startPixel + 2) + b
      else:
        linVal1 = b * math.exp(a * (i + startPixel + 0))
        linVal2 = b * math.exp(a * (i + startPixel + 1))
                               
        linVal0 = b * math.exp(a * (i + startPixel - 1))
        linVal3 = b * math.exp(a * (i + startPixel + 2))
      
      bin1valDiff = linVal1 - pixelArr[i + 0]
      bin2valDiff = linVal2 - pixelArr[i + 1]
      # WE ONLY WANT A SET OF TWO COLUMNS SO ADJACENT COLUMNS HAVE TO BE NORMAL
      bin0valDiff = linVal0 - pixelArr[i - 1]
      bin3valDiff = linVal3 - pixelArr[i + 2] 
         
      try:
        currentDoubleBinThreshold = math.sqrt((linVal1 + linVal2) * 0.5) * self.endcapInefficientDColTh
      except:
        # print(a, b, startPixel, i, linVal1, linVal2)
        continue
      
      if bin1valDiff > currentDoubleBinThreshold and bin2valDiff > currentDoubleBinThreshold and not bin3valDiff > currentDoubleBinThreshold and not bin0valDiff > currentDoubleBinThreshold:

        doubleColInRoc = ((i + startPixel) % (self.rocMaxCol)) // 2 + 1
        doubleDeadCols = doubleDeadCols + 1
        
        # outputFile.write("%s,\tX: %d-%d\tROC COLUMN: %d\tROC ROW: %d\tDOUBLE COL IN ROC: %d\tTH: %f\tLINVAL: %f\tBINVAL: %f\n" % (histName, startPixel + (i + 0), startPixel + (i + 1), rocCol, rocRow, doubleColInRoc, currentDoubleBinThreshold, linVal1, pixelArr[i]))
        rocNum, xCoordInROC = self.__convertCoordinatesFromHistToROCSpace(histName, startPixel + i, rocRow)
        outputFile.write("%s\t(x, row)->[rocNum, doubleXPixelColInROC]\t(%d, %d)->[%d, %d];\t{LIN(x) - VAL, TH}\t{%f, %f}\n" % (histName, startPixel + i, rocRow + 1, rocNum, xCoordInROC / 2, bin1valDiff, currentDoubleBinThreshold))


      # HANDLE NOISY PIXELS
      if noisyROC == True:  #don't go inside if noisyness was determined already
        continue
      
      res = self.__determineEndcapNoise(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, linVal1, pixelArr[i], i + startPixel, rocCol, rocRow)
      noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]
      if i == len(pixelArr) - 3: #  CHECK NOISYNESS IN THE RIGHTMOST INNER COL
        res = self.__determineEndcapNoise(noiseFile, columnsWithSuspiciouslyNoisyPixels, histName, meanOfPixels, linVal2, pixelArr[i + 1], i + 1 + startPixel, rocCol, rocRow)
        noisyColsNum, noisyROC = noisyColsNum + res[0], res[1]
    
    return doubleDeadCols, noisyColsNum
      
  def ReadHistograms(self):      
    doubleDeadCols, noisyColsNum = 0, 0
    
    with open(self.noiseOutputFileName, "w") as noiseFile:   
      with open(self.outputFileName, "w") as outputFile: 
        for layer in self.dicOfModuleHistograms:
          
          doubleDeadColsInLayer, noisyColsNumInLayer = 0, 0
             
          outputFile.write("-> " + layer + "\n\n")
          noiseFile.write("-> " + layer + "\n\n")
          
          for hist in self.dicOfModuleHistograms[layer]:          
            for row in range(2):          
              for rocNum in range(self.rocsInRow):
                startPixel = rocNum * self.rocMaxCol + 1
                endPixel = (rocNum + 1) * self.rocMaxCol + 1 # - 1 ???
                
                rocCol = rocNum + 1
                
                pixelArr, medFiltRes, columnsWithSuspiciouslyNoisyPixels, sciPyMedFiltRes = self.__getROCData(hist, startPixel, endPixel, row, 3, 5)
                
                if pixelArr == None:
                  continue
                  
                # meanOfPixels = sum(pixelArr) / len(pixelArr)
                # pixelArrSorted = deepcopy(pixelArr)
                # pixelArrSorted.sort()
                # outputFile.write("%s: <x> <med_min> VS. <med_max> | x_min:\t%f %f %f | %f, %f, %f, %f\n" % (hist.GetName(), meanOfPixels, min(medFiltRes), max(medFiltRes), pixelArrSorted[0], pixelArrSorted[1], pixelArrSorted[2], pixelArrSorted[3]))
                
                if "F" not in layer:
                  # pixelArrWithoutDrops = self.__getPixelArrWithRemovedDrops(pixelArr, medFiltRes)
                  pixelArrWithoutDrops = self.__getPixelArrWithRemovedDrops(pixelArr, sciPyMedFiltRes)
                  pixelArrWithoutDropsNormalized = self.__normalizeArray(pixelArrWithoutDrops)
                  # tmp_mean = sum(pixelArrWithoutDropsNormalized) / len(pixelArrWithoutDropsNormalized)
                  # pixelArrWithoutDropsNormalized = self.__setNormalizedArrayZeroInThePoint(pixelArrWithoutDropsNormalized, tmp_mean)
                  
                  # print(min(pixelArrWithoutDropsNormalized), max(pixelArrWithoutDropsNormalized))
                  result = self.__determineBarrelDColInefficiencyAndNoise(medFiltRes, hist.GetName(), pixelArr, pixelArrWithoutDropsNormalized, startPixel, rocCol, row, outputFile, columnsWithSuspiciouslyNoisyPixels, noiseFile)
                else:
                  result = self.__determineEndcapDColInefficiencyAndNoise(medFiltRes, hist.GetName(), pixelArr, startPixel, rocCol, row, outputFile, columnsWithSuspiciouslyNoisyPixels, noiseFile)
                  
                doubleDeadCols, doubleDeadColsInLayer = doubleDeadCols + result[0], doubleDeadColsInLayer + result[0]
                noisyColsNum, noisyColsNumInLayer = noisyColsNum + result[1], noisyColsNumInLayer + result[1]               
                  
          outputFile.write("\n\tTOTAL IN LAYER/DISK: %d\n\n" % (doubleDeadColsInLayer))    
          noiseFile.write("\n\tTOTAL IN LAYER/DISK: %d\n\n" % (noisyColsNumInLayer))
          
    print("Number of inefficient double columns: %d"%(doubleDeadCols))
    print("Number of noisy cols: %d"%(noisyColsNum))
      
      
#--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--
for i in range(1, len(sys.argv), 1):
  if i == 1:
    inputFileName = sys.argv[i]

runNum = ((inputFileName.split("/")[-1].split("."))[0].split("_R000"))[1]
print("Run number: %s"%(runNum))
baseRootDir = ["DQMData/Run " + runNum + "/PixelPhase1/Run summary/Phase1_MechanicalView"]
print(baseRootDir[0])
outputFileName = "inefficientDPixelColumns.txt"
noiseOutputFileName = "noisyPixelColumns.txt"

readerObj = InefficientDeadROCs(inputFileName, outputFileName, noiseOutputFileName, baseRootDir)  
readerObj.ReadHistograms()
