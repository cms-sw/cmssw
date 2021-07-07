#!/usr/bin/env python3

from __future__ import print_function
import sys
import os
from ROOT import *
from copy import deepcopy
from array import array
import six

gROOT.SetBatch() # don't pop up canvases

maxPxBarrel = 4
maxPxForward = 3
barrelLadderShift = [0, 14, 44, 90]
forwardDiskXShift = [25, 75, 125]
forwardDiskYShift = 45; # to make +DISK on top in the 'strip-like' layout
plotWidth, plotHeight = 3000, 2000
listOfDetids=[]

##############################################
#Find Data files
def getFileInPath(rfile):
##############################################
   import os 
   for dir in os.environ['CMSSW_SEARCH_PATH'].split(":"):
     if os.path.exists(os.path.join(dir,rfile)): return os.path.join(dir,rfile)                                                                          
   return None

##############################################
def __AddNamedBins(BaseTrackerMap, geoFile, tX, tY, sX, sY, applyModuleRotation = False):
##############################################
    for line in geoFile:
        lineSpl = line.strip().split("\"")
      
        detId = lineSpl[0].split(" ")[0]
      
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
      
        #print(detId, [p for p in x], [q for q in y])
        listOfDetids.append(detId)
        if applyModuleRotation:
            bin = TGraph(verNum, y, x)
        else:
            bin = TGraph(verNum, x, y)
            # bin = TGraph(verNum, y, x) # rotation by 90 deg (so that it had the same layout as for the strips)
        bin.SetName(detId)
        BaseTrackerMap.AddBin(bin)

##############################################
def main():
##############################################
    geometryFilenames = []
    for i in range(maxPxBarrel):
        geometryFilenames.append(getFileInPath("DQM/SiStripMonitorClient/data/Geometry/vertices_barrel_" + str(i + 1))) 

    for i in range(-maxPxForward, maxPxForward + 1):
        if i == 0:
            continue #there is no 0 disk
        geometryFilenames.append(getFileInPath("DQM/SiStripMonitorClient/data/Geometry/vertices_forward_" + str(i)))

    BaseTrackerMap = TH2Poly("Summary", "", -10, 160, -70, 70)
    BaseTrackerMap.SetFloat(1)
    BaseTrackerMap.GetXaxis().SetTitle("")
    BaseTrackerMap.GetYaxis().SetTitle("")
    BaseTrackerMap.SetOption("COLZ L")
    BaseTrackerMap.SetStats(0)
  
    # BARREL FIRST
    for i in range(maxPxBarrel):
       with open(geometryFilenames[i], "r") as geoFile:
          currBarrelTranslateX = 0
          currBarrelTranslateY = barrelLadderShift[i]
          __AddNamedBins(BaseTrackerMap,geoFile, currBarrelTranslateX, currBarrelTranslateY, 1, 1, True)

    # MINUS FORWARD
    for i in range(-maxPxForward, 0):
        with open(geometryFilenames[maxPxBarrel + maxPxForward + i], "r") as geoFile:
            currForwardTranslateX = forwardDiskXShift[-i - 1]
            currForwardTranslateY = -forwardDiskYShift        
            __AddNamedBins(BaseTrackerMap,geoFile, currForwardTranslateX, currForwardTranslateY, 1, 1)
        
    # PLUS FORWARD
    for i in range(maxPxForward):
        with open(geometryFilenames[maxPxBarrel + maxPxForward + i], "r") as geoFile:
            currForwardTranslateX = forwardDiskXShift[i]
            currForwardTranslateY = forwardDiskYShift
            __AddNamedBins(BaseTrackerMap,geoFile, currForwardTranslateX, currForwardTranslateY, 1, 1)
       
    print("Base Tracker Map: constructed")

    c1 = TCanvas("c1","c1", plotWidth , plotHeight)
    for detid in listOfDetids:
       BaseTrackerMap.Fill(str(detid),int(detid))
    BaseTrackerMap.Draw("AC COLZ L")        
              
    gPad.Update()
    palette = BaseTrackerMap.FindObject("palette");
    palette.SetX1NDC(0.89);
    palette.SetX2NDC(0.91);
    palette.SetLabelSize(0.05);
    gPad.Update()

    ### z arrow
    arrow = TArrow(0.05, 27.0, 0.05, -30.0, 0.02, "|>")
    arrow.SetLineWidth(4)
    arrow.Draw()
    ### phi arrow
    phiArrow = TArrow(0.0, 27.0, 30.0, 27.0, 0.02, "|>")
    phiArrow.SetLineWidth(4)
    phiArrow.Draw()
    ### x arrow
    xArrow = TArrow(25.0, 44.5, 50.0, 44.5, 0.02, "|>")
    xArrow.SetLineWidth(4)
    xArrow.Draw()
    ### y arrow
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
    txt.DrawLatex(0.5, 0.95, "Pixel Tracker Map")
    txt.SetTextSize(0.03)
    txt.DrawLatex(0.5, 0.125, "-DISK")
    txt.DrawLatex(0.5, 0.075, "NUMBER ->")
    txt.DrawLatex(0.5, 0.875, "+DISK")
    txt.DrawLatex(0.12, 0.35, "+z")
    txt.DrawLatexNDC(0.315, 0.665, "+phi") # WAY TO FORCE IT TO DRAW LATEX CORRECTLY NOT FOUND ('#' DOESN'T WORK)
    txt.DrawLatex(0.38, 0.73, "+x")
    txt.DrawLatex(0.235, 0.875, "+y")
    txt.SetTextAngle(90)
    txt.DrawLatex(0.125, 0.5, "BARREL")

    #save to the png
    c1.Print("test.png")

##################################################
if __name__ == "__main__":        
    main()
