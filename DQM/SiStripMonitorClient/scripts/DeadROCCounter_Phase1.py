#!/usr/bin/python
from ROOT import TFile, gStyle,gPad ,TObject, TCanvas, TH1, TH1F, TH2F, TLegend, TPaletteAxis, TList, TLine, TAttLine, TF1,TAxis
import re
import sys, string

def getRunNumber(filename):
    global runNumber
    pos=filename.find("__")
    runNumber=int(filename[pos-6:pos])
    #print runNumber

###########################################barrel########################################################
def countBadROCBarrel(fin, layerNo, os):
    barrelPath  = commonPath + "PXBarrel/";
    histoname = ["digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_1", "digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_2",  
                 "digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_3", "digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_4"]
    digi2D = fin.Get(barrelPath + histoname[layerNo-1])
    #return status flag is histogram is empty!
    if digi2D.GetEntries() == 0 :  
        return 1;
    Nrocspopulated = 0
    totalEntries = 0
    NexpectedROC = [1536, 3584, 5632, 8192]
    nLadders_bpx = [6, 14, 22, 32]
    nx = digi2D.GetNbinsX()
    ny = digi2D.GetNbinsY()
    for xbin in range(1,nx+1):
        if xbin >= 33 and xbin <= 40:    continue;#region of cross on x-axis
        for ybin in range(1,ny+1):
            if (ybin == 2*nLadders_bpx[layerNo-1] + 1) or (ybin == 2*nLadders_bpx[layerNo-1] + 2):    continue;#region of cross on y-axis
            bentries = digi2D.GetBinContent(xbin,ybin)
            if(bentries > 0):
                Nrocspopulated+=1
                totalEntries += bentries
    meanEntries = float(totalEntries)/Nrocspopulated
    NineffROC = 0
    #Loop to chek inefficient ROC per layer
    for xbin in range(1,nx+1):
        if xbin >= 33 and xbin <= 40:
            continue;#region of cross on x-axis
        for ybin in range(1,ny+1):
            if (ybin == 2*nLadders_bpx[layerNo-1] + 1) or (ybin == 2*nLadders_bpx[layerNo-1] + 2):    continue;#region of cross on y-axis
            bentries = digi2D.GetBinContent(xbin,ybin);
            if(bentries > 0 and bentries < meanEntries/4. ):#Assume < 25% of MEAN = inefficient
                NineffROC+=1;
    ##Printing Layer no., #dead ROC, #inefficienct ROC, #mean occupancy of Non-zer roc
    tmpstr = "BPix L" + str(layerNo)
    print >> os, tmpstr, '{0:4d} {1:4d} {2:4.1f}'.format(NexpectedROC[layerNo-1] - Nrocspopulated, NineffROC, round(meanEntries,1))
    return 0;
#############################################endacp#########################################
def countBadROCForward(fin, ringNo, os):
    forwardPath  = commonPath + "PXForward/";
    histoname = ["digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_1",
"digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_2"]
    digi2D = fin.Get(forwardPath + histoname[ringNo-1])
    #return status flag is histogram is empty!
    if digi2D.GetEntries() == 0 :  
        return 1;
    nblades_perRing_fpx = [22, 34]
    NexpectedROC_perRing = [704, 1088]
    Nrocspopulated = [0] * 6
    totalEntries = [0] * 6
    dcounter = 0
    nx = digi2D.GetNbinsX()
    ny = digi2D.GetNbinsY()
    for xbin in range(1,nx+1):
        if xbin >= 25 and xbin <= 32:    continue;#region of cross on x-axis
	if xbin > 1 and  (xbin-1)%8 == 0:    dcounter += 1; 
	for ybin in range(1,ny+1):
	    if (ybin >= 2*nblades_perRing_fpx[ringNo-1] + 1) and (ybin <= 2*nblades_perRing_fpx[ringNo-1] + 4):
	        continue;#region of cross on y-axis
	    bentries = digi2D.GetBinContent(xbin,ybin)
	    if(bentries > 0):
                Nrocspopulated[dcounter] += 1
		totalEntries[dcounter] += bentries
    #Loop to find inefficient modules
    meanEntries = [6] * 6
    for d in range(0,6):
        meanEntries[d] = float(totalEntries[d])/Nrocspopulated[d]
        NineffROC = [6] * 6
    #set disc counter to 0 since it is now 5
    dcounter = 0;
    for xbin in range(1,nx+1):
        if xbin >= 25 and xbin <= 32:    continue;#region of cross on x-axis
	if xbin > 1 and  (xbin-1)%8 == 0:    dcounter += 1 
        for ybin in range(1,ny+1):
	    if (ybin >= 2*nblades_perRing_fpx[ringNo-1] + 1) and (ybin <= 2*nblades_perRing_fpx[ringNo-1] + 4):
                continue;#region of cross on y-axis
	    bentries = digi2D.GetBinContent(xbin,ybin)
	    if(bentries > 0):#//Assume < 25% of MEAN = inefficient 
	        if bentries > 0 and bentries < meanEntries[dcounter]/4.: 
		    NineffROC[dcounter] += 1

    print >> os, "#Summary for FPix Ring", ringNo
    for d in range(0,6):
        disc = 0
	if d < 3:    disc = "M" + str(3 - d)
	else:    disc = "P" + str(d - 2)
        ##Printing Disc no., #dead ROC, #inefficienct ROC, #mean occupancy of Non-zer roc
	tmpstr = "FPix R" + str(ringNo) + "D" + str(disc)
        print >> os, '{0:10s} {1:4d} {2:4d} {3:4.1f}'.format(tmpstr, NexpectedROC_perRing[ringNo-1] - Nrocspopulated[d], NineffROC[d], round(meanEntries[d],1))
    return 0;
################################################main#######################################
fname=sys.argv[1]
getRunNumber(fname)
fin= TFile(fname)
outname="PixZeroOccROCs_run" + str(runNumber) + ".txt"
global commonPath
commonPath  = "DQMData/Run " + str(runNumber) + "/PixelPhase1/Run summary/Phase1_MechanicalView/"
#histogram of no. of pixel clusters
hnpixclus_bpix = fin.Get(commonPath + "charge_PXBarrel")
hnpixclus_fpix = fin.Get(commonPath + "charge_PXForward")

out_file = open(outname, "w")
print >> out_file, "#Layer/Disc KEY NDeadROC NineffROC MeanOccupacy"
print >> out_file, "#Pixel Barrel Summary"
for l in range(1,5):
    if countBadROCBarrel(fin, l, out_file) == 1:
        print >> out_file, "DQM histogram for Layer", str(l), " is empty!"
print >> out_file, "#Pixel Forward Summary"
for ring in range(1,3):
    if countBadROCForward(fin, ring, out_file) == 1:
        print >> out_file, "DQM histogram for Ring", str(ring), " is empty!"

print >> out_file, "Number of clusters=", int(hnpixclus_bpix.GetEntries() + hnpixclus_fpix.GetEntries())
out_file.close()	
