#!/usr/bin/python
from ROOT import TFile, gStyle,gPad ,TObject, TCanvas, TH1, TH1F, TH2F, TLegend, TPaletteAxis, TList, TLine, TAttLine, TF1,TAxis
import re
import sys, string

def getRunNumber(filename):
    global runNumber
    pos=filename.find("__")
    runNumber=filename[pos-6:pos]
    #print runNumber

def GetNonZeroOccNumber(histoname):
    global nrocs
    global fin
    nrocs=0
    histo=fin.Get(histoname)
    nx=histo.GetNbinsX()
    ny=histo.GetNbinsY()
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            value=histo.GetBinContent(i,j)
            if value>0:
                nrocs += 1

nrocs=0
fname=sys.argv[1]

runNumber="0"
getRunNumber(fname)

path="DQMData/Run " + runNumber +"/Pixel/Run summary/Clusters/OnTrack/"

labels=["BPix L1: ", "BPix L2: ", "BPix L3: ", "FPix tot: "]

histonames=[path + "pix_bar Occ_roc_ontracksiPixelDigis_layer_1",path + "pix_bar Occ_roc_ontracksiPixelDigis_layer_2",path + "pix_bar Occ_roc_ontracksiPixelDigis_layer_3",path + "ROC_endcap_occupancy"]

TotROCs=[2560-256,4096-256,5632-256,4320] #total number of ROCs in the Pixel detector layers and the FPix, the factor 256 for BPix Layer derive by half modules, left there as a reminder

DeadROCs=[0,0,0,0]

fin= TFile(fname)

#print type(fname)

outname="PixZeroOccROCs_run" + runNumber + ".txt"
out_file = open(outname, "w")

out_file.write("Pixel Zero Occupancy ROCs \n\n")
bpixtot=0

for k in range(0,4):
    GetNonZeroOccNumber(histonames[k])
    if k==3: nrocs=nrocs/2 #in FPix the histo is filled twice to have it symmetric
    DeadROCs[k]=TotROCs[k]-nrocs
    if k<3: bpixtot+=DeadROCs[k]
    tmpstr=labels[k] + str(DeadROCs[k])
    if k==3: out_file.write("\nBPix tot: %i \n" %bpixtot)	
    out_file.write("%s \n" % tmpstr)

#count entries to avoid low stat runs
clusstr=path+"charge_siPixelClusters"
nclust=fin.Get(clusstr)
nent=nclust.GetEntries()

out_file.write("\nNumber of clusters=  %i \n" % nent)

out_file.close()	

