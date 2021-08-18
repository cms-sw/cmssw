#!/bin/env python
#
# PyROOT study of standard selector performance using sim-hit matching 
# to identify fake and signal muons
#
import os, re, ROOT, sys, pickle, time
from pprint import pprint
from math import *
from array import array
from DataFormats.FWLite import Events, Handle
import numpy as np


##
## User Input
##
def getPFNs(lfns):
    files = []
    for file in lfns:
        fullpath = "/eos/cms/" + file
        if os.path.exists(fullpath):
            files.append(fullpath)
        else:
            raise Exception("File not found: %s" % fullpath)
    return files


def IsMatched(muon1,muon2,sharedFrac=0.5):
    # first check if muon share Wh/Se/St 
    if (muon1.whNum()!=muon2.whNum()): return False 
    if (muon1.scNum()!=muon2.scNum()): return False 
    if (muon1.stNum()!=muon2.stNum()): return False     
    
    # now count the number of shared hits: 
    numShared=0.
    totMuon1=0.
    for ly in range(0,7):
        if (muon1.pathWireId(ly)>=0): 
            totMuon1=totMuon1+1. 
        else:                         
            continue

        if (muon1.pathWireId(ly)!=muon2.pathWireId(ly)): continue
        if (muon1.pathTDC(ly)!=muon2.pathTDC(ly)): continue
        
        numShared = numShared+1.

    if (numShared/totMuon1 >= sharedFrac): return True

    return False

muoBayesHandle, muoBayesLabel = Handle("L1Phase2MuDTExtPhContainer"), ("dtTriggerPhase2BayesPrimitiveDigis","","L1DTTrigPhase2Prod")
muoStdHandle, muoStdLabel = Handle("L1Phase2MuDTExtPhContainer"), ("dtTriggerPhase2StdPrimitiveDigis","","L1DTTrigPhase2Prod")
genHandle, genLabel = Handle("vector<reco::GenParticle>"), "genParticles"


ROOT.gROOT.SetBatch(True)

##
## Main part
##
files = ['../../../../DTTriggerPhase2Primitives.root']

print "Number of files: %d" % len(files)

events = Events(files)

## load some histograms (and efficiencies): 
outputDict = {} 
dumpToFile = True

for frac in [0.25,0.5,0.75,1.00]:
    
    it = 0

    fracname="shared%i" %(frac*100)
    
    # Inclusive in MB
    hPhiRes_q1  = []
    hPhiRes_q3  = []
    hPhiRes_q5  = []
    hPhiRes_q8  = []
    
    hPhiBRes_q1 = []
    hPhiBRes_q3 = []
    hPhiBRes_q5 = []
    hPhiBRes_q8 = []
    
    hChi2Res_q1 = []
    hChi2Res_q3 = []
    hChi2Res_q5 = []
    hChi2Res_q8 = []
    
    hBxRes_q1 = []
    hBxRes_q3 = []
    hBxRes_q5 = []
    hBxRes_q8 = []
    
    hTimeRes_q1 = []
    hTimeRes_q3 = []
    hTimeRes_q5 = []
    hTimeRes_q8 = []

    hMatchingEff = []


    # Exclusive in MB
    hPhiRes_MB_q1  = []
    hPhiRes_MB_q3  = []
    hPhiRes_MB_q5  = []
    hPhiRes_MB_q8  = []
    
    hPhiBRes_MB_q1 = []
    hPhiBRes_MB_q3 = []
    hPhiBRes_MB_q5 = []
    hPhiBRes_MB_q8 = []
    
    hChi2Res_MB_q1 = []
    hChi2Res_MB_q3 = []
    hChi2Res_MB_q5 = []
    hChi2Res_MB_q8 = []
    
    hBxRes_MB_q1 = []
    hBxRes_MB_q3 = []
    hBxRes_MB_q5 = []
    hBxRes_MB_q8 = []
    
    hTimeRes_MB_q1 = []
    hTimeRes_MB_q3 = []
    hTimeRes_MB_q5 = []
    hTimeRes_MB_q8 = []

    hMatchingEff_MB = []

    # Split into 4 MB (1, 2, 3, 4)
    for st in range(1,5):    
        hMatchingEff.append(ROOT.TEfficiency("hMatchingEff_MB%i_%s" %(st,fracname), "",9,0.5,9.5))

        hPhiRes_q1.append(ROOT.TH1F("hPhiRes_MB%i_q1_%s" %(st,fracname),"",20,-5000,5000.))
        hPhiRes_q3.append(ROOT.TH1F("hPhiRes_MB%i_q3_%s" %(st,fracname),"",20,-5000,5000.))
        hPhiRes_q5.append(ROOT.TH1F("hPhiRes_MB%i_q5_%s" %(st,fracname),"",20,-5000,5000.))
        hPhiRes_q8.append(ROOT.TH1F("hPhiRes_MB%i_q8_%s" %(st,fracname),"",20,-5000,5000.))
        
        hPhiBRes_q1.append(ROOT.TH1F("hPhiBRes_MB%i_q1_%s" %(st,fracname),"",20,-100,100.))
        hPhiBRes_q3.append(ROOT.TH1F("hPhiBRes_MB%i_q3_%s" %(st,fracname),"",20,-100,100.))
        hPhiBRes_q5.append(ROOT.TH1F("hPhiBRes_MB%i_q5_%s" %(st,fracname),"",20,-100,100.))
        hPhiBRes_q8.append(ROOT.TH1F("hPhiBRes_MB%i_q8_%s" %(st,fracname),"",20,-100,100.))
        
        hChi2Res_q1.append(ROOT.TH1F("hChi2Res_MB%i_q1_%s" %(st,fracname),"",20,-5000,5000.))
        hChi2Res_q3.append(ROOT.TH1F("hChi2Res_MB%i_q3_%s" %(st,fracname),"",20,-5000,5000.))
        hChi2Res_q5.append(ROOT.TH1F("hChi2Res_MB%i_q5_%s" %(st,fracname),"",20,-5000,5000.))
        hChi2Res_q8.append(ROOT.TH1F("hChi2Res_MB%i_q8_%s" %(st,fracname),"",20,-5000,5000.))
            
        hBxRes_q1.append(ROOT.TH1F("hBxRes_MB%i_q1_%s" %(st,fracname),"",20,-10,10.))
        hBxRes_q3.append(ROOT.TH1F("hBxRes_MB%i_q3_%s" %(st,fracname),"",20,-10,10.))
        hBxRes_q5.append(ROOT.TH1F("hBxRes_MB%i_q5_%s" %(st,fracname),"",20,-10,10.))
        hBxRes_q8.append(ROOT.TH1F("hBxRes_MB%i_q8_%s" %(st,fracname),"",20,-10,10.))
        
        hTimeRes_q1.append(ROOT.TH1F("hTimeRes_MB%i_q1_%s" %(st,fracname),"",20,-100,100.))
        hTimeRes_q3.append(ROOT.TH1F("hTimeRes_MB%i_q3_%s" %(st,fracname),"",20,-100,100.))
        hTimeRes_q5.append(ROOT.TH1F("hTimeRes_MB%i_q5_%s" %(st,fracname),"",20,-100,100.))
        hTimeRes_q8.append(ROOT.TH1F("hTimeRes_MB%i_q8_%s" %(st,fracname),"",20,-100,100.))
    
        # Split into 5 wheels (-2, -1, 0, 1, 2)
        for wh in range(-2,3):
            hMatchingEff_MB.append(ROOT.TEfficiency("hMatchingEff_MB%i_Wh%i_%s" %(st, wh, fracname), "", 9, 0.5, 9.5))

            hPhiRes_MB_q1.append(ROOT.TH1F("hPhiRes_MB_MB%i_MB_Wh%i_MB_q1_MB_%s"   %(st, wh, fracname), "", 20, -5000, 5000.))
            hPhiRes_MB_q3.append(ROOT.TH1F("hPhiRes_MB_MB%i_MB_Wh%i_MB_q3_MB_%s"   %(st, wh, fracname), "", 20, -5000, 5000.))
            hPhiRes_MB_q5.append(ROOT.TH1F("hPhiRes_MB_MB%i_MB_Wh%i_MB_q5_MB_%s"   %(st, wh, fracname), "", 20, -5000, 5000.))
            hPhiRes_MB_q8.append(ROOT.TH1F("hPhiRes_MB_MB%i_MB_Wh%i_MB_q8_MB_%s"   %(st, wh, fracname), "", 20, -5000, 5000.))
        
            hPhiBRes_MB_q1.append(ROOT.TH1F("hPhiBRes_MB_MB%i_MB_Wh%i_MB_q1_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hPhiBRes_MB_q3.append(ROOT.TH1F("hPhiBRes_MB_MB%i_MB_Wh%i_MB_q3_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hPhiBRes_MB_q5.append(ROOT.TH1F("hPhiBRes_MB_MB%i_MB_Wh%i_MB_q5_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hPhiBRes_MB_q8.append(ROOT.TH1F("hPhiBRes_MB_MB%i_MB_Wh%i_MB_q8_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
        
            hChi2Res_MB_q1.append(ROOT.TH1F("hChi2Res_MB_MB%i_MB_Wh%i_MB_q1_MB_%s" %(st, wh, fracname), "", 20, -5000, 5000.))
            hChi2Res_MB_q3.append(ROOT.TH1F("hChi2Res_MB_MB%i_MB_Wh%i_MB_q3_MB_%s" %(st, wh, fracname), "", 20, -5000, 5000.))
            hChi2Res_MB_q5.append(ROOT.TH1F("hChi2Res_MB_MB%i_MB_Wh%i_MB_q5_MB_%s" %(st, wh, fracname), "", 20, -5000, 5000.))
            hChi2Res_MB_q8.append(ROOT.TH1F("hChi2Res_MB_MB%i_MB_Wh%i_MB_q8_MB_%s" %(st, wh, fracname), "", 20, -5000, 5000.))
            
            hBxRes_MB_q1.append(ROOT.TH1F("hBxRes_MB_MB%i_MB_Wh%i_MB_q1_MB_%s"     %(st, wh, fracname), "", 20, -10, 10.))
            hBxRes_MB_q3.append(ROOT.TH1F("hBxRes_MB_MB%i_MB_Wh%i_MB_q3_MB_%s"     %(st, wh, fracname), "", 20, -10, 10.))
            hBxRes_MB_q5.append(ROOT.TH1F("hBxRes_MB_MB%i_MB_Wh%i_MB_q5_MB_%s"     %(st, wh, fracname), "", 20, -10, 10.))
            hBxRes_MB_q8.append(ROOT.TH1F("hBxRes_MB_MB%i_MB_Wh%i_MB_q8_MB_%s"     %(st, wh, fracname), "", 20, -10, 10.))
        
            hTimeRes_MB_q1.append(ROOT.TH1F("hTimeRes_MB_MB%i_MB_Wh%i_MB_q1_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hTimeRes_MB_q3.append(ROOT.TH1F("hTimeRes_MB_MB%i_MB_Wh%i_MB_q3_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hTimeRes_MB_q5.append(ROOT.TH1F("hTimeRes_MB_MB%i_MB_Wh%i_MB_q5_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))
            hTimeRes_MB_q8.append(ROOT.TH1F("hTimeRes_MB_MB%i_MB_Wh%i_MB_q8_MB_%s" %(st, wh, fracname), "", 20, -100, 100.))


    print "now save into dictionary"
    outputDict[fracname] = {}
    for st in range(1,5):
        outputDict[fracname]['hMatchingEff_MB%i'%st] = hMatchingEff[st-1]

        outputDict[fracname]['hPhiRes_MB%i_q1'%st] = hPhiRes_q1[st-1]
        outputDict[fracname]['hPhiRes_MB%i_q3'%st] = hPhiRes_q3[st-1]
        outputDict[fracname]['hPhiRes_MB%i_q5'%st] = hPhiRes_q5[st-1]
        outputDict[fracname]['hPhiRes_MB%i_q8'%st] = hPhiRes_q8[st-1]       

        outputDict[fracname]['hPhiBRes_MB%i_q1'%st] = hPhiBRes_q1[st-1]
        outputDict[fracname]['hPhiBRes_MB%i_q3'%st] = hPhiBRes_q3[st-1]
        outputDict[fracname]['hPhiBRes_MB%i_q5'%st] = hPhiBRes_q5[st-1]
        outputDict[fracname]['hPhiBRes_MB%i_q8'%st] = hPhiBRes_q8[st-1]
    
        outputDict[fracname]['hChi2Res_MB%i_q1'%st] = hChi2Res_q1[st-1]
        outputDict[fracname]['hChi2Res_MB%i_q3'%st] = hChi2Res_q3[st-1]
        outputDict[fracname]['hChi2Res_MB%i_q5'%st] = hChi2Res_q5[st-1]
        outputDict[fracname]['hChi2Res_MB%i_q8'%st] = hChi2Res_q8[st-1]       

        outputDict[fracname]['hBxRes_MB%i_q1'%st] = hBxRes_q1[st-1]
        outputDict[fracname]['hBxRes_MB%i_q3'%st] = hBxRes_q3[st-1]
        outputDict[fracname]['hBxRes_MB%i_q5'%st] = hBxRes_q5[st-1]
        outputDict[fracname]['hBxRes_MB%i_q8'%st] = hBxRes_q8[st-1]       

        outputDict[fracname]['hTimeRes_MB%i_q1'%st] = hTimeRes_q1[st-1]
        outputDict[fracname]['hTimeRes_MB%i_q3'%st] = hTimeRes_q3[st-1]
        outputDict[fracname]['hTimeRes_MB%i_q5'%st] = hTimeRes_q5[st-1]
        outputDict[fracname]['hTimeRes_MB%i_q8'%st] = hTimeRes_q8[st-1]       

        for wh in range(-2,3):

            it = it + 1
            print("MB = {}, Wheel = {}, iteration = {}".format(st, wh, it))

            outputDict[fracname]['hMatchingEff_MB%i_Wh%i'%(st, wh)] = hMatchingEff_MB[wh+2 + 5*(st-1)]

            outputDict[fracname]['hPhiRes_MB%i_Wh%i_q1'%(st, wh)]  = hPhiRes_MB_q1[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiRes_MB%i_Wh%i_q3'%(st, wh)]  = hPhiRes_MB_q3[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiRes_MB%i_Wh%i_q5'%(st, wh)]  = hPhiRes_MB_q5[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiRes_MB%i_Wh%i_q8'%(st, wh)]  = hPhiRes_MB_q8[wh+2 + 5*(st-1)]       

            outputDict[fracname]['hPhiBRes_MB%i_Wh%i_q1'%(st, wh)] = hPhiBRes_MB_q1[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiBRes_MB%i_Wh%i_q3'%(st, wh)] = hPhiBRes_MB_q3[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiBRes_MB%i_Wh%i_q5'%(st, wh)] = hPhiBRes_MB_q5[wh+2 + 5*(st-1)]
            outputDict[fracname]['hPhiBRes_MB%i_Wh%i_q8'%(st, wh)] = hPhiBRes_MB_q8[wh+2 + 5*(st-1)]
    
            outputDict[fracname]['hChi2Res_MB%i_Wh%i_q1'%(st, wh)] = hChi2Res_MB_q1[wh+2 + 5*(st-1)]
            outputDict[fracname]['hChi2Res_MB%i_Wh%i_q3'%(st, wh)] = hChi2Res_MB_q3[wh+2 + 5*(st-1)]
            outputDict[fracname]['hChi2Res_MB%i_Wh%i_q5'%(st, wh)] = hChi2Res_MB_q5[wh+2 + 5*(st-1)]
            outputDict[fracname]['hChi2Res_MB%i_Wh%i_q8'%(st, wh)] = hChi2Res_MB_q8[wh+2 + 5*(st-1)]       

            outputDict[fracname]['hBxRes_MB%i_Wh%i_q1'%(st, wh)]   = hBxRes_MB_q1[wh+2 + 5*(st-1)]
            outputDict[fracname]['hBxRes_MB%i_Wh%i_q3'%(st, wh)]   = hBxRes_MB_q3[wh+2 + 5*(st-1)]
            outputDict[fracname]['hBxRes_MB%i_Wh%i_q5'%(st, wh)]   = hBxRes_MB_q5[wh+2 + 5*(st-1)]
            outputDict[fracname]['hBxRes_MB%i_Wh%i_q8'%(st, wh)]   = hBxRes_MB_q8[wh+2 + 5*(st-1)]       

            outputDict[fracname]['hTimeRes_MB%i_Wh%i_q1'%(st, wh)] = hTimeRes_MB_q1[wh+2 + 5*(st-1)]
            outputDict[fracname]['hTimeRes_MB%i_Wh%i_q3'%(st, wh)] = hTimeRes_MB_q3[wh+2 + 5*(st-1)]
            outputDict[fracname]['hTimeRes_MB%i_Wh%i_q5'%(st, wh)] = hTimeRes_MB_q5[wh+2 + 5*(st-1)]
            outputDict[fracname]['hTimeRes_MB%i_Wh%i_q8'%(st, wh)] = hTimeRes_MB_q8[wh+2 + 5*(st-1)]       

    # loop over events
    count = 0

    if (dumpToFile): 
        f= open("EventDumpList_StdToBayes.log","w+")

    for ev in events:
        if not count%1000:  print count, events.size()
        count = count+1
        
        ev.getByLabel(muoBayesLabel, muoBayesHandle)
        ev.getByLabel(muoStdLabel, muoStdHandle)
        ev.getByLabel(genLabel, genHandle)

        muon_bayes = muoBayesHandle.product().getContainer()
        muon_std = muoStdHandle.product().getContainer()
    
        if (dumpToFile):
            f.write( "\nInspecting Event Number %i \n" %(ev.eventAuxiliary().id().event())  )
            f.write( "        Wh   Se   St  | w1 w2 w3 w4 w5 w6 w7 w8 |  tdc1  tdc2  tdc3  tdc4  tdc5  tdc6  tdc7  tdc8 | lat1 lat2 lat3 lat4 lat5 lat6 lat7 lat8 | Q     phi  phib  phi_cmssw  phib_cmssw  bX      Chi2         x tanPsi     t0 \n" )
            f.write( "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- \n" )
            for muon in muon_bayes:
                f.write( "[Bayes]: Wh%2d Se%2d St%1d | %2d %2d %2d %2d %2d %2d %2d %2d | %5d %5d %5d %5d %5d %5d %5d %5d |  %2d   %2d   %2d   %2d   %2d   %2d   %2d   %2d | Q%1d %6d %5d     %6d       %5d %3d %9d %9d %6d %6d\n" %(muon.whNum(),muon.scNum(),muon.stNum(),muon.pathWireId(0),muon.pathWireId(1),muon.pathWireId(2),muon.pathWireId(3),muon.pathWireId(4),muon.pathWireId(5),muon.pathWireId(6),muon.pathWireId(7),muon.pathTDC(0),muon.pathTDC(1),muon.pathTDC(2),muon.pathTDC(3),muon.pathTDC(4),muon.pathTDC(5),muon.pathTDC(6),muon.pathTDC(7),muon.pathLat(0),muon.pathLat(1),muon.pathLat(2),muon.pathLat(3),muon.pathLat(4),muon.pathLat(5),muon.pathLat(6),muon.pathLat(7),muon.quality(),muon.phi(),muon.phiBend(),muon.phiCMSSW(),muon.phiBendCMSSW(),muon.bxNum()-20,muon.chi2(), muon.xLocal(), muon.tanPsi(), muon.t0() ) )
        

        for muon in muon_std:
            if (dumpToFile): 
                f.write( "[Std  ]: Wh%2d Se%2d St%1d | %2d %2d %2d %2d %2d %2d %2d %2d | %5d %5d %5d %5d %5d %5d %5d %5d |  %2d   %2d   %2d   %2d   %2d   %2d   %2d   %2d | Q%1d %6d %5d     %6d       %5d %3d %9d %9d %6d %6d\n" %(muon.whNum(),muon.scNum(),muon.stNum(),muon.pathWireId(0),muon.pathWireId(1),muon.pathWireId(2),muon.pathWireId(3),muon.pathWireId(4),muon.pathWireId(5),muon.pathWireId(6),muon.pathWireId(7),muon.pathTDC(0),muon.pathTDC(1),muon.pathTDC(2),muon.pathTDC(3),muon.pathTDC(4),muon.pathTDC(5),muon.pathTDC(6),muon.pathTDC(7),muon.pathLat(0),muon.pathLat(1),muon.pathLat(2),muon.pathLat(3),muon.pathLat(4),muon.pathLat(5),muon.pathLat(6),muon.pathLat(7),muon.quality(),muon.phi(),muon.phiBend(),muon.phiCMSSW(),muon.phiBendCMSSW(),muon.bxNum()-20,muon.chi2(), muon.xLocal(), muon.tanPsi(), muon.t0() ) )
            ## now match with the previous 
            st = muon.stNum()-1
            wh = muon.whNum()+2
            matched = False
            for muon2 in muon_bayes: 
                matched = matched or IsMatched(muon,muon2,frac)
                if not IsMatched(muon,muon2,frac): continue

                if (muon.quality()>=1) :  
                    # Inclusive in MB
                    hPhiRes_q1[st]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_q1[st] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_q1[st] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_q1[st]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_q1[st] .Fill( (muon.t0()-muon2.t0()) )

                    # Exclusive in MB
                    hPhiRes_MB_q1[5*st + wh]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_MB_q1[5*st + wh] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_MB_q1[5*st + wh] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_MB_q1[5*st + wh]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_MB_q1[5*st + wh] .Fill( (muon.t0()-muon2.t0()) )
                    
                if (muon.quality()>=3) :
                    # Inclusive in MB
                    hPhiRes_q3[st]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_q3[st] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_q3[st] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_q3[st]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_q3[st] .Fill( (muon.t0()-muon2.t0()) )

                    # Exclusive in MB
                    hPhiRes_MB_q3[5*st + wh]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_MB_q3[5*st + wh] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_MB_q3[5*st + wh] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_MB_q3[5*st + wh]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_MB_q3[5*st + wh] .Fill( (muon.t0()-muon2.t0()) )

                if (muon.quality()>=5) :
                    # Inclusive in MB
                    hPhiRes_q5[st]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_q5[st] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_q5[st] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_q5[st]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_q5[st] .Fill( (muon.t0()-muon2.t0()) )

                    # Exclusive in MB
                    hPhiRes_MB_q5[5*st + wh]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_MB_q5[5*st + wh] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_MB_q5[5*st + wh] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_MB_q5[5*st + wh]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_MB_q5[5*st + wh] .Fill( (muon.t0()-muon2.t0()) )
               
                if (muon.quality()>=8) :
                    # Inclusive in MB
                    hPhiRes_q8[st]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_q8[st] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_q8[st] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_q8[st]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_q8[st] .Fill( (muon.t0()-muon2.t0()) )

                    # Exclusive in MB
                    hPhiRes_MB_q8[5*st + wh]  .Fill( (muon.phi()-muon2.phi()) )
                    hPhiBRes_MB_q8[5*st + wh] .Fill( (muon.phiBend()-muon2.phiBend()) )
                    hChi2Res_MB_q8[5*st + wh] .Fill( (muon.chi2()-muon2.chi2()) )
                    hBxRes_MB_q8[5*st + wh]   .Fill( (muon.bxNum()-muon2.bxNum()) )
                    hTimeRes_MB_q8[5*st + wh] .Fill( (muon.t0()-muon2.t0()) )
                
            hMatchingEff[st].Fill(matched, muon.quality())
            hMatchingEff_MB[5*st + wh].Fill(matched, muon.quality())

    if (dumpToFile): f.close()
    ev.toBegin()
    dumpToFile=False

        

import pickle 
with open('GroupingComparison_StdToBayes.pickle', 'wb') as handle:
    pickle.dump(outputDict,   handle, protocol=pickle.HIGHEST_PROTOCOL)
