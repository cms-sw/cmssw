from ROOT import *

from cuts import *

## run quiet mode
import sys
sys.argv.append( '-b' )

import ROOT 
ROOT.gROOT.SetBatch(1)

#_______________________________________________________________________________
def gemChamberNumbering(plotter):
    gROOT.ForceStyle();
    gROOT.SetStyle("Plain");
    gStyle.SetPalette(1);
    gStyle.SetOptTitle(kFALSE);
    gStyle.SetOptStat(0); 
    gStyle.SetMarkerStyle(20);
    
    c1 = TCanvas("c1","c1",600,600)
    c1.cd()
    
    hist = TH1F("hist","",1000000,-10000,10000)
    
    XposGEM = 36*[None]
    YposGEM = 36*[None]
    XposCSC = 36*[None]
    YposCSC = 36*[None]

    plotter.treeGEMSimHits.Draw("globalY:globalX",AND(rm1,ri1,st1,odd)) 
    h = gPad.FindObject("Graph")
    GEMglobalzx = h.Clone()
    GEMglobalzx.SetMarkerColor(kRed)
    GEMglobalzx.SetMarkerSize(0.1)
              
    plotter.treeCSCSimHits.Draw("globalY:globalX",AND(ec2,ri1,st1,odd)) 
    h = gPad.FindObject("Graph")
    CSCglobalzx = h.Clone()
    CSCglobalzx.SetMarkerColor(kBlue)
    CSCglobalzx.SetMarkerSize(0.1)

    for i in range(1,37,2):
        plotter.treeGEMSimHits.Draw("globalY:globalX",TCut("chamber==%i"%(i)))
        h = gPad.FindObject("Graph")
        TEMPglobalzx = h.Clone()
        XposGEM[i-1] = TEMPglobalzx.GetMean(1)
        YposGEM[i-1] = TEMPglobalzx.GetMean(2)
            
        plotter.treeCSCSimHits.Draw("globalY:globalX",TCut("chamber==%i"%(i)))
        h = gPad.FindObject("Graph")
        TEMP2globalzx = h.Clone()
        XposCSC[i-1] = TEMP2globalzx.GetMean(1)
        YposCSC[i-1] = TEMP2globalzx.GetMean(2)
        
    hist.GetXaxis().SetTitle("Global x [cm]")
    hist.GetYaxis().SetTitle("Global y [cm]")

    hist.GetYaxis().SetTitleOffset(1.3)
    hist.SetAxisRange(-350,350,"X")
    hist.SetAxisRange(-350,350,"Y")
    hist.Draw("axis")

    for i in range(1,37,2):
        NumberingGEM = TText(1.7*XposGEM[i-1],1.7*YposGEM[i-1],"%d"%(i))
        NumberingGEM.SetTextSize(0.025)
        NumberingGEM.SetTextAngle(0)
        NumberingGEM.SetTextColor(kRed)
        NumberingGEM.Draw("same")
        SetOwnership(NumberingGEM, False)
        
        NumberingCSC = TText(1.05*XposCSC[i-1],1.05*YposCSC[i-1],"%d"%(i))
        NumberingCSC.SetTextSize(0.025)
        NumberingCSC.SetTextAngle(0)
        NumberingCSC.SetTextColor(kBlue)
        NumberingCSC.Draw("same")
        SetOwnership(NumberingCSC, False)
        
    CSCglobalzx.Draw("same p")
    GEMglobalzx.Draw("same p")
    
    tt_CSC = TText(295,326,"CSC")
    tt_CSC.SetTextSize(0.03)
    tt_CSC.SetTextAngle(0)
    tt_CSC.SetTextColor(kBlue)
    tt_CSC.Draw("same")
    
    tt_GEM = TText(295,305,"GEM")
    tt_GEM.SetTextSize(0.03)
    tt_GEM.SetTextAngle(0)
    tt_GEM.SetTextColor(kRed)
    tt_GEM.Draw("same")
    
    tt_TEXT = TText(-330,326,"Odd Chambers in Region -1")
    tt_TEXT.SetTextSize(0.03)
    tt_TEXT.Draw("same")
    
    c1.SaveAs("globalxy_odd_rm1" + plotter.ext)
        
