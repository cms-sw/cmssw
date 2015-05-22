#!/usr/bin/env python
import ROOT 
from math import *
import os, os.path
import sys

def doSpam(text,x1,y1,x2,y2,align=12,fill=False,textSize=0.033,_noDelete={}):
    cmsprel = ROOT.TPaveText(x1,y1,x2,y2,"NDC");
    cmsprel.SetTextSize(textSize);
    cmsprel.SetFillColor(0);
    cmsprel.SetFillStyle(1001 if fill else 0);
    cmsprel.SetLineStyle(2);
    cmsprel.SetLineColor(0);
    cmsprel.SetTextAlign(align);
    cmsprel.SetTextFont(42);
    cmsprel.AddText(text);
    cmsprel.Draw("same");
    _noDelete[text] = cmsprel; ## so it doesn't get deleted by PyROOT
    return cmsprel

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(".x ../tdrstyle.cc")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
c1 = ROOT.TCanvas("c1","c1", 600, 600)
c1.SetWindowSize(600 + (600 - c1.GetWw()), 600 + (600 - c1.GetWh()));
c1.SetTicky(0)
c1.SetTickx(0)
c1.SetLeftMargin(0.30)

results = {}
postfix = sys.argv[1] if len(sys.argv) > 1 else ""
if postfix == "":
    raise RuntimeError, "No"
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      '2l'    : [5.65773, -1.8374, 2.17504],
      'ee'    : [3.42021, -3.77313, 4.45043],
      'em'    : [2.61703, -2.15691, 2.48682],
      'mumu'  : [8.61957, -2.69825, 3.29364],
      '3l'    : [2.51768, -1.76297, 2.18453],
      'comb'  : [3.83225, -1.41544, 1.67779],
    }
elif postfix == "NJet":
    raise RuntimeError, "No"
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      'ee'    : [0.690485, -4.05982, 4.47307],
      'em'    : [2.6722, -2.24322, 2.5548],
      'mumu'  : [7.13644, -2.77257, 3.29741],
      '2l'    : [4.39539, -1.90249, 2.18146],
      '3l'    : [4.66682, -2.10897, 2.59197],
      'comb'  : [3.93816, -1.49112, 1.7396],
    }
elif postfix == "QMVA":
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      '2l'    : [5.35043, -1.83264, 2.15917],
      'ee'    : [2.81824, -4.07876, 4.57835],
      'em'    : [1.90023, -2.29834, 2.51794],
      'mumu'  : [8.38519, -2.7039, 3.2826],
      '3l'    : [2.70601, -1.76268, 2.18078],
      'comb'  : [3.74372, -1.3898, 1.64412],
    }
elif postfix == "QNJet":
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      '2l'    : [4.20321, -2.00163, 2.24926],
      'ee'    : [-0.0513303, -1.94867, 4.72933],
      'em'    : [2.40334, -2.47334, 2.68143],
      'mumu'  : [6.81641, -2.87952, 3.33022],
      '3l'    : [4.6449, -2.12436, 2.61915],
      'comb'  : [3.87704, -1.51796, 1.74815],
    }
elif postfix == "QMVA_Sip4":
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      '2l'    : [3.73785, -1.83749, 2.20754],
      'ee'    : [2.82931, -4.09548, 4.70275],
      'em'    : [1.36071, -2.29531, 2.59365],
      'mumu'  : [5.69465, -2.56476, 3.19605],
      '3l'    : [2.74969, -1.7992, 2.29945],
      'comb'  : [2.69946, -1.35242, 1.65066],
    }
elif postfix == "QMVA_SUS13":
    results = {
      '4l'   : [ -4.18   , -(6.0-4.18), (0.246+4.18) ],
      '2l'    : [5.37398, -2.25533, 2.77592],
      'ee'    : [7.90319, -4.85508, 6.24588],
      'em'    : [3.10822, -2.31458, 2.83081],
      'mumu'  : [6.98678, -3.72345, 4.58406],
      '3l'    : [1.04783, -1.55404, 2.19828],
      'comb'  : [2.48831, -1.45678, 1.7624],
    }



frame = ROOT.TH2F("frame","frame", 100, -7, 9, 4, 0, 4);
frame.GetXaxis().SetTitle("Best fit #mu = #sigma/#sigma_{SM}")
#frame.GetYaxis().SetBinLabel(4, "#splitline{combined}{   #mu = %.2f_{#scale[1.4]{-}%.2f}^{+%.2f}}" % (results['comb'][0],-results['comb'][1],results['comb'][2])  )
#frame.GetYaxis().SetBinLabel(4, "" )
frame.GetYaxis().SetBinLabel(1, "#splitline{dilepton}{   #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['2l'  ][0],-results['2l'  ][1],results['2l'  ][2])  )
frame.GetYaxis().SetBinLabel(2, "#splitline{trilepton}{  #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['3l'  ][0],-results['3l'  ][1],results['3l'  ][2])  )
frame.GetYaxis().SetBinLabel(3, "#splitline{four-lepton}{#mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['4l'  ][0],-results['4l'  ][1],results['4l'  ][2])  )
frame.GetYaxis().SetLabelSize(0.070);
frame.Draw()

combBand = ROOT.TBox( results['comb'][0]+results['comb'][1], 0, results['comb'][0]+results['comb'][2], 3);
combLine = ROOT.TLine(results['comb'][0]                   , 0, results['comb'][0]                   , 3);


points = ROOT.TGraphAsymmErrors(3)
for i,l in enumerate(["2l","3l","4l"]):
    points.SetPoint(i, results[l][0], 0.5+i)
    points.SetPointError(i, -results[l][1], results[l][2], 0, 0)

combBand.SetFillColor(80); combBand.SetFillStyle(1001); combBand.SetLineStyle(0);  
combBand.Draw()
combLine.SetLineWidth(4); 
combLine.Draw()

stopLine = ROOT.TLine(results['4l'][0]+results['4l'][1]-0.02,  2.25, results['4l'][0]+results['4l'][1]-0.02, 2.75);
stopLine.SetLineStyle(2);
stopLine.SetLineWidth(2);
stopLine.Draw()

points.SetLineColor(2)
points.SetLineWidth(5)
points.SetMarkerStyle(21)
points.SetMarkerSize(1.7)
points.Draw("P SAME");

combtext = "combined #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}" % (results['comb'][0],-results['comb'][1],results['comb'][2])
frame.Draw("AXIS SAME")
doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=0.0355)
doSpam("CMS Preliminary",                    .32, .875, .640, .927, align=21, textSize=0.045)
doSpam("m_{H} = 125.7 GeV",                  .66, .875, .950, .927, align=21, textSize=0.045)
doSpam( combtext,                            .32, .765, .950, .847, align=21, textSize=0.045)


c1.Print("muhat%s.pdf" % postfix)
c1.Print("muhat%s.png" % postfix)

frame = ROOT.TH2F("frame","frame", 100, -7, 13, 7, 0, 7);
frame.GetXaxis().SetTitle("Best fit #mu = #sigma/#sigma_{SM}")
#frame.GetYaxis().SetBinLabel(4, "#splitline{combined}{   #mu = %.2f_{#scale[1.4]{-}%.2f}^{+%.2f}}" % (results['comb'][0],-results['comb'][1],results['comb'][2])  )
#frame.GetYaxis().SetBinLabel(4, "" )
frame.GetYaxis().SetBinLabel(1, "#splitline{electron-muon}{   #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['em'  ][0],-results['em'  ][1],results['em'  ][2])  )
frame.GetYaxis().SetBinLabel(2, "#splitline{dimuon}{   #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['mumu'  ][0],-results['mumu'  ][1],results['mumu'  ][2])  )
frame.GetYaxis().SetBinLabel(3, "#splitline{dielectron}{   #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['ee'  ][0],-results['ee'  ][1],results['ee'  ][2])  )
frame.GetYaxis().SetBinLabel(4, "#splitline{trilepton}{  #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['3l'  ][0],-results['3l'  ][1],results['3l'  ][2])  )
frame.GetYaxis().SetBinLabel(5, "#splitline{four-lepton}{#mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}}" % (results['4l'  ][0],-results['4l'  ][1],results['4l'  ][2])  )
frame.GetYaxis().SetLabelSize(0.060);
frame.Draw()

combBand = ROOT.TBox( results['comb'][0]+results['comb'][1], 0, results['comb'][0]+results['comb'][2], 5);
combLine = ROOT.TLine(results['comb'][0]                   , 0, results['comb'][0]                   , 5);


points = ROOT.TGraphAsymmErrors(3)
for i,l in enumerate(["em","mumu","ee","3l","4l"]):
    points.SetPoint(i, results[l][0], 0.5+i)
    points.SetPointError(i, -results[l][1], results[l][2], 0, 0)

combBand.SetFillColor(80); combBand.SetFillStyle(1001); combBand.SetLineStyle(0);  
combBand.Draw()
combLine.SetLineWidth(4); 
combLine.Draw()

stopLine = ROOT.TLine(results['4l'][0]+results['4l'][1]-0.02,  4.25, results['4l'][0]+results['4l'][1]-0.02, 4.75);
stopLine.SetLineStyle(2);
stopLine.SetLineWidth(2);
stopLine.Draw()

points.SetLineColor(2)
points.SetLineWidth(5)
points.SetMarkerStyle(21)
points.SetMarkerSize(1.7)
points.Draw("P SAME");

combtext = "combined #mu = %.1f_{#scale[1.4]{-}%.1f}^{+%.1f}" % (results['comb'][0],-results['comb'][1],results['comb'][2])
frame.Draw("AXIS SAME")
doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=0.0355)
doSpam("CMS Preliminary",                    .32, .875, .640, .927, align=21, textSize=0.045)
doSpam("m_{H} = 125.7 GeV",                  .66, .875, .950, .927, align=21, textSize=0.045)
doSpam( combtext,                            .32, .765, .950, .847, align=21, textSize=0.045)


c1.Print("muhatSplit%s.png" % postfix)
c1.Print("muhatSplit%s.pdf" % postfix)
