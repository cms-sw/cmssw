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

def doLegend(x1,y1,x2,y2,textSize=0.035):
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetShadowColor(0)
    leg.SetTextFont(42)
    leg.SetTextSize(textSize)
    return leg



ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine(".x ../tdrstyle.cc")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
c1 = ROOT.TCanvas("c1","c1", 600, 600)
c1.SetWindowSize(600 + (600 - c1.GetWw()), 600 + (600 - c1.GetWh()));
c1.SetTicky(0)
c1.SetTickx(0)
c1.SetLeftMargin(0.37)


results = {}
postfix = sys.argv[1] if len(sys.argv) > 1 else ""
if postfix == "":
    raise RuntimeError, "NO"
    results = {
      '2l'    : [9.4723, 1.6647, 2.2588, 3.2656, 4.8407, 6.968],
      'ee'    : [12.0174, 4.3701, 6.0302, 8.8438, 13.3207, 19.7355],
      'em'    : [7.2091, 2.3699, 3.2358, 4.7031, 6.9715, 10.1497],
      'mumu'  : [14.4386, 2.1233, 2.9299, 4.2969, 6.4721, 9.5373],
      '3l'    : [6.5489, 1.7714, 2.4766, 3.6719, 5.6478, 8.5145],
      'comb'  : [6.7765, 1.1725, 1.6043, 2.3359, 3.4998, 5.1186],
      '4l'    : [6.7873, 4.0246, 5.7377, 8.8438, 14.2725, 22.653],
    }
elif postfix == "NJet":
    raise RuntimeError, "NO"
    results = {
      '4l'    : [6.7873, 4.0246, 5.7377, 8.8438, 14.2725, 22.653],
      'ee'    : [9.8822, 4.5333, 6.2024, 9.0312, 13.6031, 20.0457],
      'em'    : [7.3809, 2.474, 3.3708, 4.8906, 7.2494, 10.4353],
      'mumu'  : [12.9466, 2.7269, 3.7154, 5.3906, 7.9906, 11.5678],
      '2l'    : [8.2147, 1.9343, 2.6282, 3.7656, 5.5218, 7.8609],
      '3l'    : [9.3148, 2.0306, 2.7939, 4.1094, 6.2224, 9.2367],
      'comb'  : [6.98, 1.3819, 1.8855, 2.7109, 4.0185, 5.7514],
    }
elif postfix == "QMVA":
    results = {
      '4l'    : [6.7873, 4.0246, 5.7377, 8.8438, 14.2725, 22.653],
      '2l'    : [9.1374, 1.7152, 2.3369, 3.3906, 4.9989, 7.2605],
      'ee'    : [11.8393, 4.9224, 6.6791, 9.6562, 14.4675, 21.0426],
      'em'    : [6.7048, 2.6245, 3.5564, 5.1094, 7.5737, 10.9643],
      'mumu'  : [14.1821, 2.1696, 2.9938, 4.3906, 6.6483, 9.8688],
      '3l'    : [6.7329, 1.824, 2.5293, 3.7656, 5.792, 8.7319],
      'comb'  : [6.6331, 1.1961, 1.6411, 2.3828, 3.5701, 5.2213],
    }
elif postfix == "QNJet":
    results = {
      '4l'    : [6.7873, 4.0246, 5.7377, 8.8438, 14.2725, 22.653],
      '2l'    : [8.1465, 2.2082, 2.9759, 4.2344, 6.1754, 8.7143],
      'ee'    : [9.9407, 5.1136, 6.9385, 10.0312, 14.8694, 21.5262],
      'em'    : [7.4114, 2.8062, 3.7769, 5.4219, 7.9505, 11.3851],
      'mumu'  : [12.7018, 3.2187, 4.3554, 6.2188, 9.119, 13.0585],
      '3l'    : [9.3822, 2.185, 3.0151, 4.4219, 6.6956, 9.9391],
      'comb'  : [6.9324, 1.537, 2.0827, 2.9922, 4.4115, 6.2974],
    }
else: raise RuntimeError, "Embeh?"




frame = ROOT.TH2F("frame","frame", 100, 1.0, 30, 5, 0, 5);
frame.GetXaxis().SetTitle("95% CL upper limit on #mu = #sigma/#sigma_{SM}")
frame.GetYaxis().SetBinLabel(1, "#splitline{combined}{   #mu < %.1f (%.1f exp)}" % (results['comb'][0],results['comb'][3])  )
frame.GetYaxis().SetBinLabel(2, "#splitline{dilepton}{   #mu < %.1f (%.1f exp)}" % (results['2l'  ][0],results['2l'  ][3])  )
frame.GetYaxis().SetBinLabel(3, "#splitline{trilepton}{  #mu < %.1f (%.1f exp)}" % (results['3l'  ][0],results['3l'  ][3])  )
frame.GetYaxis().SetBinLabel(4, "#splitline{four-lepton}{#mu < %.1f (%.1f exp)}" % (results['4l'  ][0],results['4l'  ][3])  )
frame.GetYaxis().SetLabelSize(0.070);
frame.GetXaxis().SetNoExponent(1);
frame.GetXaxis().SetMoreLogLabels(1);
frame.Draw()
c1.SetLogx(1)


obs = ROOT.TGraphAsymmErrors(4)
exp = ROOT.TGraphAsymmErrors(4)
b68 = ROOT.TGraphAsymmErrors(4)
b95 = ROOT.TGraphAsymmErrors(4)
for i,l in enumerate(["comb","2l","3l","4l"]):
    obs.SetPoint(i, results[l][0], 0.5+i)
    exp.SetPoint(i, results[l][3], 0.5+i)
    b68.SetPoint(i, results[l][3], 0.5+i)
    b95.SetPoint(i, results[l][3], 0.5+i)
    b68.SetPointError(i, results[l][3]-results[l][2], results[l][4]-results[l][3], 0.5, 0.5)
    b95.SetPointError(i, results[l][3]-results[l][1], results[l][5]-results[l][3], 0.5, 0.5)
    exp.SetPointError(i, 0,0, 0.5, 0.5)
    obs.SetPointError(i, 0,0, 0.5, 0.5)

b68.SetFillColor(80);
b95.SetFillColor(90);
for X in exp, b68, b95:
    X.SetMarkerStyle(25);
    X.SetLineWidth(3);
    X.SetLineStyle(2);
obs.SetMarkerStyle(21);
obs.SetLineWidth(3);
obs.SetLineStyle(1);
b95.Draw("E2 SAME")
b68.Draw("E2 SAME")
exp.Draw("PZ SAME")
obs.Draw("PZ SAME")

#stopLine = ROOT.TLine(results['4l'][0]+results['4l'][1]-0.02,  0.25, results['4l'][0]+results['4l'][1]-0.02, 0.75);
#stopLine.SetLineStyle(2);
#stopLine.SetLineWidth(2);
#stopLine.Draw()

#combtext = "combined #mu = %.2f_{#scale[1.4]{-}%.2f}^{+%.2f}" % (results['comb'][0],-results['comb'][1],results['comb'][2])
frame.Draw("AXIS SAME")
doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=0.0355)
doSpam("CMS Preliminary",                    .38, .875, .690, .927, align=21, textSize=0.045)
doSpam("m_{H} = 125.7 GeV",                  .38, .795, .690, .867, align=21, textSize=0.045)
#doSpam( combtext,                            .32, .765, .950, .847, align=21, textSize=0.045)
leg = doLegend(.710,.795,.950,.937, textSize=0.0355)
leg.AddEntry(obs, "Observed", "LP")
leg.AddEntry(b68, "Exp. (68%)", "LPF")
leg.AddEntry(b95, "Exp. (95%)", "LPF")
leg.Draw()

c1.Print("clslimit%s.png" % postfix)
c1.Print("clslimit%s.pdf" % postfix)

frame = ROOT.TH2F("frame","frame", 100, 1.0, 30, 8, 0, 8);
frame.GetXaxis().SetTitle("95% CL upper limit on #mu = #sigma/#sigma_{SM}")
frame.GetYaxis().SetBinLabel(1, "#splitline{combined}{   #mu < %.1f (%.1f exp)}" % (results['comb'][0],results['comb'][3])  )
frame.GetYaxis().SetBinLabel(2, "#splitline{dimuon}{   #mu < %.1f (%.1f exp)}" % (results['mumu'  ][0],results['mumu'  ][3])  )
frame.GetYaxis().SetBinLabel(3, "#splitline{dielectron}{   #mu < %.1f (%.1f exp)}" % (results['ee'  ][0],results['ee'  ][3])  )
frame.GetYaxis().SetBinLabel(4, "#splitline{electron-muon}{   #mu < %.1f (%.1f exp)}" % (results['em'  ][0],results['em'  ][3])  )
frame.GetYaxis().SetBinLabel(5, "#splitline{trilepton}{  #mu < %.1f (%.1f exp)}" % (results['3l'  ][0],results['3l'  ][3])  )
frame.GetYaxis().SetBinLabel(6, "#splitline{four-lepton}{#mu < %.1f (%.1f exp)}" % (results['4l'  ][0],results['4l'  ][3])  )
frame.GetYaxis().SetLabelSize(0.070);
frame.GetXaxis().SetNoExponent(1);
frame.GetXaxis().SetMoreLogLabels(1);
frame.Draw()
c1.SetLogx(1)


obs = ROOT.TGraphAsymmErrors(6)
exp = ROOT.TGraphAsymmErrors(6)
b68 = ROOT.TGraphAsymmErrors(6)
b95 = ROOT.TGraphAsymmErrors(6)
for i,l in enumerate(["comb","mumu","ee","em","3l","4l"]):
    obs.SetPoint(i, results[l][0], 0.5+i)
    exp.SetPoint(i, results[l][3], 0.5+i)
    b68.SetPoint(i, results[l][3], 0.5+i)
    b95.SetPoint(i, results[l][3], 0.5+i)
    b68.SetPointError(i, results[l][3]-results[l][2], results[l][4]-results[l][3], 0.5, 0.5)
    b95.SetPointError(i, results[l][3]-results[l][1], results[l][5]-results[l][3], 0.5, 0.5)
    exp.SetPointError(i, 0,0, 0.5, 0.5)
    obs.SetPointError(i, 0,0, 0.5, 0.5)

b68.SetFillColor(80);
b95.SetFillColor(90);
for X in exp, b68, b95:
    X.SetMarkerStyle(25);
    X.SetLineWidth(3);
    X.SetLineStyle(2);
obs.SetMarkerStyle(21);
obs.SetLineWidth(3);
obs.SetLineStyle(1);
b95.Draw("E2 SAME")
b68.Draw("E2 SAME")
exp.Draw("PZ SAME")
obs.Draw("PZ SAME")

#stopLine = ROOT.TLine(results['4l'][0]+results['4l'][1]-0.02,  0.25, results['4l'][0]+results['4l'][1]-0.02, 0.75);
#stopLine.SetLineStyle(2);
#stopLine.SetLineWidth(2);
#stopLine.Draw()

#combtext = "combined #mu = %.2f_{#scale[1.4]{-}%.2f}^{+%.2f}" % (results['comb'][0],-results['comb'][1],results['comb'][2])
frame.Draw("AXIS SAME")
doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=0.0355)
doSpam("CMS Preliminary",                    .38, .875, .690, .927, align=21, textSize=0.045)
doSpam("m_{H} = 125.7 GeV",                  .38, .795, .690, .867, align=21, textSize=0.045)
#doSpam( combtext,                            .32, .765, .950, .847, align=21, textSize=0.045)
leg = doLegend(.710,.795,.950,.937, textSize=0.0355)
leg.AddEntry(obs, "Observed", "LP")
leg.AddEntry(b68, "Exp. (68%)", "LPF")
leg.AddEntry(b95, "Exp. (95%)", "LPF")
leg.Draw()

c1.Print("clslimitSplit%s.png" % postfix)
c1.Print("clslimitSplit%s.pdf" % postfix)
