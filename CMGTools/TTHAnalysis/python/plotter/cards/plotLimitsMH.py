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
c1.SetLeftMargin(0.17)


postfix = sys.argv[2] if len(sys.argv) > 2 else "";
f = ROOT.TFile.Open(sys.argv[1])
t = ROOT.gFile.Get("limit")
lines = {}
for q in -1, 0.025, 0.975, 0.16, 0.84, 0.5:
    t.Draw("limit:mh","abs(quantileExpected-%g) < 0.01" % q);
    lines[q] = ROOT.gROOT.FindObject("Graph").Clone()
    

frame = ROOT.TH1F("frame","frame", 100, 110, 140); #, 100, 1.0, 30);
frame.GetYaxis().SetTitle("95% CL upper limit on #mu = #sigma/#sigma_{SM}")
frame.GetXaxis().SetTitle("m_{H} (GeV)")
#frame.GetYaxis().SetLabelSize(0.070);
frame.GetYaxis().SetNoExponent(1);
frame.GetYaxis().SetRangeUser(0., 14.0);
frame.GetYaxis().SetMoreLogLabels(1);
frame.GetYaxis().SetTitleOffset(1.1);
frame.GetXaxis().SetLabelOffset(0.016);
frame.GetXaxis().SetTitleOffset(1.0);
frame.Draw()
#c1.SetLogy(1)

obs = lines[-1];
exp = ROOT.TGraphAsymmErrors(obs.GetN())
b68 = ROOT.TGraphAsymmErrors(obs.GetN())
b95 = ROOT.TGraphAsymmErrors(obs.GetN())
for i in xrange(obs.GetN()):
    exp.SetPoint(i, lines[0.5].GetX()[i], lines[0.5].GetY()[i])
    b68.SetPoint(i, lines[0.5].GetX()[i], lines[0.5].GetY()[i])
    b95.SetPoint(i, lines[0.5].GetX()[i], lines[0.5].GetY()[i])
    b68.SetPointError(i, 0, 0, -lines[0.16].GetY()[i]+lines[0.5].GetY()[i], lines[0.84].GetY()[i]-lines[0.5].GetY()[i] )
    b95.SetPointError(i, 0, 0, -lines[0.025].GetY()[i]+lines[0.5].GetY()[i], lines[0.975].GetY()[i]-lines[0.5].GetY()[i] )
    exp.SetPointError(i, 0,0, 0., 0.)
    #obs.SetPointError(i, 0,0, 0., 0.)

b68.SetFillColor(80);
b95.SetFillColor(90);
for X in exp, b68, b95:
    X.SetMarkerStyle(25);
    X.SetLineWidth(3);
    X.SetLineStyle(2);
obs.SetMarkerStyle(21);
obs.SetLineWidth(3);
obs.SetLineStyle(1);
b95.Draw("E3 SAME")
b68.Draw("E3 SAME")
exp.Draw("LPX SAME")
obs.Draw("LPX SAME")

line = ROOT.TLine(110,1,140,1);
line.SetLineColor(2);
line.SetLineWidth(2);
line.Draw()

#combtext = "combined #mu = %.2f_{#scale[1.4]{-}%.2f}^{+%.2f}" % (results['comb'][0],-results['comb'][1],results['comb'][2])
frame.Draw("AXIS SAME")
doSpam("#sqrt{s} = 8 TeV,  L = 19.6 fb^{-1}",.48, .955, .975, .995, align=32, textSize=0.0355)
doSpam("CMS Preliminary",                    .23, .875, .550, .927, align=21, textSize=0.045)
doSpam("m_{H} = 125.7 GeV",                  .23, .795, .550, .867, align=21, textSize=0.045)
#doSpam( combtext,                            .32, .765, .950, .847, align=21, textSize=0.045)
leg = doLegend(.670,.775,.945,.937, textSize=0.045)
leg.AddEntry(obs, "Observed", "LP")
leg.AddEntry(b68, "Exp. (68%)", "LPF")
leg.AddEntry(b95, "Exp. (95%)", "LPF")
leg.Draw()

c1.Print("clslimit_MH%s.png" % postfix)
c1.Print("clslimit_MH%s.pdf" % postfix)
