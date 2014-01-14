import math,ROOT
from ROOT import gStyle


ROOT.gROOT.SetStyle('Plain')
gStyle.SetOptStat(1100)
gStyle.SetOptTitle(0)
gStyle.SetPadTickY(1)
gStyle.SetPadTickX(1)
f1 = ROOT.TFile('NPVtx_new.root')
f2 = ROOT.TFile('puoutput825_25bins.root')
h1 = f1.Get('NPV')
h2 = f2.Get('pileup')
#h1.SetMarkerStyle(20)
#h1.SetMarkerSize(0.6)
#h1.SetMarkerColor(ROOT.kRed)
h1.SetLineColor(ROOT.kRed)
h1.SetFillColor(ROOT.kRed)
h1.SetFillStyle(3002)
h1.SetStats(1)
#h2.SetMarkerStyle(20)
#h2.SetMarkerSize(0.6)
#h2.SetMarkerColor(ROOT.kBlue)
h2.SetLineColor(ROOT.kBlue)
h2.SetFillColor(ROOT.kBlue)
h2.SetFillStyle(3012)
h2.SetStats(1)

h1.Sumw2()
h2.Sumw2()

h1.Scale(1/h1.Integral())
h2.Scale(1/h2.Integral())

canvas = ROOT.TCanvas()
h1.GetXaxis().SetTitle('Number of Primary Vertices')
#h1.GetXaxis().SetRangeUser(400.,800.)
#h1.GetYaxis().SetRangeUser(0.,1.5)
h1.Draw('hist e')
h2.Draw("sames hist e")

#dij = h1.Integral(300,800,0,40,"")
#sub = h2.Integral(300,800,0,40,"")

#print "dij = ", dij, "  sub = ", sub 

#h1.SetFillColor(ROOT.kRed)
#h2.SetFillColor(ROOT.kBlue)
legend = ROOT.TLegend(.15 ,.7,.25,.85)
legend.SetFillColor(ROOT.kWhite)
legend.SetLineColor(ROOT.kWhite)
legend.SetTextSize(0.04)
legend.AddEntry(h1, "MC", "f")
legend.AddEntry(h2, "Data", "f")
legend.Draw("same")


latex = ROOT.TLatex()
latex.SetNDC()
latex.SetTextSize(0.05)
#latex.DrawLatex(0.12, 0.92, 'Distance between two leading jets vs. Higgs P_{T}')
#latex.DrawLatex(0.10, 0.94, '#color[' + str(2) + ']{' + label + '}')
canvas.Update()

#canvas.SaveAs('/uscms_data/d3/ingabu/HbbSubjet/CMSSW_6_0_0_pre11/src/BDT/Plots/tmp/FJHmassvsgenHpt/GenFJmatched/'+ bin + '_dR12.gif')

raw_input('...')
