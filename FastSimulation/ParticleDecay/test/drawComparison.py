#!/usr/bin/env python

import sys,os
sys.argv.append("-b")
import ROOT as rt

ifile = sys.argv[1]
odir = sys.argv[2]

os.system("mkdir -p " + odir)
tfile = rt.TFile(ifile)
observed = tfile.Get("observed")
predicted = tfile.Get("prediction")

for key in observed.GetListOfKeys():
    name = key.GetName()
    h_observed = observed.Get(name)
    h_predicted = predicted.Get(name)
    _max = max(1,h_observed.GetMaximum())
    nbins = h_observed.GetNbinsX()
    h_observed.SetMarkerStyle(rt.kCircle)
    h_observed.SetMarkerColor(rt.kBlue)
    h_observed.SetLineColor(rt.kBlue)
    l = rt.TLegend(0.15,0.75,0.5,0.89)
    l.SetFillStyle(0)
    l.SetLineWidth(1)
    l.AddEntry(h_observed,"observed","l,p")
    if not h_predicted == None:
        _max = max(_max,h_predicted.GetMaximum())*1.3
        h_predicted.Scale(h_observed.Integral(0,nbins+1)/h_predicted.Integral(0,nbins+1))
        h_predicted.SetLineColor(rt.kRed)
        l.AddEntry(h_predicted,"predicted","l")
    h_observed.SetMaximum(_max);
    _max = _max*1.3
    h_observed.Draw("E")
    if not h_predicted == None:
        h_predicted.Draw("same")
    l.Draw()
    rt.gPad.Print(odir + "/" + name + ".png")
    

