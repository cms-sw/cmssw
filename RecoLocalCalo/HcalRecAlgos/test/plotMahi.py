#!/usr/bin/env python

# Plot MAHI multi-pulse fit results
import ROOT
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas('c','c',500,500)
c.cd()
c.SetRightMargin(0.05)
c.SetLeftMargin(0.12)
c.SetTopMargin(0.1)
c.SetTopMargin(0.05)

color = [ROOT.kAzure+3, ROOT.kAzure+2, ROOT.kAzure+1, ROOT.kYellow-7, ROOT.kRed+1, ROOT.kRed+2, ROOT.kRed+3, ROOT.kRed+4]

import os
plotdir = 'plotMahi'
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

# Parameters for Run 3 HBHE
nTS = 10 # number of TS to plot
nOOT = 7 # number of OOT samples
offset = 3 # soi

hists = {}
hists['digi'] = ROOT.TH1D('digi', ';TS;E [GeV]', nTS, -0.5, nTS-0.5)
hists['digi'].SetMarkerStyle(20)
for i in range(nOOT+1):
    hists[i] = ROOT.TH1D('bx%i' % (i-3), ';TS;E [GeV]', nTS, -0.5, nTS-0.5)
    hists[i].SetFillColor(color[i])
    hists[i].SetLineColor(ROOT.kBlack)

tags = ['']#, '_shape206', '_timeslew', '_mt6', '_qcd', '_qcdnopu']

from tqdm import tqdm

for tag in tags:
    tree = ROOT.TChain('mahiDebugger/HcalTree')
    tree.Add('mahidebugger%s.root' % tag)
    count = 0
    for rh in tqdm(tree):
        if abs(rh.ieta) > 16:
            continue
        soiEnergy = rh.mahiEnergy*rh.inGain
        if soiEnergy < 1:
            continue
        
        energy = {}
        soi = rh.soi
        assert soi == offset
        
        for i in range(nTS-1):
            hists['digi'].SetBinContent(i+1, rh.inputTS[i]*rh.inGain)
            hists[soi].SetBinContent(i+1, rh.itPulse[i]*rh.mahiEnergy*rh.inGain)
            energy[soi] = soiEnergy
        
        for o in range(nOOT):
            oh = o+1 if o>=soi else o
            ootPulse = []
            for i in range(nTS*o, nTS*(o+1)):
                ootPulse.append(rh.ootPulse[i])
            
            for i in range(nTS):
                hists[oh].SetBinContent(i+1, max(0, ootPulse[i]*rh.ootEnergy[o]*rh.inGain))
                energy[oh] = rh.ootEnergy[o]*rh.inGain
        
        stack = ROOT.THStack('stack', '')
        for i in range(nOOT+1):
            stack.Add(hists[i])
        
        hists['digi'].GetXaxis().SetRangeUser(-0.5, 7.5)
        hists['digi'].GetYaxis().SetRangeUser(0, max(hists['digi'].GetBinContent(4), hists['digi'].GetBinContent(5))*1.5)
        hists['digi'].Draw('P0')
        stack.Draw('same')
        hists['digi'].Draw('P0,same')
        
        legend = ROOT.TLegend(0.6,0.5,0.93,0.93)
        legend.SetLineWidth(0)
        legend.SetFillStyle(0)
        legend.AddEntry(hists['digi'], 'Digi', 'P')
        for i in range(nOOT+1):
            legend.AddEntry(hists[i], 'BX %+i, E = %.1f GeV' % (i-offset, energy[i]), 'F')
        
        tex = ROOT.TLatex()
        tex.SetTextSize(0.025)
        tex.DrawLatexNDC(0.15, 0.90, 'run:'+str(rh.run)+' evt:'+str(rh.evt))
        tex.DrawLatexNDC(0.15, 0.85, 'ieta:'+str(rh.ieta)+' iphi:'+str(rh.iphi)+' depth:'+str(rh.depth))
        tex.DrawLatexNDC(0.15, 0.80, '#chi^{2} = %.1f, TDC=%i, mahiTime=%.1f' % (rh.chiSq, rh.inputTDC[3], rh.arrivalTime))
        
        wTime = 0.
        sumEn = 0.
        for i in range(len(energy)):
            thisEnergy = rh.inputTS[i]*rh.inGain
            wTime += thisEnergy * i
            sumEn += thisEnergy
        if sum(energy) > 0:
            wTime /= sumEn
        tex.DrawLatexNDC(0.15, 0.75, 'Mean time = %.1f TS' % (wTime))
        
        legend.Draw()
        
        c.Print(plotdir+'/'+str(rh.run)+'_'+str(rh.evt)+'_'+str(rh.ieta)+'_'+str(rh.iphi)+'_'+str(rh.depth)+tag+'.pdf')
        c.Print(plotdir+'/'+str(rh.run)+'_'+str(rh.evt)+'_'+str(rh.ieta)+'_'+str(rh.iphi)+'_'+str(rh.depth)+tag+'.png')
        count += 1
