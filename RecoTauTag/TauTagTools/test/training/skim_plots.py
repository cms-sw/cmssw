#!/usr/bin/env python

'''

Plot information about the Tau MVA training skim process.

Author: Evan K. Friis, UC Davis

'''

import ROOT
import sys
import os

ROOT.gROOT.SetStyle("Plain")

if not os.path.exists("plots"):
    os.mkdir("plots")

config = {}

config['sig'] = {}
config['bkg'] = {}

sig = config['sig']
bkg = config['bkg']

# Load our files
sig['filename'] =  sys.argv[1]
bkg['filename'] =  sys.argv[2]
for sample in config.keys():
    config[sample]['file'] = ROOT.TFile(config[sample]['filename'], 'READ')

# Figure out where our plots are stored
for sample, name in [ ('sig', 'Signal'), ('bkg', 'Background') ]:
    config[sample]['denominator_folder'] = 'plot%sJets' % name
    config[sample]['leadobject_folder'] = 'plot%sJetsLeadObject' % name
    config[sample]['final_folder'] = 'plotPreselected%sJets' % name

#kinematic_variables = ['pt', 'eta', 'phi']
kinematic_variables = ['pt', 'eta', ]
cut_variables = ['leadobject', 'collimation']
all_variables = []
all_variables.extend(kinematic_variables)
all_variables.extend(cut_variables)

canvas = ROOT.TCanvas("c1", "canvas", 800, 600)
# Open PS file
ps = ROOT.TPostScript("plots/skim_plots.ps", 112)
canvas.Divide(2,2)

def apply_style_signal(histo):
    histo.SetLineColor(ROOT.EColor.kRed)
    histo.SetLineWidth(2)
    histo.SetFillStyle(3004)
    histo.SetFillColor(ROOT.EColor.kRed - 5)

def apply_style_background(histo):
    histo.SetLineColor(ROOT.EColor.kBlue)
    histo.SetLineWidth(2)
    histo.SetFillStyle(3005)
    histo.SetFillColor(ROOT.EColor.kBlue - 5)

keep = []
# Make control plots
for folder in ['denominator_folder', 'leadobject_folder', 'final_folder']:
    ps.NewPage()
    for index, plot in enumerate(all_variables):
        canvas.cd(index + 1)
        stack = ROOT.THStack(folder+plot, "")
        signal_histo = sig['file'].Get(os.path.join(sig[folder], plot)).Clone()
        background_histo = bkg['file'].Get(os.path.join(bkg[folder], plot)).Clone()
        signal_histo.Scale(1.0/signal_histo.Integral())
        background_histo.Scale(1.0/background_histo.Integral())
        apply_style_signal(signal_histo)
        apply_style_background(background_histo)
        stack.Add(signal_histo)
        stack.Add(background_histo)
        stack.SetTitle(signal_histo.GetTitle())
        stack.Draw("nostack")
        keep.append(stack)
    # Append to the ps file
    canvas.Update()
    #canvas.Print('plots/skim_plots.ps')

# Make final efficiency plots
canvas.Clear()
canvas.cd(0)

for numerator_folder in ['leadobject_folder', 'final_folder']:
    for var in kinematic_variables:
        ps.NewPage()
        sig_denom = sig['file'].Get(
            os.path.join(sig['denominator_folder'], var))
        sig_num = sig['file'].Get(
            os.path.join(sig[numerator_folder], var))
        bkg_denom = bkg['file'].Get(
            os.path.join(bkg['denominator_folder'], var))
        bkg_num = bkg['file'].Get(
            os.path.join(bkg[numerator_folder], var))

        sig_eff = ROOT.TGraphAsymmErrors(sig_num, sig_denom)
        bkg_eff = ROOT.TGraphAsymmErrors(bkg_num, bkg_denom)

        sig_eff.Draw("alp")
        sig_eff.GetHistogram().SetMinimum(0.0)
        sig_eff.GetHistogram().SetMaximum(1.1)
        bkg_eff.Draw("same")

        canvas.Update()
canvas.Update()
# Close ps file
ps.Close()

print "Skim statistics:"
print "Final signal jets: ", sig['file'].Get(
    os.path.join(sig['final_folder'], 'pt')).Integral()

print "Final background jets: ", bkg['file'].Get(
    os.path.join(bkg['final_folder'], 'pt')).Integral()
