#!/usr/bin/env python
'''

Produce a plot comparing the discriminator performance for a number of different
tau ID discriminator.

Author: Evan K. Friis


'''

import sys
output_file = sys.argv[1]
signal_input = sys.argv[2]
background_input = sys.argv[3]

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(0)

def make_perf_curve(signal, background, signal_denom, background_denom):
    signal_bins = []
    background_bins = []
    for bin in range(0, signal.GetNbinsX()+1):
        signal_bins.append(signal.GetBinContent(bin))
        background_bins.append(background.GetBinContent(bin))
    signal_integral= []
    background_integral = []
    signal_total = 0
    background_total = 0
    for sig, bkg in zip(signal_bins, background_bins):
        # Collapse bins with no info
        signal_total += sig
        background_total += bkg
        if (sig and bkg) and signal_total and background_total:
            signal_integral.append(signal_total)
            background_integral.append(background_total)


    points = []

    for index, (signal, background) in enumerate(zip(
        signal_integral, background_integral)[0:-1]):
        #print "Signal failing:", signal
        #print "Background failing:", background
        eff = (signal_total-signal)*1./signal_denom
        fr = (background_total-background)*1./background_denom
        if fr > 0.8:
            continue
        points.append((eff, fr))

    output = ROOT.TGraph(len(signal_integral)-1)
    for index, (eff, fr) in enumerate(points):
        output.SetPoint(index, eff, fr)

    return output

# Get the discriminators to plot
discriminators = {}
discriminators['hpsPFTauProducer'] = [
    #'hpsPFTauDiscriminationByDecayModeFinding',
    'hpsPFTauDiscriminationByLooseIsolation',
    'hpsPFTauDiscriminationByMediumIsolation',
    'hpsPFTauDiscriminationByTightIsolation',
]

discriminators['shrinkingConePFTauProducer'] = [
    #'shrinkingConePFTauDiscriminationByLeadingPionPtCut',
    'shrinkingConePFTauDiscriminationByIsolation',
    #'shrinkingConePFTauDiscriminationByTrackIsolation',
    #'shrinkingConePFTauDiscriminationByECALIsolation',
    'shrinkingConePFTauDiscriminationByTaNC',
    'shrinkingConePFTauDiscriminationByTaNCfrOnePercent',
    'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent',
    'shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent',
    #'shrinkingConePFTauDiscriminationByTaNCfrTenthPercent'
]

discriminators['hpsTancTaus'] = [
    #'hpsTancTausDiscriminationByTancRaw',
    'hpsTancTausDiscriminationByTanc',
    #'hpsTancTausDiscriminationByTancLoose',
    #'hpsTancTausDiscriminationByTancMedium',
    #'hpsTancTausDiscriminationByTancTight',
]

#del discriminators['shrinkingConePFTauProducer']
#del discriminators['hpsTancTaus']

producer_translator = {
    'hpsPFTauProducer' : 'HPS',
    'shrinkingConePFTauProducer' : 'Shrinking',
    #'hpsTancTaus' : 'HPStanc',
    'hpsTancTaus' : 'Hybrid Algo.',
}

discriminator_translator = {
    'hpsPFTauDiscriminationByDecayModeFinding' : 'decay finding',
    'hpsPFTauDiscriminationByLooseIsolation' : 'loose',
    'hpsPFTauDiscriminationByMediumIsolation' : 'medium',
    'hpsPFTauDiscriminationByTightIsolation' : 'tight',
    'shrinkingConePFTauDiscriminationByLeadingPionPtCut' : 'lead pion',
    'shrinkingConePFTauDiscriminationByIsolation' : 'comb. isolation',
    'shrinkingConePFTauDiscriminationByTrackIsolation' : 'track isolation',
    'shrinkingConePFTauDiscriminationByECALIsolation' : 'ecal isolation',
    'shrinkingConePFTauDiscriminationByTaNC' : 'TaNC ',
    'shrinkingConePFTauDiscriminationByTaNCfrOnePercent' : 'TaNC 1.00% ',
    'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent' : 'TaNC 0.50% ',
    'shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent' : 'TaNC 0.25% ',
    'shrinkingConePFTauDiscriminationByTaNCfrTenthPercent' : 'TaNC 0.10% ',
    'hpsTancTausDiscriminationByTanc' : 'scan',
    'hpsTancTausDiscriminationByTancLoose' : 'loose',
    'hpsTancTausDiscriminationByTancMedium' : 'medium',
    'hpsTancTausDiscriminationByTancTight' : 'tight',
    'hpsTancTausDiscriminationByTancRaw' : 'no transform',
}

disc_to_plot = discriminators.keys()

good_colors = [ROOT.EColor.kRed-9, ROOT.EColor.kBlue-9, ROOT.EColor.kBlack]
disc_to_plot = ['shrinkingConePFTauProducer', 'hpsPFTauProducer', 'hpsTancTaus']

#disc_to_plot = ['shrinkingConePFTauProducer', 'hpsPFTauProducer', ]
#good_colors = [ROOT.EColor.kRed, ROOT.EColor.kBlue, ROOT.EColor.kBlack]


if __name__ == "__main__":
    steering = {}
    steering['signal'] = { 'file' : ROOT.TFile(signal_input, 'READ') }
    steering['background'] = { 'file' : ROOT.TFile(background_input, 'READ') }

    print "Loading histograms"
    # Load all the histograms
    for sample, sample_info in steering.iteritems():
        sample_info['algos'] = {}
        for producer in discriminators.keys():
            sample_info['algos'][producer] = {}
            for discriminator in discriminators[producer]:
                to_get = "plot" + producer + "/" + discriminator
                print "Getting:", to_get
                sample_info['algos'][producer][discriminator] = \
                        sample_info['file'].Get(to_get)

    print "Loading normalizations"
    for sample, sample_info in steering.iteritems():
        sample_info["denominator"] = \
                sample_info["file"].Get("plotAK5PFJets/pt").GetEntries()

    # Build the master canvas and the sub pads
    canvas = ROOT.TCanvas("blah", "blah", 800, 1200)
    # Workaround so it isn't rotated..
    canvas.Divide(1, 2)
    canvas.cd(1)
    #legend_pad = ROOT.TPad("legendpad", "legendpad", 0.7, 0.1, 0.9, 0.9)
    #graph_pad = ROOT.TPad("graphpad", "graphpad", 0.1, 0.1, 0.7, 0.9)
    #graph_pad.cd()

    good_markers = [20, 21, 24, 26, 22, 26, 20, 21, 24, 26, 22 ]
    #good_colors = [ROOT.EColor.kRed, ROOT.EColor.kBlue, ROOT.EColor.kGreen + 2]
    # The background histogram
    histo = ROOT.TH1F("blank", "blank", 100, 0, 1.0)
    histo.SetMinimum(8e-4)
    histo.SetMaximum(0.5)
    histo.GetXaxis().SetTitle("Signal efficiency")
    histo.GetXaxis().SetRangeUser(0.0, 0.7)
    histo.GetYaxis().SetTitle("Fake rate")
    histo.SetTitle("")
    #histo.SetStat(0)
    histo.Draw()
    #legend = ROOT.TLegend(0.6, 0.2, 0.9, 0.7)
    legend = ROOT.TLegend(0.15, 0.4, 0.55, 0.85)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)

    graphs = {}
    for color, producer in zip(good_colors, disc_to_plot):
        graphs[producer] = {}
        for marker, discriminator in zip(good_markers,
                                         discriminators[producer]):
            print "Building graph:", producer, discriminator
            new_graph = make_perf_curve(
                steering['signal']['algos'][producer][discriminator],
                steering['background']['algos'][producer][discriminator],
                steering['signal']['denominator'],
                steering['background']['denominator'],
            )
            new_graph.SetMarkerStyle(marker)
            new_graph.SetMarkerColor(color)
            new_graph.SetMarkerSize(1.5)
            new_graph.SetLineColor(color)
            new_graph.SetLineStyle(2)
            new_graph.SetLineWidth(3)
            graphs[producer][discriminator] = new_graph
            print new_graph.GetN()
            if new_graph.GetN() > 1:
                new_graph.Draw("l")
                legend.AddEntry(
                    new_graph, "%s - %s" %
                    (producer_translator[producer],
                     discriminator_translator[discriminator]), "l")
            else:
                new_graph.Draw("P")
                legend.AddEntry(
                    new_graph, "%s - %s" %
                    (producer_translator[producer],
                     discriminator_translator[discriminator]), "p")

    ROOT.gPad.SetLogy(True)
    #legend_pad.cd()
    legend.Draw()
    ROOT.gPad.Update()
    ROOT.gPad.SaveAs(output_file)
