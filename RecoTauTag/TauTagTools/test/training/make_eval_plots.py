#!/usr/bin/env python
'''

Produce a plot comparing the discriminator performance for a number of different
tau ID discriminator.

Author: Evan K. Friis

'''
import math

import sys
output_file = sys.argv[1]
signal_input = sys.argv[2]
background_input = sys.argv[3]

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(0)

def make_perf_curve2(signal, background, signal_denom, background_denom):
    signal_integral = signal.GetIntegral()
    print signal_integral
    background_integral = background.GetIntegral()
    signal_total = signal.Integral()
    background_total = background.Integral()
    points = set([])
    for bin in range(1, signal.GetNbinsX()):
        signal_passing = int(signal_total*(1-signal_integral[bin]))
        background_passing = int(background_total*(1-background_integral[bin]))
        points.add((signal_passing, background_passing))
    points_sorted = sorted(points)
    # Take all but first and last points
    points_to_plot = points_sorted[1:-1]
    output = ROOT.TGraph(len(points_to_plot))
    for index, (sig, bkg) in enumerate(points_to_plot):
        output.SetPoint(index, sig*1.0/signal_denom, bkg*1.0/background_denom)
    return output

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
    'hpsPFTauDiscriminationByDecayModeFinding',
    'hpsPFTauDiscriminationByLooseIsolation',
    'hpsPFTauDiscriminationByMediumIsolation',
    'hpsPFTauDiscriminationByTightIsolation',
]

discriminators['shrinkingConePFTauProducer'] = [
    #'shrinkingConePFTauDiscriminationByLeadingPionPtCut',
    #'shrinkingConePFTauDiscriminationByIsolation',
    #'shrinkingConePFTauDiscriminationByTrackIsolation',
    #'shrinkingConePFTauDiscriminationByECALIsolation',
    #'shrinkingConePFTauDiscriminationByTaNC',
    #'shrinkingConePFTauDiscriminationByTaNCfrOnePercent',
    'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent',
    #'shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent',
    #'shrinkingConePFTauDiscriminationByTaNCfrTenthPercent'
]

discriminators['hpsTancTaus'] = [
    #'hpsTancTausDiscriminationByTancRaw',
    #'hpsTancTausDiscriminationByTanc',
    'hpsTancTausDiscriminationByTancLoose',
    'hpsTancTausDiscriminationByTancMedium',
    'hpsTancTausDiscriminationByTancTight',
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

good_colors = [ROOT.EColor.kRed-9, ROOT.EColor.kBlue-9, ROOT.EColor.kBlack,
               ROOT.EColor.kGreen-9, ROOT.EColor.kViolet, ROOT.EColor.kAzure]
disc_to_plot = ['shrinkingConePFTauProducer', 'hpsPFTauProducer', 'hpsTancTaus']

pt_curves_to_plot = [
    #('shrinkingConePFTauProducer',
     #'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent'),
    ('hpsPFTauProducer',
     'hpsPFTauDiscriminationByLooseIsolation'),
    ('hpsPFTauProducer',
     'hpsPFTauDiscriminationByMediumIsolation'),
    ('hpsPFTauProducer',
     'hpsPFTauDiscriminationByTightIsolation'),
    ('hpsTancTaus',
    'hpsTancTausDiscriminationByTancLoose'),
    ('hpsTancTaus',
    'hpsTancTausDiscriminationByTancMedium'),
    ('hpsTancTaus',
    'hpsTancTausDiscriminationByTancTight'),
]

_PT_CUT = 15
_DENOM_CUT = {
    'signal' : 15,
    'background' : 20
}
_DENOM_PLOT_TYPE = {
    'signal' : '_pt_embedPt', # compare to generator level pt ('embedded')
    'background' : '_pt_jetPt', # compare to reconstructed total jet pt
}

def binomial_eff_and_error(passed, total):
    if total == 0:
        return (0,0)
    eff = passed*1.0/total
    error = math.sqrt((passed*1.0/(total*total))*(1-eff))
    return (eff, error)

canvas = ROOT.TCanvas("blah", "blah", 800, 1200)
if __name__ == "__main__":
    steering = {}
    steering['signal'] = { 'file' : ROOT.TFile(signal_input, 'READ') }
    steering['background'] = { 'file' : ROOT.TFile(background_input, 'READ') }

    print "Loading histograms"
    # Load all the histograms
    for sample in ['signal', 'background']:
        sample_info = steering[sample]
        sample_info['algos'] = {}
        for producer in discriminators.keys():
            sample_info['algos'][producer] = {}
            for discriminator in discriminators[producer]:
                print "Getting %s-%s" % (producer, discriminator)
                producer_folder = sample_info['file'].Get("plot"+producer)
                to_get = discriminator + _DENOM_PLOT_TYPE[sample]
                raw_histo = producer_folder.Get(to_get)

                min_denom_bin = raw_histo.GetZaxis().FindBin(_DENOM_CUT[sample])

                # Get the minimum bin corresponding to the reco pt cut
                min_pt_bin = raw_histo.GetYaxis().FindBin(_PT_CUT)

                raw_histo.GetYaxis().SetRange(
                    min_pt_bin, raw_histo.GetNbinsY()+1)
                raw_histo.GetZaxis().SetRange(
                    min_denom_bin, raw_histo.GetNbinsZ()+1)

                projection = raw_histo.Project3D("x")

                denom = raw_histo.Integral(
                    0, raw_histo.GetNbinsX()+1,
                    0, raw_histo.GetNbinsY()+1,
                    min_denom_bin, raw_histo.GetNbinsZ()+1)

                # Now make projection for PT efficiency
                raw_histo.GetXaxis().SetRange(0, raw_histo.GetNbinsX()+1)
                raw_histo.GetYaxis().SetRange(0, raw_histo.GetNbinsY()+1)
                pt_eff_denom = raw_histo.Project3D("denom_z")

                discriminator_cut = 0.975

                raw_histo.GetXaxis().SetRange(
                    raw_histo.GetXaxis().FindBin(discriminator_cut),
                    raw_histo.GetNbinsX()+1)
                raw_histo.GetYaxis().SetRange(
                    min_pt_bin, raw_histo.GetNbinsY()+1)
                pt_eff_numerator = raw_histo.Project3D("numerator_z")

                # Get the vertex information
                vs_truePU = producer_folder.Get(
                    discriminator + "_truePU")
                vs_recoPU = producer_folder.Get(
                    discriminator + "_recoPU")

                truePU_graph = ROOT.TGraphErrors(vs_truePU.GetNbinsY())
                recoPU_graph = ROOT.TGraphErrors(vs_recoPU.GetNbinsY())

                # Compute the marginal efficiency in the numerator as a function
                # of PU.
                for histo, graph in [(vs_truePU, truePU_graph),
                                     (vs_recoPU, recoPU_graph)]:
                    for nPUVtx in range(0, histo.GetNbinsY()):
                        # TH2F bin index is offset by 1
                        ybin = nPUVtx + 1
                        vtx_projection = histo.ProjectionX(
                            "temp", ybin, ybin)
                        total_entries = vtx_projection.Integral()
                        passing_entries = vtx_projection.Integral(
                            vtx_projection.FindBin(discriminator_cut),
                            vtx_projection.GetNbinsX()+1)
                        efficiency = (
                            total_entries and passing_entries/total_entries
                            or 0.)
                        efficiency, error = binomial_eff_and_error(
                            passing_entries, total_entries)
                        graph.SetPoint(nPUVtx, nPUVtx, efficiency)
                        graph.SetPointError(nPUVtx, 0, error)
                        graph.Fit("pol1")

                #print "Mean X Proj:", projection.GetMean(1)
                #print "Post cut entries:", projection.Integral()
                sample_info['algos'][producer][discriminator] = {
                    'disc' : projection,
                    'denominator' : denom,
                    'pt_eff_num' : pt_eff_numerator,
                    'pt_eff_denom' : pt_eff_denom,
                    'truePU' : truePU_graph,
                    'recoPU' : recoPU_graph
                }
                #sample_info['algos'][producer][discriminator + '_denominator']\
                #        = projection.Integral()
                #projection.Draw()
                #canvas.SaveAs("hpstanc/eval/" +sample + "_"+ discriminator + ".png")

    # Build the master canvas and the sub pads
    #canvas = ROOT.TCanvas("blah", "blah", 800, 1200)
    # Workaround so it isn't rotated..
    canvas.Divide(1, 2)
    canvas.cd(1)
    #legend_pad = ROOT.TPad("legendpad", "legendpad", 0.7, 0.1, 0.9, 0.9)
    #graph_pad = ROOT.TPad("graphpad", "graphpad", 0.1, 0.1, 0.7, 0.9)
    #graph_pad.cd()

    good_markers = [20, 21, 24, 26, 22, 26, 20, 21, 24, 26, 22 ]
    #good_colors = [ROOT.EColor.kRed, ROOT.EColor.kBlue, ROOT.EColor.kGreen + 2]
    # The background histogram
    histo = ROOT.TH1F("blank", "blank", 200, 0, 1.0)
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
        for marker, discriminator in zip(
            good_markers, discriminators[producer]):
            print "Building perf curve graph:", producer, discriminator
            new_graph = make_perf_curve2(
                steering['signal']['algos'][producer][discriminator]['disc'],
                steering['background']['algos'][producer][discriminator]['disc'],
                steering['signal']['algos'][producer][discriminator]['denominator'],
                steering['background']['algos'][producer][discriminator]['denominator'],
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

    pt_legend = ROOT.TLegend(0.15, 0.65, 0.85, 0.85)
    pt_legend.SetFillStyle(0)
    pt_legend.SetBorderSize(0)
    pt_canvas = ROOT.TCanvas('pt', 'pt', 1000, 500)
    pt_canvas.Divide(2)
    signal_frame = ROOT.TH1F("sigbkg", "Signal efficiency", 10, 0, 200)
    signal_frame.SetMaximum(1)
    signal_frame.GetXaxis().SetTitle("True tau p_{T}")
    background_frame = ROOT.TH1F("bkgbkg", "Fake Rate", 10, 0, 200)
    background_frame.SetMaximum(1e-1)
    background_frame.GetXaxis().SetTitle("Jet p_{T}")
    background_frame.SetMinimum(1e-4)
    pt_canvas.cd(1)
    signal_frame.Draw()
    pt_canvas.cd(2)
    ROOT.gPad.SetLogy(True)
    background_frame.Draw()
    keep = []
    #pt_canvas.SaveAs(output_file.replace('.pdf', '_pt_eff.pdf'))

    for color, (producer, discriminator) in zip(good_colors, pt_curves_to_plot):
        signal_algos = steering['signal']['algos'][producer]
        background_algos = steering['background']['algos'][producer]
        signal_num = signal_algos[discriminator]['pt_eff_num']
        signal_denom = signal_algos[discriminator]['pt_eff_denom']
        signal_num.Rebin(5)
        signal_denom.Rebin(5)
        pt_canvas.cd(1)
        signal_eff = ROOT.TGraphAsymmErrors(signal_num, signal_denom)
        signal_eff.Draw("p")
        signal_eff.SetMarkerColor(color)
        signal_eff.SetMarkerStyle(20)
        pt_legend.AddEntry(
            signal_eff,
            "%s - %s" % (
                producer_translator[producer],
                discriminator_translator[discriminator],
            ), "p")
        keep.append(signal_eff)
        background_num = background_algos[discriminator]['pt_eff_num']
        background_denom = background_algos[discriminator]['pt_eff_denom']
        background_num.Rebin(20)
        background_denom.Rebin(20)
        background_eff = ROOT.TGraphAsymmErrors(background_num, background_denom)
        background_eff.SetMarkerColor(color)
        background_eff.SetMarkerStyle(20)
        pt_canvas.cd(2)
        background_eff.Draw("p")
        keep.append(background_eff)

    pt_legend.Draw()
    pt_canvas.SaveAs(output_file.replace('.pdf', '_pt_eff.pdf'))

    print "Making PU plots"
    vtx_signal = ROOT.TH1F("sig_vtx", "Efficiency vs. PU", 16, -0.5, 15.5)
    vtx_signal.SetMaximum(1.0)
    vtx_signal.SetMinimum(0)
    vtx_signal.GetXaxis().SetTitle("N_{PU}")
    vtx_background = ROOT.TH1F("bkg_vtx", "Fake rate vs. PU", 16, -0.5, 15.5)
    vtx_background.GetXaxis().SetTitle("N_{PU}")
    vtx_background.SetMaximum(1)
    vtx_background.SetMinimum(1e-3)

    # Make vtx plots
    for puType in ['truePU', 'recoPU']:
        pt_canvas.cd(1)
        vtx_signal.Draw()
        pt_canvas.cd(2)
        vtx_background.Draw()
        for color, (producer, discriminator) in zip(good_colors, pt_curves_to_plot):
            print "Making PU plots for", producer, discriminator
            signal_algos = steering['signal']['algos'][producer]
            background_algos = steering['background']['algos'][producer]
            signal_graph = signal_algos[discriminator][puType]
            background_graph = background_algos[discriminator][puType]

            signal_graph.SetMarkerStyle(20)
            signal_graph.SetMarkerColor(color)
            signal_graph.SetLineColor(color)
            signal_graph.SetLineWidth(2)

            pt_canvas.cd(1)
            signal_graph.Draw("pe")
            signal_graph.GetFunction("pol1").SetLineStyle(2)
            signal_graph.GetFunction("pol1").SetLineWidth(1)
            signal_graph.GetFunction("pol1").SetLineColor(color)
            keep.append(signal_graph)

            background_graph.SetMarkerStyle(20)
            background_graph.SetMarkerColor(color)
            background_graph.SetLineColor(color)
            background_graph.SetLineWidth(2)
            background_graph.GetFunction("pol1").SetLineStyle(2)
            background_graph.GetFunction("pol1").SetLineWidth(1)
            background_graph.GetFunction("pol1").SetLineColor(color)

            pt_canvas.cd(2)
            background_graph.Draw("ep")
            keep.append(background_graph)

        pt_legend.Draw()
        pt_canvas.SaveAs(output_file.replace('.pdf', '_%s.pdf' % puType))


