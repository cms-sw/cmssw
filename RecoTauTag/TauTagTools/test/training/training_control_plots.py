#!/usr/bin/env python

from __future__ import print_function
import sys
import re
import os
import six

input_file = sys.argv[1]
output_dir = sys.argv[2]

sys.argv[:] = []

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetPalette(1)


def get_by_type(directory, type):
    for key in directory.GetListOfKeys():
        object = key.ReadObj()
        if isinstance(object, type):
            yield object

def gini_index(signal, background):
    signal_integral = signal.GetIntegral()
    background_integral = background.GetIntegral()
    total = signal.Integral() + background.Integral()
    linear = ROOT.TGraph(signal.GetNbinsX()+1)
    signal_fraction = ROOT.TGraph(signal.GetNbinsX()+1)
    for ibin in range(0, signal.GetNbinsX()+1):
        total_fraction_of_sample = (
        signal_integral[ibin] + background_integral[ibin])/2.0
        linear.SetPoint(
            ibin, total_fraction_of_sample, total_fraction_of_sample)
        total_fraction_of_signal = signal_integral[ibin]
        signal_fraction.SetPoint(
            ibin, total_fraction_of_sample, total_fraction_of_signal)
    return 0.5-signal_fraction.Integral()

colors = {
    'Signal' : ROOT.EColor.kRed,
    'Background' : ROOT.EColor.kBlue,
}


if __name__ == "__main__":
    # Exit gracefully
    if not os.path.exists(input_file):
        print("WARNING: no training control plot .root file found!")
        sys.exit(0)
    file = ROOT.TFile(input_file)

    correlation_canvas = ROOT.TCanvas("corr", "corr", 2000, 1000)
    correlation_canvas.Divide(2)
    signal_correlation = file.Get("CorrelationMatrixS")
    background_correlation = file.Get("CorrelationMatrixB")
    background_correlation.SetMarkerColor(ROOT.EColor.kBlack)
    for index, plot in enumerate([signal_correlation, background_correlation]):
        correlation_canvas.cd(index+1)
        plot.SetMarkerColor(ROOT.EColor.kBlack)
        plot.Draw("col")
        plot.Draw("text, same")
        plot.GetXaxis().SetLabelSize(0.03)
        plot.GetYaxis().SetLabelSize(0.03)
        ROOT.gPad.SetMargin(0.2, 0.1, 0.2, 0.1)

    correlation_canvas.SaveAs(os.path.join(
        output_dir, "correlations.png"))

    method_canvas = ROOT.TCanvas("method", "method", 800, 800)
    method_canvas.cd()
    # Find the MVA result directory
    for method_dir in [dir for dir in get_by_type(file, ROOT.TDirectory)
                       if 'Method_' in dir.GetName()]:
        method_canvas.SetLogy(False)
        # Strip prefix
        method_type = method_dir.GetName().replace('Method_', '')
        print(method_type)
        result_dir = method_dir.Get(method_type)
        signal_mva_out = result_dir.Get("MVA_%s_S" % method_type)
        background_mva_out = result_dir.Get("MVA_%s_B" % method_type)
        signal_mva_out.SetLineColor(colors['Signal'])
        background_mva_out.SetLineColor(colors['Background'])
        stack = ROOT.THStack("stack", "MVA Output")
        stack.Add(signal_mva_out, "HIST")
        stack.Add(background_mva_out, "HIST")
        stack.Draw("nostack")
        method_canvas.SaveAs(os.path.join(
            output_dir, "%s_mva_output.png" % method_type))

        perf_curve = result_dir.Get("MVA_%s_effBvsS" % method_type)
        perf_curve.Draw()
        perf_curve.SetMinimum(1e-4)
        method_canvas.SetLogy(True)
        method_canvas.SaveAs(os.path.join(output_dir, "%s_performance.png"
                                         % method_type))

    input_var_dir = file.Get("InputVariables_NoTransform")
    if not input_var_dir:
        input_var_dir = file.Get("InputVariables_Id")

    matcher = re.compile("(?P<name>[^_]*)__(?P<type>[A-Za-z0-9]*)_Id")

    input_distributions = {}
    for histo in get_by_type(input_var_dir, ROOT.TH1F):
        rawname = histo.GetName()
        match = matcher.match(rawname)
        name = match.group('name')
        type = match.group('type')
        histo.Scale(1.0/histo.Integral())
        histo.SetLineColor(colors[type])
        histo_info = input_distributions.setdefault(name, {})
        histo_info[type] = histo

    variable_canvas = ROOT.TCanvas("var", "var", 1000, 1000)
    for variable, histograms in six.iteritems(input_distributions):
        maximum = max(histograms[type].GetMaximum()
                      for type in ['Signal', 'Background'])
        for type in ['Signal', 'Background']:
            histograms[type].SetLineWidth(2)
        # Tgraph integral not in ROOT 5.27?
        gini = gini_index(histograms['Signal'], histograms['Background'])
        histograms['Signal'].SetMaximum(1.2*maximum)
        histograms['Signal'].SetTitle(variable + " gini: %0.2f" % gini)
        histograms['Signal'].Draw()
        histograms['Background'].Draw('same')
        variable_canvas.SaveAs(os.path.join(
            output_dir, variable + ".png"))
