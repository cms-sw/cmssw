#!/usr/bin/env python

## Utility used to overlay muon HLT plots from several releases

## For more information about available options, use the -h option:
##     ./compareReleases.py -h

## Import python libraries
import sys
import optparse
import os
import re

## Set up ROOT in batch mode
if '-h' not in sys.argv:
    sys.argv.append('-b')
    import ROOT
    ROOT.gROOT.Macro('rootlogon.C')
    sys.argv.remove('-b')
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(0)
    ROOT.gStyle.SetErrorX(0.5)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    c1 = ROOT.TCanvas("c1","c1",800,600)
    c1.GetFrame().SetBorderSize(6)
    c1.GetFrame().SetBorderMode(-1)



class RootFile:
    def __init__(self, file_name):
        self.name = file_name[0:file_name.find(".root")]
        self.file = ROOT.TFile(file_name, "read")
        if self.file.IsZombie():
            print "Error opening %s, exiting..." % file_name
            sys.exit(1)
    def Get(self, object_name):
        return self.file.Get(object_name)



def main():
    ## Parse command line options
    usage="usage: %prog [options] file1.root file2.root file3.root ..."
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-g', '--gen', action="store_true", default=False,
                      help="skip plots which match to reconstructed muons")
    parser.add_option('-r', '--rec', action="store_true", default=False,
                      help="skip plots which match to generated muons")
    parser.add_option('-s', '--stats', action="store_true", default=False,
                      help="print a table rather than plotting")
    parser.add_option('--stats-format', default="twiki",
                      help="choose 'twiki' (default) or 'html' table format")
    parser.add_option('-p', '--path', default="HLT_Mu9",
                      help="specify which HLT path to focus on")
    parser.add_option('--file-type', default="pdf", metavar="EXT", 
                      help="choose an output format other than pdf")
    options, arguments = parser.parse_args()
    path = options.path
    source_types = ["gen", "rec"]
    if options.gen: source_types.remove("rec")
    if options.rec: source_types.remove("gen")
    if len(arguments) == 0:
        print "Please provide at least one file as an argument"
        return

    ## Define constants
    global file_type, path_muon, path_dist, cross_channel_format, colors
    file_type = options.file_type
    path_muon = "/DQMData/Run 1/HLT/Run summary/Muon"
    path_dist = "%s/Distributions" % path_muon
    cross_channel_format = re.compile("HLT_[^_]*_[^_]*$")
    colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kMagenta+2,
              ROOT.kYellow+2, ROOT.kCyan+2, ROOT.kBlue+3, ROOT.kRed+3]

    ## Make table/plots
    files, path_names = get_files_and_path_names(arguments)
    efficiencies = get_global_efficiencies(files, path)
    filter_names, me_values = get_filters_and_MEs(files, path)
    if options.stats:
        for source_type in source_types:
            make_efficiency_table(files, source_type, filter_names,
                                  efficiencies, options.stats_format)
        sys.exit(0)
    print "Generating plots for %s path" % path
    make_efficiency_summary_plot(files, me_values)
    if len(files) == 1:
        for source_type in source_types:
            for filter in filter_names:
                make_overlaid_turnon_plot(files, source_type, filter, path,
                                          efficiencies, me_values)
    plot_vars = ["TurnOn1", "EffEta", "EffPhi"]
    if "SingleMu" in files[0].name or len(files) == 1:
        plot_vars.remove("TurnOn1")
    for source_type in source_types:
        for plot_var in plot_vars:
            for filter in filter_names:
                plot_name = "%s%s_%s" % (source_type, plot_var, filter)
                make_efficiency_plot(files, plot_name, path,
                                     efficiencies, me_values)
    if "SingleMu" not in files[0].name:
        for source_type in source_types[0:1]:
            for path in path_names:
                if "HighMultiplicity" in path: continue # Kludge
                filter_names, me_values = get_filters_and_MEs(files, path)
                for filter in filter_names:
                    plot_name = "%s%s_%s" % ("gen", "TurnOn1", filter)
                    make_efficiency_plot(files, plot_name, path,
                                         efficiencies, me_values)
    if file_type == "pdf":
        merge_pdf_output(files)



def counter_generator():
    k = 0
    while True:
        k += 1
        yield k
next_counter = counter_generator().next



def get_files_and_path_names(arguments):
    files = [RootFile(filename) for filename in arguments]
    path_names = []
    keys = files[0].file.GetDirectory(path_dist).GetListOfKeys()
    obj = keys.First()
    while obj:
        if obj.IsFolder():
            path_names.append(obj.GetName())
        obj = keys.After(obj)
    return files, path_names



def get_global_efficiencies(files, path):
    efficiencies = []
    for file_index, file in enumerate(files):
        try:
            eff_hist = ROOT.TH1F(file.Get("%s/%s/globalEfficiencies" %
                                               (path_dist, path)))
        except:
            print "No global efficiency in %s" % file.name
            sys.exit(1)
        eff_hist.LabelsDeflate()
        efficiencies.append({})
        for i in range(eff_hist.GetNbinsX()):
            label = eff_hist.GetXaxis().GetBinLabel(i + 1)
            eff = 100. * eff_hist.GetBinContent(i + 1)
            err = 100. * eff_hist.GetBinError(i + 1)
            efficiencies[file_index][label] = [eff, err]
    return efficiencies



def get_filters_and_MEs(files, path):
    filter_names = []
    me_values = {}
    regexp = ROOT.TPRegexp("^<(.*)>(i|f|s)=(.*)</\\1>$")
    full_path = "%s/%s" % (path_dist, path)
    keys = files[0].file.GetDirectory(full_path).GetListOfKeys()
    obj = keys.First()
    while obj:
        name = obj.GetName()
        filter_name = name[name.find("_") + 1:len(name)]
        if "genPassPhi" in name:
            if "L1" in name or "L2" in name or "L3" in name:
                filter_names.append(filter_name)
        if regexp.Match(name):
            array = ROOT.TObjArray(regexp.MatchS(name))
            me_values[array.At(1).GetName()] = float(array.At(3).GetName())
        obj = keys.After(obj)
    return filter_names, me_values



def make_efficiency_table(files, source_type, filter_names,
                          efficiencies, stats_format):
    if "twiki" in stats_format:
        format = "| %s | %s |"
        separator = " | "
    if "html" in stats_format:
        format = "<tr><th>%s<td>%s"
        separator = "<td>"
        print "<table>"
    sample = files[0].name[0:files[0].name.find("_")]
    print (format % (sample, separator.join(filter_names))).replace('td', 'th')
    for file_index, file in enumerate(files):
        entries = []
        for filter in filter_names:
            label = "%sEffPhi_%s" % (source_type, filter)
            eff, err = efficiencies[file_index][label]
            entries.append("%.1f &plusmn; %.1f" % (eff, err))
        version = file.name
        version = version[version.find("_")+1:version.find("-")]
        print format % (version, separator.join(entries))
    if "html" in stats_format:
        print "</table>"


def make_efficiency_summary_plot(files, me_values):
    class Bin:
        def __init__(self, label, content, error):
            self.label = label
            self.content = float(content)
            self.error = float(error)
    hists = ROOT.TList()
    hist_path = "%s/Summary/Efficiency_Summary" % path_muon
    for file_index, file in enumerate(files):
        hist = file.Get(hist_path)
        if not hist:
            hist = file.Get(hist_path + "_Muon")
        if not hist:
            print "No efficiency summary plot found in %s" % file.name
            return
        n_bins = hist.GetNbinsX()
        bins = [Bin(hist.GetXaxis().GetBinLabel(i + 1),
                    hist.GetBinContent(i + 1),
                    ## hist.GetBinError(i + 1) kludge below!
                    hist.GetBinContent(i + 1)/1000)
                for i in range(n_bins)]
        n_cross = i = 0
        while i != (len(bins) - n_cross):
            if cross_channel_format.match(bins[i].label):
                bins.append(bins.pop(i))
                n_cross += 1
            else:
                i += 1
        new_hist = ROOT.TH1F("%s_clone%i" % (hist.GetName(), file_index),
                             file.name, n_bins,
                             hist.GetXaxis().GetBinLowEdge(1),
                             hist.GetXaxis().GetBinUpEdge(n_bins))
        for i, bin in enumerate(bins):
            new_hist.GetXaxis().SetBinLabel(i + 1, bin.label)
            new_hist.SetBinContent(i + 1, bin.content)
            new_hist.SetBinError(i + 1, bin.error)
        hists.Add(new_hist)
    if hists.At(0):
        plot(hists, "Summary", "Efficiency Summary")



def make_efficiency_plot(files, plot_name, path, efficiencies, me_values):
    hists = ROOT.TList()
    hist_path = "%s/%s/%s" % (path_dist, path, plot_name)
    hist_title = files[0].Get(hist_path).GetTitle()
    for file in files:
        try: hist = ROOT.TH1F(file.Get(hist_path))
        except:
            try: hist = ROOT.TProfile(file.Get(hist_path))
            except: print "Failed to find %s!!" % plot_name; return
        new_hist = hist.Clone()
        new_hist.SetTitle(file.name)
        hists.Add(new_hist)
    if hists.At(0):
        plot(hists, plot_name, hist_title, efficiencies, me_values, path)



def make_overlaid_turnon_plot(files, source_type, filter, path,
                              efficiencies, me_values):
    hists = ROOT.TList()
    for plot_name in ["%sTurnOn%s_%s" % (source_type, number, filter)
                      for number in [1, 2]]:
        hist_path = "%s/%s/%s" % (path_dist, path, plot_name)
        try: hist = ROOT.TH1F(files[0].Get(hist_path))
        except: hist = ROOT.TProfile(files[0].Get(hist_path))
        new_hist = hist.Clone()
        x_title = hist.GetXaxis().GetTitle()
        x_title = x_title.replace("Leading", "Leading/Next-to-Leading")
        new_hist.SetTitle("%s;%s" % (new_hist.GetTitle(), x_title))
        hists.Add(new_hist)
    if hists.At(0):
        plot(hists, "%sTurnOn", "Turn-On Curves for %s" % filter,
             efficiencies, me_values, path)



def plot(hists, hist_type, hist_title,
         efficiencies=None, me_values=None, path=None):
    plot_num = next_counter()
    first = hists.First()
    last = hists.Last()
    hist = last
    x_title = first.GetXaxis().GetTitle()
    if   "Gen" in x_title: source_type = "gen"
    elif "Rec" in x_title: source_type = "reco"
    expression = ("0.5 * [2] * (" +
                  "TMath::Erf((x / [0] + 1.) / (TMath::Sqrt(2.) * [1])) + " +
                  "TMath::Erf((x / [0] - 1.) / (TMath::Sqrt(2.) * [1])))")
    function_turn_on = ROOT.TF1("turnOn", expression, 10, 40)
    while hist:
        hist_index = int(hists.IndexOf(hist))
        hist.Draw()
        ROOT.gPad.SetLogx(False)
        ROOT.gPad.SetTickx(1)
        title = hist.GetTitle()
        hist.SetLineWidth(2)
        hist.SetLineColor(colors[hist_index % len(colors)])
        if hist.GetMaximum() <= 1.0:
            hist.Scale(100.)
        hist.GetYaxis().SetRangeUser(0., 100.)
        if "Summary" in hist_type:
            hist.GetXaxis().LabelsOption("u")
            c1.SetTickx(0)
        if "Eff" in hist_type or "TurnOn" in hist_type:
            yTitle = hist.GetYaxis().GetTitle()
            slashIndex = yTitle.find("/")
            yTitle = "#frac{%s}{%s} (%%)" % (yTitle[0:slashIndex - 1],
                                            yTitle[slashIndex + 2:100])
            hist.GetYaxis().SetTitle(yTitle)
            hist.GetYaxis().SetTitleOffset(1.5)
            hist.GetYaxis().SetTitleSize(0.04)
        if "Eff" in hist_type:
            eff, err = efficiencies[hist_index][hist_type]
            hist.SetTitle("%s (%.1f#pm%.1f%%)" % (title, eff, err))
        if "TurnOn" in hist_type:
            ROOT.gPad.SetLogx(True)
            hist.GetXaxis().SetRangeUser(2., 300.)
            function_turn_on.SetParameters(1,20,100)
            function_turn_on.SetLineColor(hist.GetLineColor())
            hist.Fit("turnOn","q")
            hist.Draw()
            eff = function_turn_on.GetParameter(2)
            if eff < 100 and eff > 0:
                hist.SetTitle("%s (%.1f%%)" % (title, eff))
        hist = hists.Before(hist)
    last.Draw()
    hist = hists.Before(last)
    while hist:
        hist.Draw("same")
        hist = hists.Before(hist)
    lower_bound_y = 0.15
    upper_bound_y = lower_bound_y + (0.055 * hists.Capacity())
    if   "Summary" in hist_type:
        cuts = None
    elif "TurnOn"  in hist_type:
        cuts = "|#eta| < %.1f" % me_values["CutMaxEta"]
    else:
        cuts = "p_{T} #geq %.0f, |#eta| < %.1f" % (me_values["CutMinPt"],
                                                me_values["CutMaxEta"])
    legend = ROOT.gPad.BuildLegend(0.22, lower_bound_y, 0.90, upper_bound_y)
    legend.SetFillColor(0)
    legend.SetFillStyle(1001)
    if "Summary" not in hist_type:
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextAlign(31)
        latex.DrawLatex(0.93, upper_bound_y + 0.015,
                        "Cuts on #mu_{%s}: %s" % (source_type, cuts))
    hist_title = "%.3i: %s" % (plot_num, hist_title)
    if path: hist_title += " step in %s" % path
    last.SetTitle(hist_title)
    ROOT.gPad.Update()
    c1.SaveAs("%.3i.%s" % (plot_num, file_type))



def merge_pdf_output(files):
    os.system("gs -q -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dAutoRotatePages=/All  -sOutputFile=merged.pdf [0-9][0-9][0-9].pdf")
    if len(files) == 1:
        pdfname = "%s.pdf" % files[0].name
    elif len(files) == 2:
        pdfname = "%s_vs_%s.pdf" % (files[0].name, files[1].name)
    else:
        pdfname = "%s_vs_Many.pdf" % files[0].name
    os.system("cp merged.pdf %s" % pdfname)
    os.system("rm [0-9]*.pdf")
    print "Wrote %i plots to %s" % (next_counter() - 1, pdfname)



if __name__ == "__main__":
    sys.exit(main())

