#!/bin/env python

import argparse
import ROOT as r
from array import array
import os
import sys
from math import sqrt

sel_choices = ["base", "loweta", "xtr", "vtr", "none"]
metric_choices = ["eff", "fakerate", "duplrate"]
variable_choices = ["pt", "ptmtv", "ptlow", "eta", "phi", "dxy", "dz", "vxy"]
objecttype_choices = ["TC", "pT5", "T5", "pT3", "pLS", "pT5_lower", "pT3_lower", "T5_lower"]
#lowerObjectType = ["pT5_lower", "pT3_lower", "T5_lower"]

r.gROOT.SetBatch(True)

# Argument parsing
parser = argparse.ArgumentParser(description="What are we wanting to graph?")
# Input (input is a root file with numerator and denominator histograms
parser.add_argument('inputs', nargs='+', help='input num_den_hist.root files')
parser.add_argument('--tag'         , '-t' , dest='tag'         , type=str , default='v0'        , help='tag of the run [DEFAULT=v0]', required=True)
# When more than two input files are provided it is to compare the two sets of performance plots
# The provided input file must contain the TGraphAsymmError plots when comparing
parser.add_argument('--metric'      , '-m' , dest='metric'      , type=str            , help='{}'.format(','.join(metric_choices), metric_choices[0]))
parser.add_argument('--objecttype'  , '-o' , dest='objecttype'  , type=str            , help='{}'.format(','.join(objecttype_choices), objecttype_choices[0]), default=objecttype_choices[0])
parser.add_argument('--selection'   , '-s' , dest='selection'   , type=str            , help='{}'.format(','.join(sel_choices), sel_choices[0]), default=sel_choices[0])
parser.add_argument('--pdgid'       , '-g' , dest='pdgid'       , type=int            , help='pdgid (efficiency plots only)')
parser.add_argument('--charge'      , '-c' , dest='charge'      , type=int            , help='charge (efficiency plots only) (0: both, 1: positive, -1:negative', default=0)
parser.add_argument('--variable'    , '-v' , dest='variable'    , type=str            , help='{}'.format(','.join(variable_choices), variable_choices[0]))
parser.add_argument('--individual'  , '-b' , dest='individual'  , action="store_true" , help='plot not the breakdown but individual')
parser.add_argument('--yzoom'       , '-y' , dest='yzoom'       , action="store_true" , help='zoom in y')
parser.add_argument('--xcoarse'     , '-x' , dest='xcoarse'     , action="store_true" , help='coarse in x')
parser.add_argument('--sample_name' , '-S' , dest='sample_name' , type=str            , help='sample name in case one wants to override')
parser.add_argument('--pt_cut'      ,        dest='pt_cut'      , type=float          , default=0.9, help='transverse momentum cut [DEFAULT=0.9]')
parser.add_argument('--eta_cut'     ,        dest='eta_cut'     , type=float          , default=4.5, help='pseudorapidity cut [DEFAULT=4.5]')
parser.add_argument('--compare'     , '-C' , dest='compare'     , action="store_true" , help='plot comparisons of input files')
parser.add_argument('--comp_labels' , '-L' , dest='comp_labels' , type=str            , help='comma separated legend labels for comparison plots (e.g. reference,pT5_update')


#______________________________________________________________________________________________________
def main():

    # Plotting a set of performance plots
    #
    #   $ lst_plot_performance.py -t myRun1 INPUT.root
    #
    # where INPUT.root contains a set of numerator and denominator histograms named in following convention
    #
    #  Efficiency histogram:
    #
    #     Root__<OBJ>_<SEL>_<PDGID>_<CHARGE>_ef_numer_<VARIABLE>
    #     Root__<OBJ>_<SEL>_<PDGID>_<CHARGE>_ef_denom_<VARIABLE>
    #
    #     e.g. Root__pT5_loweta_321_-1_ef_numer_eta
    #
    #          OBJ=pT5
    #          SEL=loweta
    #          PDGID=Kaon
    #          CHARGE=-1
    #          METRIC=ef (efficiency)
    #          VARIABLE=eta
    #
    #  Fake rate or duplicate rate histogram:
    #
    #     Root__<OBJ>_fr_numer_<VARIABLE> (fake rate numerator)
    #     Root__<OBJ>_fr_denom_<VARIABLE> (fake rate denominator)
    #     Root__<OBJ>_dr_numer_<VARIABLE> (duplicate rate numerator)
    #     Root__<OBJ>_dr_denom_<VARIABLE> (duplicate rate denominator)
    #
    # Once ran, the output will be stored in performance/myRun1_<githash>
    # Also a ROOT file with the efficiency graphs will be saved to performance/myRun1_<githash>/efficiency.root
    #

    args = parser.parse_args()

    plot_standard_performance_plots(args)

#______________________________________________________________________________________________________
def plot(args):

    params = process_arguments_into_params(args)

    if params["metric"] == "eff":
        params["output_name"] = "{objecttype}_{selection}_{pdgid}_{charge}_{metric}_{variable}".format(**params)
    else:
        params["output_name"] = "{objecttype}_{metric}_{variable}".format(**params)

    if params["xcoarse"]:
        params["output_name"] += "coarse"
    if params["yzoom"]:
        params["output_name"] += "zoom"
    # if params["breakdown"]:
    #     params["output_name"] += "_breakdown"
    # if params["compare"]:
    #     params["output_name"] += "_breakdown"
    # Get histogram names
    if params["metric"] == "eff":
        params["denom"] = "Root__{objecttype}_{selection}_{pdgid}_{charge}_{metricsuffix}_denom_{variable}".format(**params)
        params["numer"] = "Root__{objecttype}_{selection}_{pdgid}_{charge}_{metricsuffix}_numer_{variable}".format(**params)
    else:
        params["denom"] = "Root__{objecttype}_{metricsuffix}_denom_{variable}".format(**params)
        params["numer"] = "Root__{objecttype}_{metricsuffix}_numer_{variable}".format(**params)

    # print(params["numer"])
    # print(params["denom"])
    # print(params["output_name"])

    #skip if histograms not found!
    if (not params["input_file"].GetListOfKeys().Contains(params["numer"])) or (not params["input_file"].GetListOfKeys().Contains(params["denom"])):
        return

    # Denom histogram
    denom = []
    denom.append(params["input_file"].Get(params["denom"]).Clone())

    # Numerator histograms
    numer = []
    numer.append(params["input_file"].Get(params["numer"]).Clone())

    breakdown_hist_types = ["pT5", "pT3", "T5", "pLS"]
    print("breakdown = ", params["breakdown"])
    if params["breakdown"]:
        for breakdown_hist_type in breakdown_hist_types:
            breakdown_histname = params["numer"].replace("TC", breakdown_hist_type)
            hist = params["input_file"].Get(breakdown_histname)
            numer.append(hist.Clone())
            denom.append(params["input_file"].Get(params["denom"]).Clone())

    if params["compare"]:
        for f in params["additional_input_files"]:
            hist = f.Get(params["numer"])
            numer.append(hist.Clone())
            hist = f.Get(params["denom"])
            denom.append(hist.Clone())


    if params["breakdown"]:
        params["legend_labels"] = ["TC" ,"pT5" ,"pT3" ,"T5" ,"pLS"]
    else:
        params["legend_labels"] = [args.objecttype]

    if params["compare"]:
        params["legend_labels"] = ["reference"]
        for i, f in enumerate(params["additional_input_files"]):
            params["legend_labels"].append("{i}".format(i=i))
        if params["comp_labels"]:
            params["legend_labels"] = params["comp_labels"]


    draw_ratio(
            numer, # numerator histogram(s)
            denom, # denominator histogram
            params,
            )

    DIR = os.environ["LSTPERFORMANCEWEBDIR"]
    perfwebpath = os.path.normpath("{}".format(DIR))
    os.system("cd {}; ln -sf {}/summary".format(params["output_dir"], perfwebpath))
    os.system("cd {}; ln -sf {}/compare".format(params["output_dir"], perfwebpath))

#______________________________________________________________________________________________________
def process_arguments_into_params(args):

    params = {}

    # parse metric
    if args.metric not in metric_choices:
        print("metric", args.metric)
        parser.print_help(sys.stderr)
        sys.exit(1)
    params["metric"] = args.metric

    # parse the selection type
    if args.selection not in sel_choices:
        print("selection", args.selection)
        parser.print_help(sys.stderr)
        sys.exit(1)
    params["selection"] = args.selection

    params["pdgid"] = args.pdgid

    params["charge"] = args.charge

    # parse the object type
    if args.objecttype not in objecttype_choices:
        print("objjecttype", args.objjecttype)
        parser.print_help(sys.stderr)
        sys.exit(1)
    params["objecttype"] = args.objecttype

    # parse variable
    if args.variable not in variable_choices:
        print("variable", args.variable)
        parser.print_help(sys.stderr)
        sys.exit(1)
    params["variable"] = args.variable

    if args.yzoom:
        params["yzoom"] = True
    else:
        params["yzoom"] = False
    if args.xcoarse:
        params["xcoarse"] = True
    else:
        params["xcoarse"] = False

    # Parse from input file
    root_file_name = args.inputs[0]
    f = r.TFile(root_file_name)
    params["input_file"] = f

    # git version hash
    git_hash = f.Get("githash").GetTitle()
    params["git_hash"] = git_hash

    params["pt_cut"] = args.pt_cut
    params["eta_cut"] = args.eta_cut

    # sample name
    sample_name = f.Get("input").GetTitle()
    if args.sample_name:
        sample_name = args.sample_name
    params["sample_name"] = sample_name

    if len(args.inputs) > 1: # if more than 1 save them to separate params
        params["additional_input_files"] = []
        params["additional_git_hashes"] = []
        params["additional_sample_names"] = []
        for i in args.inputs[1:]:
            params["additional_input_files"].append(r.TFile(i))
            params["additional_git_hashes"].append(params["additional_input_files"][-1].Get("githash").GetTitle())
            params["additional_sample_names"].append(params["additional_input_files"][-1].Get("input").GetTitle())

    if params["metric"] == "eff": params["metricsuffix"] = "ef"
    if params["metric"] == "duplrate": params["metricsuffix"] = "dr"
    if params["metric"] == "fakerate": params["metricsuffix"] = "fr"

    # If breakdown it must be object type of TC
    params["breakdown"] = args.breakdown
#    if args.individual:
#        params["breakdown"] = False
#    else:
        # if params["objecttype"] != "TC":
        #     print("Warning! objecttype is set to \"TC\" because individual is False!")
        # params["objecttype"] = "TC"
#        params["breakdown"] = True

    # If compare we compare the different files
    params["compare"] = False
    if args.compare:
        params["breakdown"] = False
        params["compare"] = True

    if args.comp_labels:
        params["comp_labels"] = args.comp_labels.split(",")

    # process tags
    tag_ = os.path.normpath(args.tag)
    bn = os.path.basename(tag_)
    dn = os.path.dirname(tag_)
    params["tagbase"] = bn
    params["tagdir"] = dn
    params["tag"] = args.tag

    # Create output_dir
    params["lstoutputdir"] = os.environ["LSTOUTPUTDIR"]
    params["output_dir"] = os.path.normpath("{lstoutputdir}/performance/{tagdir}/{tagbase}_{git_hash}-{sample_name}".format(**params))
    if params["compare"]:
        for gg, ii in zip(params["additional_git_hashes"], params["additional_sample_names"]):
            params["output_dir"] += "_{}-{}".format(gg, ii)
    os.system("mkdir -p {output_dir}/mtv/var".format(**params))
    os.system("mkdir -p {output_dir}/mtv/num".format(**params))
    os.system("mkdir -p {output_dir}/mtv/den".format(**params))
    os.system("mkdir -p {output_dir}/mtv/ratio".format(**params))

    # Output file
    params["output_file"] = r.TFile("{output_dir}/efficiency.root".format(**params), "update")

    # git version hash
    params["input_file"].Get("githash").Write("", r.TObject.kOverwrite)

    # sample name
    params["input_file"].Get("input").Write("", r.TObject.kOverwrite)

    # Obtain the number of events used
    params["nevts"] = str(int(f.Get("nevts").GetBinContent(1)))

    return params

#______________________________________________________________________________________________________
def draw_ratio(nums, dens, params):

    # Rebin if necessary
    if "scalar" in params["output_name"] and "ptscalar" not in params["output_name"]:
        for num in nums:
            num.Rebin(180)
        for den in dens:
            den.Rebin(180)

    # Rebin if necessary
    if "coarse" in params["output_name"] and "ptcoarse" not in params["output_name"]:
        for num in nums:
            num.Rebin(6)
        for den in dens:
            den.Rebin(6)

    # Deal with overflow bins for pt plots
    if "pt" in params["output_name"] or "vxy" in params["output_name"]:
        for num in nums:
            overFlowBin = num.GetBinContent(num.GetNbinsX() + 1)
            lastBin = num.GetBinContent(num.GetNbinsX())
            num.SetBinContent(num.GetNbinsX(), lastBin + overFlowBin)
            num.SetBinError(num.GetNbinsX(), sqrt(lastBin + overFlowBin))
        for den in dens:
            overFlowBin = den.GetBinContent(den.GetNbinsX() + 1)
            lastBin = den.GetBinContent(den.GetNbinsX())
            den.SetBinContent(den.GetNbinsX(), lastBin + overFlowBin)
            den.SetBinError(den.GetNbinsX(), sqrt(lastBin + overFlowBin))

    # Create efficiency graphs
    teffs = []
    effs = []
    for num, den in zip(nums, dens):
        teff = r.TEfficiency(num, den)
        eff = teff.CreateGraph()
        teffs.append(teff)
        effs.append(eff)

    # print(effs)

    #
    hist_name_suffix = ""
    if params["xcoarse"]:
        hist_name_suffix += "coarse"
    if params["yzoom"]:
        hist_name_suffix += "zoom"

    params["output_file"].cd()
    outputname = params["output_name"]
    for den in dens:
        den.Write(den.GetName() + hist_name_suffix, r.TObject.kOverwrite)
        # eff_den = r.TGraphAsymmErrors(den)
        # eff_den.SetName(outputname+"_den")
        # eff_den.Write("", r.TObject.kOverwrite)
    for num in nums:
        num.Write(num.GetName() + hist_name_suffix, r.TObject.kOverwrite)
        # eff_num = r.TGraphAsymmErrors(num)
        # eff_num.SetName(outputname+"_num")
        # eff_num.Write("", r.TObject.kOverwrite)
    for eff in effs:
        eff.SetName(outputname)
        eff.Write("", r.TObject.kOverwrite)

    draw_plot(effs, nums, dens, params)


#______________________________________________________________________________________________________
def parse_plot_name(output_name):
    if "fake" in output_name:
        rtnstr = ["Fake Rate of"]
    elif "dup" in output_name:
        rtnstr = ["Duplicate Rate of"]
    elif "inefficiency" in output_name:
        rtnstr = ["Inefficiency of"]
    else:
        rtnstr = ["Efficiency of"]
    if "MD_" in output_name:
        rtnstr.append("Mini-Doublet")
    elif "LS_" in output_name and "pLS" not in output_name:
        rtnstr.append("Line Segment")
    elif "pT4_" in output_name:
        rtnstr.append("Quadruplet w/ Pixel LS")
    elif "T4_" in output_name:
        rtnstr.append("Quadruplet w/o gap")
    elif "T4x_" in output_name:
        rtnstr.append("Quadruplet w/ gap")
    elif "pT3_" in output_name:
        rtnstr.append("Pixel Triplet")
    elif "pT5_" in output_name:
        rtnstr.append("Pixel Quintuplet")
    elif "T3_" in output_name:
        rtnstr.append("Triplet")
    elif "TCE_" in output_name:
        rtnstr.append("Extended Track")
    elif "pureTCE_" in output_name:
        rtnstr.append("Pure Extensions")
    elif "TC_" in output_name:
        rtnstr.append("Track Candidate")
    elif "T4s_" in output_name:
        rtnstr.append("Quadruplet w/ or w/o gap")
    elif "pLS_" in output_name:
        rtnstr.append("Pixel Line Segment")
    elif "T5_" in output_name:
        rtnstr.append("Quintuplet")
    return " ".join(rtnstr)

#______________________________________________________________________________________________________
def get_pdgidstr(pdgid):
    if abs(pdgid) == 0: return "All"
    elif abs(pdgid) == 11: return "Electron"
    elif abs(pdgid) == 13: return "Muon"
    elif abs(pdgid) == 211: return "Pion"
    elif abs(pdgid) == 321: return "Kaon"

#______________________________________________________________________________________________________
def get_chargestr(charge):
    if charge == 0: return "All"
    elif charge == 1: return "Positive"
    elif charge == -1: return "Negative"

#______________________________________________________________________________________________________
def set_label(eff, output_name, raw_number):
    if "phi" in output_name:
        title = "#phi"
    elif "_dz" in output_name:
        title = "z [cm]"
    elif "_dxy" in output_name:
        title = "d0 [cm]"
    elif "_vxy" in output_name:
        title = "r_{vertex} [cm]"
    elif "_pt" in output_name:
        title = "p_{T} [GeV]"
    elif "_hit" in output_name:
        title = "hits"
    elif "_lay" in output_name:
        title = "layers"
    else:
        title = "#eta"
    eff.GetXaxis().SetTitle(title)
    if "fakerate" in output_name:
        eff.GetYaxis().SetTitle("Fake Rate")
    elif "duplrate" in output_name:
        eff.GetYaxis().SetTitle("Duplicate Rate")
    elif "inefficiency" in output_name:
        eff.GetYaxis().SetTitle("Inefficiency")
    else:
        eff.GetYaxis().SetTitle("Efficiency")
    if raw_number:
        eff.GetYaxis().SetTitle("# of objects of interest")
    eff.GetXaxis().SetTitleSize(0.05)
    eff.GetYaxis().SetTitleSize(0.05)
    eff.GetXaxis().SetLabelSize(0.05)
    eff.GetYaxis().SetLabelSize(0.05)

#______________________________________________________________________________________________________
def draw_label(params):
    version_tag = params["git_hash"]
    sample_name = params["sample_name"]
    pdgidstr = get_pdgidstr(params["pdgid"])
    chargestr = get_chargestr(params["charge"])
    output_name = params["output_name"]
    n_events_processed = params["nevts"]
    ptcut = params["pt_cut"]
    etacut = params["eta_cut"]
    # Label
    t = r.TLatex()

    # Draw information about sample, git version, and types of particles
    t.SetTextAlign(11) # align bottom left corner of text
    t.SetTextColor(r.kBlack)
    t.SetTextSize(0.04)
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.09 + 0.03
    sample_name_label = "Sample:" + sample_name
    sample_name_label += "   Version tag:" + version_tag
    if n_events_processed:
        sample_name_label +="  N_{evt}:" + n_events_processed
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % sample_name_label)

    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.045 + 0.03

    etacutstr = "|#eta| < 4.5"
    if params["selection"] == "loweta":
        etacutstr = "|#eta| < 2.4"
    if params["selection"] == "xtr":
        etacutstr = "x-reg"
    if params["selection"] == "vtr":
        etacutstr = "not x-reg"
    if "eff" in output_name:
        if "_pt" in output_name:
            fiducial_label = "{etacutstr}, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(etacutstr=etacutstr)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut)
        elif "_dz" in output_name:
            fiducial_label = "{etacutstr}, p_{{T}} > {pt} GeV, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, etacutstr=etacutstr)
        elif "_dxy" in output_name:
            fiducial_label = "{etacutstr}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, etacutstr=etacutstr)
        elif "_vxy" in output_name:
            fiducial_label = "{etacutstr}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm".format(pt=ptcut, etacutstr=etacutstr)
        else:
            fiducial_label = "{etacutstr}, p_{{T}} > {pt} GeV, |Vtx_{{z}}| < 30 cm, |Vtx_{{xy}}| < 2.5 cm".format(pt=ptcut, etacutstr=etacutstr)
        particleselection = ((", Particle:" + pdgidstr) if pdgidstr else "" ) + ((", Charge:" + chargestr) if chargestr else "" )
        fiducial_label += particleselection
    # If fake rate or duplicate rate plot follow the following fiducial label rule
    elif "fakerate" in output_name or "duplrate" in output_name:
        if "_pt" in output_name:
            fiducial_label = "|#eta| < {eta}".format(eta=etacut)
        elif "_eta" in output_name:
            fiducial_label = "p_{{T}} > {pt} GeV".format(pt=ptcut)
        else:
            fiducial_label = "|#eta| < {eta}, p_{{T}} > {pt} GeV".format(pt=ptcut, eta=etacut)
    t.DrawLatexNDC(x,y,"#scale[0.9]{#font[42]{%s}}" % fiducial_label)

    # Draw CMS label
    cms_label = "Simulation"
    x = r.gPad.GetX1() + r.gPad.GetLeftMargin()
    y = r.gPad.GetY2() - r.gPad.GetTopMargin() + 0.005
    t.DrawLatexNDC(x,y,"#scale[1.25]{#font[61]{CMS}} #scale[1.1]{#font[52]{%s}}" % cms_label)

#______________________________________________________________________________________________________
def draw_plot(effs, nums, dens, params):

    legend_labels = params["legend_labels"]
    output_dir = params["output_dir"]
    output_name = params["output_name"]
    sample_name = params["sample_name"]
    version_tag = params["git_hash"]
    pdgidstr = get_pdgidstr(params["pdgid"])
    chargestr = get_chargestr(params["charge"])

    # Get Canvas
    c1 = r.TCanvas()
    c1.SetBottomMargin(0.15)
    c1.SetLeftMargin(0.15)
    c1.SetTopMargin(0.22)
    c1.SetRightMargin(0.15)

    # Set logx
    if "_pt" in output_name:
        c1.SetLogx()

    # Set title
    # print(output_name)
    # print(parse_plot_name(output_name))
    effs[0].SetTitle(parse_plot_name(output_name))

    # Draw the efficiency graphs
    colors = [1, 2, 3, 4, 6]
    markerstyles = [20, 26, 28, 24, 27]
    markersize = 1.2
    linewidth = 2
    for i, eff in enumerate(effs):
        if i == 0:
            eff.Draw("epa")
        else:
            eff.Draw("epsame")
        eff.SetMarkerStyle(markerstyles[i])
        eff.SetMarkerSize(markersize)
        eff.SetLineWidth(linewidth)
        eff.SetMarkerColor(colors[i])
        eff.SetLineColor(colors[i])
        set_label(eff, output_name, raw_number=False)

    nleg = len(legend_labels)
    legend = r.TLegend(0.15,0.75-nleg*0.04,0.25,0.75)
    for i, label in enumerate(legend_labels):
        legend.AddEntry(effs[i], label)
    legend.Draw("same")

    # Compute the yaxis_max
    yaxis_max = 0
    for eff in effs:
        for i in range(0, eff.GetN()):
            if yaxis_max < eff.GetY()[i]:
                yaxis_max = eff.GetY()[i]

    # Compute the yaxis_min
    yaxis_min = 999
    for i in range(0, effs[0].GetN()):
        if yaxis_min > effs[0].GetY()[i] and effs[0].GetY()[i] != 0:
            yaxis_min = effs[0].GetY()[i]

    # Set Yaxis range
    effs[0].GetYaxis().SetRangeUser(0, 1.02)
    if "zoom" not in output_name:
        effs[0].GetYaxis().SetRangeUser(0, 1.02)
    else:
        if "fakerate" in output_name:
            effs[0].GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        elif "duplrate" in output_name:
            effs[0].GetYaxis().SetRangeUser(0.0, yaxis_max * 1.1)
        else:
            effs[0].GetYaxis().SetRangeUser(0.6, 1.02)

    # Set xaxis range
    if "_eta" in output_name:
        effs[0].GetXaxis().SetLimits(-4.5, 4.5)

    # Draw label
    draw_label(params)

    # Output file path
    output_fullpath = output_dir + "/mtv/var/" + output_name + ".pdf"

    # Save
    c1.SetGrid()
    c1.SaveAs("{}".format(output_fullpath))
    c1.SaveAs("{}".format(output_fullpath.replace(".pdf", ".png")))
    effs[0].SetName(output_name)

    for i, num in enumerate(nums):
        set_label(num, output_name, raw_number=True)
        num.Draw("hist")
        c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/num/").replace(".pdf", "_num{}.pdf".format(i))))
        c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/num/").replace(".pdf", "_num{}.png".format(i))))

    for i, den in enumerate(dens):
        set_label(den, output_name, raw_number=True)
        den.Draw("hist")
        c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/den/").replace(".pdf", "_den{}.pdf".format(i))))
        c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/den/").replace(".pdf", "_den{}.png".format(i))))

    # Double ratio if more than one nums are provided
    # Take the first num as the base
    if len(nums) > 1:
        base = nums[0].Clone()
        base.Divide(nums[0], dens[0], 1, 1, "B") #Binomial
        others = []
        for num, den in zip(nums[1:], dens[1:]):
            other = num.Clone()
            other.Divide(other, den, 1, 1, "B")
            others.append(other)

        # Take double ratio
        for other in others:
            other.Divide(base)

        for i, other in enumerate(others):
            other.Draw("ep")
            other.GetYaxis().SetTitle("{} / {}".format(legend_labels[i+1], legend_labels[0]))
            other.SetMarkerStyle(markerstyles[i+1])
            other.SetMarkerSize(markersize)
            other.SetMarkerColor(colors[i+1])
            other.SetLineWidth(linewidth)
            other.SetLineColor(colors[i+1])
            c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/ratio/").replace(".pdf", "_ratio{}.pdf".format(i))))
            c1.SaveAs("{}".format(output_fullpath.replace("/mtv/var/", "/mtv/ratio/").replace(".pdf", "_ratio{}.png".format(i))))

#______________________________________________________________________________________________________
def plot_standard_performance_plots(args):

    # Efficiency plots
    metrics = metric_choices
    yzooms = [False, True]
    variables = {
            "eff": ["pt", "ptlow", "ptmtv", "eta", "phi", "dxy", "dz", "vxy"],
            "fakerate": ["pt", "ptlow", "ptmtv", "eta", "phi"],
            "duplrate": ["pt", "ptlow", "ptmtv", "eta", "phi"],
            }
    sels = {
            "eff": ["base", "loweta"],
            "fakerate": ["none"],
            "duplrate": ["none"],
            }
    xcoarses = {
            "pt": [False],
            "ptlow": [False],
            "ptmtv": [False],
            "eta": [False, True],
            "phi": [False, True],
            "dxy": [False, True],
            "vxy": [False, True],
            "dz": [False, True],
            }
    types = objecttype_choices
    breakdowns = {
            "eff":{
                "TC": [True, False],
                "pT5": [False],
                "pT3": [False],
                "T5": [False],
                "pLS": [False],
                "pT5_lower":[False],
                "pT3_lower":[False],
                "T5_lower":[False],
                },
            "fakerate":{
                "TC": [True, False],
                "pT5": [False],
                "pT3": [False],
                "T5": [False],
                "pLS": [False],
                "pT5_lower":[False],
                "pT3_lower":[False],
                "T5_lower":[False],
                },
            "duplrate":{
                "TC": [True, False],
                "pT5": [False],
                "pT3": [False],
                "T5": [False],
                "pLS": [False],
                "pT5_lower":[False],
                "pT3_lower":[False],
                "T5_lower":[False],
            },
    }
    pdgids = {
            "eff": [0, 11, 13, 211, 321],
            "fakerate": [0],
            "duplrate": [0],
            }
    charges = {
            "eff":[0, 1, -1],
            "fakerate":[0],
            "duplrate":[0],
            }

    if args.metric:
        metrics = [args.metric]

    if args.objecttype:
        types = args.objecttype.split(',')

    if args.compare:
        types = args.objecttype.split(',')

    if args.selection:
        sels["eff"] = [args.selection]

    if args.pdgid != None:
        pdgids["eff"] = [args.pdgid]
        pdgids["fakerate"] = [args.pdgid]
        pdgids["duplrate"] = [args.pdgid]

    if args.charge != None:
        charges["eff"] = [args.charge]
        charges["fakerate"] = [args.charge]
        charges["duplrate"] = [args.charge]

    if args.variable:
        # dxy and dz are only in efficiency
        if args.variable != "dxy" and args.variable != "dz" and args.variable != "vxy":
            variables["eff"] = [args.variable]
            variables["fakerate"] = [args.variable]
            variables["suplrate"] = [args.variable]
        else:
            variables["eff"] = [args.variable]
            variables["fakerate"] = []
            variables["suplrate"] = []

    if args.individual:
        # Only eff / TC matters here
        breakdowns = {"eff":{"TC":[False], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]},
                "fakerate": {"TC":[False], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]},
                "duplrate": {"TC":[False], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]}}


    else:
        # Only eff / TC matters here
        breakdowns = {"eff":{"TC":[True], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]},
                "fakerate": {"TC":[True], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]},
                "duplrate": {"TC":[True], "pT5_lower":[False], "pT3_lower":[False], "T5_lower":[False]}}
    if args.yzoom:
        args.yzooms = [args.yzoom]

    if args.xcoarse:
        args.xcoarses = [args.xcoarse]

    for metric in metrics:
        for sel in sels[metric]:
            for variable in variables[metric]:
                for yzoom in yzooms:
                    for xcoarse in xcoarses[variable]:
                        for typ in types:
                            print("type = ",typ)
                            for breakdown in breakdowns[metric][typ]:
                                for pdgid in pdgids[metric]:
                                    for charge in charges[metric]:
                                        args.metric = metric
                                        args.objecttype = typ
                                        args.selection = sel
                                        args.pdgid = pdgid
                                        args.charge = charge
                                        args.variable = variable
                                        args.breakdown = breakdown
                                        args.yzoom = yzoom
                                        args.xcoarse = xcoarse
                                        print(args)
                                        plot(args)

if __name__ == "__main__":

    main()

