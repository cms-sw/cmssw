import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Higgs.hltHiggsPostProcessor_cfi import *

# Build the standard strings to the DQM
def make_efficiency_string(objtype,plot_type,triggerpath):
    # --- IMPORTANT: Add here a elif if you are introduce a new collection
    #                (see EVTColContainer::getTypeString) 
    if objtype == "Mu":
        objtypeLatex="#mu"
    elif objtype == "Photon": 
        objtypeLatex="#gamma"
    elif objtype == "Ele": 
        objtypeLatex="e"
    elif objtype == "MET" :
        objtypeLatex="MET"
    elif objtype == "PFMET" :
        objtypeLatex="PFMET"
    elif objtype == "PFTau": 
        objtypeLatex="#tau"
    elif objtype == "Jet": 
        objtypeLatex="jet"
    else:
        objtypeLatex=objtype

    numer_description = "# gen %s passed the %s" % (objtypeLatex,triggerpath)
    denom_description = "# gen %s " % (objtypeLatex)

    if plot_type == "TurnOn1":
        title = "pT Turn-On"
        xAxis = "p_{T} of Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt1" % (objtype)
    elif plot_type == "TurnOn2":
        title = "Next-to-Leading pT Turn-On"
        xAxis = "p_{T} of Next-to-Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt2" % (objtype)
    elif plot_type == "EffEta":
        title = "#eta Efficiency"
        xAxis = "#eta of Generated %s " % (objtype)
        input_type = "gen%sEta" % (objtype)
    elif plot_type == "EffPhi":
        title = "#phi Efficiency"
        xAxis = "#phi of Generated %s " % (objtype)
        input_type = "gen%sPhi" % (objtype)     
    elif "TurnOn" in plot_type:
        title = "%sth Leading pT Turn-On" % (plot_type[-1])
        xAxis = "p_{T} of %sth Leading Generated %s (GeV/c)" % (plot_type[-1], objtype)
        input_type = "gen%sMaxPt%s" % (objtype, plot_type[-1])
    elif plot_type == "EffdEtaqq":
        title = "#Delta #eta_{qq} Efficiency"
        xAxis = "#Delta #eta_{qq} of Generated %s " % (objtype)
        input_type = "gen%sdEtaqq" % (objtype)
    elif plot_type == "Effmqq":
        title = "m_{qq} Efficiency"
        xAxis = "m_{qq} of Generated %s " % (objtype)
        input_type = "gen%smqq" % (objtype)
    elif plot_type == "EffdPhibb":
        title = "#Delta #phi_{bb} Efficiency"
        xAxis = "#Delta #phi_{bb} of Generated %s " % (objtype)
        input_type = "gen%sdPhibb" % (objtype)
    elif plot_type == "EffCSV1":
        title = "CSV1 Efficiency"
        xAxis = "CSV1 of Generated %s " % (objtype)
        input_type = "gen%sCSV1" % (objtype)
    elif plot_type == "EffCSV2":
        title = "CSV2 Efficiency"
        xAxis = "CSV2 of Generated %s " % (objtype)
        input_type = "gen%sCSV2" % (objtype)
    elif plot_type == "EffCSV3":
        title = "CSV3 Efficiency"
        xAxis = "CSV3 of Generated %s " % (objtype)
        input_type = "gen%sCSV3" % (objtype)
    elif plot_type == "EffmaxCSV":
        title = "max CSV Efficiency"
        xAxis = "max CSV of Generated %s " % (objtype)
        input_type = "gen%smaxCSV" % (objtype)
        
    yAxis = "%s / %s" % (numer_description, denom_description)
    all_titles = "%s for trigger %s; %s; %s" % (title, triggerpath,
                                        xAxis, yAxis)
    return "Eff_%s_%s '%s' %s_%s %s" % (input_type,triggerpath,
            all_titles,input_type,triggerpath,input_type)

#--- IMPORTANT: Trigger are extracted from the hltHiggsValidator_cfi.py module
from HLTriggerOffline.Higgs.hltHiggsValidator_cfi import hltHiggsValidator as _config
def make_higgs_postprocessor(analysis_name, plot_types=["TurnOn1", "TurnOn2", "EffEta", "EffPhi"], object_types=["Mu","Ele","Photon","MET","PFMET","PFTau","Jet"], extra_str_templates=[]):
    postprocessor = hltHiggsPostProcessor.clone()
    postprocessor.subDirs = ["HLT/Higgs/" + analysis_name]
    efficiency_strings = [] # List of plots to look for. This is quite a bit larger than the number of plots that will be made.

    efficiency_summary_string = "EffSummaryPaths_" + analysis_name + "_gen ' Efficiency of paths used in " + analysis_name + " ; trigger path ' SummaryPaths_" + analysis_name + "_gen_passingHLT SummaryPaths_" + analysis_name + "_gen"
    efficiency_strings.append(efficiency_summary_string)
    efficiency_strings.append(efficiency_summary_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))

    for plot_type in plot_types:
        for object_type in object_types:
            for trigger in [x.replace("_v", "") for x in _config.__getattribute__(analysis_name).hltPathsToCheck]:
                this_efficiency_string = make_efficiency_string(object_type, plot_type, trigger)
                efficiency_strings.append(this_efficiency_string)
                efficiency_strings.append(this_efficiency_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))

                for str_template in extra_str_templates:
                    this_extra_string = str_template.replace("@ANALYSIS@", analysis_name).replace("@TRIGGER@", trigger)
                    efficiency_strings.append(this_extra_string)
                    efficiency_strings.append(this_extra_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))

    postprocessor.efficiencyProfile = efficiency_strings
    return postprocessor

truevtx_string_template = "Eff_trueVtxDist_@ANALYSIS@_gen_@TRIGGER@ ' Efficiency of @TRIGGER@ vs nb of interactions ; nb events passing each path ' trueVtxDist_@ANALYSIS@_gen_@TRIGGER@ trueVtxDist_@ANALYSIS@_gen"
hltHiggsPostHWW = make_higgs_postprocessor("HWW", extra_str_templates=[truevtx_string_template])
hltHiggsPostHZZControlPaths = make_higgs_postprocessor("HZZControlPaths", extra_str_templates=[truevtx_string_template])
hltHiggsPostHZZ = make_higgs_postprocessor("HZZ", extra_str_templates=[truevtx_string_template])
hltHiggsPostHgg = make_higgs_postprocessor("Hgg", extra_str_templates=[truevtx_string_template])
#hltHiggsPostHggControlPaths = make_higgs_postprocessor("HggControlPaths", extra_str_templates=[truevtx_string_template])
hltHiggsPostMuonJet = make_higgs_postprocessor("MuonJet", extra_str_templates=[truevtx_string_template])
hltHiggsPostDoubleHinTaus = make_higgs_postprocessor("DoubleHinTaus", extra_str_templates=[truevtx_string_template])
hltHiggsPostHiggsDalitz = make_higgs_postprocessor("HiggsDalitz", extra_str_templates=[truevtx_string_template])
hltHiggsPostH2tau = make_higgs_postprocessor("H2tau", extra_str_templates=[truevtx_string_template])
hltHiggsPostHtaunu = make_higgs_postprocessor("Htaunu", extra_str_templates=[truevtx_string_template])
hltHiggsPostVBFHToInv = make_higgs_postprocessor("VBFHToInv", extra_str_templates=[truevtx_string_template])

TTHbbej_HtDist_template = "Eff_HtDist_@ANALYSIS@_gen_@TRIGGER@ ' Efficiency of @TRIGGER@ vs sum pT of jets ; sum pT of jets ' HtDist_@ANALYSIS@_gen_@TRIGGER@ HtDist_@ANALYSIS@_gen"
hltHiggsPostTTHbbej = make_higgs_postprocessor("TTHbbej", extra_str_templates=[truevtx_string_template, TTHbbej_HtDist_template])
hltHiggsPostAHttH = make_higgs_postprocessor("AHttH", extra_str_templates=[truevtx_string_template, TTHbbej_HtDist_template])

#Specific plots for VBFHbb_2btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
NminOneCutNames = ("EffdEtaqq", "Effmqq", "EffdPhibb", "EffCSV1", "EffCSV2", "EffCSV3",  "EffmaxCSV", "", "", "TurnOn1", "TurnOn2", "TurnOn3", "TurnOn4")
VBFHbb_2btag_plot_types = []
NminOneCuts =_config.VBFHbb_2btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                VBFHbb_2btag_plot_types.pop()
            VBFHbb_2btag_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostVBFHbb_2btag = make_higgs_postprocessor("VBFHbb_2btag", object_types=["Jet"], plot_types=VBFHbb_2btag_plot_types, extra_str_templates=[truevtx_string_template])

#Specific plots for VBFHbb_1btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
VBFHbb_1btag_plot_types = []
NminOneCuts = _config.VBFHbb_1btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                VBFHbb_1btag_plot_types.pop()
            VBFHbb_1btag_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostVBFHbb_1btag = make_higgs_postprocessor("VBFHbb_1btag", plot_types=VBFHbb_1btag_plot_types, object_types=["Jet"], extra_str_templates=[truevtx_string_template])

#Specific plots for VBFHbb_0btag  
#dEtaqq, mqq, dPhibb, CVS1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
VBFHbb_0btag_plot_types = []
NminOneCuts = _config.VBFHbb_0btag.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            if( NminOneCutNames[iCut] == "EffmaxCSV" ):
                VBFHbb_0btag_plot_types.pop()
            VBFHbb_0btag_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostVBFHbb_0btag = make_higgs_postprocessor("VBFHbb_0btag", plot_types=VBFHbb_0btag_plot_types, object_types=["Jet"], extra_str_templates=[truevtx_string_template])


#Specific plots for ZnnHbb
ZnnHbb_plot_types = ["EffEta", "EffPhi", "TurnOn1"]
NminOneCuts = _config.ZnnHbb.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            ZnnHbb_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostZnnHbb = make_higgs_postprocessor("ZnnHbb", plot_types=ZnnHbb_plot_types, object_types=["Jet", "PFMET"], extra_str_templates=[truevtx_string_template])

#Specific plots for X4b
#Jet plots
X4b_plot_types = ["EffEta", "EffPhi", "TurnOn1"]
NminOneCuts = _config.X4b.NminOneCuts
if NminOneCuts: 
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            X4b_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostX4b = make_higgs_postprocessor("X4b", plot_types=X4b_plot_types, object_types=["Jet"], extra_str_templates=[truevtx_string_template])

#Specific plots for WH -> ev + bb
hltHiggsPostWHToENuBB = make_higgs_postprocessor("WHToENuBB", extra_str_templates=[truevtx_string_template, TTHbbej_HtDist_template])

#Specific plots for MSSMHbb 
#Jet plots
MSSMHbb_plot_types = ["EffEta", "EffPhi", "TurnOn1"]
NminOneCuts = _config.MSSMHbb.NminOneCuts
if NminOneCuts:
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            MSSMHbb_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostMSSMHbb = make_higgs_postprocessor("MSSMHbb", plot_types=MSSMHbb_plot_types, object_types=["Jet"], extra_str_templates=[truevtx_string_template])

#Specific plots for MSSMHbbmu
MSSMHbbmu_plot_types = ["EffEta", "EffPhi", "TurnOn1"]
NminOneCuts = _config.MSSMHbbmu.NminOneCuts
if NminOneCuts:
    for iCut in range(0,len(NminOneCuts)):
        if( NminOneCuts[iCut] and NminOneCutNames[iCut] ):
            MSSMHbbmu_plot_types.append(NminOneCutNames[iCut])
hltHiggsPostMSSMHbbmu = make_higgs_postprocessor("MSSMHbbmu", plot_types=MSSMHbbmu_plot_types, object_types=["Jet"], extra_str_templates=[truevtx_string_template])

hltHiggsPostProcessors = cms.Sequence(
        hltHiggsPostHWW+
        hltHiggsPostHZZ+
        hltHiggsPostHZZControlPaths+
        hltHiggsPostHgg+
        #hltHiggsPostHggControlPaths+
        hltHiggsPostMuonJet+
        hltHiggsPostHtaunu+
        hltHiggsPostH2tau+
        hltHiggsPostTTHbbej+
        hltHiggsPostAHttH+
        hltHiggsPostVBFHbb_0btag+
        hltHiggsPostVBFHbb_1btag+
        hltHiggsPostVBFHbb_2btag+
        hltHiggsPostZnnHbb+
        hltHiggsPostDoubleHinTaus+
        hltHiggsPostHiggsDalitz+
        hltHiggsPostX4b+
        hltHiggsPostWHToENuBB+
        hltHiggsPostMSSMHbb+
        hltHiggsPostMSSMHbbmu+ 
        hltHiggsPostVBFHToInv
)


