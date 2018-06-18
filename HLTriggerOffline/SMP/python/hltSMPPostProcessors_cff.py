import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.SMP.hltSMPPostProcessor_cfi import *

# Build the standard strings to the DQM
def make_efficiency_string(objtype,plot_type,triggerpath):
    # --- IMPORTANT: Add here a elif if you are introduce a new collection
    #                (see EVTColContainer::getTypeString) 
    if objtype == "Mu" :
      objtypeLatex="#mu"
    elif objtype == "Photon": 
      objtypeLatex="#gamma"
    elif objtype == "Ele": 
      objtypeLatex="e"
    elif objtype == "MET" :
      objtypeLatex="MET"
    elif objtype == "PFTau": 
      objtypeLatex="#tau"
    else:
      objtypeLatex=objtype

    numer_description = "# gen %s passed the %s" % (objtypeLatex,triggerpath)
    denom_description = "# gen %s " % (objtypeLatex)

    if plot_type == "TurnOn1":
        title = "pT Turn-On"
        xAxis = "p_{T} of Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt1" % (objtype)
    if plot_type == "TurnOn2":
        title = "Next-to-Leading pT Turn-On"
        xAxis = "p_{T} of Next-to-Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt2" % (objtype)
    if plot_type == "EffEta":
        title = "#eta Efficiency"
        xAxis = "#eta of Generated %s " % (objtype)
        input_type = "gen%sEta" % (objtype)
    if plot_type == "EffPhi":
        title = "#phi Efficiency"
        xAxis = "#phi of Generated %s " % (objtype)
        input_type = "gen%sPhi" % (objtype)

    yAxis = "%s / %s" % (numer_description, denom_description)
    all_titles = "%s for trigger %s; %s; %s" % (title, triggerpath,
                                        xAxis, yAxis)
    return "Eff_%s_%s '%s' %s_%s %s" % (input_type,triggerpath,
                all_titles,input_type,triggerpath,input_type)

from HLTriggerOffline.SMP.hltSMPValidator_cfi import hltSMPValidator as _config
def make_smp_postprocessor(analysis_name, plot_types=["TurnOn1", "TurnOn2", "EffEta", "EffPhi"], object_types=["Mu","Ele","Photon","MET","PFMET","PFTau","Jet"], extra_str_templates=[]):
    postprocessor = hltSMPPostProcessor.clone()
    postprocessor.subDirs = ["HLT/SMP/" + analysis_name]
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


plot_types = ["TurnOn1", "TurnOn2", "EffEta", "EffPhi"]
#--- IMPORTANT: Update this collection whenever you introduce a new object
#               in the code (from EVTColContainer::getTypeString)
object_types  = ["Mu","Ele","Photon","MET","PFTau"]
truevtx_string_template = "Eff_trueVtxDist_@ANALYSIS@_gen_@TRIGGER@ ' Efficiency of @TRIGGER@ vs nb of interactions ; nb events passing each path ' trueVtxDist_@ANALYSIS@_gen_@TRIGGER@ trueVtxDist_@ANALYSIS@_gen"


hltSMPPostSingleEle = make_smp_postprocessor("SingleEle", plot_types=plot_types, object_types=object_types, extra_str_templates=[truevtx_string_template])
#hltSMPPostSingleMu = make_smp_postprocessor("SingleMu", plot_types=plot_types, object_types=object_types, extra_str_templates=[truevtx_string_template])
hltSMPPostSinglePhoton = make_smp_postprocessor("SinglePhoton", plot_types=plot_types, object_types=object_types, extra_str_templates=[truevtx_string_template])

# hltSMPPostSingleMu = hltSMPPostProcessor.clone()
# hltSMPPostSingleMu.subDirs = ['HLT/SMP/SingleMu']
# hltSMPPostSingleMu.efficiencyProfile = efficiency_strings

hltSMPPostProcessors = cms.Sequence(
    hltSMPPostSingleEle+
#    hltSMPPostSingleMu+
    hltSMPPostSinglePhoton
)


