import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaPostProcessor_cfi import *

# Build the standard strings to the DQM
def make_efficiency_string(objtype, plot_type, triggerpath):
    # --- IMPORTANT: Add here a elif if you are introduce a new collection
    #                (see EVTColContainer::getTypeString) 
    if objtype == "Mu" :
        objtypeLatex="#mu"
    elif objtype == "refittedStandAloneMuons": 
        objtypeLatex="refittedStandAlone #mu"
    elif objtype == "Ele": 
        objtypeLatex="e"
    elif objtype == "Photon": 
        objtypeLatex="#gamma"
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
    if plot_type == "TurnOn3":
        title = "Next-to-next-to-Leading pT Turn-On"
        xAxis = "p_{T} of Next-to-next-to-Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sMaxPt3" % (objtype)
    if plot_type == "TurnOn4":
        title = "SumEt Turn-On"
        xAxis = "SumEt of Leading Generated %s (GeV/c)" % (objtype)
        input_type = "gen%sSumEt" % (objtype)
    if plot_type == "EffEta":
        title = "#eta Efficiency"
        xAxis = "#eta of Generated %s " % (objtype)
        input_type = "gen%sEta" % (objtype)
    if plot_type == "EffPhi":
        title = "#phi Efficiency"
        xAxis = "#phi of Generated %s " % (objtype)
        input_type = "gen%sPhi" % (objtype)
    if plot_type == "EffDxy":
        title = "Dxy Efficiency"
        xAxis = "Dxy of Generated %s " % (objtype)
        input_type = "gen%sDxy" % (objtype)

    yAxis = "%s / %s" % (numer_description, denom_description)
    all_titles = "%s for trigger %s; %s; %s" % (title, triggerpath,
                                        xAxis, yAxis)
    return "Eff_%s_%s '%s' %s_%s %s" % (input_type,triggerpath,
                    all_titles,input_type,triggerpath,input_type)


#--- IMPORTANT: Trigger are extracted from the hltExoticaValidator_cfi.py module
from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import hltExoticaValidator as _config
#------------------------------------------------------------
#--- IMPORTANT: Update this collection whenever you introduce a new object
#               in the code (from EVTColContainer::getTypeString)
def make_exo_postprocessor(analysis_name, plot_types=["TurnOn1", "TurnOn2", "TurnOn3", "TurnOn4", "EffEta", "EffPhi", "EffDxy"], object_types=["Mu","refittedStandAloneMuons","Track","Ele","Photon","PFTau","PFJet","MET","PFMET","PFMHT","GenMET","CaloJet","CaloMET","CaloMHT","l1MET"], extra_str_templates=[]):
    postprocessor = hltExoticaPostProcessor.clone()
    postprocessor.subDirs = ["HLT/Exotica/" + analysis_name]
    efficiency_strings = [] # List of plots to look for. This is quite a bit larger than the number of plots that will be made.

    # Higgs and SMP postprocessors use this string, but exo does not, for now.
    #efficiency_summary_string = "EffSummaryPaths_" + analysis_name + "_gen ' Efficiency of paths used in " + analysis_name + " ; trigger path ' SummaryPaths_" + analysis_name + "_gen_passingHLT SummaryPaths_" + analysis_name + "_gen"
    #efficiency_strings.append(efficiency_summary_string)
    #efficiency_strings.append(efficiency_summary_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))

    for plot_type in plot_types:
        for object_type in object_types:
            for trigger in [x.replace("_v", "") for x in _config.__getattribute__(analysis_name).hltPathsToCheck]:
                this_efficiency_string = make_efficiency_string(object_type, plot_type, trigger)
                efficiency_strings.append(this_efficiency_string)
                efficiency_strings.append(this_efficiency_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))

                for str_template in extra_str_templates:
                    this_extra_string = str_template.replace("@ANALYSIS@", analysis_name).replace("@TRIGGER@", trigger)
                    efficiency_strings.append(this_extra_string)
                    efficiency_strings.append(this_extra_stringreplace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))
    postprocessor.efficiencyProfile = efficiency_strings
    return postprocessor

hltExoticaPostLowPtTrimuon = make_exo_postprocessor("LowPtTrimuon")
hltExoticaPostHighPtDimuon = make_exo_postprocessor("HighPtDimuon")
hltExoticaPostHighPtDielectron = make_exo_postprocessor("HighPtDielectron")
hltExoticaPostHighPtElectron = make_exo_postprocessor("HighPtElectron")
hltExoticaPostLowPtElectron = make_exo_postprocessor("LowPtElectron")
hltExoticaPostLowPtDimuon = make_exo_postprocessor("LowPtDimuon")
hltExoticaPostLowPtDielectron = make_exo_postprocessor("LowPtDielectron")
hltExoticaPostHighPtPhoton = make_exo_postprocessor("HighPtPhoton")
hltExoticaPostDiPhoton = make_exo_postprocessor("DiPhoton")
hltExoticaPostSingleMuon = make_exo_postprocessor("SingleMuon")
hltExoticaPostPFHT = make_exo_postprocessor("PFHT")
hltExoticaPostCaloHT = make_exo_postprocessor("CaloHT")
hltExoticaPostJetNoBptx = make_exo_postprocessor("JetNoBptx")
hltExoticaPostMuonNoBptx = make_exo_postprocessor("MuonNoBptx")
hltExoticaPostDisplacedMuEG = make_exo_postprocessor("DisplacedMuEG")
hltExoticaPostDisplacedDimuon = make_exo_postprocessor("DisplacedDimuon")
hltExoticaPostMonojet = make_exo_postprocessor("Monojet")
hltExoticaPostMonojetBackup = make_exo_postprocessor("MonojetBackup")
hltExoticaPostPureMET = make_exo_postprocessor("PureMET")
hltExoticaPostMETplusTrack = make_exo_postprocessor("METplusTrack")
hltExoticaEleMu = make_exo_postprocessor("EleMu")
hltExoticaPhotonMET = make_exo_postprocessor("PhotonMET")
hltExoticaHTDisplacedJets = make_exo_postprocessor("HTDisplacedJets")
hltExoticaDSTJets = make_exo_postprocessor("DSTJets")
hltExoticaDSTMuons = make_exo_postprocessor("DSTMuons")
hltExoticaTracklessJets = make_exo_postprocessor("TracklessJets")

hltExoticaPostProcessors = cms.Sequence(
    # Tri-lepton paths
    hltExoticaPostLowPtTrimuon +
    # Di-lepton paths
    hltExoticaPostHighPtDimuon +
    hltExoticaPostHighPtDielectron +
    hltExoticaPostLowPtDimuon +
    hltExoticaPostLowPtDielectron +
    # Single Lepton paths
    hltExoticaPostHighPtElectron +
    hltExoticaPostLowPtElectron +
    # Photon paths
    hltExoticaPostHighPtPhoton +
    hltExoticaPostDiPhoton +
    # HT path
    hltExoticaPostPFHT +
    hltExoticaPostCaloHT +
    # NoBptx paths
    hltExoticaPostJetNoBptx +
    hltExoticaPostMuonNoBptx +
    # Displaced paths
    hltExoticaPostDisplacedMuEG +
    hltExoticaPostDisplacedDimuon +
    # Others
    hltExoticaPostMonojet +
    hltExoticaPostMonojetBackup +
    hltExoticaPostPureMET +
    hltExoticaPostMETplusTrack +
    hltExoticaEleMu +
    hltExoticaPhotonMET +
    hltExoticaHTDisplacedJets +
    hltExoticaTracklessJets +
    # scouting triggers
    hltExoticaDSTJets +
    hltExoticaDSTMuons 
    )


    #for analysis in _config.analyses:
#    hltExoticaPostProcessors *= analysis_postprocessors[analysis]

