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

plot_types = ["TurnOn1", "TurnOn2", "TurnOn3", "TurnOn4", "EffEta", "EffPhi", "EffDxy"]
#--- IMPORTANT: Update this collection whenever you introduce a new object
#               in the code (from EVTColContainer::getTypeString)
obj_types  = ["Mu","refittedStandAloneMuons","Track","Ele","Photon","PFTau","PFJet","MET","PFMET","PFMHT","GenMET","CaloJet"
             ,"CaloMET","CaloMHT","l1MET"]

#--- IMPORTANT: Trigger are extracted from the hltExoticaValidator_cfi.py module
from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import hltExoticaValidator as _config
#------------------------------------------------------------

def make_postprocessor(analysis_name):
    postprocessor = hltExoticaPostProcessor.clone()
    postprocessor.subDirs = ["HLT/Exotica/" + analysis_name]
    efficiency_strings = [] # List of plots to look for. This is quite a bit larger than the number of plots that will be made.
    for plot_type in plot_types:
        for object_type in obj_types:
            for trigger in [x.replace("_v", "") for x in _config.__getattribute__(analysis_name).hltPathsToCheck]:
                this_efficiency_string = make_efficiency_string(object_type, plot_type, trigger)
                efficiency_strings.append(this_efficiency_string)
                efficiency_strings.append(this_efficiency_string.replace("Generated", "Reconstructed").replace("Gen", "Reco").replace("gen", "rec"))
    postprocessor.efficiencyProfile = efficiency_strings
    return postprocessor

hltExoticaPostLowPtTrimuon = make_postprocessor("LowPtTrimuon")
hltExoticaPostHighPtDimuon = make_postprocessor("HighPtDimuon")
hltExoticaPostHighPtDielectron = make_postprocessor("HighPtDielectron")
hltExoticaPostHighPtElectron = make_postprocessor("HighPtElectron")
#hltExoticaPostLowPtElectron = make_postprocessor("LowPtElectron")
hltExoticaPostLowPtDimuon = make_postprocessor("LowPtDimuon")
hltExoticaPostLowPtDielectron = make_postprocessor("LowPtDielectron")
hltExoticaPostHighPtPhoton = make_postprocessor("HighPtPhoton")
hltExoticaPostDiPhoton = make_postprocessor("DiPhoton")
hltExoticaPostSingleMuon = make_postprocessor("SingleMuon")
hltExoticaPostPFHT = make_postprocessor("PFHT")
hltExoticaPostCaloHT = make_postprocessor("CaloHT")
hltExoticaPostJetNoBptx = make_postprocessor("JetNoBptx")
hltExoticaPostMuonNoBptx = make_postprocessor("MuonNoBptx")
hltExoticaPostDisplacedMuEG = make_postprocessor("DisplacedMuEG")
hltExoticaPostDisplacedDimuon = make_postprocessor("DisplacedDimuon")
hltExoticaPostMonojet = make_postprocessor("Monojet")
hltExoticaPostMonojetBackup = make_postprocessor("MonojetBackup")
hltExoticaPostPureMET = make_postprocessor("PureMET")
hltExoticaPostMETplusTrack = make_postprocessor("METplusTrack")
hltExoticaEleMu = make_postprocessor("EleMu")
hltExoticaPhotonMET = make_postprocessor("PhotonMET")
hltExoticaHTDisplacedJets = make_postprocessor("HTDisplacedJets")
hltExoticaDSTJets = make_postprocessor("DSTJets")
hltExoticaDSTMuons = make_postprocessor("DSTMuons")
hltExoticaTracklessJets = make_postprocessor("TracklessJets")

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
    #hltExoticaPostLowPtElectron +
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

