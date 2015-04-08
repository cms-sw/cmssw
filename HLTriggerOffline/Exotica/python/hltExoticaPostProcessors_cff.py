import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaPostProcessor_cfi import *

# Build the standard strings to the DQM
def efficiency_string(objtype,plot_type,triggerpath):
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

# Adding the reco objects
def add_reco_strings(strings):
    reco_strings = []
    for entry in strings:
        reco_strings.append(entry
                            .replace("Generated", "Reconstructed")
                            .replace("Gen", "Reco")
                            .replace("gen", "rec"))
    strings.extend(reco_strings)


plot_types = ["TurnOn1", "TurnOn2", "TurnOn3", "TurnOn4", "EffEta", "EffPhi", "EffDxy"]
#--- IMPORTANT: Update this collection whenever you introduce a new object
#               in the code (from EVTColContainer::getTypeString)
obj_types  = ["Mu","refittedStandAloneMuons","Track","Ele","Photon","PFTau","PFJet","MET","PFMET","PFMHT","GenMET","CaloJet"
             ,"CaloMET","l1MET"]
#--- IMPORTANT: Trigger are extracted from the hltExoticaValidator_cfi.py module
triggers = [ ] 
efficiency_strings = []

# Extract the triggers used in the hltExoticaValidator, for each path
from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import hltExoticaValidator as _config
triggers = set([])
for an in _config.analysis:
	s = _config.__getattribute__(an)
	vstr = s.__getattribute__("hltPathsToCheck")
	map(lambda x: triggers.add(x.replace("_v","")),vstr)
triggers = list(triggers)
print triggers
#------------------------------------------------------------

# Generating the list with all the efficiencies
for type in plot_types:
	for obj in obj_types:
		for trig in triggers:
			efficiency_strings.append(efficiency_string(obj,type,trig))
#for item in efficiency_strings:
#    print item

add_reco_strings(efficiency_strings)

#--- IMPORTANT: Here you have to add the analyses one by one.
hltExoticaPostLowPtTrimuon = hltExoticaPostProcessor.clone()
hltExoticaPostLowPtTrimuon.subDirs = ['HLT/Exotica/LowPtTrimuon']
hltExoticaPostLowPtTrimuon.efficiencyProfile = efficiency_strings

hltExoticaPostHighPtDimuon = hltExoticaPostProcessor.clone()
hltExoticaPostHighPtDimuon.subDirs = ['HLT/Exotica/HighPtDimuon']
hltExoticaPostHighPtDimuon.efficiencyProfile = efficiency_strings

hltExoticaPostHighPtDielectron = hltExoticaPostProcessor.clone()
hltExoticaPostHighPtDielectron.subDirs = ['HLT/Exotica/HighPtDielectron']
hltExoticaPostHighPtDielectron.efficiencyProfile = efficiency_strings

hltExoticaPostHighPtElectron = hltExoticaPostProcessor.clone()
hltExoticaPostHighPtElectron.subDirs = ['HLT/Exotica/HighPtElectron']
hltExoticaPostHighPtElectron.efficiencyProfile = efficiency_strings

hltExoticaPostLowPtElectron = hltExoticaPostProcessor.clone()
hltExoticaPostLowPtElectron.subDirs = ['HLT/Exotica/LowPtElectron']
hltExoticaPostLowPtElectron.efficiencyProfile = efficiency_strings

hltExoticaPostLowPtDimuon = hltExoticaPostProcessor.clone()
hltExoticaPostLowPtDimuon.subDirs = ['HLT/Exotica/LowPtDimuon']
hltExoticaPostLowPtDimuon.efficiencyProfile = efficiency_strings

hltExoticaPostLowPtDielectron = hltExoticaPostProcessor.clone()
hltExoticaPostLowPtDielectron.subDirs = ['HLT/Exotica/LowPtDielectron']
hltExoticaPostLowPtDielectron.efficiencyProfile = efficiency_strings

hltExoticaPostHighPtPhoton = hltExoticaPostProcessor.clone()
hltExoticaPostHighPtPhoton.subDirs = ['HLT/Exotica/HighPtPhoton']
hltExoticaPostHighPtPhoton.efficiencyProfile = efficiency_strings

hltExoticaPostDiPhoton = hltExoticaPostProcessor.clone()
hltExoticaPostDiPhoton.subDirs = ['HLT/Exotica/DiPhoton']
hltExoticaPostDiPhoton.efficiencyProfile = efficiency_strings

hltExoticaPostSingleMuon = hltExoticaPostProcessor.clone()
hltExoticaPostSingleMuon.subDirs = ['HLT/Exotica/SingleMuon']
hltExoticaPostSingleMuon.efficiencyProfile = efficiency_strings

hltExoticaPostHT = hltExoticaPostProcessor.clone()
hltExoticaPostHT.subDirs = ['HLT/Exotica/HT']
hltExoticaPostHT.efficiencyProfile = efficiency_strings

hltExoticaPostJetNoBptx = hltExoticaPostProcessor.clone()
hltExoticaPostJetNoBptx.subDirs = ['HLT/Exotica/JetNoBptx']
hltExoticaPostJetNoBptx.efficiencyProfile = efficiency_strings

hltExoticaPostMuonNoBptx = hltExoticaPostProcessor.clone()
hltExoticaPostMuonNoBptx.subDirs = ['HLT/Exotica/MuonNoBptx']
hltExoticaPostMuonNoBptx.efficiencyProfile = efficiency_strings

hltExoticaPostDisplacedMuEG = hltExoticaPostProcessor.clone()
hltExoticaPostDisplacedMuEG.subDirs = ['HLT/Exotica/DisplacedMuEG']
hltExoticaPostDisplacedMuEG.efficiencyProfile = efficiency_strings

hltExoticaPostDisplacedDimuon = hltExoticaPostProcessor.clone()
hltExoticaPostDisplacedDimuon.subDirs = ['HLT/Exotica/DisplacedDimuon']
hltExoticaPostDisplacedDimuon.efficiencyProfile = efficiency_strings

hltExoticaPostDisplacedL2Dimuon = hltExoticaPostProcessor.clone()
hltExoticaPostDisplacedL2Dimuon.subDirs = ['HLT/Exotica/DisplacedL2Dimuon']
hltExoticaPostDisplacedL2Dimuon.efficiencyProfile = efficiency_strings

hltExoticaPostMonojet = hltExoticaPostProcessor.clone()
hltExoticaPostMonojet.subDirs = ['HLT/Exotica/Monojet']
hltExoticaPostMonojet.efficiencyProfile = efficiency_strings

hltExoticaPostMonojetBackup = hltExoticaPostProcessor.clone()
hltExoticaPostMonojetBackup.subDirs = ['HLT/Exotica/MonojetBackup']
hltExoticaPostMonojetBackup.efficiencyProfile = efficiency_strings

hltExoticaPostPureMET = hltExoticaPostProcessor.clone()
hltExoticaPostPureMET.subDirs = ['HLT/Exotica/PureMET']
hltExoticaPostPureMET.efficiencyProfile = efficiency_strings

hltExoticaPostMETplusTrack = hltExoticaPostProcessor.clone()
hltExoticaPostMETplusTrack.subDirs = ['HLT/Exotica/METplusTrack']
hltExoticaPostMETplusTrack.efficiencyProfile = efficiency_strings

hltExoticaDisplacedDimuonDijet = hltExoticaPostProcessor.clone()
hltExoticaDisplacedDimuonDijet.subDirs = ['HLT/Exotica/DisplacedDimuonDijet']
hltExoticaDisplacedDimuonDijet.efficiencyProfile = efficiency_strings

hltExoticaEleMu = hltExoticaPostProcessor.clone()
hltExoticaEleMu.subDirs = ['HLT/Exotica/EleMu']
hltExoticaEleMu.efficiencyProfile = efficiency_strings

hltExoticaPhotonMET = hltExoticaPostProcessor.clone()
hltExoticaPhotonMET.subDirs = ['HLT/Exotica/PhotonMET']
hltExoticaPhotonMET.efficiencyProfile = efficiency_strings

hltExoticaHTDisplacedJets = hltExoticaPostProcessor.clone()
hltExoticaHTDisplacedJets.subDirs = ['HLT/Exotica/HTDisplacedJets']
hltExoticaHTDisplacedJets.efficiencyProfile = efficiency_strings

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
    hltExoticaPostHT +
    # NoBptx paths
    hltExoticaPostJetNoBptx +
    hltExoticaPostMuonNoBptx +
    # Displaced paths
    hltExoticaPostDisplacedMuEG +
    hltExoticaPostDisplacedDimuon +
    hltExoticaPostDisplacedL2Dimuon +
    # Others
    hltExoticaPostMonojet +
    hltExoticaPostMonojetBackup +
    hltExoticaPostPureMET +
    hltExoticaPostMETplusTrack +
    hltExoticaDisplacedDimuonDijet +
    hltExoticaEleMu +
    hltExoticaPhotonMET +
    hltExoticaHTDisplacedJets
    )
