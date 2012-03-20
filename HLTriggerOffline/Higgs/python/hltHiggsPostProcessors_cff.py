import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Higgs.hltHiggsPostProcessor_cfi import *

# Build the standard strings to the DQM
def efficiency_string(objtype,plot_type,triggerpath):
    if objtype == "Mu" :
	objtypeLatex="#mu"
    elif objtype == "Photon": 
	objtypeLatex="#gamma"
    elif objtype == "Ele": 
	objtypeLatex="e"

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

# Adding the reco 
def add_reco_strings(strings):
    reco_strings = []
    for entry in strings:
        reco_strings.append(entry
                            .replace("Generated", "Reconstructed")
                            .replace("Gen", "Reco")
                            .replace("gen", "rec"))
    strings.extend(reco_strings)


plot_types = ["TurnOn1", "TurnOn2", "EffEta", "EffPhi"]
obj_types  = ["Mu","Ele","Photon"]
triggers = [ "HLT_Photon26_Photon18", 
		"HLT_Photon36_Photon22",
		"HLT_Mu17_Mu8",
		"HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL",
		"HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL",
		"HLT_Ele17_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_Ele0_Mass50",
		]
efficiency_strings = []

for type in plot_types:
    for obj in obj_types:
	for trig in triggers:
	    efficiency_strings.append(efficiency_string(obj,type,trig))

add_reco_strings(efficiency_strings)

hltHiggsPostHWW = hltHiggsPostProcessor.clone()
hltHiggsPostHWW.subDirs = ['HLT/Higgs/HWW']
hltHiggsPostHWW.efficiencyProfile = efficiency_strings

hltHiggsPostHgg = hltHiggsPostProcessor.clone()
hltHiggsPostHgg.subDirs = ['HLT/Higgs/Hgg']
hltHiggsPostHgg.efficiencyProfile = efficiency_strings

hltHiggsPostProcessors = cms.Sequence(
    hltHiggsPostHWW+
    hltHiggsPostHgg
)
