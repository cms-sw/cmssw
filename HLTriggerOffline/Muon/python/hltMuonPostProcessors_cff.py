import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Muon.hltMuonPostProcessor_cfi import *

def efficiency_string(numer_label, denom_label,
                      plot_type, output_label=None):
    
    if output_label is None:
        output_label = numer_label
    if denom_label == "All" and numer_label != "L1":
        output_description = "Full Path"
    else:
        output_description = numer_label
    if denom_label == "All":
        denom_description = "# Gen #mu"
    else:
        denom_description = "# Gen #mu Matched to %s" % (denom_label)
    numer_description     = "# Gen #mu Matched to %s" % (numer_label)

    if plot_type == "TurnOn1":
        title = "pT Turn-On"
        xAxis = "p_{T} of Leading Generated Muon (GeV)"
        input_type = "PassMaxPt1"
    if plot_type == "TurnOn2":
        title = "Next-to-Leading pT Turn-On"
        xAxis = "p_{T} of Next-to-Leading Generated Muon (GeV)"
        input_type = "PassMaxPt2"
    if plot_type == "EffEta":
        title = "#eta Efficiency"
        xAxis = "#eta of Generated Muon (GeV)"
        input_type = "PassEta"
    if plot_type == "EffPhi":
        title = "#phi Efficiency"
        xAxis = "#phi of Generated Muon (GeV)"
        input_type = "PassPhi"

    yAxis = "%s / %s" % (numer_description, denom_description)
    all_titles = "%s for %s; %s; %s" % (title, output_description,
                                        xAxis, yAxis)
    return "gen%s_%s '%s' gen%s_%s gen%s_%s" % (plot_type, output_label,
                                                all_titles, input_type,
                                                numer_label, input_type,
                                                denom_label)


def add_reco_strings(strings):
    reco_strings = []
    for entry in strings:
        reco_strings.append(entry
                            .replace("Generated", "Reconstructed")
                            .replace("Gen", "Reco")
                            .replace("gen", "rec"))
    strings.extend(reco_strings)


plot_types = ["TurnOn1", "TurnOn2", "EffEta", "EffPhi"] 
efficiency_strings = []
iso_strings = []
noniso_strings = []

for type in plot_types:
    efficiency_strings.append(efficiency_string("L1", "All", type))
    for step in ["L2", "L2Iso", "L3", "Tk", "L3EcalIso", "L3HcalIso", "L3TkIso"]:
        efficiency_strings.append(efficiency_string(step, "L1", type))
    noniso_strings.append(efficiency_string("L3", "All", type, "Total"))
    iso_strings.append(efficiency_string("L3EcalIso", "All", type, "Total"))
    iso_strings.append(efficiency_string("L3HcalIso", "All", type, "Total"))
    iso_strings.append(efficiency_string("L3TkIso", "All", type, "Total"))

add_reco_strings(efficiency_strings)
add_reco_strings(noniso_strings)
add_reco_strings(iso_strings)

hltMuonPostMain = hltMuonPostProcessor.clone()
hltMuonPostMain.subDirs = ['HLT/Muon/Distributions/*']
hltMuonPostMain.efficiencyProfile = efficiency_strings

hltMuonPostNonIso = hltMuonPostMain.clone()
#hltMuonPostNonIso.subDirs = ['HLT/Muon/Distributions/^(?:[^I]+|I(?!so))*$']
hltMuonPostNonIso.subDirs = ['HLT/Muon/Distributions/((?!Iso).)*$']
hltMuonPostNonIso.efficiencyProfile = noniso_strings

hltMuonPostIso = hltMuonPostMain.clone()
hltMuonPostIso.subDirs = ['HLT/Muon/Distributions/.*Iso.*$']
hltMuonPostIso.efficiencyProfile = iso_strings

hltMuonPostProcessors = cms.Sequence(
    hltMuonPostNonIso *
    hltMuonPostIso    *
    hltMuonPostMain
)
