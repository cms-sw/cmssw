def hltlayout(i, p, *rows): i["00 Shift/HLT/" + p] = DQMItem(layout=rows)
  

#hltlayout(dqmitems,"01 HLT Stream A Composition", [{'path': "HLT/TrigResults/PathsSummary/HLT LS/HLT_A_LS", 'description': "Shows total rate of Stream A (top Y bin) and the PD's in stream A. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])
#hltlayout(dqmitems,"02 HLT Stream A Composition", [{'path': "HLT/TrigResults/PathsSummary/HLT Correlations/HLT_A_Pass_Normalized_Any", 'description': "Shows relative fraction of the PD's in stream A. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])


def hltlayout(i, p, *rows): i["00 Shift/HLT/Cosmics/" + p] = DQMItem(layout=rows)

  
hltlayout(dqmitems,"01 HLT_Commissioning_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_Commissioning_Pass_Any", 'description': "Shows total number of HLT Commissioning trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"02 HLT_Cosmics_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_Cosmics_Pass_Any", 'description': "Shows total number of HLT Cosmics trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"03 HLT_ForwardTriggers_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_ForwardTriggers_Pass_Any", 'description': "Shows total number of HLT ForwardTriggers trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"04 HLT_HcalHPDNoise_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_HcalHPDNoise_Pass_Any", 'description': "Shows total number of HLT HcalHPDNoise trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"05 HLT_HcalNZS_Pass_Any",
    [{'path': "HLT/FourVector/PathsSummary/HLT_HcalNZS_Pass_Any", 'description': "Shows total number of HLT HcalNZS trigger accepts and the total number of any HLT accepts. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])



def hltlayout(i, p, *rows): i["00 Shift/HLT/Collisions/" + p] = DQMItem(layout=rows)
  

# slaunwhj -- updated March 9 2011 for new PDs
# hdyoo -- update May 22 for new PDs (remove forward/METBtag, add MET, BTag)
# slaunwhj -- update June 30 change to xsec plots

#hltlayout(dqmitems,"01 HLT_BTag_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_BTag_Xsec", 'description': "Shows total number of BTag PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"02 HLT_Commissioning_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Commissioning_Xsec", 'description': "Shows total number of Commissioning PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"03 HLT_Cosmics_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Cosmics_Xsec", 'description': "Shows total number of Cosmics PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"04 HLT_DoubleElectron_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_DoubleElectron_Xsec", 'description': "Shows total number of DoubleElectron PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"05 HLT_DoubleMu_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_DoubleMu_Xsec", 'description': "Shows total number of DoubleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"06 HLT_ElectronHad_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_ElectronHad_Xsec", 'description': "Shows total number of ElectronHad PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])


#hltlayout(dqmitems,"07 HLT_HT_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_HT_Xsec", 'description': "Shows total number of HT PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"08 HLT_HcalHPDNoise_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_HcalHPDNoise_Xsec", 'description': "Shows total number of HcalHPDNoise PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"09 HLT_HcalNZS_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_HcalNZS_Xsec", 'description': "Shows total number of HcalNZS PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"01 HLT_Jet_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Jet_Xsec", 'description': "Shows total number of Jet PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"11 HLT_MET_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MET_Xsec", 'description': "Shows total number of MET PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"12 HLT_MinimumBias_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MinimumBias_Xsec", 'description': "Shows total number of MinimumBias PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"13 HLT_MuEG_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MuEG_Xsec", 'description': "Shows total number of MuEG PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"14 HLT_MuHad_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MuHad_Xsec", 'description': "Shows total number of MuHad PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"15 HLT_MuOnia_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MuOnia_Xsec", 'description': "Shows total number of MuOnia PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"16 HLT_MultiJet_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_MultiJet_Xsec", 'description': "Shows total number of MultiJet PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"17 HLT_Photon_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Photon_Xsec", 'description': "Shows total number of Photon PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"18 HLT_PhotonHad_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_PhotonHad_Xsec", 'description': "Shows total number of PhotonHad PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"02 HLT_SingleElectron_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_SingleElectron_Xsec", 'description': "Shows total number of SingleElectron PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"03 HLT_SingleMu_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_SingleMu_Xsec", 'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])


#hltlayout(dqmitems,"21 HLT_Tau_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_Tau_Xsec", 'description': "Shows total number of Tau PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

#hltlayout(dqmitems,"22 HLT_TauPlusX_Xsec", [{'path': "HLT/TrigResults/PathsSummary/HLT Counts/HLT_TauPlusX_Xsec", 'description': "Shows total number of TauPlusX PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])

hltlayout(dqmitems,"04 HLT_Jet_Occupancy", [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_Jet_EtaVsPhiFine", 'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])
hltlayout(dqmitems,"05 HLT_SingleElectron_Occupancy", [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_SingleElectron_EtaVsPhiFine", 'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])
hltlayout(dqmitems,"06 HLT_SingleMu_Occupancy", [{'path': "HLT/OccupancyPlots/PD_EtaVsPhi/HLT_SingleMu_EtaVsPhiFine", 'description': "Shows total number of SingleMu PD accepts and the total number of any HLT accepts in this PD. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/TriggerShiftHLTGuide\">here</a>."}])
