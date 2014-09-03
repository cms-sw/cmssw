

###---- GENERIC - FourVector selection goes here: ####
######################################################

###---- GENERIC - FourVector Muon
def trigHltMuonOfflineSummaryNew(i, p, *rows):
    i["HLT/Layouts/" + p] = DQMItem(layout=rows)

## ## Layout for MuMonitor sample
## paths = ["HLT_L1Mu7_v1", "HLT_L2Mu7_v1", "HLT_Mu3_v2",
##          "HLT_L1DoubleMuOpen", "HLT_L2DoubleMu0", "HLT_DoubleMu0"]
## for i, path in enumerate(paths):
##     trigHltMuonOfflineSummaryNew(dqmitems, "Muon (MuMonitor)/%02d - %s" % (i, path),
##         [{'path': "HLT/Muon/DistributionsVbtf/%s/efficiencyTurnOn" % path, 'description':"Efficiency for VBTF muons to match HLT"}],
##         [{'path': "HLT/Muon/DistributionsVbtf/%s/efficiencyPhiVsEta" % path, 'description':"Efficiency for VBTF muons to match HLT"}])

## Layout for SingleMu sample
trigHltMuonOfflineSummaryNew(dqmitems, "Muon (SingleMu)/Tag and Probe",
    [{'path': "HLT/Muon/DistributionsVbtf/HLT_Mu15_v1/massVsEta_efficiency", 'description':"Tag and probe"}])
paths = ["HLT_Mu24_v2", "HLT_Mu30_v2", "HLT_IsoMu17_v6"]
for i, path in enumerate(paths):
    trigHltMuonOfflineSummaryNew(dqmitems, "Muon (Mu)/%02d - %s" % (i, path),
        [{'path': "HLT/Muon/DistributionsVbtf/%s/efficiencyTurnOn" % path, 'description':"Efficiency for VBTF muons to match HLT"}],
        [{'path': "HLT/Muon/DistributionsVbtf/%s/efficiencyPhiVsEta" % path, 'description':"Efficiency for VBTF muons to match HLT"}])
                  

###---- GENERIC - FourVector Muon
def trigHltMuonOfflineSummary(i, p, *rows): 
   i["HLT/Muon/MuonHLTSummary/" + p] = DQMItem(layout=rows)


######################################################

trigHltMuonOfflineSummary(dqmitems,"01 - HLT_Mu3_v4",
   [{'path': "HLT/Muon/Distributions/HLT_Mu3_v4/allMuons/recEffPt_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_Mu3_v4/allMuons/recEffPhiVsEta_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

######################################################

trigHltMuonOfflineSummary(dqmitems,"02 - HLT_Mu5_v4",
   [{'path': "HLT/Muon/Distributions/HLT_Mu5_v4/allMuons/recEffPt_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_Mu5_v4/allMuons/recEffPhiVsEta_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

######################################################

trigHltMuonOfflineSummary(dqmitems,"03 - HLT_Mu12_v2",
   [{'path': "HLT/Muon/Distributions/HLT_Mu12_v2/allMuons/recEffPt_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_Mu12_v2/allMuons/recEffPhiVsEta_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

######################################################

trigHltMuonOfflineSummary(dqmitems,"04 - HLT_Mu30_v2",
   [{'path': "HLT/Muon/Distributions/HLT_Mu30_v2/allMuons/recEffPt_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_Mu30_v2/allMuons/recEffPhiVsEta_L3Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

######################################################

trigHltMuonOfflineSummary(dqmitems,"05 - HLT_L2Mu10_v2",
   [{'path': "HLT/Muon/Distributions/HLT_L2Mu10_v2/allMuons/recEffPt_L2Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_L2Mu10_v2/allMuons/recEffPhiVsEta_L2Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

######################################################

trigHltMuonOfflineSummary(dqmitems,"06 - HLT_L2Mu20_v2",
   [{'path': "HLT/Muon/Distributions/HLT_L2Mu20_v2/allMuons/recEffPt_L2Filtered", 'description':"Efficiency for RECO muons to match HLT"}],
   [{'path': "HLT/Muon/Distributions/HLT_L2Mu20_v2/allMuons/recEffPhiVsEta_L2Filtered", 'description':"Efficiency for RECO muons to match HLT"}])

