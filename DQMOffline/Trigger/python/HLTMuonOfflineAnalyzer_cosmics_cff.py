import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cfi import hltMuonOfflineAnalyzer

barrelMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoCuts = cms.untracked.string("isStandAloneMuon && abs(eta) < 0.9"),
    hltCuts  = cms.untracked.string("abs(eta) < 0.9"),
)

endcapMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoCuts = cms.untracked.string("isStandAloneMuon && abs(eta) > 1.4 && "
                                    "abs(eta) < 2.0"),
    hltCuts  = cms.untracked.string("abs(eta) > 1.4 && abs(eta) < 2.0"),
)

allMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoCuts = cms.untracked.string("isStandAloneMuon && abs(eta) < 2.0"),
    hltCuts  = cms.untracked.string("abs(eta) < 2.0"),
)

barrelAnalyzer = hltMuonOfflineAnalyzer.clone(
    destination = "HLT/Muon/DistributionsBarrel",
    targetParams = barrelMuonParams
)
barrelAnalyzer.probeParams = cms.PSet()

endcapAnalyzer = hltMuonOfflineAnalyzer.clone(
    destination = "HLT/Muon/DistributionsEndcap",
    targetParams = endcapMuonParams
)
endcapAnalyzer.probeParams = cms.PSet()

allAnalyzer = hltMuonOfflineAnalyzer.clone(
    destination = "HLT/Muon/DistributionsAll",
    targetParams = allMuonParams,
    probeParams = allMuonParams
)
hltMuonOfflineAnalyzers = cms.Sequence(
    barrelAnalyzer *
    endcapAnalyzer *
    allAnalyzer
)
