import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cfi import hltMuonOfflineAnalyzer

barrelMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoMaxEtaCut = cms.untracked.double(0.9),
    recoMinEtaCut = cms.untracked.double(0.0),
    recoGlbMuCut = cms.untracked.bool(False), #is a SA muon
    hltMaxEtaCut  = cms.untracked.double(0.9),
    hltMinEtaCut  = cms.untracked.double(0.0),
)

endcapMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoMaxEtaCut = cms.untracked.double(2.0),
    recoMinEtaCut = cms.untracked.double(1.4),
    recoGlbMuCut = cms.untracked.bool(False), #is a SA muon
    hltMaxEtaCut  = cms.untracked.double(2.0),
    hltMinEtaCut  = cms.untracked.double(1.4),
)

allMuonParams = cms.PSet(
    d0Cut = cms.untracked.double(1000.0),
    z0Cut = cms.untracked.double(1000.0),
    recoMaxEtaCut = cms.untracked.double(2.0),
    recoMinEtaCut = cms.untracked.double(0.0),
    recoGlbMuCut = cms.untracked.bool(False), #is a SA muon
    hltMaxEtaCut  = cms.untracked.double(2.0),
    hltMinEtaCut  = cms.untracked.double(0.0),
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
