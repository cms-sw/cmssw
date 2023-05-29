import FWCore.ParameterSet.Config as cms

l1tDoublePFPuppiJets112offMaxDeta1p6 = cms.EDFilter("HLT2CaloJetCaloJet",
    MaxDelR = cms.double(1000.0),
    MaxDeta = cms.double(1.6),
    MaxDphi = cms.double(10000000.0),
    MaxMinv = cms.double(10000000.0),
    MaxPt = cms.double(10000000.0),
    MinDelR = cms.double(0.0),
    MinDeta = cms.double(-1000.0),
    MinDphi = cms.double(0.0),
    MinMinv = cms.double(0.0),
    MinN = cms.int32(1),
    MinPt = cms.double(0.0),
    inputTag1 = cms.InputTag("l1tDoublePFPuppiJet112offMaxEta2p4"),
    inputTag2 = cms.InputTag("l1tDoublePFPuppiJet112offMaxEta2p4"),
    originTag1 = cms.VInputTag(cms.InputTag("l1tPhase1JetCalibrator9x9trimmed","Phase1L1TJetFromPfCandidates")),
    originTag2 = cms.VInputTag(cms.InputTag("l1tPhase1JetCalibrator9x9trimmed","Phase1L1TJetFromPfCandidates")),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(-116),
    triggerType2 = cms.int32(-116)
)
