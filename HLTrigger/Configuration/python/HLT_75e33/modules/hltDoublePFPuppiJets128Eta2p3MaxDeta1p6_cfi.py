import FWCore.ParameterSet.Config as cms

hltDoublePFPuppiJets128Eta2p3MaxDeta1p6 = cms.EDFilter("HLT2PFJetPFJet",
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
    inputTag1 = cms.InputTag("hltDoublePFPuppiJets128MaxEta2p4"),
    inputTag2 = cms.InputTag("hltDoublePFPuppiJets128MaxEta2p4"),
    originTag1 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    originTag2 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(86),
    triggerType2 = cms.int32(86)
)
