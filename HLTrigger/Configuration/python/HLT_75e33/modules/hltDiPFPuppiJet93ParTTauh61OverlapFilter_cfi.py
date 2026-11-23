import FWCore.ParameterSet.Config as cms

# The two VBF jets must be at dR > 0.5 from the tau jet.
# saveTags must be True: HLTDoubletSinglet skips the selection otherwise.
hltDiPFPuppiJet93ParTTauh61OverlapFilter = cms.EDFilter("HLT3TriplePFJet",
    MaxDeta = cms.double(-1.0),
    MaxDelR = cms.double(9999.0),
    MaxDphi = cms.double(-1.0),
    MaxMinv = cms.double(-1.0),
    MaxPt = cms.double(-1.0),
    MinDeta = cms.double(1.0),
    MinDelR = cms.double(0.5),
    MinDphi = cms.double(1.0),
    MinMinv = cms.double(1.0),
    MinN = cms.int32(1),
    MinPt = cms.double(1.0),
    inputTag1 = cms.InputTag("hltPFPuppiJetCategoriesVBF93Mjj600"),
    inputTag2 = cms.InputTag("hltPFPuppiJetCategoriesVBF93Mjj600"),
    inputTag3 = cms.InputTag("hltSinglePFPuppiJet61ParTTauhTagMediumWP"),
    originTag1 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    originTag2 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    originTag3 = cms.VInputTag("hltAK4PFPuppiJets"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(85),
    triggerType2 = cms.int32(85),
    triggerType3 = cms.int32(84)
)
