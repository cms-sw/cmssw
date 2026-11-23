import FWCore.ParameterSet.Config as cms

# The two VBF jets must be at dR > 0.5 from the photon.
# saveTags must stay True: the plugin only loops over the triplets when it is set.
hltDiPFPuppiJet70Photon22OverlapFilter = cms.EDFilter("HLT3DoublePFJetPhoton",
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
    inputTag1 = cms.InputTag("hltPFPuppiJetCategoriesVBF1000"),
    inputTag2 = cms.InputTag("hltPFPuppiJetCategoriesVBF1000"),
    inputTag3 = cms.InputTag("hltPhoton22VBFHEL1SeededFilter"),
    originTag1 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    originTag2 = cms.VInputTag("hltAK4PFPuppiJetsCorrected"),
    originTag3 = cms.VInputTag("hltEgammaHoverEL1Seeded"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(85),
    triggerType2 = cms.int32(85),
    triggerType3 = cms.int32(81)
)
