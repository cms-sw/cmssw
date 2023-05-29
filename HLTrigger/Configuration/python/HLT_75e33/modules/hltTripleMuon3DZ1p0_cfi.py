import FWCore.ParameterSet.Config as cms

hltTripleMuon3DZ1p0 = cms.EDFilter("HLT2L1TkMuonL1TkMuonDZ",
    MaxDZ = cms.double(1.0),
    MinDR = cms.double(-1),
    MinN = cms.int32(3),
    inputTag1 = cms.InputTag("hltL1TripleMuFiltered3"),
    inputTag2 = cms.InputTag("hltL1TripleMuFiltered3"),
    originTag1 = cms.VInputTag("l1tTkMuonsGmt"),
    originTag2 = cms.VInputTag("l1tTkMuonsGmt"),
    saveTags = cms.bool(True),
    triggerType1 = cms.int32(-114),
    triggerType2 = cms.int32(-114)
)
