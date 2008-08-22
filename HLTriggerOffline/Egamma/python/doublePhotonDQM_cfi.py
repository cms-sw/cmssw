import FWCore.ParameterSet.Config as cms

doublePhotonDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedDouble"), cms.InputTag("hltL1IsoDoublePhotonL1MatchFilterRegional"), cms.InputTag("hltL1IsoDoublePhotonEtFilter"), cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter"), 
        cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(83, 100, 100, 100, 100, 
        100, 100),
    reqNum = cms.uint32(2),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(22)
)



