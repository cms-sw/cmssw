import FWCore.ParameterSet.Config as cms

singlePhotonDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedSingle"), cms.InputTag("hltL1IsoSinglePhotonL1MatchFilter"), cms.InputTag("hltL1IsoSinglePhotonEtFilter"), cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter"), 
        cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(83, 100, 100, 100, 100, 
        100),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(22)
)



