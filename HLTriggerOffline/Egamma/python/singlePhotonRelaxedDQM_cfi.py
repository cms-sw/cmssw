import FWCore.ParameterSet.Config as cms

singlePhotonRelaxedDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedSingle"), cms.InputTag("hltL1NonIsoSinglePhotonL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoSinglePhotonEtFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter"), 
        cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100, 100, 100, 
        100),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(22)
)



