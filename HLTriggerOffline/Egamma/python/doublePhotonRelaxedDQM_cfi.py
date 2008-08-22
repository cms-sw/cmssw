import FWCore.ParameterSet.Config as cms

doublePhotonRelaxedDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedDouble"), cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoDoublePhotonEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter"), cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter"), 
        cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100, 100, 100, 
        100, 100),
    reqNum = cms.uint32(2),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(22)
)



