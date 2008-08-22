import FWCore.ParameterSet.Config as cms

highEtDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedSingle"), cms.InputTag("hltL1NonIsoSingleEMHighEtL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoSinglePhotonEMHighEtEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtEcalIsolFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtHOEFilter"), 
        cms.InputTag("hltL1NonIsoSingleEMHighEtHcalDBCFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100, 100, 100, 
        100, 100),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(1000.0),
    pdgGen = cms.int32(11)
)



