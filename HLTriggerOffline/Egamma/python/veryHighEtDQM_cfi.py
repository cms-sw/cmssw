import FWCore.ParameterSet.Config as cms

veryHighEtDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedSingle"), cms.InputTag("hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(4000.0),
    pdgGen = cms.int32(11)
)



