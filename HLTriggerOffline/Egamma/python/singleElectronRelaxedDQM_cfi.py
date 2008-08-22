import FWCore.ParameterSet.Config as cms

singleElectronRelaxedDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedSingle"), cms.InputTag("hltL1NonIsoSingleElectronL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoSingleElectronEtFilter"), cms.InputTag("hltL1NonIsoSingleElectronHcalIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronPixelMatchFilter"), 
        cms.InputTag("hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100, 100, 100, 
        92, 92),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)



