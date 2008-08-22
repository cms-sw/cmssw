import FWCore.ParameterSet.Config as cms

doubleElectronRelaxedDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedRelaxedDouble"), cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional"), cms.InputTag("hltL1NonIsoDoubleElectronEtFilter"), cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter"), 
        cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(82, 100, 100, 100, 100, 
        92, 92),
    reqNum = cms.uint32(2),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)



