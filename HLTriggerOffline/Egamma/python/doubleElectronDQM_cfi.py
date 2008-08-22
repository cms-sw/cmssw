import FWCore.ParameterSet.Config as cms

doubleElectronDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedDouble"), cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional"), cms.InputTag("hltL1IsoDoubleElectronEtFilter"), cms.InputTag("hltL1IsoDoubleElectronHcalIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronPixelMatchFilter"), 
        cms.InputTag("hltL1IsoDoubleElectronEoverpFilter"), cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(83, 100, 100, 100, 100, 
        92, 92),
    reqNum = cms.uint32(2),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)



