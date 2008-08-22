import FWCore.ParameterSet.Config as cms

singleElectronDQM = cms.EDFilter("EmDQM",
    genEtaAcc = cms.double(2.5),
    HLTCollectionLabels = cms.VInputTag(cms.InputTag("l1seedSingle"), cms.InputTag("hltL1IsoSingleL1MatchFilter"), cms.InputTag("hltL1IsoSingleElectronEtFilter"), cms.InputTag("hltL1IsoSingleElectronHcalIsolFilter"), cms.InputTag("hltL1IsoSingleElectronPixelMatchFilter"), 
        cms.InputTag("hltL1IsoSingleElectronHOneOEMinusOneOPFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter")),
    genEtAcc = cms.double(2.0),
    theHLTOutputTypes = cms.vint32(83, 100, 100, 100, 100, 
        92, 92),
    reqNum = cms.uint32(1),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11)
)



