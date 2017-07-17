import FWCore.ParameterSet.Config as cms

WZInterestingEventSelector = cms.EDFilter(
    "WZInterestingEventSelector",
    electronCollection = cms.untracked.InputTag('gedGsfElectrons'),
    pfMetCollection = cms.untracked.InputTag('pfMet'),
    offlineBSCollection = cms.untracked.InputTag('offlineBeamSpot'),
    ptCut = cms.double(20.),
    missHitsCut = cms.int32(1),
    eb_trIsoCut = cms.double(0.1),
    eb_ecalIsoCut = cms.double(0.1),
    eb_hcalIsoCut = cms.double(0.1),
    eb_hoeCut = cms.double(0.1),
    eb_seeCut = cms.double(0.014),
    ee_trIsoCut = cms.double(0.1),
    ee_ecalIsoCut = cms.double(0.1),
    ee_hcalIsoCut = cms.double(0.1),
    ee_hoeCut = cms.double(0.1),
    ee_seeCut = cms.double(0.035),
    metCut = cms.double(12.),
    invMassCut = cms.double(40.)
    )
