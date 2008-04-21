import FWCore.ParameterSet.Config as cms

ecalMIPRecHitFilter = cms.EDFilter("EcalMIPRecHitFilter",
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    #These are in GeV
    AmpMinSeed = cms.untracked.double(0.045),
    SingleAmpMin = cms.untracked.double(0.108),
    side = cms.untracked.int32(3),
    #
    maskedChannels = cms.untracked.vint32(),
    AmpMin2 = cms.untracked.double(0.045)
)


