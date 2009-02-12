import FWCore.ParameterSet.Config as cms

pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    GsfTracks = cms.InputTag(""),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    AddGSFTkColl = cms.bool(False),
    GsfElectrons =cms.InputTag(""),                   
    ModeMomentum = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GsfTrackModuleLabel = cms.InputTag("electronGsfTracks"),
    Propagator = cms.string('fwdElectronPropagator'),
    PFRecTrackLabel = cms.InputTag("trackerDrivenElectronSeeds")
)


