import FWCore.ParameterSet.Config as cms

pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    GsfTracks = cms.InputTag("pixelMatchGsfFit"),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    AddGSFTkColl = cms.bool(True),
    GsfElectrons =cms.InputTag("pixelMatchGsfElectrons"),                   
    ModeMomentum = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GsfTrackModuleLabel = cms.InputTag("gsfPFtracks"),
    Propagator = cms.string('fwdElectronPropagator'),
    PFRecTrackLabel = cms.InputTag("elecpreid")
)


