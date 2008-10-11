import FWCore.ParameterSet.Config as cms

pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    GsfColList = cms.VInputTag(cms.InputTag("pixelMatchGsfFit")),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    AddGSFTkColl = cms.bool(False),
    ModeMomentum = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    GsfTrackModuleLabel = cms.InputTag("gsfPFtracks"),
    Propagator = cms.string('fwdElectronPropagator'),
    PFRecTrackLabel = cms.InputTag("elecpreid")
)


