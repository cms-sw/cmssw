import FWCore.ParameterSet.Config as cms

pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    GsfTrackCandidateModuleLabel = cms.string('gsfElCandidates'),
    ModeMomentum = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdElectronPropagator'),
    GsfTrackModuleLabel = cms.string('gsfPFtracks')
)


