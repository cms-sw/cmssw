import FWCore.ParameterSet.Config as cms

pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    ModeMomentum = cms.bool(True),
    applyEGSelection = cms.bool(True),        
    applyGsfTrackCleaning = cms.bool(True),
    useFifthTrackingStep = cms.bool(False),
    useFifthTrackingStepForSecondaries = cms.bool(True),                   
    MinDEtaGsfSC = cms.double(0.06),
    MinDPhiGsfSC = cms.double(0.15),
    MinSCEnergy = cms.double(4.0),                         
    TTRHBuilder = cms.string('WithTrackAngle'),
    GsfTrackModuleLabel = cms.InputTag("electronGsfTracks"),
    Propagator = cms.string('fwdElectronPropagator'),
    PFRecTrackLabel = cms.InputTag("trackerDrivenElectronSeeds")
)


