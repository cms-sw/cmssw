import FWCore.ParameterSet.Config as cms

lowPtGsfElectronSeeds = cms.EDProducer(
    "LowPtGsfElectronSeedProducer",
    tracks = cms.InputTag("generalTracks"),
    pfTracks = cms.InputTag("lowPtGsfElePfTracks"),
    ecalClusters = cms.InputTag("particleFlowClusterECAL"),
    hcalClusters = cms.InputTag("particleFlowClusterHCAL"),
    EBRecHits = cms.InputTag('reducedEcalRecHitsEB'), 
    EERecHits = cms.InputTag('reducedEcalRecHitsEE'),
    rho = cms.InputTag('fixedGridRhoFastjetAllTmp'),
    BeamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('GsfTrajectoryFitter_forPreId'),
    Smoother = cms.string('GsfTrajectorySmoother_forPreId'),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    ModelNames = cms.vstring([
            'unbiased',
            'ptbiased',
            ]),
    ModelWeights = cms.vstring([
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Fall17_LowPtElectrons_unbiased.xml.gz',
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Fall17_LowPtElectrons_displaced_pt_eta_biased.xml.gz',
            ]),
    ModelThresholds = cms.vdouble([
            {"L": 1.03,"M":1.75,"T":2.61}["L"], # unbiased
            {"L":-0.48,"M":0.76,"T":1.83}["L"], # ptbiased
            ]),
    PassThrough = cms.bool(False),
    UsePfTracks = cms.bool(True),
    MinPtThreshold = cms.double(0.5),
    MaxPtThreshold = cms.double(15.),
    )
