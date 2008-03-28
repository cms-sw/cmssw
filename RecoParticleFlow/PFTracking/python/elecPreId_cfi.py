import FWCore.ParameterSet.Config as cms

elecpreid = cms.EDProducer("GoodSeedProducer",
    ProduceCkfPFTracks = cms.untracked.bool(True),
    MaxEOverP = cms.double(3.0),
    Smoother = cms.string('GsfTrajectorySmoother_forPreId'),
    UseQuality = cms.bool(True),
    PFPSClusterLabel = cms.InputTag("particleFlowClusterPS"),
    ThresholdFile = cms.string('RecoParticleFlow/PFTracking/data/Threshold.dat'),
    TMVAMethod = cms.string('BDT'),
    MaxEta = cms.double(2.4),
    EtaMap = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    PhiMap = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
    PreCkfLabel = cms.string('SeedsForCkf'),
    NHitsInSeed = cms.int32(3),
    Fitter = cms.string('GsfTrajectoryFitter_forPreId'),
    PreGsfLabel = cms.string('SeedsForGsf'),
    MinEOverP = cms.double(0.3),
    Weights = cms.string('RecoParticleFlow/PFTracking/data/BDT_weights.txt'),
    PFEcalClusterLabel = cms.InputTag("particleFlowClusterECAL"),
    PSThresholdFile = cms.string('RecoParticleFlow/PFTracking/data/PSThreshold.dat'),
    MinPt = cms.double(2.0),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep")),
    UseTMVA = cms.untracked.bool(False),
    TrackQuality = cms.string('highPurity'),
    MaxPt = cms.double(50.0),
    ClusterThreshold = cms.double(0.5)
)


