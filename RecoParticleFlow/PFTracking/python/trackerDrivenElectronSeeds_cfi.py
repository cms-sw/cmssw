import FWCore.ParameterSet.Config as cms

trackerDrivenElectronSeeds = cms.EDProducer("GoodSeedProducer",
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
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    PreGsfLabel = cms.string('SeedsForGsf'),
    MinEOverP = cms.double(0.3),
    Weights1 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat1.xml'),
    Weights2 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat2.xml'),
    Weights3 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat3.xml'),
    Weights4 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat4.xml'),
    Weights5 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat5.xml'),
    Weights6 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat6.xml'),
    Weights7 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat7.xml'),
    Weights8 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat8.xml'),
    Weights9 = cms.string('RecoParticleFlow/PFTracking/data/MVA_BDTTrackDrivenSeed_cat9.xml'),                                        
    PFEcalClusterLabel = cms.InputTag("particleFlowClusterECAL"),
    PFHcalClusterLabel = cms.InputTag("particleFlowClusterHCAL"),
    PSThresholdFile = cms.string('RecoParticleFlow/PFTracking/data/PSThreshold.dat'),
    MinPt = cms.double(2.0),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks")),
    UseTMVA = cms.untracked.bool(True),
    TrackQuality = cms.string('highPurity'),
    MaxPt = cms.double(50.0),
    ApplyIsolation = cms.bool(False),
    EcalStripSumE_deltaPhiOverQ_minValue = cms.double(-0.1),
    EcalStripSumE_minClusEnergy = cms.double(0.1),
    EcalStripSumE_deltaEta = cms.double(0.03),
    EcalStripSumE_deltaPhiOverQ_maxValue = cms.double(0.5),
    EOverPLead_minValue = cms.double(0.95),
    HOverPLead_maxValue = cms.double(0.05),
    HcalWindow=cms.double(0.184),                       
    ClusterThreshold = cms.double(0.5),
    UsePreShower =cms.bool(False),
    PreIdLabel = cms.string('preid'),
    ProducePreId = cms.untracked.bool(True),
    PtThresholdSavePreId = cms.untracked.double(1.0),
    Min_dr = cms.double(0.2)
)

# This customization will be removed once we get the templates for
# phase2 pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(trackerDrivenElectronSeeds, TTRHBuilder  = 'WithTrackAngle') # FIXME

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
pp_on_XeXe_2017.toModify(trackerDrivenElectronSeeds, MinPt = 5.0) 
