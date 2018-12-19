import FWCore.ParameterSet.Config as cms

def thresholds( wp ) :
    return cms.vdouble([{"L": 1.03,"M":1.75,"T":2.61}.get(wp,1.e6),  # unbiased
                        {"L":-0.48,"M":0.76,"T":1.83}.get(wp,1.e6)]) # ptbiased

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
    ModelThresholds = thresholds("T"),
    PassThrough = cms.bool(False),
    UsePfTracks = cms.bool(True),
    MinPtThreshold = cms.double(0.5),
    MaxPtThreshold = cms.double(15.),
    )

# Modifiers for FastSim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
lowPtGsfElectronSeedsTmp = lowPtGsfElectronSeeds.clone(tracks = cms.InputTag("generalTracksBeforeMixing"))
import FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi
_fastSim_lowPtGsfElectronSeeds = FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi.fixedTrackerDrivenElectronSeeds.clone()
_fastSim_lowPtGsfElectronSeeds.seedCollection = cms.InputTag("lowPtGsfElectronSeedsTmp","")
_fastSim_lowPtGsfElectronSeeds.idCollection = cms.VInputTag("lowPtGsfElectronSeedsTmp","lowPtGsfElectronSeedsTmp:HCAL")
_fastSim_lowPtGsfElectronSeeds.PreIdLabel = cms.vstring("","HCAL")
_fastSim_lowPtGsfElectronSeeds.PreGsfLabel = cms.string("")
fastSim.toReplaceWith(lowPtGsfElectronSeeds,_fastSim_lowPtGsfElectronSeeds)

# Modifiers for Phase2
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(lowPtGsfElectronSeeds, TTRHBuilder  = 'WithTrackAngle')

# Modifiers for BParking
from Configuration.Eras.Modifier_BParking_cff import BParking
BParking.toModify(lowPtGsfElectronSeeds, ModelThresholds = thresholds("L") )
