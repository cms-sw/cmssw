import FWCore.ParameterSet.Config as cms

def thresholds( wp ) :
    return cms.vdouble([{"VL": 0.19,"L":1.20,"M":2.02,"T":3.05}.get(wp,1.e6),  # unbiased
                        {"VL":-1.99,"L":0.01,"M":1.29,"T":2.42}.get(wp,1.e6)]) # ptbiased

lowPtGsfElectronSeeds = cms.EDProducer(
    "LowPtGsfElectronSeedProducer",
    tracks = cms.InputTag("generalTracks"),
    pfTracks = cms.InputTag("lowPtGsfElePfTracks"),
    ecalClusters = cms.InputTag("particleFlowClusterECAL"),
    hcalClusters = cms.InputTag("particleFlowClusterHCAL"),
    EBRecHits = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EERecHits = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    rho = cms.InputTag('fixedGridRhoFastjetAllTmp'),
    BeamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.ESInputTag("", 'GsfTrajectoryFitter_forPreId'),
    Smoother = cms.ESInputTag("", 'GsfTrajectorySmoother_forPreId'),
    TTRHBuilder = cms.ESInputTag("", 'WithAngleAndTemplate'),
    ModelNames = cms.vstring([
            'unbiased',
            'ptbiased',
            ]),
    ModelWeights = cms.vstring([
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_unbiased.xml.gz',
            'RecoEgamma/ElectronIdentification/data/LowPtElectrons/RunII_Autumn18_LowPtElectrons_displaced_pt_eta_biased.xml.gz',
            ]),
    ModelThresholds = thresholds("T"),
    PassThrough = cms.bool(False),
    UsePfTracks = cms.bool(True),
    MinPtThreshold = cms.double(1.0),
    MaxPtThreshold = cms.double(15.),
    )

# Modifiers for FastSim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
lowPtGsfElectronSeedsTmp = lowPtGsfElectronSeeds.clone(tracks = "generalTracksBeforeMixing")
import FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi
_fastSim_lowPtGsfElectronSeeds = FastSimulation.Tracking.ElectronSeedTrackRefFix_cfi.fixedTrackerDrivenElectronSeeds.clone(
    seedCollection = "lowPtGsfElectronSeedsTmp:",
    idCollection   = ["lowPtGsfElectronSeedsTmp","lowPtGsfElectronSeedsTmp:HCAL"],
    PreIdLabel     = ["","HCAL"],
    PreGsfLabel    = ""
)
fastSim.toReplaceWith(lowPtGsfElectronSeeds,_fastSim_lowPtGsfElectronSeeds)

# Modifiers for BParking
from Configuration.Eras.Modifier_bParking_cff import bParking
bParking.toModify(lowPtGsfElectronSeeds, 
    ModelThresholds = thresholds("VL"), 
)
