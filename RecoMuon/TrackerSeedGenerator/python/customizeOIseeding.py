import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *
import RecoMuon.TrackerSeedGenerator.tsgForOIDNN_cfi as _mod

def customizeOIseeding(process):
    """
    - adds doublet-like HB seeds (maxHitDoubletSeeds)
    - HL seeds from two trajectories (IP, MuS) are considered separate types
    - Number of seeds of each type and error SF for HL seeds  can be determined individually for each L2 muon using a DNN
    """

    process.hltIterL3OISeedsFromL2Muons = _mod.tsgForOIDNN.clone(
        src = "hltL2Muons:UpdatedAtVtx",
        MeasurementTrackerEvent = "hltSiStripClusters",
        debug = False,
        estimator = 'hltESPChi2MeasurementEstimator100',
        fixedErrorRescaleFactorForHitless = 2.0,
        hitsToTry = 1,
        layersToTry = 2,
        maxEtaForTOB = 1.8,
        minEtaForTEC = 0.7,
        maxHitDoubletSeeds = 0,
        maxHitSeeds = 1,
        maxHitlessSeedsIP = 5,
        maxHitlessSeedsMuS = 0,
        maxSeeds = 20,
        propagatorName = 'PropagatorWithMaterialParabolicMf',
        getStrategyFromDNN = True,  # will override max nSeeds of all types and SF
        useRegressor = False,
        dnnMetadataPath = 'RecoMuon/TrackerSeedGenerator/data/OIseeding/DNNclassifier_Run3_metadata.json',
        # dnnMetadataPath = 'RecoMuon/TrackerSeedGenerator/data/OIseeding/DNNregressor_Run3_metadata.json'
    )

    return process
