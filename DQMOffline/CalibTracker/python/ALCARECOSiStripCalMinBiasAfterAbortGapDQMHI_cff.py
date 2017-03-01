import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi
import DQMOffline.Alignment.TkAlCaRecoMonitor_cfi

#---------------
# AlCaReco DQM #
#---------------

__selectionName = 'SiStripCalMinBiasAfterAbortGap'
ALCARECOSiStripCalMinBiasAfterAbortGapTrackingDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    SeedProducer = 'hiPixelTrackSeeds',
    TCProducer = 'hiPrimTrackCandidates',
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    BSFolderName = "AlCaReco/"+__selectionName+"/BeamSpot",
    primaryVertex = "hiSelectedVertex",
    allTrackProducer = "hiGeneralTracks",
# margins and settings
    TkSizeBin = 300,
    TkSizeMin = -0.5,
    TkSizeMax = 299.5,
    TrackPtMax = 30
)

ALCARECOSiStripCalMinBiasAfterAbortGapTrackerDQM = DQMOffline.Alignment.TkAlCaRecoMonitor_cfi.TkAlCaRecoMonitor.clone(
#names and desigantions
    TrackProducer = 'ALCARECO'+__selectionName,
    AlgoName = 'ALCARECO'+__selectionName,
    FolderName = "AlCaReco/"+__selectionName,
    ReferenceTrackProducer = 'hiGeneralTracks',
# margins and settings
    fillInvariantMass = False,
    TrackPtMax = 30,
    SumChargeBin = 101,
    SumChargeMin = -50.5,
    SumChargeMax = 50.5
)


#------------
# Sequence #
#------------

ALCARECOSiStripCalMinBiasAfterAbortGapDQM = cms.Sequence( ALCARECOSiStripCalMinBiasAfterAbortGapTrackingDQM + 
                                                          ALCARECOSiStripCalMinBiasAfterAbortGapTrackerDQM)
