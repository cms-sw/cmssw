import FWCore.ParameterSet.Config as cms
import DQM.TrackingMonitor.TrackingMonitor_cfi
import DQMOffline.Alignment.TkAlCaRecoMonitor_cfi

#---------------
# AlCaReco DQM #
#---------------

__selectionName = 'SiStripCalMinBias'
ALCARECOSiStripCalMinBiasTrackingDQM = DQM.TrackingMonitor.TrackingMonitor_cfi.TrackMon.clone(
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

ALCARECOSiStripCalMinBiasTrackerDQM = DQMOffline.Alignment.TkAlCaRecoMonitor_cfi.TkAlCaRecoMonitor.clone(
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

# Short-term workaround to preserve the "run for every event" while removing the use of convertToUnscheduled()
# To be reverted in a subsequent PR
ALCARECOSiStripCalMinBiasDQMTask = cms.Task(ALCARECOSiStripCalMinBiasTrackingDQM , ALCARECOSiStripCalMinBiasTrackerDQM)
ALCARECOSiStripCalMinBiasDQM = cms.Sequence(ALCARECOSiStripCalMinBiasDQMTask)
