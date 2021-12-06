import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
from DQM.TrackingMonitor.TrackingMonitor_cfi import *

# properties
TrackMon.MeasurementState       = cms.string('ImpactPoint')

# which plots to do
TrackMon.doTrackerSpecific      = cms.bool(True)
TrackMon.doAllPlots             = cms.bool(True)
TrackMon.doBeamSpotPlots        = cms.bool(True)
TrackMon.doSeedParameterHistos  = cms.bool(False)

# out of the box
# ---------------------------------------------------------------------------#

# generalTracks
TrackMonGenTk = TrackMon.clone(
    TrackProducer = "generalTracks",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/GenTk/GlobalParameters',
    BSFolderName = 'Tracking/GenTk/BeamSpotParameters',
    AlgoName = 'GenTk',
    doSeedParameterHistos = False
)



# Step0 
TrackMonStep0 = TrackMon.clone(
    TrackProducer = "zeroStepTracksWithQuality",
    SeedProducer = "initialStepSeeds",
    TCProducer = "initialStepTrackCandidates",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/Step0/GlobalParameters',
    BSFolderName = 'Tracking/Step0/BeamSpotParameters',
    AlgoName = 'Step0',
    doSeedParameterHistos = True,
    doTrackCandHistos = True
)

# Step1 
TrackMonStep1 = TrackMon.clone(
    TrackProducer = "preMergingFirstStepTracksWithQuality",
    SeedProducer = "newSeedFromPairs",
    TCProducer = "stepOneTrackCandidateMaker",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/Step1/GlobalParameters',
    BSFolderName = 'Tracking/Step1/BeamSpotParameters',
    AlgoName = 'Step1',
    doSeedParameterHistos = True,
    doTrackCandHistos = True
)

# Step2 
TrackMonStep2 = TrackMon.clone(
    TrackProducer = "secStep",
    SeedProducer = "secTriplets",
    TCProducer = "secTrackCandidates",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/Step2/GlobalParameters',
    BSFolderName = 'Tracking/Step2/BeamSpotParameters',
    AlgoName = 'Step2',
    doSeedParameterHistos = True,
    doTrackCandHistos = True
)

# Step4 
TrackMonStep4 = TrackMon.clone(
    TrackProducer = "pixellessStep",
    SeedProducer = "fourthPLSeeds",
    TCProducer = "fourthTrackCandidates",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/Step4/GlobalParameters',
    BSFolderName = 'Tracking/Step4/BeamSpotParameters',
    AlgoName = 'Step4',
    doSeedParameterHistos = True,
    doTrackCandHistos = True
)

# Step4 
TrackMonStep5 = TrackMon.clone(
    TrackProducer = "tobtecStep",
    SeedProducer = "fifthSeeds",
    TCProducer = "fifthTrackCandidates",
    beamSpot = "offlineBeamSpot",
    FolderName = 'Tracking/Step5/GlobalParameters',
    BSFolderName = 'Tracking/Step5/BeamSpotParameters',
    AlgoName = 'Step5',
    doSeedParameterHistos = True,
    doTrackCandHistos = True
)

# high Purity 
# ---------------------------------------------------------------------------#





#-------------------------------------------------
# Paths 
#-------------------------------------------------

# out of the box
trkmonootb = cms.Sequence(
      TrackMonGenTk
    * TrackMonStep0
    * TrackMonStep1
    * TrackMonStep2
#    * TrackMonStep3
    * TrackMonStep4
    * TrackMonStep5 
)



# all paths
trkmon = cms.Sequence(
      trkmonootb
   # * trkmonhp
   # * trkmontight
   # * trkmonloose
)

