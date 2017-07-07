import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
from DQM.TrackingMonitor.TrackingMonitor_cfi import *

# properties
TrackMon.OutputMEsInRootFile    = cms.bool(False)
TrackMon.OutputFileName         = cms.string('TrackingMonitorAllSequences.root')
TrackMon.MeasurementState       = cms.string('ImpactPoint')

# which plots to do
TrackMon.doTrackerSpecific      = cms.bool(True)
TrackMon.doAllPlots             = cms.bool(True)
TrackMon.doBeamSpotPlots        = cms.bool(True)
TrackMon.doSeedParameterHistos  = cms.bool(False)

# out of the box
# ---------------------------------------------------------------------------#

# generalTracks
TrackMonGenTk = TrackMon.clone()
TrackMonGenTk.TrackProducer         = cms.InputTag("generalTracks")
TrackMonGenTk.beamSpot              = cms.InputTag("offlineBeamSpot")
TrackMonGenTk.FolderName            = cms.string('Tracking/GenTk/GlobalParameters')
TrackMonGenTk.BSFolderName          = cms.string('Tracking/GenTk/BeamSpotParameters')
TrackMonGenTk.AlgoName              = cms.string('GenTk')
TrackMonGenTk.doSeedParameterHistos = cms.bool(False)



# Step0 
TrackMonStep0 = TrackMon.clone()
TrackMonStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackMonStep0.SeedProducer  = cms.InputTag("initialStepSeeds")
TrackMonStep0.TCProducer    = cms.InputTag("initialStepTrackCandidates")
TrackMonStep0.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep0.FolderName    = cms.string('Tracking/Step0/GlobalParameters')
TrackMonStep0.BSFolderName  = cms.string('Tracking/Step0/BeamSpotParameters')
TrackMonStep0.AlgoName      = cms.string('Step0')
TrackMonStep0.doSeedParameterHistos = cms.bool(True)
TrackMonStep0.doTrackCandHistos = cms.bool(True)

# Step1 
TrackMonStep1 = TrackMon.clone()
TrackMonStep1.TrackProducer = cms.InputTag("preMergingFirstStepTracksWithQuality")
TrackMonStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackMonStep1.TCProducer    = cms.InputTag("stepOneTrackCandidateMaker")
TrackMonStep1.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep1.FolderName    = cms.string('Tracking/Step1/GlobalParameters')
TrackMonStep1.BSFolderName  = cms.string('Tracking/Step1/BeamSpotParameters')
TrackMonStep1.AlgoName      = cms.string('Step1')
TrackMonStep1.doSeedParameterHistos = cms.bool(True)
TrackMonStep1.doTrackCandHistos = cms.bool(True)

# Step2 
TrackMonStep2 = TrackMon.clone()
TrackMonStep2.TrackProducer = cms.InputTag("secStep")
TrackMonStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackMonStep2.TCProducer    = cms.InputTag("secTrackCandidates")
TrackMonStep2.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep2.FolderName    = cms.string('Tracking/Step2/GlobalParameters')
TrackMonStep2.BSFolderName  = cms.string('Tracking/Step2/BeamSpotParameters')
TrackMonStep2.AlgoName      = cms.string('Step2')
TrackMonStep2.doSeedParameterHistos = cms.bool(True)
TrackMonStep2.doTrackCandHistos = cms.bool(True)

# Step4 
TrackMonStep4 = TrackMon.clone()
TrackMonStep4.TrackProducer = cms.InputTag("pixellessStep")
TrackMonStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackMonStep4.TCProducer    = cms.InputTag("fourthTrackCandidates")
TrackMonStep4.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep4.FolderName    = cms.string('Tracking/Step4/GlobalParameters')
TrackMonStep4.BSFolderName  = cms.string('Tracking/Step4/BeamSpotParameters')
TrackMonStep4.AlgoName      = cms.string('Step4')
TrackMonStep4.doSeedParameterHistos = cms.bool(True)
TrackMonStep4.doTrackCandHistos = cms.bool(True)

# Step4 
TrackMonStep5 = TrackMon.clone()
TrackMonStep5.TrackProducer = cms.InputTag("tobtecStep")
TrackMonStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackMonStep5.TCProducer    = cms.InputTag("fifthTrackCandidates")
TrackMonStep5.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep5.FolderName    = cms.string('Tracking/Step5/GlobalParameters')
TrackMonStep5.BSFolderName  = cms.string('Tracking/Step5/BeamSpotParameters')
TrackMonStep5.AlgoName      = cms.string('Step5')
TrackMonStep5.doSeedParameterHistos = cms.bool(True)
TrackMonStep5.doTrackCandHistos = cms.bool(True)

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

