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

# Pixel less Tracks
TrackMonNoPxl = TrackMon.clone()
TrackMonNoPxl.TrackProducer = cms.InputTag("ctfPixelLess")
TrackMonNoPxl.SeedProducer  = cms.InputTag("globalPixelLessSeeds")
TrackMonNoPxl.TCProducer    = cms.InputTag("ckfTrackCandidatesPixelLess")
TrackMonNoPxl.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonNoPxl.FolderName    = cms.string('Tracking/NoPxl/GlobalParameters')
TrackMonNoPxl.BSFolderName  = cms.string('Tracking/NoPxl/BeamSpotParameters')
TrackMonNoPxl.AlgoName      = cms.string('NoPxl')
TrackMonNoPxl.doSeedParameterHistos = cms.bool(True)

# Step0 
TrackMonStep0 = TrackMon.clone()
TrackMonStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackMonStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackMonStep0.TCProducer    = cms.InputTag("newTrackCandidateMaker")
TrackMonStep0.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep0.FolderName    = cms.string('Tracking/Step0/GlobalParameters')
TrackMonStep0.BSFolderName  = cms.string('Tracking/Step0/BeamSpotParameters')
TrackMonStep0.AlgoName      = cms.string('Step0')
TrackMonStep0.doSeedParameterHistos = cms.bool(True)

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

# Step3 
TrackMonStep3 = TrackMon.clone()
TrackMonStep3.TrackProducer = cms.InputTag("thStep")
TrackMonStep3.SeedProducer  = cms.InputTag("thPLSeeds")
TrackMonStep3.TCProducer    = cms.InputTag("thTrackCandidates")
TrackMonStep3.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonStep3.FolderName    = cms.string('Tracking/Step3/GlobalParameters')
TrackMonStep3.BSFolderName  = cms.string('Tracking/Step3/BeamSpotParameters')
TrackMonStep3.AlgoName      = cms.string('Step3')
TrackMonStep3.doSeedParameterHistos = cms.bool(True)

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

# high Purity 
# ---------------------------------------------------------------------------#

# generalTracks
TrackMonHpGenTk = TrackMon.clone()
TrackMonHpGenTk.TrackProducer         = cms.InputTag("generalTracks")
TrackMonHpGenTk.beamSpot              = cms.InputTag("offlineBeamSpot")
TrackMonHpGenTk.FolderName            = cms.string('Tracking/GenTk/GlobalParameters')
TrackMonHpGenTk.BSFolderName          = cms.string('Tracking/GenTk/BeamSpotParameters')
TrackMonHpGenTk.AlgoName              = cms.string('GenTk')
TrackMonHpGenTk.Quality               = cms.string('highPurity')
TrackMonHpGenTk.doSeedParameterHistos = cms.bool(False)

# Pixel less Tracks
TrackMonHpNoPxl = TrackMon.clone()
TrackMonHpNoPxl.TrackProducer = cms.InputTag("ctfPixelLess")
TrackMonHpNoPxl.SeedProducer  = cms.InputTag("globalPixelLessSeeds")
TrackMonHpNoPxl.TCProducer    = cms.InputTag("ckfTrackCandidatesPixelLess")
TrackMonHpNoPxl.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpNoPxl.FolderName    = cms.string('Tracking/NoPxl/GlobalParameters')
TrackMonHpNoPxl.BSFolderName  = cms.string('Tracking/NoPxl/BeamSpotParameters')
TrackMonHpNoPxl.Quality       = cms.string('highPurity')
TrackMonHpNoPxl.AlgoName      = cms.string('NoPxl')

# Step0 
TrackMonHpStep0 = TrackMon.clone()
TrackMonHpStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackMonHpStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackMonHpStep0.TCProducer    = cms.InputTag("newTrackCandidateMaker")
TrackMonHpStep0.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep0.FolderName    = cms.string('Tracking/Step0/GlobalParameters')
TrackMonHpStep0.BSFolderName  = cms.string('Tracking/Step0/BeamSpotParameters')
TrackMonHpStep0.Quality       = cms.string('highPurity')
TrackMonHpStep0.AlgoName      = cms.string('Step0')

# Step1 
TrackMonHpStep1 = TrackMon.clone()
TrackMonHpStep1.TrackProducer = cms.InputTag("preMergingFirstStepTracksWithQuality")
TrackMonHpStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackMonHpStep1.TCProducer    = cms.InputTag("stepOneTrackCandidateMaker")
TrackMonHpStep1.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep1.FolderName    = cms.string('Tracking/Step1/GlobalParameters')
TrackMonHpStep1.BSFolderName  = cms.string('Tracking/Step1/BeamSpotParameters')
TrackMonHpStep1.Quality       = cms.string('highPurity')
TrackMonHpStep1.AlgoName      = cms.string('Step1')

# Step2 
TrackMonHpStep2 = TrackMon.clone()
TrackMonHpStep2.TrackProducer = cms.InputTag("secStep")
TrackMonHpStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackMonHpStep2.TCProducer    = cms.InputTag("secTrackCandidates")
TrackMonHpStep2.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep2.FolderName    = cms.string('Tracking/Step2/GlobalParameters')
TrackMonHpStep2.BSFolderName  = cms.string('Tracking/Step2/BeamSpotParameters')
TrackMonHpStep2.Quality       = cms.string('highPurity')
TrackMonHpStep2.AlgoName      = cms.string('Step2')

# Step3 
TrackMonHpStep3 = TrackMon.clone()
TrackMonHpStep3.TrackProducer = cms.InputTag("thStep")
TrackMonHpStep3.SeedProducer  = cms.InputTag("thPLSeeds")
TrackMonHpStep3.TCProducer    = cms.InputTag("thTrackCandidates")
TrackMonHpStep3.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep3.FolderName    = cms.string('Tracking/Step3/GlobalParameters')
TrackMonHpStep3.BSFolderName  = cms.string('Tracking/Step3/BeamSpotParameters')
TrackMonHpStep3.Quality       = cms.string('highPurity')
TrackMonHpStep3.AlgoName      = cms.string('Step3')

# Step4 
TrackMonHpStep4 = TrackMon.clone()
TrackMonHpStep4.TrackProducer = cms.InputTag("pixellessStep")
TrackMonHpStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackMonHpStep4.TCProducer    = cms.InputTag("fourthTrackCandidates")
TrackMonHpStep4.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep4.FolderName    = cms.string('Tracking/Step4/GlobalParameters')
TrackMonHpStep4.BSFolderName  = cms.string('Tracking/Step4/BeamSpotParameters')
TrackMonHpStep4.Quality       = cms.string('highPurity')
TrackMonHpStep4.AlgoName      = cms.string('Step4')

# Step4 
TrackMonHpStep5 = TrackMon.clone()
TrackMonHpStep5.TrackProducer = cms.InputTag("tobtecStep")
TrackMonHpStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackMonHpStep5.TCProducer    = cms.InputTag("fifthTrackCandidates")
TrackMonHpStep5.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonHpStep5.FolderName    = cms.string('Tracking/Step5/GlobalParameters')
TrackMonHpStep5.BSFolderName  = cms.string('Tracking/Step5/BeamSpotParameters')
TrackMonHpStep5.Quality       = cms.string('highPurity')
TrackMonHpStep5.AlgoName      = cms.string('Step5')

# tight 
# ---------------------------------------------------------------------------#

# generalTracks
TrackMonTightGenTk = TrackMon.clone()
TrackMonTightGenTk.TrackProducer         = cms.InputTag("generalTracks")
TrackMonTightGenTk.beamSpot              = cms.InputTag("offlineBeamSpot")
TrackMonTightGenTk.FolderName            = cms.string('Tracking/GenTk/GlobalParameters')
TrackMonTightGenTk.BSFolderName          = cms.string('Tracking/GenTk/BeamSpotParameters')
TrackMonTightGenTk.AlgoName              = cms.string('GenTk')
TrackMonTightGenTk.Quality               = cms.string('tight')
TrackMonTightGenTk.doSeedParameterHistos = cms.bool(False)

# Pixel less Tracks
TrackMonTightNoPxl = TrackMon.clone()
TrackMonTightNoPxl.TrackProducer = cms.InputTag("ctfPixelLess")
TrackMonTightNoPxl.SeedProducer  = cms.InputTag("globalPixelLessSeeds")
TrackMonTightNoPxl.TCProducer    = cms.InputTag("ckfTrackCandidatesPixelLess")
TrackMonTightNoPxl.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightNoPxl.FolderName    = cms.string('Tracking/NoPxl/GlobalParameters')
TrackMonTightNoPxl.BSFolderName  = cms.string('Tracking/NoPxl/BeamSpotParameters')
TrackMonTightNoPxl.Quality       = cms.string('tight')
TrackMonTightNoPxl.AlgoName      = cms.string('NoPxl')
TrackMonTightNoPxl.doSeedParameterHistos = cms.bool(True)

# Step0 
TrackMonTightStep0 = TrackMon.clone()
TrackMonTightStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackMonTightStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackMonTightStep0.TCProducer    = cms.InputTag("newTrackCandidateMaker")
TrackMonTightStep0.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep0.FolderName    = cms.string('Tracking/Step0/GlobalParameters')
TrackMonTightStep0.BSFolderName  = cms.string('Tracking/Step0/BeamSpotParameters')
TrackMonTightStep0.Quality       = cms.string('tight')
TrackMonTightStep0.AlgoName      = cms.string('Step0')

# Step1 
TrackMonTightStep1 = TrackMon.clone()
TrackMonTightStep1.TrackProducer = cms.InputTag("preMergingFirstStepTracksWithQuality")
TrackMonTightStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackMonTightStep1.TCProducer    = cms.InputTag("stepOneTrackCandidateMaker")
TrackMonTightStep1.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep1.FolderName    = cms.string('Tracking/Step1/GlobalParameters')
TrackMonTightStep1.BSFolderName  = cms.string('Tracking/Step1/BeamSpotParameters')
TrackMonTightStep1.Quality       = cms.string('tight')
TrackMonTightStep1.AlgoName      = cms.string('Step1')

# Step2 
TrackMonTightStep2 = TrackMon.clone()
TrackMonTightStep2.TrackProducer = cms.InputTag("secStep")
TrackMonTightStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackMonTightStep2.TCProducer    = cms.InputTag("secTrackCandidates")
TrackMonTightStep2.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep2.FolderName    = cms.string('Tracking/Step2/GlobalParameters')
TrackMonTightStep2.BSFolderName  = cms.string('Tracking/Step2/BeamSpotParameters')
TrackMonTightStep2.Quality       = cms.string('tight')
TrackMonTightStep2.AlgoName      = cms.string('Step2')

# Step3 
TrackMonTightStep3 = TrackMon.clone()
TrackMonTightStep3.TrackProducer = cms.InputTag("thStep")
TrackMonTightStep3.SeedProducer  = cms.InputTag("thPLSeeds")
TrackMonTightStep3.TCProducer    = cms.InputTag("thTrackCandidates")
TrackMonTightStep3.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep3.FolderName    = cms.string('Tracking/Step3/GlobalParameters')
TrackMonTightStep3.BSFolderName  = cms.string('Tracking/Step3/BeamSpotParameters')
TrackMonTightStep3.Quality       = cms.string('tight')
TrackMonTightStep3.AlgoName      = cms.string('Step3')

# Step4 
TrackMonTightStep4 = TrackMon.clone()
TrackMonTightStep4.TrackProducer = cms.InputTag("pixellessStep")
TrackMonTightStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackMonTightStep4.TCProducer    = cms.InputTag("fourthTrackCandidates")
TrackMonTightStep4.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep4.FolderName    = cms.string('Tracking/Step4/GlobalParameters')
TrackMonTightStep4.BSFolderName  = cms.string('Tracking/Step4/BeamSpotParameters')
TrackMonTightStep4.Quality       = cms.string('tight')
TrackMonTightStep4.AlgoName      = cms.string('Step4')

# Step4 
TrackMonTightStep5 = TrackMon.clone()
TrackMonTightStep5.TrackProducer = cms.InputTag("tobtecStep")
TrackMonTightStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackMonTightStep5.TCProducer    = cms.InputTag("fifthTrackCandidates")
TrackMonTightStep5.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonTightStep5.FolderName    = cms.string('Tracking/Step5/GlobalParameters')
TrackMonTightStep5.BSFolderName  = cms.string('Tracking/Step5/BeamSpotParameters')
TrackMonTightStep5.Quality       = cms.string('tight')
TrackMonTightStep5.AlgoName      = cms.string('Step5')

# loose 
# ---------------------------------------------------------------------------#

# generalTracks
TrackMonLooseGenTk = TrackMon.clone()
TrackMonLooseGenTk.TrackProducer         = cms.InputTag("generalTracks")
TrackMonLooseGenTk.beamSpot              = cms.InputTag("offlineBeamSpot")
TrackMonLooseGenTk.FolderName            = cms.string('Tracking/GenTk/GlobalParameters')
TrackMonLooseGenTk.BSFolderName          = cms.string('Tracking/GenTk/BeamSpotParameters')
TrackMonLooseGenTk.AlgoName              = cms.string('GenTk')
TrackMonLooseGenTk.Quality               = cms.string('loose')
TrackMonLooseGenTk.doSeedParameterHistos = cms.bool(False)

# Pixel less Tracks
TrackMonLooseNoPxl = TrackMon.clone()
TrackMonLooseNoPxl.TrackProducer = cms.InputTag("ctfPixelLess")
TrackMonLooseNoPxl.SeedProducer  = cms.InputTag("globalPixelLessSeeds")
TrackMonLooseNoPxl.TCProducer    = cms.InputTag("ckfTrackCandidatesPixelLess")
TrackMonLooseNoPxl.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseNoPxl.FolderName    = cms.string('Tracking/NoPxl/GlobalParameters')
TrackMonLooseNoPxl.BSFolderName  = cms.string('Tracking/NoPxl/BeamSpotParameters')
TrackMonLooseNoPxl.Quality       = cms.string('loose')
TrackMonLooseNoPxl.AlgoName      = cms.string('NoPxl')

# Step0 
TrackMonLooseStep0 = TrackMon.clone()
TrackMonLooseStep0.TrackProducer = cms.InputTag("zeroStepTracksWithQuality")
TrackMonLooseStep0.SeedProducer  = cms.InputTag("newSeedFromTriplets")
TrackMonLooseStep0.TCProducer    = cms.InputTag("newTrackCandidateMaker")
TrackMonLooseStep0.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep0.FolderName    = cms.string('Tracking/Step0/GlobalParameters')
TrackMonLooseStep0.BSFolderName  = cms.string('Tracking/Step0/BeamSpotParameters')
TrackMonLooseStep0.Quality       = cms.string('loose')
TrackMonLooseStep0.AlgoName      = cms.string('Step0')

# Step1 
TrackMonLooseStep1 = TrackMon.clone()
TrackMonLooseStep1.TrackProducer = cms.InputTag("preMergingFirstStepTracksWithQuality")
TrackMonLooseStep1.SeedProducer  = cms.InputTag("newSeedFromPairs")
TrackMonLooseStep1.TCProducer    = cms.InputTag("stepOneTrackCandidateMaker")
TrackMonLooseStep1.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep1.FolderName    = cms.string('Tracking/Step1/GlobalParameters')
TrackMonLooseStep1.BSFolderName  = cms.string('Tracking/Step1/BeamSpotParameters')
TrackMonLooseStep1.Quality       = cms.string('loose')
TrackMonLooseStep1.AlgoName      = cms.string('Step1')

# Step2 
TrackMonLooseStep2 = TrackMon.clone()
TrackMonLooseStep2.TrackProducer = cms.InputTag("secStep")
TrackMonLooseStep2.SeedProducer  = cms.InputTag("secTriplets")
TrackMonLooseStep2.TCProducer    = cms.InputTag("secTrackCandidates")
TrackMonLooseStep2.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep2.FolderName    = cms.string('Tracking/Step2/GlobalParameters')
TrackMonLooseStep2.BSFolderName  = cms.string('Tracking/Step2/BeamSpotParameters')
TrackMonLooseStep2.Quality       = cms.string('loose')
TrackMonLooseStep2.AlgoName      = cms.string('Step2')

# Step3 
TrackMonLooseStep3 = TrackMon.clone()
TrackMonLooseStep3.TrackProducer = cms.InputTag("thStep")
TrackMonLooseStep3.SeedProducer  = cms.InputTag("thPLSeeds")
TrackMonLooseStep3.TCProducer    = cms.InputTag("thTrackCandidates")
TrackMonLooseStep3.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep3.FolderName    = cms.string('Tracking/Step3/GlobalParameters')
TrackMonLooseStep3.BSFolderName  = cms.string('Tracking/Step3/BeamSpotParameters')
TrackMonLooseStep3.Quality       = cms.string('loose')
TrackMonLooseStep3.AlgoName      = cms.string('Step3')

# Step4 
TrackMonLooseStep4 = TrackMon.clone()
TrackMonLooseStep4.TrackProducer = cms.InputTag("pixellessStep")
TrackMonLooseStep4.SeedProducer  = cms.InputTag("fourthPLSeeds")
TrackMonLooseStep4.TCProducer    = cms.InputTag("fourthTrackCandidates")
TrackMonLooseStep4.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep4.FolderName    = cms.string('Tracking/Step4/GlobalParameters')
TrackMonLooseStep4.BSFolderName  = cms.string('Tracking/Step4/BeamSpotParameters')
TrackMonLooseStep4.Quality       = cms.string('loose')
TrackMonLooseStep4.AlgoName      = cms.string('Step4')

# Step4 
TrackMonLooseStep5 = TrackMon.clone()
TrackMonLooseStep5.TrackProducer = cms.InputTag("tobtecStep")
TrackMonLooseStep5.SeedProducer  = cms.InputTag("fifthSeeds")
TrackMonLooseStep5.TCProducer    = cms.InputTag("fifthTrackCandidates")
TrackMonLooseStep5.beamSpot      = cms.InputTag("offlineBeamSpot")
TrackMonLooseStep5.FolderName    = cms.string('Tracking/Step5/GlobalParameters')
TrackMonLooseStep5.BSFolderName  = cms.string('Tracking/Step5/BeamSpotParameters')
TrackMonLooseStep5.Quality       = cms.string('loose')
TrackMonLooseStep5.AlgoName      = cms.string('Step5')


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# only do this for one of the steps (work around)
TrackMonGenTk.OutputMEsInRootFile = cms.bool(True)  

#-------------------------------------------------
# Paths 
#-------------------------------------------------

# out of the box
trkmonootb = cms.Sequence(
      TrackMonGenTk
    * TrackMonNoPxl
    * TrackMonStep0
    * TrackMonStep1
    * TrackMonStep2
    * TrackMonStep3
    * TrackMonStep4
    * TrackMonStep5 
)

# high purity 
trkmonhp = cms.Sequence(
      TrackMonHpGenTk
    * TrackMonHpStep0
    * TrackMonHpStep1
    * TrackMonHpStep2
    * TrackMonHpStep3
    * TrackMonHpStep4
    * TrackMonHpStep5 
)

# tight 
trkmontight = cms.Sequence(
      TrackMonTightGenTk
    * TrackMonTightStep0
    * TrackMonTightStep1
    * TrackMonTightStep2
    * TrackMonTightStep3
    * TrackMonTightStep4
    * TrackMonTightStep5 
)

# loose
trkmonloose = cms.Sequence(
      TrackMonLooseGenTk
    * TrackMonLooseStep0
    * TrackMonLooseStep1
    * TrackMonLooseStep2
    * TrackMonLooseStep3
    * TrackMonLooseStep4
    * TrackMonLooseStep5 
)

# all paths
trkmon = cms.Sequence(
      trkmonootb
    * trkmonhp
    * trkmontight
    * trkmonloose
)

