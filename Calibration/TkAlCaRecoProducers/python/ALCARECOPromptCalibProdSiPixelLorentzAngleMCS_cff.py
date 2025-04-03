import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by SiPixelCalCosmics AlcaReco
from  HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalCosmicsFilterForSiPixelLorentzAngleMCS = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOSiPixelCalCosmics'],
    throw = True, ## dont throw on unknown path names
    TriggerResultsTag = ("TriggerResults","","RECO")
)
# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiPixelCalSingleMuonLoose
# to have access to transient objects produced during RECO step and not saved

from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOPixelLACalibrationTracksMCS = AlignmentTrackSelector.clone(
    src = 'ALCARECOSiPixelCalCosmics',
    filter = True,
    applyBasicCuts = True,
    ptMin = 3.
)

# FIXME: the beam-spot should be kept in the AlCaReco (if not already there) and dropped from here
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *

ALCARECOPixelLACalibrationTracksRefitMCS = TrackRefitter.clone(src = "ALCARECOPixelLACalibrationTracksMCS",
                                                               TrajectoryInEvent = True,
                                                               NavigationSchool = ""
                                                           )

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOPixelLATrackFilterRefitMCS = cms.Sequence(ALCARECOPixelLACalibrationTracksMCS +
                                                  offlineBeamSpot +
                                                  ALCARECOPixelLACalibrationTracksRefitMCS)

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration
from CalibTracker.SiPixelLorentzAngle.SiPixelLorentzAnglePCLWorker_cfi import SiPixelLorentzAnglePCLWorker 
ALCARECOSiPixelLACalibMCS = SiPixelLorentzAnglePCLWorker.clone(
    folder = 'AlCaReco/SiPixelLorentzAngle',
    src = 'ALCARECOPixelLACalibrationTracksRefitMCS',
    analysisType = 'MinimalClusterSize'
)
# ----------------------------------------------------------------------------

# ****************************************************************************
# ** Conversion for the SiPixelLorentzAngle DQM dir                         **
# ****************************************************************************
MEtoEDMConvertSiPixelLorentzAngleMCS = cms.EDProducer("MEtoEDMConverter",
                                                      Name = cms.untracked.string('MEtoEDMConverter'),
                                                      Verbosity = cms.untracked.int32(0), # 0 provides no output
                                                      # 1 provides basic output
                                                      # 2 provide more detailed output
                                                      Frequency = cms.untracked.int32(50),
                                                      MEPathToSave = cms.untracked.string('AlCaReco/SiPixelLorentzAngle'),
                                                   )

# The actual sequence
seqALCARECOPromptCalibProdSiPixelLorentzAngleMCS = cms.Sequence(
    ALCARECOCalCosmicsFilterForSiPixelLorentzAngleMCS *
    ALCARECOPixelLATrackFilterRefitMCS *
    ALCARECOSiPixelLACalibMCS *
    MEtoEDMConvertSiPixelLorentzAngleMCS
   )
