import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by SiStripCalCosmics AlcaReco
from  HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalCosmicsFilterForSiStripLorentzAngle = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOSiStripCalCosmics'],
    throw = True, ## dont throw on unknown path names
    TriggerResultsTag = ("TriggerResults","","RECO")
)
# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiStripCalCosmics
# to have access to transient objects produced during RECO step and not saved

from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOSiStripLACalibrationTracks = AlignmentTrackSelector.clone(
    src = 'ALCARECOSiStripCalCosmics',
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

ALCARECOSiStripLACalibrationTracksRefit = TrackRefitter.clone(src = "ALCARECOSiStripLACalibrationTracks",
                                                              TrajectoryInEvent = True,
                                                              NavigationSchool = "")

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOSiStripLATrackFilterRefit = cms.Sequence(ALCARECOSiStripLACalibrationTracks +
                                                 offlineBeamSpot +
                                                 ALCARECOSiStripLACalibrationTracksRefit)

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration
from CalibTracker.SiStripLorentzAngle.SiStripLorentzAnglePCLMonitor_cfi import SiStripLorentzAnglePCLMonitor
ALCARECOSiStripLACalib = SiStripLorentzAnglePCLMonitor.clone(
    folder = 'AlCaReco/SiStripLorentzAngle',
    Tracks = 'ALCARECOSiStripLACalibrationTracksRefit'
)
# ----------------------------------------------------------------------------

# ****************************************************************************
# ** Conversion for the SiStripLorentzAngle DQM dir                         **
# ****************************************************************************
MEtoEDMConvertSiStripLorentzAngle = cms.EDProducer("MEtoEDMConverter",
                                                   Name = cms.untracked.string('MEtoEDMConverter'),
                                                   Verbosity = cms.untracked.int32(0), # 0 provides no output
                                                   # 1 provides basic output
                                                   # 2 provide more detailed output
                                                   Frequency = cms.untracked.int32(50),
                                                   MEPathToSave = cms.untracked.string('AlCaReco/SiStripLorentzAngle'))

# The actual sequence
seqALCARECOPromptCalibProdSiStripLorentzAngle = cms.Sequence(
    ALCARECOCalCosmicsFilterForSiStripLorentzAngle *
    ALCARECOSiStripLATrackFilterRefit *
    ALCARECOSiStripLACalib *
    MEtoEDMConvertSiStripLorentzAngle 
   )
