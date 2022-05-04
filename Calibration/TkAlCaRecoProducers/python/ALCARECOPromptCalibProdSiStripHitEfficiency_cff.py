import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripHitEff = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOSiStripCalMinBias'],
    throw = True, ## throw on unknown path names
    TriggerResultsTag = ("TriggerResults","","RECO")
)

# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiStripCalMinBias
# to have access to transient objects produced during RECO step and not saved
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOMonitoringTracks = AlignmentTrackSelector.clone(
    #    src = 'generalTracks',
    src = 'ALCARECOSiStripCalMinBias',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.8,
    nHitMin = 6,
    chi2nMax = 10.)

# FIXME: the beam-spot should be kept in the AlCaReco (if not already there) and dropped from here
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
from DQM.SiStripCommon.TkHistoMap_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *

ALCARECOMonitoringTracksRefit = TrackRefitter.clone(
         src = "ALCARECOMonitoringTracks",
         NavigationSchool = cms.string("")
)

# ------------------------------------------------------------------------------
# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOTrackFilterRefit = cms.Sequence(ALCARECOMonitoringTracks +
                                        MeasurementTrackerEvent +
                                        offlineBeamSpot +
                                        ALCARECOMonitoringTracksRefit)

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration
from CalibTracker.SiStripHitEfficiency.siStripHitEfficiencyWorker_cfi import siStripHitEfficiencyWorker
ALCARECOSiStripHitEff =  siStripHitEfficiencyWorker.clone(
    lumiScalers= "scalersRawToDigi",
    addLumi = True,
    commonMode = "siStripDigis:CommonMode",
    addCommonMode= False,
    combinatorialTracks = "ALCARECOMonitoringTracksRefit",
    trajectories        = "ALCARECOMonitoringTracksRefit",
    siStripClusters     = "siStripClusters",
    siStripDigis        = "siStripDigis",
    trackerEvent        = "MeasurementTrackerEvent",
    # part 2
    Layer = 0, # =0 means do all layers
    Debug = True,
    # do not cut on the total number of tracks
    cutOnTracks = True,
    trackMultiplicity = 100,
    # use or not first and last measurement of a trajectory (biases), default is false
    useFirstMeas = False,
    useLastMeas = False,
    useAllHitsFromTracksWithMissingHits = False,
    ## non-default settings
    ClusterMatchingMethod = 4,  # default 0  case0,1,2,3,4
    ClusterTrajDist       = 15, # default 64
)

# ----------------------------------------------------------------------------
MEtoEDMConvertSiStripHitEff = cms.EDProducer("MEtoEDMConverter",
                                             Name = cms.untracked.string('MEtoEDMConverter'),
                                             Verbosity = cms.untracked.int32(0), # 0 provides no output
                                             # 1 provides basic output
                                             # 2 provide more detailed output
                                             Frequency = cms.untracked.int32(50),
                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripHitEfficiency'))

# The actual sequence
seqALCARECOPromptCalibProdSiStripHitEfficiency = cms.Sequence(
    ALCARECOCalMinBiasFilterForSiStripHitEff *
    ALCARECOTrackFilterRefit *
    ALCARECOSiStripHitEff *
    MEtoEDMConvertSiStripHitEff)
