import FWCore.ParameterSet.Config as cms
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGains_cff import ALCARECOCalibrationTracks,ALCARECOCalibrationTracksRefit

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripHitEff = hltHighLevel.clone(
    HLTPaths = ['pathALCARECOSiStripCalMinBias'],
    throw = True, ## throw on unknown path names
    TriggerResultsTag = ("TriggerResults","","RECO")
)

# FIXME: the beam-spot should be kept in the AlCaReco (if not already there) and dropped from here
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *
from DQM.SiStripCommon.TkHistoMap_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *

# ------------------------------------------------------------------------------
# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOTrackFilterRefitForSiStripHitEff = cms.Sequence(ALCARECOCalibrationTracks  +
                                                        MeasurementTrackerEvent +
                                                        offlineBeamSpot +
                                                        ALCARECOCalibrationTracksRefit)

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration
from CalibTracker.SiStripHitEfficiency.siStripHitEfficiencyWorker_cfi import siStripHitEfficiencyWorker
ALCARECOSiStripHitEff =  siStripHitEfficiencyWorker.clone(
    dqmDir = "AlCaReco/SiStripHitEfficiency",
    lumiScalers= "scalersRawToDigi",
    addLumi = True,
    commonMode = "siStripDigis:CommonMode",
    addCommonMode= False,
    combinatorialTracks = "ALCARECOCalibrationTracksRefit",
    trajectories        = "ALCARECOCalibrationTracksRefit",
    siStripClusters     = "siStripClusters",
    siStripDigis        = "siStripDigis",
    trackerEvent        = "MeasurementTrackerEvent",
    # part 2
    Layer = 0, # =0 means do all layers
    Debug = True,
    # do not cut on the total number of tracks
    cutOnTracks = False,
    trackMultiplicity = 1000,
    # use or not first and last measurement of a trajectory (biases), default is false
    useFirstMeas = False,
    useLastMeas = False,
    useAllHitsFromTracksWithMissingHits = False,
    doMissingHitsRecovery = False,
    ## non-default settings
    ClusterMatchingMethod = 4,  # default 0  case0,1,2,3,4
    ClusterTrajDist       = 15, # default 64
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(ALCARECOSiStripHitEff,
                     useAllHitsFromTracksWithMissingHits = True,
                     doMissingHitsRecovery = True)

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
    ALCARECOTrackFilterRefitForSiStripHitEff *
    ALCARECOSiStripHitEff *
    MEtoEDMConvertSiStripHitEff)
