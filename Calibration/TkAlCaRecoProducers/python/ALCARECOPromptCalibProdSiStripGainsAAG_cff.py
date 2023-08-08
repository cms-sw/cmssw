import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi import *
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripGainsAAG = copy.deepcopy(hltHighLevel)
ALCARECOCalMinBiasFilterForSiStripGainsAAG.HLTPaths = ['pathALCARECOSiStripCalMinBiasAAG']
ALCARECOCalMinBiasFilterForSiStripGainsAAG.throw = True ## dont throw on unknown path names
ALCARECOCalMinBiasFilterForSiStripGainsAAG.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


# ****************************************************************************
# ** Uncomment the following lines to set the LVL1 bit filter for the HTTxx **
# ****************************************************************************

#from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
#from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
#HTTFilter = hltLevel1GTSeed.clone(  
#              #L1SeedsLogicalExpression = cms.string("L1_HTT125 OR L1_HTT150 OR L1_HTT175" ),
#              L1SeedsLogicalExpression = cms.string("L1_HTT125 OR L1_HTT150"),
#              L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
#            )
# ----------------------------------------------------------------------------



# FIXME: are the following blocks needed?

#this block is there to solve issue related to SiStripQualityRcd
#process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
#process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
#process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")


# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiStripCalMinBias
# to have access to transient objects produced during RECO step and not saved

from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOCalibrationTracksAAG = AlignmentTrackSelector.clone(
    #    src = 'generalTracks',
    src = 'ALCARECOSiStripCalMinBiasAAG',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0.8,
    nHitMin = 6,
    chi2nMax = 10.,
    )

# FIXME: the beam-spot should be kept in the AlCaReco (if not already there) and dropped from here
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoTracker.TrackProducer.TrackRefitter_cfi import *

ALCARECOCalibrationTracksRefitAAG = TrackRefitter.clone(src = cms.InputTag("ALCARECOCalibrationTracksAAG"),
                                                     NavigationSchool = cms.string("")
                                                     )

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no other users
from RecoLocalTracker.SiPixelRecHits.SiPixelTemplateStoreESProducer_cfi import SiPixelTemplateStoreESProducer
ALCARECOTrackFilterRefitAAG = cms.Sequence(ALCARECOCalibrationTracksAAG +
                                           offlineBeamSpot +
                                           ALCARECOCalibrationTracksRefitAAG,
                                           cms.Task(SiPixelTemplateStoreESProducer) )

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration
from CalibTracker.SiStripChannelGain.SiStripGainsPCLWorker_cfi import SiStripGainsPCLWorker
ALCARECOSiStripCalibAAG = SiStripGainsPCLWorker.clone(
        tracks              = cms.InputTag('ALCARECOCalibrationTracksRefitAAG'),
        FirstSetOfConstants = cms.untracked.bool(False),
        DQMdir              = cms.untracked.string('AlCaReco/SiStripGainsAAG'),
        calibrationMode     = cms.untracked.string('AagBunch')
        )

# ----------------------------------------------------------------------------

MEtoEDMConvertSiStripGainsAAG = cms.EDProducer("MEtoEDMConverter",
                                            Name = cms.untracked.string('MEtoEDMConverter'),
                                            Verbosity = cms.untracked.int32(1), # 0 provides no output
                                            # 1 provides basic output
                                            # 2 provide more detailed output
                                            Frequency = cms.untracked.int32(50),
                                            MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsAAG'),
)

# The actual sequence
seqALCARECOPromptCalibProdSiStripGainsAAG = cms.Sequence(
   ALCARECOCalMinBiasFilterForSiStripGainsAAG *
   ALCARECOTrackFilterRefitAAG *
   ALCARECOSiStripCalibAAG *
   MEtoEDMConvertSiStripGainsAAG
)
