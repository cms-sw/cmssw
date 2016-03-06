import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi import *
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripGains = copy.deepcopy(hltHighLevel)
ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = ['pathALCARECOSiStripCalMinBias']
ALCARECOCalMinBiasFilterForSiStripGains.throw = True ## dont throw on unknown path names
ALCARECOCalMinBiasFilterForSiStripGains.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
ZeroBiasGC                 = triggerResultsFilter.clone(
                                triggerConditions = cms.vstring("HLT_ZeroBias_v*"),
                                hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                l1tResults = cms.InputTag( "" ),
                                throw = cms.bool(False)
                             )

ZeroBiasIsolatedBunchGC    = triggerResultsFilter.clone(
                                triggerConditions = cms.vstring("HLT_ZeroBias_IsolatedBunches_*"),
                                hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                l1tResults = cms.InputTag( "" ),
                                throw = cms.bool(False)
                             )

HLTPhysicsGC               = triggerResultsFilter.clone(
                                triggerConditions = cms.vstring("HLT_Physics_v*"),
                                hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                l1tResults = cms.InputTag( "" ),
                                throw = cms.bool(False)
                             )

HLTPhysicsIsolatedBunchGC  = triggerResultsFilter.clone(
                                triggerConditions = cms.vstring("HLT_Physics_IsolatedBunches_*"),
                                hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                l1tResults = cms.InputTag( "" ),
                                throw = cms.bool(False)
                             )

#from L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff import *
#from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import hltLevel1GTSeed
#HTTFilter = hltLevel1GTSeed.clone(  
#              #L1SeedsLogicalExpression = cms.string("L1_HTT125 OR L1_HTT150 OR L1_HTT175" ),
#              L1SeedsLogicalExpression = cms.string("L1_HTT125 OR L1_HTT150"),
#              #saveTags = cms.bool( True ),
#              #L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
#              #L1UseL1TriggerObjectMaps = cms.bool( True ),
#              #L1UseAliasesForSeeding = cms.bool( True ),
#              #L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
#              #L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
#              #L1NrBxInEvent = cms.int32( 3 ),
#              L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
#              #L1TechTriggerSeeding = cms.bool( False )
#            )
# ------------------------------------------------------------------------------


# FIXME: are the following blocks needed?

#this block is there to solve issue related to SiStripQualityRcd
#process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
#process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
#process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")


#process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")



# ------------------------------------------------------------------------------
# This is the sequence for track refitting of the track saved by SiStripCalMinBias
# to have access to transient objects produced during RECO step and not saved

from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOCalibrationTracks = AlignmentTrackSelector.clone(
    #    src = 'generalTracks',
    src = 'ALCARECOSiStripCalMinBias',
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

ALCARECOCalibrationTracksRefit = TrackRefitter.clone(src = cms.InputTag("ALCARECOCalibrationTracks"),
                                                     NavigationSchool = cms.string("")
                                                     )

# refit and BS can be dropped if done together with RECO.
# track filter can be moved in acalreco if no otehr users
ALCARECOTrackFilterRefit = cms.Sequence(ALCARECOCalibrationTracks +
                                        offlineBeamSpot +
                                        ALCARECOCalibrationTracksRefit )

# ------------------------------------------------------------------------------
# Get the information you need from the tracks, calibTree-style to have no code difference
from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import shallowEventRun
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import shallowTracks
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import shallowGainCalibration
ALCARECOShallowEventRun = shallowEventRun.clone()
ALCARECOShallowTracks = shallowTracks.clone(Tracks=cms.InputTag('ALCARECOCalibrationTracksRefit'))
ALCARECOShallowGainCalibration = shallowGainCalibration.clone(Tracks=cms.InputTag('ALCARECOCalibrationTracksRefit'))
ALCARECOShallowSequence = cms.Sequence(ALCARECOShallowEventRun*ALCARECOShallowTracks*ALCARECOShallowGainCalibration)

# ------------------------------------------------------------------------------
# This is the module actually doing the calibration

from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib
ALCARECOSiStripCalibAllBunch = SiStripCalib.clone()
ALCARECOSiStripCalibAllBunch.AlgoMode            = cms.untracked.string('PCL')
ALCARECOSiStripCalibAllBunch.Tracks              = cms.untracked.InputTag('ALCARECOCalibrationTracksRefit')
ALCARECOSiStripCalibAllBunch.FirstSetOfConstants = cms.untracked.bool(False)
ALCARECOSiStripCalibAllBunch.harvestingMode      = cms.untracked.bool(False)
ALCARECOSiStripCalibAllBunch.splitDQMstat        = cms.untracked.bool(True)
ALCARECOSiStripCalibAllBunch.calibrationMode     = cms.untracked.string('AllBunch')
ALCARECOSiStripCalibAllBunch.doStoreOnDB         = cms.bool(False)
ALCARECOSiStripCalibAllBunch.gain.label          = cms.untracked.string('ALCARECOShallowGainCalibration')
ALCARECOSiStripCalibAllBunch.evtinfo.label       = cms.untracked.string('ALCARECOShallowEventRun')
ALCARECOSiStripCalibAllBunch.tracks.label        = cms.untracked.string('ALCARECOShallowTracks')


ALCARECOSiStripCalibAllBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('AllBunch0T'))
ALCARECOSiStripCalibIsoBunch   = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('IsoBunch'))
ALCARECOSiStripCalibIsoBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('IsoBunch0T'))

#ALCARECOSiStripCalibHLTAllBunch   = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HLTAllBunch'))
#ALCARECOSiStripCalibHLTAllBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HLTAllBunch0T'))
#ALCARECOSiStripCalibHLTIsoBunch   = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HLTIsoBunch'))
#ALCARECOSiStripCalibHLTIsoBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HLTIsoBunch0T'))

#ALCARECOSiStripCalibHTTAllBunch   = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HTTAllBunch'))
#ALCARECOSiStripCalibHTTAllBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HTTAllBunch0T'))
#ALCARECOSiStripCalibHTTIsoBunch   = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HTTIsoBunch'))
#ALCARECOSiStripCalibHTTIsoBunch0T = ALCARECOSiStripCalibAllBunch.clone(calibrationMode=cms.untracked.string('HTTIsoBunch0T'))
# ------------------------------------------------------------------------------
MEtoEDMConvertSiStripGains = cms.EDProducer("MEtoEDMConverter",
                                            Name = cms.untracked.string('MEtoEDMConverter'),
                                            Verbosity = cms.untracked.int32(2), # 0 provides no output
                                            # 1 provides basic output
                                            # 2 provide more detailed output
                                            Frequency = cms.untracked.int32(50),
                                            MEPathToSave = cms.untracked.string('AlCaReco/SiStripGains'),
                                            deleteAfterCopy = cms.untracked.bool(False)
)

seqALCARECOPromptCalibProdSiStripGainsMEtoEDM = cms.Sequence( MEtoEDMConvertSiStripGains )

MEtoEDMConvertSiStripGainsAllBunch = cms.EDProducer("MEtoEDMConverter",
                                            Name = cms.untracked.string('MEtoEDMConverter'),
                                            Verbosity = cms.untracked.int32(2), # 0 provides no output
                                            # 1 provides basic output
                                            # 2 provide more detailed output
                                            Frequency = cms.untracked.int32(50),
                                            MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsAllBunch'),
                                            deleteAfterCopy = cms.untracked.bool(True)
)

MEtoEDMConvertSiStripGainsAllBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsAllBunch0T')
                                                                               )

MEtoEDMConvertSiStripGainsIsoBunch   = MEtoEDMConvertSiStripGainsAllBunch.clone(
                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsIsoBunch')
                                                                               )

MEtoEDMConvertSiStripGainsIsoBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsIsoBunch0T')
                                                                               )


#MEtoEDMConvertSiStripGainsHLTAllBunch   = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHLTAllBunch')
#                                                                               )

#MEtoEDMConvertSiStripGainsHLTAllBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHLTAllBunch0T')
#                                                                               )

#MEtoEDMConvertSiStripGainsHLTIsoBunch   = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHLTIsoBunch')
#                                                                               )

#MEtoEDMConvertSiStripGainsHLTIsoBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHLTIsoBunch0T')
#                                                                               )


#MEtoEDMConvertSiStripGainsHTTAllBunch   = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHTTAllBunch')
#                                                                               )

#MEtoEDMConvertSiStripGainsHTTAllBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHTTAllBunch0T')
#                                                                               )

#MEtoEDMConvertSiStripGainsHTTIsoBunch   = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHTTIsoBunch')
#                                                                               )

#MEtoEDMConvertSiStripGainsHTTIsoBunch0T = MEtoEDMConvertSiStripGainsAllBunch.clone(
#                                             MEPathToSave = cms.untracked.string('AlCaReco/SiStripGainsHTTIsoBunch0T')
#                                                                               )




# the actual sequence
seqALCARECOPromptCalibProdSiStripGainsAllBunch = cms.Sequence(
   ALCARECOCalMinBiasFilterForSiStripGains * ZeroBiasGC * siStripBFieldOnFilter *
   ALCARECOTrackFilterRefit *
   ALCARECOShallowSequence *
   ALCARECOSiStripCalibAllBunch *
   MEtoEDMConvertSiStripGainsAllBunch
)

seqALCARECOPromptCalibProdSiStripGainsAllBunch0T = cms.Sequence(
   ALCARECOCalMinBiasFilterForSiStripGains * ZeroBiasGC * siStripBFieldOffFilter *
   ALCARECOTrackFilterRefit *
   ALCARECOShallowSequence *
   ALCARECOSiStripCalibAllBunch0T *
   MEtoEDMConvertSiStripGainsAllBunch0T
)

seqALCARECOPromptCalibProdSiStripGainsIsoBunch = cms.Sequence(
   ALCARECOCalMinBiasFilterForSiStripGains * ZeroBiasIsolatedBunchGC * siStripBFieldOnFilter *
   ALCARECOTrackFilterRefit *
   ALCARECOShallowSequence *
   ALCARECOSiStripCalibIsoBunch *
   MEtoEDMConvertSiStripGainsIsoBunch
)

seqALCARECOPromptCalibProdSiStripGainsIsoBunch0T = cms.Sequence(
   ALCARECOCalMinBiasFilterForSiStripGains * ZeroBiasIsolatedBunchGC * siStripBFieldOffFilter *
   ALCARECOTrackFilterRefit *
   ALCARECOShallowSequence *
   ALCARECOSiStripCalibIsoBunch0T *
   MEtoEDMConvertSiStripGainsIsoBunch0T
)

#seqALCARECOPromptCalibProdSiStripGainsHLTAllBunch = cms.Sequence(
#  ALCARECOCalMinBiasFilterForSiStripGains * HLTPhysicsGC * siStripBFieldOnFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHLTAllBunch *
#   MEtoEDMConvertSiStripGainsHLTAllBunch
#)

#seqALCARECOPromptCalibProdSiStripGainsHLTAllBunch0T = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HLTPhysicsGC * siStripBFieldOffFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHLTAllBunch0T *
#   MEtoEDMConvertSiStripGainsHLTAllBunch0T
#)

#seqALCARECOPromptCalibProdSiStripGainsHLTIsoBunch = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HLTPhysicsIsolatedBunchGC * siStripBFieldOnFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHLTIsoBunch *
#   MEtoEDMConvertSiStripGainsHLTIsoBunch
#)

#seqALCARECOPromptCalibProdSiStripGainsHLTIsoBunch0T = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HLTPhysicsIsolatedBunchGC * siStripBFieldOffFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHLTIsoBunch0T *
#   MEtoEDMConvertSiStripGainsHLTIsoBunch0T
#)

#seqALCARECOPromptCalibProdSiStripGainsHTTAllBunch = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HTTFilter * ZeroBiasGC * siStripBFieldOnFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHTTAllBunch *
#   MEtoEDMConvertSiStripGainsHTTAllBunch
#)

#seqALCARECOPromptCalibProdSiStripGainsHTTAllBunch0T = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HTTFilter * ZeroBiasGC * siStripBFieldOffFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHTTAllBunch0T *
#   MEtoEDMConvertSiStripGainsHTTAllBunch0T
#)

#seqALCARECOPromptCalibProdSiStripGainsHTTIsoBunch = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HTTFilter * ZeroBiasIsolatedBunchGC * siStripBFieldOnFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHTTIsoBunch *
#   MEtoEDMConvertSiStripGainsHTTIsoBunch
#)

#seqALCARECOPromptCalibProdSiStripGainsHTTIsoBunch0T = cms.Sequence(
#   ALCARECOCalMinBiasFilterForSiStripGains * HTTFilter * ZeroBiasIsolatedBunchGC * siStripBFieldOffFilter *
#   ALCARECOTrackFilterRefit *
#   ALCARECOShallowSequence *
#   ALCARECOSiStripCalibHTTIsoBunch0T *
#   MEtoEDMConvertSiStripGainsHTTIsoBunch0T
#)
