import FWCore.ParameterSet.Config as cms

# ------------------------------------------------------------------------------
# configure a filter to run only on the events selected by TkAlMinBias AlcaReco
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ALCARECOCalMinBiasFilterForSiStripGains = copy.deepcopy(hltHighLevel)
ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = ['pathALCARECOSiStripCalMinBias']
ALCARECOCalMinBiasFilterForSiStripGains.throw = True ## dont throw on unknown path names
ALCARECOCalMinBiasFilterForSiStripGains.TriggerResultsTag = cms.InputTag("TriggerResults","","RECO")
#process.TkAlMinBiasFilterForBS.eventSetupPathsKey = 'pathALCARECOTkAlMinBias:RECO'
#ALCARECODtCalibHLTFilter.andOr = True ## choose logical OR between Triggerbits


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
# This is the module actually doing the calibration

from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib as ALCARECOSiStripCalib
ALCARECOSiStripCalib.AlgoMode = cms.untracked.string('PCL')
ALCARECOSiStripCalib.Tracks   = cms.untracked.InputTag('ALCARECOCalibrationTracksRefit')
ALCARECOSiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)
ALCARECOSiStripCalib.harvestingMode    = cms.untracked.bool(False)
ALCARECOSiStripCalib.doStoreOnDB         = cms.bool(False)


# ------------------------------------------------------------------------------
MEtoEDMConvertSiStripGains = cms.EDProducer("MEtoEDMConverter",
                                            Name = cms.untracked.string('MEtoEDMConverter'),
                                            Verbosity = cms.untracked.int32(0), # 0 provides no output
                                            # 1 provides basic output
                                            # 2 provide more detailed output
                                            Frequency = cms.untracked.int32(50),
                                            MEPathToSave = cms.untracked.string('AlCaReco/SiStripGains'),
                                            deleteAfterCopy = cms.untracked.bool(False)
)






# the actual sequence
seqALCARECOPromptCalibProdSiStripGains = cms.Sequence(ALCARECOCalMinBiasFilterForSiStripGains *
                                                      ALCARECOTrackFilterRefit *
                                                      ALCARECOSiStripCalib *
                                                      MEtoEDMConvertSiStripGains)


