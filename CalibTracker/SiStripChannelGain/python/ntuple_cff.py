import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import *
from CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi import *

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
IsolatedBunch = triggerResultsFilter.clone(
#                     triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                      triggerConditions = cms.vstring("HLT_ZeroBias_IsolatedBunches_*"),
                      hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                      l1tResults = cms.InputTag( "" ),
                      throw = cms.bool(False)
                   )


OfflineChannelGainOutputCommands =  [
                                     'keep *_shallowEventRun_*_*',
                                     'keep *_shallowTracks_trackchi2ndof_*',
                                     'keep *_shallowTracks_trackmomentum_*',
                                     'keep *_shallowTracks_trackpt_*',
                                     'keep *_shallowTracks_tracketa_*',
                                     'keep *_shallowTracks_trackphi_*',
                                     'keep *_shallowTracks_trackhitsvalid_*',
                                     'keep *_shallowTracks_trackalgo_*',
                                     'keep *_shallowGainCalibration_*_*'
                                    ]

gainCalibrationTreeIsoBunch   = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoBunch.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeIsoBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoBunch0T.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeAllBunch = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAllBunch.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeAllBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAllBunch0T.outputCommands += OfflineChannelGainOutputCommands


OfflineGainNtuple_AllBunch = cms.Sequence( siStripBFieldOnFilter + 
                                          (shallowEventRun +
                                           shallowTracks +
                                           shallowGainCalibration) *
                                          gainCalibrationTreeAllBunch
                                         )

OfflineGainNtuple_AllBunch0T = cms.Sequence( siStripBFieldOffFilter + 
                                            (shallowEventRun +
                                             shallowTracks +
                                             shallowGainCalibration) *
                                            gainCalibrationTreeAllBunch0T
                                           )

OfflineGainNtuple_IsoBunch = cms.Sequence( siStripBFieldOnFilter + IsolatedBunch +
                                          (shallowEventRun +
                                           shallowTracks +
                                           shallowGainCalibration) *
                                          gainCalibrationTreeIsoBunch
                                         )

OfflineGainNtuple_IsoBunch0T = cms.Sequence( siStripBFieldOffFilter + IsolatedBunch +
                                            (shallowEventRun +
                                             shallowTracks +
                                             shallowGainCalibration) *
                                            gainCalibrationTreeIsoBunch0T
                                           )

#OfflineGainNtuple = cms.Sequence( (shallowEventRun+
#                        shallowTracks +
#                        shallowGainCalibration) *
#                        gainCalibrationTree
#                       )
