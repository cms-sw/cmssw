import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import *
from CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi import *

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
AfterAbortGapFilter = triggerResultsFilter.clone(
#                       triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                        triggerConditions = cms.vstring("HLT_ZeroBias_FirstCollisionAfterAbortGap_*"),
                        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                        l1tResults = cms.InputTag( "" ),
                        throw = cms.bool(False)
                   )

IsolatedMuonFilter = triggerResultsFilter.clone(
#                       triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                        triggerConditions = cms.vstring("HLT_IsoMu20_*"),
                        hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                        l1tResults = cms.InputTag( "" ),
                        throw = cms.bool(False)
                   )

ZeroBiasFilter = triggerResultsFilter.clone(
#                       triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                        triggerConditions = cms.vstring("HLT_ZeroBias_*"),
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

gainCalibrationTreeAagBunch = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAagBunch.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeAagBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAagBunch0T.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeStdBunch = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeStdBunch.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeStdBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeStdBunch0T.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeIsoMuon = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoMuon.outputCommands += OfflineChannelGainOutputCommands

gainCalibrationTreeIsoMuon0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoMuon0T.outputCommands += OfflineChannelGainOutputCommands



inputDataSequence = cms.Sequence( shallowEventRun + shallowTracks + shallowGainCalibration )

OfflineGainNtuple_StdBunch = cms.Sequence( ZeroBiasFilter + siStripBFieldOnFilter + 
                                           inputDataSequence * gainCalibrationTreeStdBunch )

OfflineGainNtuple_StdBunch0T = cms.Sequence( ZeroBiasFilter + siStripBFieldOffFilter + 
                                           inputDataSequence * gainCalibrationTreeStdBunch0T )

OfflineGainNtuple_AagBunch = cms.Sequence( siStripBFieldOnFilter + AfterAbortGapFilter +
                                           inputDataSequence * gainCalibrationTreeAagBunch )

OfflineGainNtuple_AagBunch0T = cms.Sequence( siStripBFieldOffFilter + AfterAbortGapFilter +
                                             inputDataSequence * gainCalibrationTreeAagBunch0T )

OfflineGainNtuple_IsoMuon = cms.Sequence( siStripBFieldOnFilter + AfterAbortGapFilter +
                                           inputDataSequence * gainCalibrationTreeIsoMuon )

OfflineGainNtuple_IsoMuon0T = cms.Sequence( siStripBFieldOffFilter + AfterAbortGapFilter +
                                             inputDataSequence * gainCalibrationTreeIsoMuon0T )

#OfflineGainNtuple = cms.Sequence( (shallowEventRun+
#                        shallowTracks +
#                        shallowGainCalibration) *
#                        gainCalibrationTree
#                       )
