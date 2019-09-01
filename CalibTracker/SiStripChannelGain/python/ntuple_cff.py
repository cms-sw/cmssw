import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import *
from CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi import *

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
AAGFilter = triggerResultsFilter.clone(triggerConditions = cms.vstring("HLT_ZeroBias_FirstCollisionAfterAbortGap_*"),
                                       hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                       l1tResults = cms.InputTag( "" ),
                                       throw = cms.bool(False)
                                       )

IsolatedMuonFilter = triggerResultsFilter.clone(triggerConditions = cms.vstring("HLT_IsoMu20_*"),
                                                hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                                l1tResults = cms.InputTag( "" ),
                                                throw = cms.bool(False)
                                                )

ZeroBiasFilter = triggerResultsFilter.clone(triggerConditions = cms.vstring("HLT_ZeroBias_*",),
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


# Use compressiong settings of TFile
# see https://root.cern.ch/root/html534/TFile.html#TFile:SetCompressionSettings
# settings = 100 * algorithm + level
# level is from 1 (small) to 9 (large compression)
# algo: 1 (ZLIB), 2 (LMZA)
# see more about compression & performance: https://root.cern.ch/root/html534/guides/users-guide/InputOutput.html#compression-and-performance
compressionSettings = 201

gainCalibrationTreeAagBunch = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAagBunch.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeAagBunch.CompressionSettings = cms.untracked.int32(compressionSettings)

gainCalibrationTreeAagBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeAagBunch0T.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeAagBunch0T.CompressionSettings = cms.untracked.int32(compressionSettings)

gainCalibrationTreeStdBunch = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeStdBunch.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeStdBunch.CompressionSettings = cms.untracked.int32(compressionSettings)

gainCalibrationTreeStdBunch0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeStdBunch0T.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeStdBunch0T.CompressionSettings = cms.untracked.int32(compressionSettings)

gainCalibrationTreeIsoMuon = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoMuon.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeIsoMuon.CompressionSettings = cms.untracked.int32(compressionSettings)

gainCalibrationTreeIsoMuon0T = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTreeIsoMuon0T.outputCommands += OfflineChannelGainOutputCommands
gainCalibrationTreeIsoMuon0T.CompressionSettings = cms.untracked.int32(compressionSettings)


inputDataSequence = cms.Sequence( shallowEventRun + shallowTracks + shallowGainCalibration )

OfflineGainNtuple_StdBunch = cms.Sequence( ZeroBiasFilter + ~AAGFilter + siStripBFieldOnFilter + 
                                           inputDataSequence * gainCalibrationTreeStdBunch )

OfflineGainNtuple_StdBunch0T = cms.Sequence( ZeroBiasFilter + ~AAGFilter + siStripBFieldOffFilter + 
                                           inputDataSequence * gainCalibrationTreeStdBunch0T )

OfflineGainNtuple_AagBunch = cms.Sequence( siStripBFieldOnFilter + AAGFilter +
                                           inputDataSequence * gainCalibrationTreeAagBunch )

OfflineGainNtuple_AagBunch0T = cms.Sequence( siStripBFieldOffFilter + AAGFilter +
                                             inputDataSequence * gainCalibrationTreeAagBunch0T )

OfflineGainNtuple_IsoMuon = cms.Sequence( siStripBFieldOnFilter + IsolatedMuonFilter +
                                           inputDataSequence * gainCalibrationTreeIsoMuon )

OfflineGainNtuple_IsoMuon0T = cms.Sequence( siStripBFieldOffFilter + IsolatedMuonFilter +
                                             inputDataSequence * gainCalibrationTreeIsoMuon0T )

#OfflineGainNtuple = cms.Sequence( (shallowEventRun+
#                        shallowTracks +
#                        shallowGainCalibration) *
#                        gainCalibrationTree
#                       )
