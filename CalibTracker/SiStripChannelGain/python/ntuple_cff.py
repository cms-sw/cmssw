import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import *

OfflineChannelGainOutputCommands =  [
                                     'keep *_shallowEventRun_*_*',
                                     'keep *_shallowTracks_trackchi2ndof_*',
                                     'keep *_shallowTracks_trackmomentum_*',
                                     'keep *_shallowTracks_trackpt_*',
                                     'keep *_shallowTracks_tracketa_*',
                                     'keep *_shallowTracks_trackphi_*',
                                     'keep *_shallowTracks_trackhitsvalid_*',
                                     'keep *_shallowGainCalibration_*_*'
                                    ]

gainCalibrationTree = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
gainCalibrationTree.outputCommands += OfflineChannelGainOutputCommands

OfflineGainNtuple = cms.Sequence( (shallowEventRun+
                        shallowTracks +
                        shallowGainCalibration) *
                        gainCalibrationTree
                       )
