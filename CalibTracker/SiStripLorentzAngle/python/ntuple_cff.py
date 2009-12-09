import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi import *

LorentzAngleOutputCommands =  [ 'keep *_*_clusterdetid_*',
                                'keep *_*_clusterwidth_*',
                                'keep *_*_clustervariance_*',
                                'keep *_*_tsostrackmulti_*',
                                'keep *_*_tsosdriftx_*',
                                'keep *_*_tsosdriftz_*',
                                'keep *_*_tsoslocalpitch_*',
                                'keep *_*_tsoslocaltheta_*',
                                'keep *_*_tsoslocalphi_*',
                                'keep *_*_tsosBdotY_*',
                                #'keep *_*_tsoslocaly_*',
                                'keep *_*_tsosglobalZofunitlocalY_*']

laCalibrationTree = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
laCalibrationTree.outputCommands += LorentzAngleOutputCommands

LorentzAngleNtuple = cms.Sequence( (shallowClusters +
                                    shallowTrackClusters) *
                                   laCalibrationTree
                                   )
