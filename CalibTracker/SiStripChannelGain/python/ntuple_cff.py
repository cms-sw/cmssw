import FWCore.ParameterSet.Config as cms

#Gain
from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowGainCalibration_cfi import *


#LA
from CalibTracker.SiStripCommon.ShallowClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi import *


commonCalibTreeOutputCommands =  [
#CommonCalibration
                                     'keep *_shallowEventRun_*_*',
                                     'keep *_shallowTracks_trackchi2ndof_*',
                                     'keep *_shallowTracks_trackmomentum_*',
                                     'keep *_shallowTracks_trackpt_*',
                                     'keep *_shallowTracks_tracketa_*',
                                     'keep *_shallowTracks_trackphi_*',
                                     'keep *_shallowTracks_trackhitsvalid_*',
                                     'keep *_shallowGainCalibration_*_*',
                                     'keep *_shallowClusters_clusterdetid_*',
                                     'keep *_shallowClusters_clusterwidth_*',
                                     'keep *_shallowClusters_clustervariance_*',
                                     'keep *_shallowTrackClusters_tsostrackmulti_*',
                                     'keep *_shallowTrackClusters_tsosdriftx_*',
                                     'keep *_shallowTrackClusters_tsosdriftz_*',
                                     'keep *_shallowTrackClusters_tsoslocaltheta_*',
                                     'keep *_shallowTrackClusters_tsoslocalphi_*',
                                     'keep *_shallowTrackClusters_tsosBdotY_*',
                                    #'keep *_shallowTrackClusters_tsoslocaly_*',
                                     'keep *_shallowTrackClusters_tsosglobalZofunitlocalY_*',
                                     'keep *_shallowTrackClusters_tsostrackindex_*',
                                ]


commonCalibrationTree = cms.EDAnalyzer("ShallowTree", outputCommands = cms.untracked.vstring('drop *'))
commonCalibrationTree.outputCommands += commonCalibTreeOutputCommands

OfflineGainNtuple = cms.Sequence( (shallowEventRun+  #Gain
                        shallowTracks +              #Gain
                        shallowClusters +            #LA
                        shallowTrackClusters +       #LA
                        shallowGainCalibration) *    #GAIN
                        commonCalibrationTree
                       )
