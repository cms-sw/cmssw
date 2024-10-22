import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pixel_lumi_dqm  = DQMEDAnalyzer('PixelLumiDQM',
                                 pixelClusterLabel = cms.untracked.InputTag("siPixelClusters"),
                                 includePixelClusterInfo = cms.untracked.bool(True),
                                 includePixelQualCheckHistos = cms.untracked.bool(True),
                                 # This is the correct list of modules to be ignored for 2012.
                                 deadModules = cms.untracked.vuint32(),
                                 # Only count pixel clusters with a minimum number of pixels.
                                 minNumPixelsPerCluster = cms.untracked.int32(2),
                                 # Only count pixel clusters with a minimum charge.
                                 minChargePerCluster = cms.untracked.double(15000.),
                                 #log file defined in class but not here as parameter
                                 logFileName = cms.untracked.string('/tmp/pixel_lumi.txt')
                                 )
