import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagCommon_cff import *

bTagAnalysis = cms.EDAnalyzer("BTagPerformanceAnalyzerOnData",
                              bTagCommonBlock,
                              )

bTagHarvest = DQMEDHarvester("BTagPerformanceHarvester",
                             bTagCommonBlock,
                             produceEps = cms.bool(False),
                             producePs = cms.bool(False),
                             flavPlots = cms.string("all"),
                             differentialPlots = cms.bool(False), #not needed in validation procedure, put True to produce them 
                             )


