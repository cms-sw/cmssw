import FWCore.ParameterSet.Config as cms
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerTrendTask = DQMEDAnalyzer('ESTrendTask',
                                        RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
                                        ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                        prefixME = cms.untracked.string("EcalPreshower") 
                                        )

# foo bar baz
# 3rx4J6JOZI8zK
# B3thOBctItpH2
