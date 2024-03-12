import FWCore.ParameterSet.Config as cms
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerFEDIntegrityTask = DQMEDAnalyzer('ESFEDIntegrityTask',
                                               prefixME = cms.untracked.string('EcalPreshower'),
                                               ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                               ESKChipCollections = cms.InputTag("ecalPreshowerDigis"),
                                               FEDRawDataCollection = cms.InputTag("rawDataCollector"),
                                               OutputFile = cms.untracked.string(""),
                                               FEDDirName =cms.untracked.string("FEDIntegrity")
                                               )

# foo bar baz
# 4bisli9BOlGzQ
# 9qy3fjbgVPWQG
