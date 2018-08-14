import FWCore.ParameterSet.Config as cms
    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerIntegrityTask = DQMEDAnalyzer('ESIntegrityTask',
                                            LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            ESDCCCollections = cms.InputTag("ecalPreshowerDigis"),
                                            ESKChipCollections = cms.InputTag("ecalPreshowerDigis"),
                                            OutputFile = cms.untracked.string(""),
                                            DoLumiAnalysis = cms.bool(False)
                                            )

