import FWCore.ParameterSet.Config as cms

ecalPreshowerMonitorClient = cms.EDAnalyzer('EcalPreshowerMonitorClient',	
                                            LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
                                            OutputFile = cms.untracked.string(''),
                                            InputFile = cms.untracked.string(''),
                                            enableCleanup = cms.untracked.bool(False),
                                            enabledClients = cms.untracked.vstring('Integrity',
                                                                                   'Summary'
                                                                                   ),
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            prescaleFactor = cms.untracked.int32(1),
                                            verbose = cms.untracked.bool(False),
                                            debug = cms.untracked.bool(False),
                                            fitPedestal = cms.untracked.bool(True)
                                            
                                            )
