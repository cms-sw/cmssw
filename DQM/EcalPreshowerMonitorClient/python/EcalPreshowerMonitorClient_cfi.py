import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ecalPreshowerMonitorClient = DQMEDHarvester('EcalPreshowerMonitorClient',	
                                            LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
                                            enabledClients = cms.untracked.vstring('Integrity',
                                                                                   'Summary'
                                                                                   ),
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            verbose = cms.untracked.bool(False),
                                            debug = cms.untracked.bool(False),
                                            fitPedestal = cms.untracked.bool(True),
                                            cloneME = cms.untracked.bool(True)
                                            )
