import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ecalPreshowerLocalMonitorClient = DQMEDHarvester('EcalPreshowerMonitorClient',	
                                            LookupTable = cms.untracked.FileInPath('EventFilter/ESDigiToRaw/data/ES_lookup_table.dat'),
                                            enabledClients = cms.untracked.vstring('Pedestal'),
                                            prefixME = cms.untracked.string('EcalPreshower'),
                                            prescaleFactor = cms.untracked.int32(1),
                                            verbose = cms.untracked.bool(True),
                                            debug = cms.untracked.bool(True),
                                            fitPedestal = cms.untracked.bool(False)
                                            )
