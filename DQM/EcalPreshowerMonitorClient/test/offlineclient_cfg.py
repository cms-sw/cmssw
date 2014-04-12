import FWCore.ParameterSet.Config as cms

process = cms.Process("ESDQMCL")
  
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )


process.load("FWCore.MessageService.MessageLogger_cfi")

### include to get DQM histogramming services
#process.load("DQMServices.Core.DQM_cfg")
#process.load("DQMServices.Components.DQMStoreStats_cfi")


process.source = cms.Source("EmptySource")
    
    
process.load("DQM.EcalPreshowerMonitorClient.EcalPreshowerMonitorClient_cfi")
process.ecalPreshowerMonitorClient.OutputFile = 'clientoutput.root'
process.ecalPreshowerMonitorClient.InputFile = 'taskoutput.root'
process.ecalPreshowerMonitorClient.debug = True
process.ecalPreshowerMonitorClient.enableMonitorDaemon = False
process.ecalPreshowerMonitorClient.prescaleFactor = 50
process.ecalPreshowerMonitorClient.LookupTable = 'EventFilter/ESDigiToRaw/data/ES_lookup_table.dat' 

process.DQMStore = cms.Service("DQMStore")


process.p = cms.Path(process.ecalPreshowerMonitorClient)



