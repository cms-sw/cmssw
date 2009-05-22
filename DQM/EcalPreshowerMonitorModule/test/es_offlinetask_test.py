import FWCore.ParameterSet.Config as cms

process = cms.Process("ESDQMTK")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
  
process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource", 
	fileNames = cms.untracked.vstring(
	    "file:raw2digi.root"	    
	) 
    )
    
process.load("DQM.EcalPreshowerMonitorModule.EcalPreshowerMonitorTasks_cfi")

process.DQMStore = cms.Service("DQMStore")

process.p = cms.Path(process.ecalPreshowerDefaultTasksSequence)



