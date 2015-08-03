import FWCore.ParameterSet.Config as cms

DQMDbHarvester = cms.EDAnalyzer("DQMDbHarvester",
       numMonitorName = cms.string("Physics/TopTest/ElePt_leading_HLT_matched"),
       denMonitorName = cms.string("Physics/TopTest/ElePt_leading"),   
	   #histogramsPerLumi = cms.vstring("Physics\/TopTest\/Vertex_number","Physics\/TopTest\/pfMet","Physics\/TopTest\/NElectrons","Physics\/TopTest\/ElePt_leading","Physics\/TopTest\/EleEta_leading","Physics\/TopTest\/ElePhi_leading"),
	   histogramsPerLumi = cms.vstring("Physics\/TopTest\/Vertex_number","Physics\/TopTest\/pfMet","Physics\/TopTest\/NElectrons"),
	   histogramsPerRun = cms.vstring("Physics\/TopTest\/ElePt_leading","Physics\/TopTest\/EleEta_leading","Physics\/TopTest\/ElePhi_leading"),
	   
       DBParameters = cms.PSet(
       authenticationPath = cms.untracked.string(''),
       messageLevel = cms.untracked.int32(3),
       enableConnectionSharing = cms.untracked.bool(True),
       connectionTimeOut = cms.untracked.int32(60),
       enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False),
       connectionRetrialTimeOut = cms.untracked.int32(60),
       connectionRetrialPeriod = cms.untracked.int32(10),
       enablePoolAutomaticCleanUp = cms.untracked.bool(False),
       ),
       connect = cms.string('sqlite_file:testDQM2DB_3.db'),
)
