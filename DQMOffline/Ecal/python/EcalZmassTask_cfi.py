import FWCore.ParameterSet.Config as cms

ecalzmasstask = cms.EDAnalyzer("EcalZmassTask",
	  prefixME = cms.untracked.string('EcalCalibration'),
	  electronCollection    = cms.InputTag("gsfElectrons"), 
	  gsftracks_h           = cms.InputTag("electronGsfTracks")
 )

