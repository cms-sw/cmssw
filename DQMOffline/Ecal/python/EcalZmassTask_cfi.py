import FWCore.ParameterSet.Config as cms

ecalzmasstask = DQMStep1Module('EcalZmassTask',
	  prefixME = cms.untracked.string('EcalCalibration'),
	  electronCollection    = cms.InputTag("gedGsfElectrons"),
    trackCollection = cms.InputTag("electronGsfTracks")
 )

