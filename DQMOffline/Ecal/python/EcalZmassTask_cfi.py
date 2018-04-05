import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalzmasstask = DQMEDAnalyzer('EcalZmassTask',
	  prefixME = cms.untracked.string('EcalCalibration'),
	  electronCollection    = cms.InputTag("gedGsfElectrons"),
    trackCollection = cms.InputTag("electronGsfTracks")
 )

