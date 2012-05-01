import FWCore.ParameterSet.Config as cms

ecalzmassclient = cms.EDAnalyzer('EcalZmassClient',
      prefixME = cms.untracked.string('EcalCalibration')

)
