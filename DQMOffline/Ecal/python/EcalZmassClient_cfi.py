import FWCore.ParameterSet.Config as cms

ecalzmassclient = cms.EDProducer('EcalZmassClient',
      prefixME = cms.untracked.string('EcalCalibration')

)
