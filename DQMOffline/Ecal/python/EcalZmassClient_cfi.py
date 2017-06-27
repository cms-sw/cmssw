import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

ecalzmassclient = DQMEDHarvester('EcalZmassClient',
      prefixME = cms.untracked.string('EcalCalibration')

)
