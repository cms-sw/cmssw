import FWCore.ParameterSet.Config as cms

ecalCommonParams = cms.untracked.PSet(
    onlineMode = cms.untracked.bool(False),
    willConvertToEDM = cms.untracked.bool(True)
)
<<<<<<< HEAD
=======

#ecaldqmLaserWavelengths = cms.untracked.vint32(1, 2, 3)
>>>>>>> af90af1e2e742d36094250a06c7d8522d5d7a9c8
ecaldqmLaserWavelengths = 1, 2, 3
ecaldqmLedWavelengths = 1, 2
ecaldqmMGPAGains = 12
ecaldqmMGPAGainsPN = 16

