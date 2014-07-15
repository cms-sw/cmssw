import FWCore.ParameterSet.Config as cms

ecalCommonParams = cms.untracked.PSet(
    onlineMode = cms.untracked.bool(False),
    willConvertToEDM = cms.untracked.bool(True)
)

ecaldqmLaserWavelengths = cms.untracked.vint32(1, 2, 3)
ecaldqmLedWavelengths = cms.untracked.vint32(1, 2)
ecaldqmMGPAGains = cms.untracked.vint32(12)
ecaldqmMGPAGainsPN = cms.untracked.vint32(16)    
