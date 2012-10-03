import FWCore.ParameterSet.Config as cms

ecalCalibCommonParams = cms.untracked.PSet(
#    laserWavelengths = cms.untracked.vint32(1, 2, 3, 4),
    laserWavelengths = cms.untracked.vint32(1, 2, 3),
    ledWavelengths = cms.untracked.vint32(1, 2),
#    MGPAGains = cms.untracked.vint32(1, 6, 12),
    MGPAGains = cms.untracked.vint32(12),
#    MGPAGainsPN = cms.untracked.vint32(1, 16),
    MGPAGainsPN = cms.untracked.vint32(16)
)
