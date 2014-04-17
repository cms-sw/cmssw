import FWCore.ParameterSet.Config as cms

# time resolution
_timeResolutionECALBarrel = cms.PSet(
    noiseTerm = cms.double(26.4021428571 * 0.042),
    constantTerm = cms.double(0.428192),
    noiseTermLowE = cms.double(31.4007142857 * 0.042),
    corrTermLowE = cms.double(0.0510871),
    constantTermLowE = cms.double(0.),
    threshLowE = cms.double(0.5),
    threshHighE = cms.double(5.)
  )

_timeResolutionECALEndcap = cms.PSet(
    noiseTerm = cms.double(40.8921428571 * 0.14),
    constantTerm = cms.double(0.),
    noiseTermLowE = cms.double(49.4773571429 * 0.14),
    corrTermLowE = cms.double(0.),
    constantTermLowE = cms.double(0.),
    threshLowE = cms.double(1.),
    threshHighE = cms.double(10.)
  )
