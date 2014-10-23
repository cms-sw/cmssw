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

_timeResolutionHCAL = cms.PSet(
    noiseTerm = cms.double(15.92),
    constantTerm = cms.double(2.59),
    noiseTermLowE = cms.double(3.313),
    corrTermLowE = cms.double(0.0),
    constantTermLowE = cms.double(4.088),
    threshLowE = cms.double(5.0),
    threshHighE = cms.double(20.0)
  )

_timeResolutionHCALMaxSample = cms.PSet(
    noiseTerm = cms.double(21.86),
    constantTerm = cms.double(2.82),
    noiseTermLowE = cms.double(8),
    corrTermLowE = cms.double(0.0),
    constantTermLowE = cms.double(4.24),
    threshLowE = cms.double(6.0),
    threshHighE = cms.double(15.0)
  )

