import FWCore.ParameterSet.Config as cms

pset = cms.PSet(
    IDname = cms.string('ByTightCombinedIsolationDBSumPtCorr3HitsdR03'),
    maximumAbsoluteValues = cms.vdouble(0.8, 1000000000.0),
    maximumRelativeValues = cms.vdouble(-1.0, 0.1),
    referenceRawIDNames = cms.vstring(
        'ByRawCombinedIsolationDBSumPtCorr3HitsdR03',
        'PhotonPtSumOutsideSignalConedR03'
    )
)