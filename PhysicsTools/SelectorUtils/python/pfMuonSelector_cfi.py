import FWCore.ParameterSet.Config as cms
 
pfMuonSelector = cms.PSet(
    version = cms.string('SPRING11'),
    Chi2 = cms.double(10.0),
    D0 = cms.double(0.02),
    NHits = cms.int32(11),
    NValMuHits = cms.int32(0),
    PFIso = cms.double(0.2),
    nPixelHits = cms.int32(1),
    nMatchedStations=cms.int32(1),
    cutsToIgnore=cms.vstring()
    )

