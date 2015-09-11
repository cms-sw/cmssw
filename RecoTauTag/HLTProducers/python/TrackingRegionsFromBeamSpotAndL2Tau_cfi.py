import FWCore.ParameterSet.Config as cms

TrackingRegionsFromBeamSpotAndL2TauBlock = cms.PSet(
    RegionPSet = cms.PSet(
        ptMin = cms.double( 5. ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 24.0 ),
        deltaEta = cms.double( 0.3 ),
        deltaPhi = cms.double( 0.3 ),
        JetSrc = cms.InputTag( "hltFilterL2EtCutDoublePFIsoTau25Trk5" ),
        JetMinPt = cms.double( 25. ),
        JetMaxEta = cms.double( 2.1 ),
        JetMaxN = cms.int32( 10 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        precise = cms.bool( True ),
        whereToUseMeasurementTracker = cms.string("Never"),
    )
)

