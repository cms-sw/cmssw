import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

hltPixelTracks = cms.EDProducer("PixelTracksProducer",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('WithoutRefit')
    ),
    SeedProducer = cms.InputTag("hltPixelTripletSeeds"),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    FilterPSet = cms.PSet(
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
        ptMin = cms.double(0.1),
        tipMax = cms.double(1.0)
    )
)

hltPixelTracksReg = hltPixelTracks.clone()
hltPixelTracksReg.FilterPSet.ptMin = 0.1
hltPixelTracksReg.FilterPSet.chi2 = 50.
hltPixelTracksReg.RegionFactoryPSet.ComponentName = "CandidateSeededTrackingRegionsProducer"
hltPixelTracksReg.RegionFactoryPSet.RegionPSet = cms.PSet( 
    precise = cms.bool( True ),
    originRadius = cms.double( 0.2 ),
    ptMin = cms.double( 0.9 ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    mode = cms.string( "BeamSpotSigma" ),
    input = cms.InputTag( "hltL2TausForPixelIsolation" ),
    maxNRegions = cms.int32( 10 ),
    vertexCollection = cms.InputTag( "" ),
    maxNVertices = cms.int32( 1 ),
    zErrorBeamSpot = cms.double( 24.2 ),
    deltaEta = cms.double( 0.5 ),
    deltaPhi = cms.double( 0.5 ),
    nSigmaZVertex = cms.double( 3.0 ),
    zErrorVertex = cms.double( 0.2 ),
    nSigmaZBeamSpot = cms.double( 4.0 ),
    whereToUseMeasurementTracker = cms.string("Never"))

hltPixelTracksForHighMult = hltPixelTracks.clone()
hltPixelTracksForHighMult.FilterPSet.ptMin = 0.4

