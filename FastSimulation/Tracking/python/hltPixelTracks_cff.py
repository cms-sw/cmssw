import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *

# Note: naming convention of pixel track filter needs to follow HLT,
# current it is set to what gets used in customizeHLTforCMSSW
from RecoTracker.PixelTrackFitting.pixelFitterByHelixProjections_cfi import pixelFitterByHelixProjections as _pixelFitterByHelixProjections
hltPixelTracksFitter = _pixelFitterByHelixProjections.clone()

from RecoTracker.PixelTrackFitting.pixelTrackFilterByKinematics_cfi import pixelTrackFilterByKinematics as _pixelTrackFilterByKinematics
hltPixelTracksFilter = _pixelTrackFilterByKinematics.clone()

hltPixelTracks = cms.EDProducer("PixelTracksProducer",
    Fitter = cms.InputTag("hltPixelTracksFitter"),
    SeedProducer = cms.InputTag("hltPixelTripletSeeds"),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    Filter = cms.InputTag("hltPixelTracksFilter"),
)

hltPixelTracksRegFilter = hltPixelTracksFilter.clone(
    ptMin = 0.1,
    chi2 = 50.,
)
hltPixelTracksReg = hltPixelTracks.clone(
    Filter = "hltPixelTracksRegFilter",
)
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

hltPixelTracksForHighMultFilter = hltPixelTracksFilter.clone(ptMin = 0.4)
hltPixelTracksForHighMult = hltPixelTracks.clone(Filter = "hltPixelTracksForHighMultFilter")

