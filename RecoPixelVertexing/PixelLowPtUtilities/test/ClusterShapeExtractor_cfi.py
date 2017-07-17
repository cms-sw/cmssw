import FWCore.ParameterSet.Config as cms

clusterShapeExtractor = cms.EDAnalyzer("ClusterShapeExtractor",
    trackProducer  = cms.string('allTracks'),
    clusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    pixelSimLinkSrc = cms.InputTag('simSiPixelDigis', 'Pixel'),
    hasSimHits     = cms.bool(True),
    hasRecTracks   = cms.bool(False),
    associateStrip      = cms.bool(False),
    associatePixel      = cms.bool(True),
    associateRecoTracks = cms.bool(False),
    ROUList = cms.vstring(
      'TrackerHitsPixelBarrelLowTof',
      'TrackerHitsPixelBarrelHighTof',
      'TrackerHitsPixelEndcapLowTof',
      'TrackerHitsPixelEndcapHighTof')
)

