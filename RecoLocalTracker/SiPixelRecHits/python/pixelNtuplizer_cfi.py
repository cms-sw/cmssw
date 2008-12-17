import FWCore.ParameterSet.Config as cms

PixelNtuplizer = cms.EDFilter("PixelNtuplizer",
    OutputFile = cms.string('pixel_ntuplizer_ntuple.root'),
    src = cms.InputTag("siPixelRecHits"),
    # check that the simHit associated with recHit is of the expected particle type
    checkType = cms.bool(True),
    # the type of particle that the simHit associated with recHits should be
    genType = cms.int32(13),
    associatePixel = cms.bool(True),
    associateStrip = cms.bool(False),
    associateRecoTracks = cms.bool(False),

  ROUList = cms.vstring(
    #BPIX
    "TrackerHitsPixelBarrelLowTof","TrackerHitsPixelBarrelHighTof",
    #FPIX
    "TrackerHitsPixelEndcapLowTof","TrackerHitsPixelEndcapHighTof"
    )


)



