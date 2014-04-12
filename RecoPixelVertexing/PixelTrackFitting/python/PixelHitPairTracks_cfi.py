import FWCore.ParameterSet.Config as cms

pixelTracks = cms.EDProducer("PixelHitPairTrackProducer",
    Fitter = cms.string('PixelFitterByHelixProjections'),
    HitCollectionLabel = cms.InputTag("siPixelRecHits"),
    FitterPSet = cms.PSet(

    ),
    Filter = cms.string('PixelTrackFilterByKinematics'),
    FilterPSet = cms.PSet(
        chi2 = cms.double(1000.0),
        ptMin = cms.double(0.0),
        tipMax = cms.double(1.0)
    )
)


