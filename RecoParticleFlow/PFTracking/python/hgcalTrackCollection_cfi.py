import FWCore.ParameterSet.Config as cms

hgcalTrackCollection = cms.EDProducer(
    "HGCalTrackCollectionProducer",
    src = cms.InputTag("pfTrack"),
    trackQuality = cms.string("highPurity"),
    # From GeneralTracksImporter
    useIterativeTracking = cms.bool(True),
    DPtOverPtCuts_byTrackAlgo = cms.vdouble(10.0,10.0,10.0,10.0,10.0,5.0),
    NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3,3),

    # From HGCClusterizer
    hgcalGeometryNames = cms.PSet( HGC_ECAL  = cms.string('HGCalEESensitive'),
                                    ),

)
