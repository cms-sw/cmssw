import FWCore.ParameterSet.Config as cms

hgcalTrackCollection = cms.EDProducer(
    "HGCalTrackCollectionProducer",
    src = cms.InputTag("pfTrack"),
    debug = cms.bool(False),
    # From GeneralTracksImporter

    useIterativeTracking = cms.bool(True),
    DPtOverPtCuts_byTrackAlgo = cms.vdouble(-1.0,-1.0,-1.0,
                                             1.0,1.0),
    NHitCuts_byTrackAlgo = cms.vuint32(3,3,3,3,3),

    # From HGCClusterizer
    hgcalGeometryNames = cms.PSet( HGC_ECAL  = cms.string('HGCalEESensitive'),
                                    ),

)
