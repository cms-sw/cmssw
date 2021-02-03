import FWCore.ParameterSet.Config as cms

hgcalTrackCollection = cms.EDProducer("HGCalTrackCollectionProducer",
    DPtOverPtCuts_byTrackAlgo = cms.vdouble(
        10.0, 10.0, 10.0, 10.0, 10.0,
        5.0
    ),
    NHitCuts_byTrackAlgo = cms.vuint32(
        3, 3, 3, 3, 3,
        32700
    ),
    hgcalGeometryNames = cms.PSet(
        HGC_ECAL = cms.string('HGCalEESensitive')
    ),
    src = cms.InputTag("pfTrack"),
    trackQuality = cms.string('highPurity'),
    useIterativeTracking = cms.bool(True)
)
