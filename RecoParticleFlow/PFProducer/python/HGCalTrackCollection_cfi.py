import FWCore.ParameterSet.Config as cms

HGCalTrackCollection = cms.EDProducer(
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
#     HGC_HCALF = cms.string('HGCalHESiliconSensitive'),                                                
#     HGC_HCALB = cms.string('HGCalHEScintillatorSensitive') ),                                          
                                    ),
#    UseFirstLayerOnly = cms.bool(True) # always true, no longer needed
)
