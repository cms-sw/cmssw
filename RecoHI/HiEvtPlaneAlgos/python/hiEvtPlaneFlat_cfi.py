import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlat = cms.EDProducer('HiEvtPlaneFlatProducer',
                                centralityVariable = cms.string("HFtowers"),
                                centralityBinTag = cms.InputTag("centralityBin","HFtowers"),
                                centralityTag = cms.InputTag("hiCentrality"),
                                vertexTag = cms.InputTag("offlinePrimaryVertices"),
                                inputPlanesTag = cms.InputTag("hiEvtPlane"),
                                nonDefaultGlauberModel = cms.string(""),
                                trackTag = cms.InputTag("generalTracks"),
                                FlatOrder = cms.int32(9),
                                NumFlatBins = cms.int32(40),
                                flatnvtxbins = cms.int32(10),
                                flatminvtx = cms.double(-15.0),
                                flatdelvtx = cms.double(3.0),
                                caloCentRef = cms.double(-1.),
                                caloCentRefWidth = cms.double(-1.),
                                CentBinCompression = cms.int32(5),
                                useOffsetPsi = cms.bool(True)
                                )
