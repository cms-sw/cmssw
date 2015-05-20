import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlat = cms.EDProducer('HiEvtPlaneFlatProducer',
                                vertexTag = cms.InputTag("hiSelectedVertex"),
                                centralityTag = cms.InputTag("hiCentrality"),
                                centralityBinTag = cms.InputTag("centralityBin","HFtowers"),
                                centralityVariable = cms.string("HFtowers"),
                                nonDefaultGlauberModel = cms.string(""),
                                inputPlanesTag = cms.InputTag("hiEvtPlane"),
                                trackTag = cms.InputTag("hiGeneralTracks"),
                                FlatOrder = cms.int32(9),
                                NumFlatBins = cms.int32(40),
                                Noffmin = cms.int32 (-1),
                                Noffmax = cms.int32 (10000),
                                CentBinCompression = cms.int32(5),
                                caloCentRef = cms.double(80.),
                                caloCentRefWidth = cms.double(5.0),
                                useOffsetPsi = cms.bool(True)
                                )
