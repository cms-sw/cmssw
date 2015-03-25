import FWCore.ParameterSet.Config as cms

hiEvtPlaneFlat = cms.EDProducer('HiEvtPlaneFlatProducer',
                                vertexTag_=cms.InputTag("hiSelectedVertex"),
                                centralityTag_=cms.InputTag("hiCentrality"),
                                centralityBinTag_ = cms.InputTag("centralityBin","HFtowers"),
                                centralityVariable = cms.string("HFtowers"),
                                nonDefaultGlauberModel = cms.string(""),
                                inputPlanesTag_ = cms.InputTag("hiEvtPlane","recoLevel"),
                                trackTag_=cms.InputTag("hiGeneralTracks"),                           
                                FlatOrder_ = cms.untracked.int32(9),
                                NumFlatBins_ = cms.untracked.int32(40),
                                Noffmin_ = cms.untracked.int32 (-1),
                                Noffmax_ = cms.untracked.int32 (10000),
                                CentBinCompression_ = cms.untracked.int32(5),
                                caloCentRef_ = cms.untracked.double(80.),
                                caloCentRefWidth_ = cms.untracked.double(5.0),
                                HFEtScale_ = cms.untracked.int32(3800),
                                useOffsetPsi_ = cms.untracked.bool(True)
                                )
