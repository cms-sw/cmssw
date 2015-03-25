import FWCore.ParameterSet.Config as cms

hiEvtPlane = cms.EDProducer("EvtPlaneProducer",
                            vertexTag_=cms.InputTag("hiSelectedVertex"),
                            centralityTag_=cms.InputTag("hiCentrality"),
                            caloTag_=cms.InputTag("towerMaker"),
                            castorTag_=cms.InputTag("CastorTowerReco"),
                            trackTag_=cms.InputTag("hiGeneralTracks"),         
                            centralityBinTag_ = cms.InputTag("centralityBin","HFtowers"),
                            centralityVariable = cms.string("HFtowers"),
                            nonDefaultGlauberModel = cms.string(""),
                            FlatOrder_ = cms.untracked.int32(9),
                            NumFlatBins_ = cms.untracked.int32(40),
                            CentBinCompression_ = cms.untracked.int32(5),
                            caloCentRef_ = cms.untracked.double(80.),
                            caloCentRefWidth_ = cms.untracked.double(5.0),
                            HFEtScale_ = cms.untracked.int32(3800),
                            loadDB_ = cms.untracked.bool(True),                 
                            minet_ = cms.untracked.double(-1.),
                            maxet_ = cms.untracked.double(-1.),
                            minpt_ = cms.untracked.double(0.3),
                            maxpt_ = cms.untracked.double(3.0),
                            minvtx_ = cms.untracked.double(-25.),
                            maxvtx_ = cms.untracked.double(25.),
                            dzerr_ = cms.untracked.double(10.),
                            chi2_ = cms.untracked.double(40.)
                            )
                            




    
