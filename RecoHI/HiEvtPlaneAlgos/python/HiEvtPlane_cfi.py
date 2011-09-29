import FWCore.ParameterSet.Config as cms

hiEvtPlane = cms.EDProducer("EvtPlaneProducer",
                            vtxCollection_=cms.untracked.string("hiSelectedVertex"),
                            caloCollection_=cms.untracked.string("towerMaker"),
                            trackCollection_=cms.untracked.string("hiGoodTightMergedTracks"),
                            useECAL_ = cms.untracked.bool(True),
                            useHCAL_ = cms.untracked.bool(True),
                            useTrackPtWeight_ = cms.untracked.bool(True),
                            minet_ = cms.untracked.double(0.2),
                            maxet_ = cms.untracked.double(500.0),
                            minpt_ = cms.untracked.double(0.3),
                            maxpt_ = cms.untracked.double(2.6),
                            minvtx_ = cms.untracked.double(-10.),
                            maxvtx_ = cms.untracked.double(10.),
                            dzerr_ = cms.untracked.double(10.),
                            chi2_ = cms.untracked.double(40.)
                            )
                            




    
