import FWCore.ParameterSet.Config as cms



goodHwwMuons = cms.EDFilter("MuonRefSelector",
                         src = cms.InputTag("muons"),
                         cut = cms.string("obj.pt() > 10 && " +
                                                      "(obj.isolationR03().sumPt+obj.isolationR03().emEt+obj.isolationR03().hadEt)/obj.pt() < 1.0 && " +
                                                      "obj.isGlobalMuon() && obj.isTrackerMuon()"
                                                      ),
                                 )

              
goodHwwElectrons = cms.EDFilter("GsfElectronRefSelector",
                             src = cms.InputTag("gedGsfElectrons"),
                             cut = cms.string(    "obj.pt() > 10 &&" +
                                                  " std::abs(obj.deltaEtaSuperClusterTrackAtVtx()) < 0.010 &&" +
                                                  " (( obj.isEB() && obj.sigmaIetaIeta() < 0.011) ||" +
                                                  "  (!obj.isEB() && obj.sigmaIetaIeta() < 0.031))"),
                             )

diHwwMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                                     decay       = cms.string("goodHwwMuons goodHwwMuons"),
                                     checkCharge = cms.bool(False),
                                     cut         = cms.string("obj.mass() > 5"),
                                 )

diHwwElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                         decay       = cms.string("goodHwwElectrons goodHwwElectrons"),
                                         checkCharge = cms.bool(False),
                                         cut         = cms.string("obj.mass() > 5"),
                                     )
crossHwwLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay       = cms.string("goodHwwMuons goodHwwElectrons"),
                                           checkCharge = cms.bool(False),
                                           cut         = cms.string("obj.mass() > 1"),
                                       )

diHwwMuonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("diHwwMuons"),
                             minNumber = cms.uint32(1)
)
diHwwElectronsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("diHwwElectrons"),
                             minNumber = cms.uint32(1)
)
crossHwwLeptonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("crossHwwLeptons"),
                             minNumber = cms.uint32(1)
)

diMuonSequence = cms.Sequence( goodHwwMuons * diHwwMuons * diHwwMuonsFilter )

diElectronSequence = cms.Sequence( goodHwwElectrons * diHwwElectrons * diHwwElectronsFilter )

EleMuSequence = cms.Sequence( goodHwwMuons * goodHwwElectrons * crossHwwLeptons * crossHwwLeptonsFilter )

