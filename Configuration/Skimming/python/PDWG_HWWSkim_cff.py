import FWCore.ParameterSet.Config as cms



goodHwwMuons = cms.EDFilter("MuonRefSelector",
                         src = cms.InputTag("muons"),
                         cut = cms.string("pt > 10 && " +
                                                      "(isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)/pt < 1.0 && " +
                                                      "isGlobalMuon && isTrackerMuon"
                                                      ),
                                 )

              
goodHwwElectrons = cms.EDFilter("GsfElectronRefSelector",
                             src = cms.InputTag("gedGsfElectrons"),
                             cut = cms.string(    "pt > 10 &&" +
                                                  " abs(deltaEtaSuperClusterTrackAtVtx) < 0.010 &&" +
                                                  " (( isEB && sigmaIetaIeta < 0.011) ||" +
                                                  "  (!isEB && sigmaIetaIeta < 0.031))"),
                             )

diHwwMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                                     decay       = cms.string("goodHwwMuons goodHwwMuons"),
                                     checkCharge = cms.bool(False),
                                     cut         = cms.string("mass > 5"),
                                 )

diHwwElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                         decay       = cms.string("goodHwwElectrons goodHwwElectrons"),
                                         checkCharge = cms.bool(False),
                                         cut         = cms.string("mass > 5"),
                                     )
crossHwwLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay       = cms.string("goodHwwMuons goodHwwElectrons"),
                                           checkCharge = cms.bool(False),
                                           cut         = cms.string("mass > 1"),
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

