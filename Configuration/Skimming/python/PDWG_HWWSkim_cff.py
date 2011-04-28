import FWCore.ParameterSet.Config as cms


MUON_CUT=("pt > 10 && " +
          "(isolationR03().sumPt+isolationR03().emEt+isolationR03().hadEt)/pt < 1.0 && " +
          "isGlobalMuon && isTrackerMuon");
goodMuons = cms.EDFilter("MuonRefSelector",
                                     src = cms.InputTag("muons"),
                                     cut = cms.string(MUON_CUT),
                                 )
ELECTRON_CUT=("pt > 10 &&" +
              " abs(deltaEtaSuperClusterTrackAtVtx) < 0.010 &&" +
              " (( isEB && sigmaIetaIeta < 0.011) ||" +
              "  (!isEB && sigmaIetaIeta < 0.031))");

goodElectrons = cms.EDFilter("GsfElectronRefSelector",
                                         src = cms.InputTag("gsfElectrons"),
                                         cut = cms.string(ELECTRON_CUT),
                                     )

diMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                                     decay       = cms.string("goodMuons goodMuons"),
                                     checkCharge = cms.bool(False),
                                     cut         = cms.string("mass > 5"),
                                 )

diElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                         decay       = cms.string("goodElectrons goodElectrons"),
                                         checkCharge = cms.bool(False),
                                         cut         = cms.string("mass > 5"),
                                     )
crossLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
                                           decay       = cms.string("goodMuons goodElectrons"),
                                           checkCharge = cms.bool(False),
                                           cut         = cms.string("mass > 1"),
                                       )

diMuonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("diMuons"),
                             minNumber = cms.uint32(1)
)
diElectronsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("diElectrons"),
                             minNumber = cms.uint32(1)
)
crossLeptonsFilter = cms.EDFilter("CandViewCountFilter",
                             src = cms.InputTag("crossLeptons"),
                             minNumber = cms.uint32(1)
)

diMuonSequence = cms.Sequence( goodMuons * diMuons * diMuonsFilter )

diElectronSequence = cms.Sequence( goodElectrons * diElectrons * diElectronsFilter )

EleMuSequence = cms.Sequence( goodMuons * goodElectrons * crossLeptons * crossLeptonsFilter )

