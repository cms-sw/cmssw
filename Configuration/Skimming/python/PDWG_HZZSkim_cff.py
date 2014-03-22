import FWCore.ParameterSet.Config as cms

# cuts
MUON_CUT=("pt > 7 && abs(eta)<2.5 && (isGlobalMuon || isTrackerMuon)")
ELECTRON_CUT=("pt > 10 && abs(eta)<2.5")
DIMUON_CUT=("mass > 40 && daughter(0).pt>20 && daughter(1).pt()>7")
DIELECTRON_CUT=("mass > 40 && daughter(0).pt>20 && daughter(1).pt()>10")
EMU_CUT=("mass > 40 && ((daughter(0).pt>7 && daughter(1).pt()>20) || (daughter(0).pt>20 && daughter(1).pt()>10))")

# single lepton selectors
goodHzzMuons = cms.EDFilter("MuonRefSelector",
                            src = cms.InputTag("muons"),
                            cut = cms.string(MUON_CUT)
                            )
goodHzzElectrons = cms.EDFilter("GsfElectronRefSelector",
                                src = cms.InputTag("gedGsfElectrons"),
                                cut = cms.string(ELECTRON_CUT)
                                )

# dilepton selectors
diHzzMuons = cms.EDProducer("CandViewShallowCloneCombiner",
                            decay       = cms.string("goodHzzMuons goodHzzMuons"),
                            checkCharge = cms.bool(False),
                            cut         = cms.string(DIMUON_CUT)
                            )
diHzzElectrons = cms.EDProducer("CandViewShallowCloneCombiner",
                                decay       = cms.string("goodHzzElectrons goodHzzElectrons"),
                                checkCharge = cms.bool(False),
                                cut         = cms.string(DIELECTRON_CUT)
                                )
crossHzzLeptons  = cms.EDProducer("CandViewShallowCloneCombiner",
                                  decay       = cms.string("goodHzzMuons goodHzzElectrons"),
                                  checkCharge = cms.bool(False),
                                  cut         = cms.string(EMU_CUT)
                                  )

# dilepton counters
diHzzMuonsFilter = cms.EDFilter("CandViewCountFilter",
                                src = cms.InputTag("diHzzMuons"),
                                minNumber = cms.uint32(1)
                                )
diHzzElectronsFilter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("diHzzElectrons"),
                                    minNumber = cms.uint32(1)
                                    )
crossHzzLeptonsFilter = cms.EDFilter("CandViewCountFilter",
                                     src = cms.InputTag("crossHzzLeptons"),
                                     minNumber = cms.uint32(1)
                                     )

#sequences
zzdiMuonSequence = cms.Sequence( goodHzzMuons * diHzzMuons * diHzzMuonsFilter )
zzdiElectronSequence = cms.Sequence( goodHzzElectrons * diHzzElectrons * diHzzElectronsFilter )
zzeleMuSequence = cms.Sequence( goodHzzMuons * goodHzzElectrons * crossHzzLeptons * crossHzzLeptonsFilter )

