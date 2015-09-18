import FWCore.ParameterSet.Config as cms

# cuts
MUON_CUT=("obj.pt() > 7 && std::abs(obj.eta())<2.5 && (obj.isGlobalMuon() || obj.isTrackerMuon())")
ELECTRON_CUT=("obj.pt() > 10 && std::abs(obj.eta())<2.5")
DIMUON_CUT=("obj.mass() > 40 && obj.daughter(0)->pt()>20 && obj.daughter(1)->pt()>7")
DIELECTRON_CUT=("obj.mass() > 40 && obj.daughter(0)->pt()>20 && obj.daughter(1)->pt()>10")
EMU_CUT=("obj.mass() > 40 && ((obj.daughter(0)->pt()>7 && obj.daughter(1)->pt()>20) || (obj.daughter(0)->pt()>20 && obj.daughter(1)->pt()>10))")

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

