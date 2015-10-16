import FWCore.ParameterSet.Config as cms

TauMCProducer  = cms.EDProducer("HLTTauMCProducer",
                              GenParticles  = cms.untracked.InputTag("genParticles"),
			      GenMET        = cms.untracked.InputTag("genMetTrue"),
                              ptMinTau      = cms.untracked.double(15),
                              ptMinMuon     = cms.untracked.double(15),
                              ptMinElectron = cms.untracked.double(15),
                              BosonID       = cms.untracked.vint32(23,24,25,32,33,34,35,36,37),
                              EtaMax        = cms.untracked.double(2.5)
)



#Create LorentzVectors for offline objects
TauRelvalRefProducer = cms.EDProducer("HLTTauRefProducer",

                                PFTaus = cms.untracked.PSet(
                                   PFTauDiscriminators = cms.untracked.VInputTag(
                                                    cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),
                                                    cms.InputTag("hpsPFTauDiscriminationByLooseIsolation")
                                   ),
                                   doPFTaus = cms.untracked.bool(True),
                                   ptMin = cms.untracked.double(15.0),
                                   PFTauProducer = cms.untracked.InputTag("hpsPFTauProducer")
                                   ),
                                Electrons = cms.untracked.PSet(
                                   ElectronCollection = cms.untracked.InputTag("gsfElectrons"),
                                   doID = cms.untracked.bool(False),
                                   InnerConeDR = cms.untracked.double(0.02),
                                   MaxIsoVar = cms.untracked.double(0.02),
                                   doElectrons = cms.untracked.bool(True),
                                   TrackCollection = cms.untracked.InputTag("generalTracks"),
                                   OuterConeDR = cms.untracked.double(0.6),
                                   ptMin = cms.untracked.double(15.0),
                                   doTrackIso = cms.untracked.bool(True),
                                   ptMinTrack = cms.untracked.double(1.5),
                                   lipMinTrack = cms.untracked.double(0.2),
                                   IdCollection = cms.untracked.InputTag("elecIDext")
                                   ),
                                Jets = cms.untracked.PSet(
                                  JetCollection = cms.untracked.InputTag("iterativeCone5CaloJets"),
                                  etMin = cms.untracked.double(10.0),
                                  doJets = cms.untracked.bool(True)
                                  ),
                                Towers = cms.untracked.PSet(
                                        TowerCollection = cms.untracked.InputTag("towerMaker"),
                                        etMin = cms.untracked.double(10.0),
                                        doTowers = cms.untracked.bool(True),
                                        towerIsolation = cms.untracked.double(5.0)
                                ),
                                Muons = cms.untracked.PSet(
                                       doMuons = cms.untracked.bool(True),
                                       MuonCollection = cms.untracked.InputTag("muons"),
                                       ptMin = cms.untracked.double(15.0)
                                ),
                                Photons = cms.untracked.PSet(
                                          doPhotons = cms.untracked.bool(True),
                                          PhotonCollection = cms.untracked.InputTag("photons"),
                                          etMin = cms.untracked.double(10.0),
                                          ECALIso = cms.untracked.double(3.0)
                                          ),
                                MET = cms.untracked.PSet(
                                          doMET = cms.untracked.bool(True),
                                          METCollection = cms.untracked.InputTag("caloMet"),
                                          ptMin = cms.untracked.double(15.0)
                                ),
                                EtaMax = cms.untracked.double(2.5)
)


#Match PF Taus to MC
TauRefCombiner = cms.EDProducer("HLTTauRefCombiner",
                                InputCollections = cms.VInputTag(
                                        cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
                                        cms.InputTag("TauRelvalRefProducer","PFTaus")
                                ),
                                MatchDeltaR = cms.double(0.15),
                                OutputCollection = cms.string("")
)                                




hltTauRef = cms.Sequence(TauMCProducer*TauRelvalRefProducer*TauRefCombiner)








                              
