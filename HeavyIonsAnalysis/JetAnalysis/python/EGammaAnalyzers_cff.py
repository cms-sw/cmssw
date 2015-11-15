import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.PhotonAnalysis.MultiPhotonAnalyzer_cfi import multiPhotonAnalyzer
multiPhotonAnalyzer.GenParticleProducer = cms.InputTag("hiGenParticles")
multiPhotonAnalyzer.PhotonProducer = cms.InputTag("selectedPatPhotons")
multiPhotonAnalyzer.VertexProducer = cms.InputTag("hiSelectedVertex")
multiPhotonAnalyzer.TrackProducer = cms.InputTag("hiGeneralTracks")
multiPhotonAnalyzer.OutputFile = cms.string('mpa.root')
multiPhotonAnalyzer.doStoreCompCone = cms.untracked.bool(False)
multiPhotonAnalyzer.doStoreJets = cms.untracked.bool(False)

multiPhotonAnalyzer.gsfElectronCollection = cms.untracked.InputTag("ecalDrivenGsfElectrons")
multiPhotonAnalyzer.GammaEtaMax = cms.untracked.double(100)
multiPhotonAnalyzer.GammaPtMin = cms.untracked.double(10)

from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import photonMatch, patPhotons, selectedPatPhotons

photonMatch.matched = cms.InputTag("hiGenParticles")
photonMatch.src = cms.InputTag("photons")

patPhotons.addPhotonID = cms.bool(False)
patPhotons.photonSource = cms.InputTag("photons")
patPhotons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEB")
patPhotons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit:EcalRecHitsEE")
patPhotons.isoDeposits = cms.PSet()
patPhotons.isolationValues = cms.PSet()

from RecoHI.HiEgammaAlgos.HiEgamma_cff import hiPhotonSequence

photonStep_withReco = cms.Sequence(
                                   hiPhotonSequence * # need to re-run to get track info
                                   photonMatch *
                                   patPhotons *
                                   selectedPatPhotons *
                                   multiPhotonAnalyzer)

photonStep = cms.Sequence(
                          photonMatch *
                          patPhotons *
                          selectedPatPhotons *
                          multiPhotonAnalyzer)
