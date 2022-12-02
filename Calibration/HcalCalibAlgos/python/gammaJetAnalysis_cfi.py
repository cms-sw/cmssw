import FWCore.ParameterSet.Config as cms
from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.RecoPFJets_cff import *
from CommonTools.ParticleFlow.pfNoPileUp_cff import *

GammaJetAnalysis = cms.EDAnalyzer('GammaJetAnalysis',
                                  rhoColl             = cms.InputTag("fixedGridRhoFastjetAll"),
                                  PFMETColl           = cms.InputTag("pfMet"),
                                  PFMETTYPE1Coll      = cms.InputTag("pfType1CorrectedMet"),
                                  photonCollName      = cms.string('gedPhotons'),
                                  pfJetCollName       = cms.string('ak4PFJetsCHS'),
                                  JetCorrections      = cms.InputTag("ak4PFCHSL2L3Corrector"),
                                  genJetCollName      = cms.string('ak4GenJets'),
                                  genParticleCollName = cms.string('genParticles'),
                                  genEventInfoName    = cms.string('generator'),
                                  hbheRecHitName      = cms.string('hbhereco'),
                                  hfRecHitName        = cms.string('hfreco'),
                                  hoRecHitName        = cms.string('horeco'),
                                  rootHistFilename    = cms.string('PhotonPlusJet_tree.root'),
                                  pvCollName = cms.string('offlinePrimaryVertices'),
                                  beamSpotName= cms.string('offlineBeamSpot'),
                                  conversionsName= cms.string('allConversions'),
                                  electronCollName= cms.string('gedGsfElectrons'),
                                  photonIdTightName= cms.InputTag('PhotonIDProdGED','PhotonCutBasedIDTight'),
                                  photonIdLooseName= cms.InputTag('PhotonIDProdGED','PhotonCutBasedIDLoose'),
                                  prodProcess = cms.untracked.string('reRECO'),
                                  allowNoPhoton       = cms.bool(False),
                                  photonJetDPhiMin    = cms.double(2.0),  # 0.75 pi= 2.356, 0.7 pi=2.2
                                  photonPtMin         = cms.double(15.),
                                  jetEtMin            = cms.double(15.),
                                  jet2EtMax            = cms.double(100.),
                                  jet3EtMax            = cms.double(50.),
                                  photonTriggers      = cms.vstring(''), #HLT_Photon20_*, HLT_Photon135*'),
                                  jetTriggers         = cms.vstring(''), #HLT_Jet30*'),
                                  writeTriggerPrescale= cms.bool(False),
                                  doPFJets            = cms.bool(True),
                                  doGenJets           = cms.bool(True),
                                  debug               = cms.untracked.int32(0),
                                  debugHLTTrigNames   = cms.untracked.int32(2),
                                  workOnAOD           = cms.int32(0),
                                  stageL1Trigger = cms.uint32(1)
                                  )
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(GammaJetAnalysis, stageL1Trigger = 2)
