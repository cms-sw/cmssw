import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfElectronTranslatorMVACut_cfi import *

pfElectronTranslator = cms.EDProducer("PFElectronTranslator",
                                      PFCandidate = cms.InputTag("pfSelectedElectrons"),
                                      PFCandidateElectron = cms.InputTag("particleFlowTmp:electrons"),
                                      GSFTracks = cms.InputTag("electronGsfTracks"),
                                      PFBasicClusters = cms.string("pf"),
                                      PFPreshowerClusters = cms.string("pf"),
                                      PFSuperClusters = cms.string("pf"),
                                      ElectronMVA = cms.string("pf"),
                                      ElectronSC = cms.string("pf"),
                                      PFGsfElectron = cms.string("pf"),
                                      PFGsfElectronCore = cms.string("pf"),
                                      MVACutBlock = cms.PSet(pfElecMva),
                                      CheckStatusFlag = cms.bool(True),

				      useIsolationValues = cms.bool(False),
                                      isolationValues = cms.PSet(
                                               pfSumChargedHadronPt = cms.InputTag('elPFIsoValueCharged04PFId'),
                                               pfSumPhotonEt = cms.InputTag('elPFIsoValueGamma04PFId'),
                                               pfSumNeutralHadronEt= cms.InputTag('elPFIsoValueNeutral04PFId'),
                                               pfSumPUPt= cms.InputTag('elPFIsoValuePU04PFId'),
                                          )
                                      )
