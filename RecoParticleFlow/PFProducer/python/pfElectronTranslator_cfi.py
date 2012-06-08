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
                                      isolationValues = cms.PSet(
                                               pfChargedHadrons = cms.InputTag('elPFIsoValueCharged04'),
                                               pfPhotons = cms.InputTag('elPFIsoValueGamma04'),
                                               pfNeutralHadrons= cms.InputTag('elPFIsoValueNeutral04'))
                                      )
