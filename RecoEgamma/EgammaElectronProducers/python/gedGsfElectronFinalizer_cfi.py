import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import *

gedGsfElectrons = cms.EDProducer("GEDGsfElectronFinalizer",
                                 previousGsfElectronsTag = cms.InputTag("gedGsfElectronsTmp"),
                                 pfCandidatesTag = cms.InputTag("particleFlowTmp"),
                                 regressionConfig = regressionModifier.clone(),
                                 pfIsolationValues = cms.PSet(
                                       pfSumChargedHadronPt = cms.InputTag('egmElectronIsolationCITK:h+-DR030-'),
                                       pfSumPhotonEt = cms.InputTag('egmElectronIsolationCITK:gamma-DR030-'),
                                       pfSumNeutralHadronEt= cms.InputTag('egmElectronIsolationCITK:h0-DR030-'),
                                       pfSumPUPt = cms.InputTag('egmElectronIsolationPileUpCITK:h+-DR030-'),
                                       pfSumEcalClusterEt = cms.InputTag("electronEcalPFClusterIsolationProducer"),
                                       pfSumHcalClusterEt = cms.InputTag("electronHcalPFClusterIsolationProducer")),
                                 outputCollectionLabel = cms.string("")
                                 )
