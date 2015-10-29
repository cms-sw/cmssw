import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import *

gedGsfElectrons = cms.EDProducer("GEDGsfElectronFinalizer",
                                 previousGsfElectronsTag = cms.InputTag("gedGsfElectronsTmp"),
                                 pfCandidatesTag = cms.InputTag("particleFlowTmp"),
                                 regressionConfig = regressionModifier.clone(rhoCollection=cms.InputTag("fixedGridRhoFastjetAllTmp")),
                                 pfIsolationValues = cms.PSet(
                                       pfSumChargedHadronPt = cms.InputTag('gedElPFIsoValueCharged03'),
                                       pfSumPhotonEt = cms.InputTag('gedElPFIsoValueGamma03'),
                                       pfSumNeutralHadronEt= cms.InputTag('gedElPFIsoValueNeutral03'),
                                       pfSumPUPt = cms.InputTag('gedElPFIsoValuePU03')),
                                 outputCollectionLabel = cms.string("")
                                 )
