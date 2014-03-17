import FWCore.ParameterSet.Config as cms

gedGsfElectrons = cms.EDProducer("GEDGsfElectronFinalizer",
                                 previousGsfElectronsTag = cms.InputTag("gedGsfElectronsTmp"),
                                 pfCandidatesTag = cms.InputTag("particleFlowTmp"),
                                 pfIsolationValues = cms.PSet(
                                       pfSumChargedHadronPt = cms.InputTag('gedElPFIsoValueCharged03'),
                                       pfSumPhotonEt = cms.InputTag('gedElPFIsoValueGamma03'),
                                       pfSumNeutralHadronEt= cms.InputTag('gedElPFIsoValueNeutral03'),
                                       pfSumPUPt = cms.InputTag('gedElPFIsoValuePU03')),
                                 outputCollectionLabel = cms.string("")
                                 )
