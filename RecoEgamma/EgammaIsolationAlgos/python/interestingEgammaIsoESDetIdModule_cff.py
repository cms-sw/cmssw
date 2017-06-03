import FWCore.ParameterSet.Config as cms

interestingEgammaIsoESDetId = cms.EDProducer("EgammaIsoESDetIdCollectionProducer",
                                             eeClusToESMapLabel=cms.InputTag("particleFlowClusterECALRemade"),
                                             ecalPFClustersLabel=cms.InputTag("particleFlowClusterECALRemade"),
                                             ootEEClusToESMapLabel=cms.InputTag("particleFlowClusterOOTECAL"),
                                             ootEcalPFClustersLabel=cms.InputTag("particleFlowClusterOOTECAL"),
                                             elesLabel=cms.InputTag("gedGsfElectrons"),
                                             phosLabel=cms.InputTag("gedPhotons"),
                                             ootPhosLabel=cms.InputTag("ootPhotons"),
                                             superClustersLabel=cms.InputTag("particleFlowEGamma"),
                                             minSCEt=cms.double(500),
                                             minEleEt=cms.double(20),
                                             minPhoEt=cms.double(20),
                                             minOOTPhoEt=cms.double(20),
                                             maxDR=cms.double(0.4),
                                             interestingDetIdCollection=cms.string("")
                                             )
                                                
