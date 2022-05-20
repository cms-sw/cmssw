import FWCore.ParameterSet.Config as cms

interestingEgammaIsoESDetId = cms.EDProducer("EgammaIsoESDetIdCollectionProducer",
                                             eeClusToESMapLabel=cms.InputTag(""),
                                             ecalPFClustersLabel=cms.InputTag(""),
                                             elesLabel=cms.InputTag("gedGsfElectrons"),
                                             phosLabel=cms.InputTag("gedPhotons"),
                                             superClustersLabel=cms.InputTag("particleFlowEGamma"),
                                             minSCEt=cms.double(500),
                                             minEleEt=cms.double(20),
                                             minPhoEt=cms.double(20),
                                             maxDR=cms.double(0.4),
                                             interestingDetIdCollection=cms.string("")
                                             )
                                                
