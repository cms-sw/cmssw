import FWCore.ParameterSet.Config as cms


interestingEgammaHCALDetIds= cms.EDProducer("InterestingHcalDetIdCollectionProducer",
                                             recHitsLabel=cms.InputTag("hbhereco"),
                                            elesLabel=cms.InputTag("gedGsfElectrons"),
                                            phosLabel=cms.InputTag("gedPhotons"),
                                            superClustersLabel=cms.InputTag("particleFlowEGamma"),
                                            minSCEt=cms.double(20),
                                            minEleEt=cms.double(-1),
                                            minPhoEt=cms.double(-1),
                                            maxDIEta=cms.int32(5),
                                            maxDIPhi=cms.int32(5),
                                            interestingDetIdCollection=cms.string("")
                                            )

