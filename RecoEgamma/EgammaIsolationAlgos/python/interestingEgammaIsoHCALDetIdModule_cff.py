import FWCore.ParameterSet.Config as cms

interestingEgammaIsoHCALDetId = cms.EDProducer("EgammaIsoHcalDetIdCollectionProducer",
                                               recHitsLabel=cms.InputTag("hbhereco"),
                                               elesLabel=cms.InputTag("gedGsfElectrons"),
                                               phosLabel=cms.InputTag("gedPhotons"),
                                               superClustersLabel=cms.InputTag("particleFlowEGamma"),
                                               minSCEt=cms.double(20),
                                               minEleEt=cms.double(20),
                                               minPhoEt=cms.double(20),
                                               maxDIEta=cms.int32(6),
                                               maxDIPhi=cms.int32(6),
                                               minEnergyHCAL = cms.double(0.8),
                                               interestingDetIdCollection=cms.string("")
                                               )
