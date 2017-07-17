import FWCore.ParameterSet.Config as cms

QJetsAdder = cms.EDProducer("QjetsAdder",
                            src=cms.InputTag("ak8PFJetsCHS"),
                            zcut=cms.double(0.1),
                            dcutfctr=cms.double(0.5),
                            expmin=cms.double(0.0),
                            expmax=cms.double(0.0),
                            rigidity=cms.double(0.1),
                            ntrial = cms.int32(50),
                            cutoff=cms.double(10.0),
                            jetRad= cms.double(0.8),
                            jetAlgo=cms.string("CA"),
                            preclustering = cms.int32(50),
                            )
