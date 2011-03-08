import FWCore.ParameterSet.Config as cms

pfGsfElectronLinker = cms.EDProducer("GsfElectronLinker",
                                     PFCandidate = cms.InputTag("particleFlow"),
                                     GsfElectrons = cms.InputTag("gsfElectrons"),
                                     OutputPF = cms.string("withElec"))
