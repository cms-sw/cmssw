import FWCore.ParameterSet.Config as cms

egammaLinker = cms.EDProducer("EgammaPFLinker",
                              PFCandidate = cms.InputTag("particleFlow"),
                              GsfElectrons = cms.InputTag("gsfElectrons"),
                              Photons = cms.InputTag("pfPhotonTranslator:pfphot"),
                              ProducePFCandidates = cms.bool(True),
                              OutputPF = cms.string(""),
                              ValueMapElectrons = cms.string("electrons"),                              
                              ValueMapPhotons = cms.string("photons"))
