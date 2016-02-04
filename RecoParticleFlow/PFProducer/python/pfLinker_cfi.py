import FWCore.ParameterSet.Config as cms

pfLinker = cms.EDProducer("PFLinker",
                          PFCandidate = cms.VInputTag(cms.InputTag("particleFlow")),
                          GsfElectrons = cms.InputTag("gsfElectrons"),
                          Photons = cms.InputTag("pfPhotonTranslator:pfphot"),
                          Muons = cms.InputTag("muons","muons1stStep2muonsMap"),
                          ProducePFCandidates = cms.bool(True),
                          FillMuonRefs = cms.bool(True),
                          OutputPF = cms.string(""),
                          ValueMapElectrons = cms.string("electrons"),                              
                          ValueMapPhotons = cms.string("photons"),
                          ValueMapMerged = cms.string("all")
                          )
