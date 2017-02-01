import FWCore.ParameterSet.Config as cms

pfGSFixLinker = cms.EDProducer("PFGSFixLinker",
    PFCandidate = cms.InputTag("particleFlow", '', cms.InputTag.skipCurrentProcess()),
    GsfElectrons = cms.InputTag("gsFixedGsfElectrons"),
    Photons = cms.InputTag("gsFixedGEDPhotons"),
    ValueMapElectrons = cms.string("electrons"),                              
    ValueMapPhotons = cms.string("photons")
)
