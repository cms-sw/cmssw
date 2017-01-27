import FWCore.ParameterSet.Config as cms

pfEGFootprintGSFixLinker = cms.EDProducer("PFEGFootprintGSFixLinker",
    PFCandidate = cms.InputTag("particleFlow", '', cms.InputTag.skipCurrentProcess()),
    GsfElectrons = cms.InputTag("gsFixedGsfElectrons"),
    Photons = cms.InputTag("gsFixedGEDPhotons"),
    GsfElectronsFootprint = cms.InputTag('particleBasedIsolation', 'gedGsfElectrons', cms.InputTag.skipCurrentProcess()),
    PhotonsFootprint = cms.InputTag('particleBasedIsolation', 'gedPhotons', cms.InputTag.skipCurrentProcess()),
    ValueMapElectrons = cms.string("gedGsfElectrons"),
    ValueMapPhotons = cms.string("gedPhotons")
)
