import FWCore.ParameterSet.Config as cms

# (Bz, Br) field map of the BrokenLine fit, for the vector-hits CA
hltESPBLBFieldMap = cms.ESProducer('BLBFieldMapESProducerAlpaka@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
