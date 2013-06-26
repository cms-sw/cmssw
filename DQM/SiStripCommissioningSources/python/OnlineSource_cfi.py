import FWCore.ParameterSet.Config as cms

# input source from shared memory
source = cms.Source("DaqSource",
    readerPluginName = cms.untracked.string('FUShmReader'),
    evtsPerLS = cms.untracked.uint32(50)
)

