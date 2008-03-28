import FWCore.ParameterSet.Config as cms

source = cms.Source("DaqSource",
    pset = cms.PSet(
        dummy = cms.untracked.int32(0)
    ),
    reader = cms.string('FUReader')
)


