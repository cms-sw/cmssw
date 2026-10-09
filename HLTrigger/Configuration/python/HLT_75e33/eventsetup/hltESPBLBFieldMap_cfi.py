import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting

# (Bz, Br) field map of the BrokenLine fit, for the vector-hits CA
def _addHltESPBLBFieldMap(process):
    process.hltESPBLBFieldMap = cms.ESProducer('BLBFieldMapESProducerAlpaka@alpaka',
        appendToDataLabel = cms.string(''),
        alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
    )

addHltESPBLBFieldMap = (phase2CAVectorHits | ngtScouting).makeProcessModifier(_addHltESPBLBFieldMap)
