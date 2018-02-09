import FWCore.ParameterSet.Config as cms

# With an ideal geometry (for the reconstruction)
from FastSimulation.TrackerSetup.TrackerMaterial_cfi import *
FastSimGeometryESProducer = cms.ESProducer("FastSimGeometryESProducer",
    TrackerMaterialBlock
)

# The same as above but with a misaligned tracker geometry (for the simulation)
misalignedGeometry = cms.ESProducer("FastSimGeometryESProducer",
    TrackerMaterialBlock,
    trackerGeometryLabel = cms.untracked.string('MisAligned'),
    appendToDataLabel = cms.string('MisAligned')
)


