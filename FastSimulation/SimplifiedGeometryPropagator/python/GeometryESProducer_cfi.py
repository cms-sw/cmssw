import FWCore.ParameterSet.Config as cms

# With an ideal geometry (for the reconstruction)
from FastSimulation.TrackerSetup.TrackerMaterial_cfi import *
GeometryESProducer = cms.ESProducer("GeometryESProducer",
    TrackerMaterialBlock
)

# The same as above but with a misaligned tracker geometry (for the simulation)
misalignedGeometry = cms.ESProducer("GeometryESProducer",
    TrackerMaterialBlock,
    trackerGeometryLabel = cms.untracked.string('MisAligned'),
    appendToDataLabel = cms.string('MisAligned')
)


