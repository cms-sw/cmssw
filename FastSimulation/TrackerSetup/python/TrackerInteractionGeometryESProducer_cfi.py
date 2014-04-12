import FWCore.ParameterSet.Config as cms

# With an ideal geometry (for the reconstruction)
from FastSimulation.TrackerSetup.TrackerMaterial_cfi import *
TrackerInteractionGeometryESProducer = cms.ESProducer("TrackerInteractionGeometryESProducer",
    TrackerMaterialBlock
)

# The same as above but with a misaligned tracker geometry (for the simulation)
misalignedTrackerInteractionGeometry = cms.ESProducer("TrackerInteractionGeometryESProducer",
    TrackerMaterialBlock,
    trackerGeometryLabel = cms.untracked.string('MisAligned'),
    appendToDataLabel = cms.string('MisAligned')
)


