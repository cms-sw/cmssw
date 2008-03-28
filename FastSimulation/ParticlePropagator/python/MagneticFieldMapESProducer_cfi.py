import FWCore.ParameterSet.Config as cms

# With an ideal geometry (for the reconstruction)
MagneticFieldMapESProducer = cms.ESProducer("MagneticFieldMapESProducer")

# The same as above but with a misaligned tracker geometry (for the simulation)
misalignedMagneticFieldMap = cms.ESProducer("MagneticFieldMapESProducer",
    trackerGeometryLabel = cms.untracked.string('MisAligned'),
    appendToDataLabel = cms.string('MisAligned')
)


