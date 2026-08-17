import FWCore.ParameterSet.Config as cms

# Stacked outer-tracker module geometry (sensor separation, radii, module classification),
# consumed by the stub producers and the stub cellular automaton.
hltESPStackedModuleGeometry = cms.ESProducer('StackedModuleGeometryESProducer@alpaka',
    appendToDataLabel = cms.string(''),
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)
