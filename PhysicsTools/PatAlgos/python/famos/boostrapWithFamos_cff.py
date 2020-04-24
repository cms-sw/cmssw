import FWCore.ParameterSet.Config as cms

#
# General fast simulation configuration ###
#
# Random number generator service
from FastSimulation.Configuration.RandomServiceInitialization_cff import *
# Generate ttbar events
from FastSimulation.Configuration.ttbar_cfi import *
# Famos sequences
from FastSimulation.Configuration.CommonInputsFake_cff import *
from FastSimulation.Configuration.FamosSequences_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
# If you want to turn on/off pile-up (e.g. default low lumi: 5.0)
famosPileUp.PileUpSimulator.averageNumber = 0
# You may not want to simulate everything for your study
fastSimProducer.SimulateCalorimetry = True
for layer in process.fastSimProducer.detectorDefinition.BarrelLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
for layer in process.fastSimProducer.detectorDefinition.ForwardLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
fastSimProducer.SimulateMuons = True
VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

