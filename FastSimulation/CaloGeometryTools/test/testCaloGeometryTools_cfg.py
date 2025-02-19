# The following comments couldn't be translated into the new config version:

# timing and memory checks

import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryTest")
process.load("Geometry.CMSCommonData.ecalhcalGeometryXML_cfi")

# Magnetic field full setup
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

# FastCalorimetry
process.load("FastSimulation.Calorimetry.Calorimetry_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    # To restore the status of the last event, just un-comment the
    # following line (and comment the saveFileName line!)
    # untracked string restoreFileName = "RandomEngineState.log"
    # To reproduce events using the RandomEngineStateProducer (source
    # excluded), comment the sourceSeed definition, and un-comment 
    # the restoreStateLabel
    # untracked string restoreStateLabel = "randomEngineStateProducer"
    # This is to initialize the random engine of the source
    prod = cms.PSet(
        initialSeed = cms.untracked.uint32(123456789),
        engineName = cms.untracked.string('TRandom3')
    ),
    # To save the status of the last event (useful for crashes)
    # Just give a name to the file you want the status to be saved
    # otherwise just put saveFileName = ""
    saveFileName = cms.untracked.string('')
)

process.source = cms.Source("EmptySource")

process.prod = cms.EDAnalyzer("testCaloGeometryTools")

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.p1 = cms.Path(process.prod)


