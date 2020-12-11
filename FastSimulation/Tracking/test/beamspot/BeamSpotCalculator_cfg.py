import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# Number of events to be generated
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1500)
)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate ttbar events
process.load("Configuration.Generator.QCD_Pt_80_120_cfi")

# Famos sequences (NO HLT)
process.load("FastSimulation.Configuration.CommonInputs_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

process.BeamSpotDBSource = cms.ESSource("PoolDBESSource",
                                        process.CondDBSetup,
                                        toGet = cms.VPSet(cms.PSet(
    record = cms.string('BeamSpotObjectsRcd'),
    tag = cms.string('Early900GeVCollision_7p4cm_V1_IDEAL_V10'))),
    connect = cms.string('sqlite_file:EarlyCollision.db'))
                                        
process.es_prefer_beamspot = cms.ESPrefer("PoolDBESSource","BeamSpotDBSource")                                         

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
process.famosPileUp.PileUpSimulator.averageNumber = 5.0
# You may not want to simulate everything for your study
process.fastSimProducer.SimulateCalorimetry = True
for layer in process.fastSimProducer.detectorDefinition.BarrelLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
for layer in process.fastSimProducer.detectorDefinition.ForwardLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")

# Get frontier conditions    - not applied in the HCAL, see below
# Values for globaltag are "STARTUP_V5::All", "1PB::All", "10PB::All", "IDEAL_V5::All"
process.GlobalTag.globaltag = "STARTUP_V5::All"

# Apply ECAL miscalibration
process.caloRecHits.RecHitsFactory.doMiscalib = True

# Apply Tracker misalignment
process.fastSimProducer.detectorDefinition.trackerAlignmentLabel = cms.untracked.string("MisAligned")
process.misalignedTrackerGeometry.applyAlignment = True

# Apply HCAL miscalibration (not ideal in that case) . Choose between hcalmiscalib_startup.xml , hcalmiscalib_1pb.xml , hcalmiscalib_10pb.xml (startup is the default)
process.caloRecHits.RecHitsFactory.HCAL.Refactor = 1.0
process.caloRecHits.RecHitsFactory.HCAL.Refactor_mean = 1.0
#process.caloRecHits.RecHitsFactory.HCAL.fileNameHcal = "hcalmiscalib_startup.xml"

process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

# Famos with everything !
process.p1 = cms.Path(process.famosWithTracks*process.d0_phi_analyzer)



# To write out events
process.load("FastSimulation.Configuration.EventContent_cff")
process.o1 = cms.OutputModule("PoolOutputModule",
    process.AODSIMEventContent,
    fileName = cms.untracked.string('AODIntegrationTest.root')
)
process.outpath = cms.EndPath(process.o1)

# Keep output to a nice level
# process.Timing =  cms.Service("Timing")
process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.pyDetailedInfo=dict(extension="txt")

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
