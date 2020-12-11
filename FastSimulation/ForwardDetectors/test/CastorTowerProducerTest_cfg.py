import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")

# The number of events to be processed.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
    
# For valgrind studies
# process.ProfilerService = cms.Service("ProfilerService",
#    lastEvent = cms.untracked.int32(13),
#    firstEvent = cms.untracked.int32(3),
#    paths = cms.untracked.vstring('p1')
#)

# Include the RandomNumberGeneratorService definition
process.load("IOMC.RandomEngine.IOMC_cff")

# Generate H -> ZZ -> l+l- l'+l'- (l,l'=e or mu), with mH=200GeV/c2
#process.load("Configuration.Generator.H200ZZ4L_cfi")
# Generate ttbar events
#  process.load("FastSimulation/Configuration/ttbar_cfi")
# Generate multijet events with different ptHAT bins
#  process.load("FastSimulation/Configuration/QCDpt80-120_cfi")
#  process.load("FastSimulation/Configuration/QCDpt600-800_cfi")
# Generate Minimum Bias Events
process.load("FastSimulation/Configuration/MinBiasEvents_cfi")
# Generate muons with a flat pT particle gun, and with pT=10.
# process.load("FastSimulation/Configuration/FlatPtMuonGun_cfi")
# replace FlatRandomPtGunProducer.PGunParameters.PartID={130}
# Generate di-electrons with pT=35 GeV
#process.load("FastSimulation/Configuration/DiElectrons_cfi")

# Famos sequences (Frontier conditions)
process.load("FastSimulation/Configuration/CommonInputs_cff")
process.GlobalTag.globaltag = "IDEAL_30X::All"
process.load("FastSimulation/Configuration/FamosSequences_cff")

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# If you want to turn on/off pile-up
#process.famosPileUp.PileUpSimulator.averageNumber = 5.0    
# You may not want to simulate everything for your study
process.fastSimProducer.SimulateCalorimetry = True
for layer in process.fastSimProducer.detectorDefinition.BarrelLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
for layer in process.fastSimProducer.detectorDefinition.ForwardLayers: 
    layer.interactionModels = cms.untracked.vstring("pairProduction", "nuclearInteraction", "bremsstrahlung", "energyLoss", "multipleScattering", "trackerSimHits")
# process.fastSimProducer.SimulateMuons = False

# include Castor fast sim
process.load("FastSimulation.ForwardDetectors.CastorFastReco_cff")

# Produce Tracks and Clusters
process.p1 = cms.Path(process.ProductionFilterSequence*process.famosWithEverything*process.CastorFastReco)

# To write out events (not need: FastSimulation _is_ fast!)
process.o1 = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("MyFirstFamosFile.root"),
    outputCommands = cms.untracked.vstring("keep *_Castor*Reco_*_*",
                                           "drop *_mix_*_*")
    )

process.outpath = cms.EndPath(process.o1)

# Keep the logging output to a nice level #

#process.Timing =  cms.Service("Timing")
#process.load("FWCore/MessageService/MessageLogger_cfi")
#process.MessageLogger.destinations = cms.untracked.vstring("pyDetailedInfo.txt","cout")
#
#process.MessageLogger.cout = cms.untracked.PSet(threshold=cms.untracked.string("INFO"),
#                                                default=cms.untracked.PSet(limit=cms.untracked.int32(0)),
#                                                FamosManager=cms.untracked.PSet(limit=cms.untracked.int32(100000)))

# Make the job crash in case of missing product
process.options = cms.untracked.PSet( Rethrow = cms.untracked.vstring('ProductNotFound') )
