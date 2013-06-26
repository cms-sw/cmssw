# Test script for Calibratable
# Jamie Ballin
# Imperial College London, November 2008

# Generates Fast sim dipions, and extracts sim particle info and PFCandidate info


import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        caloRecHits = cms.untracked.uint32(754321),
        VtxSmeared = cms.untracked.uint32(223458),
        muonCSCDigis = cms.untracked.uint32(525432),
        muonDTDigis = cms.untracked.uint32(67673876),
        famosSimHits = cms.untracked.uint32(235791312),
        MuonSimHits = cms.untracked.uint32(834032),
        famosPileUp = cms.untracked.uint32(918273),
        muonRPCDigis = cms.untracked.uint32(524964),
        siTrackerGaussianSmearingRecHits = cms.untracked.uint32(34680)
    ),
    sourceSeed = cms.untracked.uint32(1234)
)

#fastsim
process.load("FastSimulation.Configuration.RandomServiceInitialization_cff")
process.load("FastSimulation.Configuration.CommonInputsFake_cff")
process.load("FastSimulation.Configuration.FamosSequences_cff")

process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0

process.famosSimHits.VertexGenerator.BetaStar = 0.00001
process.famosSimHits.VertexGenerator.SigmaZ = 0.00001

# Parametrized magnetic field (new mapping, 4.0 and 3.8T)
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True

# process.famosSimHits.MaterialEffects.PairProduction = false
# process.famosSimHits.MaterialEffects.Bremsstrahlung = false
# process.famosSimHits.MaterialEffects.EnergyLoss = false
# process.famosSimHits.MaterialEffects.MultipleScattering = false
process.famosSimHits.MaterialEffects.NuclearInteraction = False

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")

process.source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(0.0),
        MinE = cms.untracked.double(0.0),
        DoubleParticle = cms.untracked.bool(True),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxE = cms.untracked.double(30.0)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(True)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        PFBlockProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PFClusterProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        PFProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring('PFClusterProducer', 
        'PFBlockProducer', 
        'PFProducer'),
    destinations = cms.untracked.vstring('cout')
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.extraction = cms.EDAnalyzer("CalibratableTest",
    #Increase this above zero for increasing amounts of debug output
    debug = cms.int32(3),
    deltaRCandToSim = cms.double(0.4),
    PFCandidates = cms.InputTag("particleFlow"),
    PFSimParticles = cms.InputTag("particleFlowSimParticle"),
    PFClustersEcal = cms.InputTag("particleFlowClusterECAL"),
    PFClustersHcal = cms.InputTag("particleFlowClusterHCAL")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('CalibratableTest_tree.root')
)

process.p1 = cms.Path(process.famosWithElectrons+process.famosWithCaloTowersAndParticleFlow+process.caloJetMetGen*process.particleFlowSimParticle*process.extraction)



