# The following comments couldn't be translated into the new config version:

# Work-around because of a bug in HLT 
# Reconstruction sequence
import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")
process.load("Configuration.Generator.PythiaUESettings_cfi")

# Famos sequences (With HLT)
process.load("FastSimulation.Configuration.CommonInputsFake_cff")

process.load("FastSimulation.Configuration.FamosSequences_cff")

# L1 Emulator and HLT Setup
process.load("FastSimulation.HighLevelTrigger.common.HLTSetup_cff")

process.load("PhysicsTools.HepMCCandAlgos.genEventWeight_cfi")

process.load("PhysicsTools.HepMCCandAlgos.genEventScale_cfi")

# HLT paths
process.load("HLTrigger.Configuration.main.HLTpaths_cff")

# To write out events 
process.load("FastSimulation.Configuration.EventContent_cff")

# Keep the logging output to a nice level #
process.load("FWCore.MessageService.MessageLogger_cfi")

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.4 $'),
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/FastSimulation/Validation/data/ZPrimeEEM4000.cfg,v $'),
    annotation = cms.untracked.string('RelVal FastSim SSM ZPrime, M=4000 GeV, with decay to electrons')
)
process.ReleaseValidation = cms.untracked.PSet(
    eventsPerJob = cms.untracked.uint32(1000),
    totalNumberOfEvents = cms.untracked.uint32(25000),
    primaryDatasetName = cms.untracked.string('RelValFastSimZPrimeEEM4000')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    # This is to initialize the random engines of Famos
    moduleSeeds = cms.PSet(
        l1ParamMuons = cms.untracked.uint32(54525),
        caloRecHits = cms.untracked.uint32(654321),
        MuonSimHits = cms.untracked.uint32(97531),
        muonCSCDigis = cms.untracked.uint32(525432),
        muonDTDigis = cms.untracked.uint32(67673876),
        famosSimHits = cms.untracked.uint32(13579),
        paramMuons = cms.untracked.uint32(54525),
        famosPileUp = cms.untracked.uint32(918273),
        VtxSmeared = cms.untracked.uint32(123456789),
        muonRPCDigis = cms.untracked.uint32(524964),
        siTrackerGaussianSmearingRecHits = cms.untracked.uint32(24680)
    ),
    # This is to initialize the random engine of the source
    sourceSeed = cms.untracked.uint32(123456789)
)

process.source = cms.Source("PythiaSource",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL = 0 ', 
            'MSUB(141) = 1 !ff gamma z0 Z0', 
            'MSTP(44) = 3 !only select the Z process', 
            'PMAS(32,1) = 4000. !mass of Zprime', 
            'CKIN(1) = 400 !(D=2. GeV)', 
            'MDME(289,1)= 0 !d dbar', 
            'MDME(290,1)= 0 !u ubar', 
            'MDME(291,1)= 0 !s sbar', 
            'MDME(292,1)= 0 !c cbar', 
            'MDME(293,1)= 0 !b bar', 
            'MDME(294,1)= 0 !t tbar', 
            'MDME(295,1)= 0 !4th gen Q Qbar', 
            'MDME(296,1)= 0 !4th gen Q Qbar', 
            'MDME(297,1)= 1 !e e', 
            'MDME(298,1)= 0 !neutrino e e', 
            'MDME(299,1)= 0 ! mu mu', 
            'MDME(300,1)= 0 !neutrino mu mu', 
            'MDME(301,1)= 0 !tau tau', 
            'MDME(302,1)= 0 !neutrino tau tau', 
            'MDME(303,1)= 0 !4th generation lepton', 
            'MDME(304,1)= 0 !4th generation neutrino', 
            'MDME(305,1)= 0 !W W', 
            'MDME(306,1)= 0 !H charged higgs', 
            'MDME(307,1)= 0 !Z', 
            'MDME(308,1)= 0 !Z', 
            'MDME(309,1)= 0 !sm higgs', 
            'MDME(310,1)= 0 !weird neutral higgs HA'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.Timing = cms.Service("Timing")

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO')
    ),
    fileName = cms.untracked.string('FEVTWithHLT.root')
)

process.simulation = cms.Path(process.simulationWithFamos+process.genEventScale+process.genEventWeight)
process.hltEnd = cms.Sequence(process.dummyModule)
process.reconstruction = cms.Path(process.doCalo+process.towerMakerForAll+process.reconstructionWithFamos)
process.outpath = cms.EndPath(process.o1)
process.famosPileUp.UseTRandomEngine = True
process.famosSimHits.UseTRandomEngine = True
process.siTrackerGaussianSmearingRecHits.UseTRandomEngine = True
process.caloRecHits.UseTRandomEngine = True
process.paramMuons.UseTRandomEngine = True
process.famosPileUp.PileUpSimulator.averageNumber = 0.0
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = True
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = True
process.MessageLogger.destinations = ['detailedInfo.txt']

