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
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/FastSimulation/Validation/data/WM.cfg,v $'),
    annotation = cms.untracked.string('RelVal FastSim W to mu nu')
)
process.ReleaseValidation = cms.untracked.PSet(
    eventsPerJob = cms.untracked.uint32(1000),
    totalNumberOfEvents = cms.untracked.uint32(25000),
    primaryDatasetName = cms.untracked.string('RelValFastSimWM')
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
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    crossSection = cms.untracked.double(17120.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL = 0 !User defined processes', 
            'MSUB(2) = 1 !W production', 
            'MDME(190,1) = 0 !W decay into dbar u', 
            'MDME(191,1) = 0 !W decay into dbar c', 
            'MDME(192,1) = 0 !W decay into dbar t', 
            'MDME(194,1) = 0 !W decay into sbar u', 
            'MDME(195,1) = 0 !W decay into sbar c', 
            'MDME(196,1) = 0 !W decay into sbar t', 
            'MDME(198,1) = 0 !W decay into bbar u', 
            'MDME(199,1) = 0 !W decay into bbar c', 
            'MDME(200,1) = 0 !W decay into bbar t', 
            'MDME(205,1) = 0 !W decay into bbar tp', 
            'MDME(206,1) = 0 !W decay into e+ nu_e', 
            'MDME(207,1) = 1 !W decay into mu+ nu_mu', 
            'MDME(208,1) = 0 !W decay into tau+ nu_tau'),
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

