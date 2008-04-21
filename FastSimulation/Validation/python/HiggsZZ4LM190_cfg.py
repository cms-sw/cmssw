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
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/FastSimulation/Validation/data/HiggsZZ4LM190.cfg,v $'),
    annotation = cms.untracked.string('RelVal FastSim Higgs to ZZ to 4 leptons at 190 GeV')
)
process.ReleaseValidation = cms.untracked.PSet(
    eventsPerJob = cms.untracked.uint32(1000),
    totalNumberOfEvents = cms.untracked.uint32(25000),
    primaryDatasetName = cms.untracked.string('RelValFastSimHiggsZZ4LM190')
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
        processParameters = cms.vstring('PMAS(25,1)=190.0 !mass of Higgs', 
            'MSEL=0 !(D=1) to select between full user control (0, then use MSUB) and some preprogrammed alternative: QCD hight pT processes (1, then ISUB=11, 12, 13, 28, 53, 68), QCD low pT processes (2, then ISUB=11, 12, 13, 28, 53, 68, 91, 92, 94, 95)', 
            'MSUB(102)=1 !ggH', 
            'MSUB(123)=1 !ZZ fusion to H', 
            'MSUB(124)=1 !WW fusion to H', 
            'CKIN(45)=5. !high mass cut on m2 in 2 to 2 process Registered by Chris.Seez@cern.ch', 
            'MSTP(25)=2 !Angular decay correlations in H->ZZ->4fermions Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(46)=150. !high mass cut on secondary resonance m1 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(47)=5. !low mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'CKIN(48)=150. !high mass cut on secondary resonance m2 in 2->1->2 process Registered by Alexandre.Nikitenko@cern.ch', 
            'MDME(174,1)=0 !Z decay into d dbar', 
            'MDME(175,1)=0 !Z decay into u ubar', 
            'MDME(176,1)=0 !Z decay into s sbar', 
            'MDME(177,1)=0 !Z decay into c cbar', 
            'MDME(178,1)=0 !Z decay into b bbar', 
            'MDME(179,1)=0 !Z decay into t tbar', 
            'MDME(182,1)=1 !Z decay into e- e+', 
            'MDME(183,1)=0 !Z decay into nu_e nu_ebar', 
            'MDME(184,1)=1 !Z decay into mu- mu+', 
            'MDME(185,1)=0 !Z decay into nu_mu nu_mubar', 
            'MDME(186,1)=1 !Z decay into tau- tau+', 
            'MDME(187,1)=0 !Z decay into nu_tau nu_taubar', 
            'MDME(210,1)=0 !Higgs decay into dd', 
            'MDME(211,1)=0 !Higgs decay into uu', 
            'MDME(212,1)=0 !Higgs decay into ss', 
            'MDME(213,1)=0 !Higgs decay into cc', 
            'MDME(214,1)=0 !Higgs decay into bb', 
            'MDME(215,1)=0 !Higgs decay into tt', 
            'MDME(216,1)=0 !Higgs decay into', 
            'MDME(217,1)=0 !Higgs decay into Higgs decay', 
            'MDME(218,1)=0 !Higgs decay into e nu e', 
            'MDME(219,1)=0 !Higgs decay into mu nu mu', 
            'MDME(220,1)=0 !Higgs decay into tau nu tau', 
            'MDME(221,1)=0 !Higgs decay into Higgs decay', 
            'MDME(222,1)=0 !Higgs decay into g g', 
            'MDME(223,1)=0 !Higgs decay into gam gam', 
            'MDME(224,1)=0 !Higgs decay into gam Z', 
            'MDME(225,1)=1 !Higgs decay into Z Z', 
            'MDME(226,1)=0 !Higgs decay into W W'),
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

