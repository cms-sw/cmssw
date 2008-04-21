# The following comments couldn't be translated into the new config version:

# "TAUO = 0 0 ! Registered by Alexandre.Nikitenko@cern.ch",

# higgs decays

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
    name = cms.untracked.string('$Source: /cvs_server/repositories/CMSSW/CMSSW/FastSimulation/Validation/data/HiggsChargedTausM200.cfg,v $'),
    annotation = cms.untracked.string('RelVal FastSim charge Higgs to taus, mHiggs=200GeV')
)
process.ReleaseValidation = cms.untracked.PSet(
    eventsPerJob = cms.untracked.uint32(1000),
    totalNumberOfEvents = cms.untracked.uint32(25000),
    primaryDatasetName = cms.untracked.string('RelValFastSimHiggsChargedTausM200')
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
        processParameters = cms.vstring('MSEL = 0 ! user control', 
            'MSUB(401) = 1 ! gg->tbH+ Registered by Alexandre.Nikitenko@cern.ch', 
            'MSUB(402) = 1 ! qq->tbH+ Registered by Alexandre.Nikitenko@cern.ch', 
            'IMSS(1)= 1 ! MSSM ', 
            'RMSS(5) = 30. ! TANBETA', 
            'RMSS(19) = 200. ! (D=850.) m_A', 
            'MDME(503,1)=0 !Higgs(H+) decay into dbar u', 
            'MDME(504,1)=0 !Higgs(H+) decay into sbar c', 
            'MDME(505,1)=0 !Higgs(H+) decay into bbar t', 
            'MDME(506,1)=0 !Higgs(H+) decay into b bar t', 
            'MDME(507,1)=0 !Higgs(H+) decay into e+ nu_e', 
            'MDME(508,1)=0 !Higgs(H+) decay into mu+ nu_mu', 
            'MDME(509,1)=1 !Higgs(H+) decay into tau+ nu_tau', 
            'MDME(510,1)=0 !Higgs(H+) decay into tau prime+ nu_tau', 
            'MDME(511,1)=0 !Higgs(H+) decay into W+ h0', 
            'MDME(512,1)=0 !Higgs(H+) decay into ~chi_10 ~chi_1+', 
            'MDME(513,1)=0 !Higgs(H+) decay into ~chi_10 ~chi_2+', 
            'MDME(514,1)=0 !Higgs(H+) decay into ~chi_20 ~chi_1+', 
            'MDME(515,1)=0 !Higgs(H+) decay into ~chi_20 ~chi_2+', 
            'MDME(516,1)=0 !Higgs(H+) decay into ~chi_30 ~chi_1+', 
            'MDME(517,1)=0 !Higgs(H+) decay into ~chi_30 ~chi_2+', 
            'MDME(518,1)=0 !Higgs(H+) decay into ~chi_40 ~chi_1+', 
            'MDME(519,1)=0 !Higgs(H+) decay into ~chi_40 ~chi_2+', 
            'MDME(520,1)=0 !Higgs(H+) decay into ~t_1 ~b_1bar', 
            'MDME(521,1)=0 !Higgs(H+) decay into ~t_2 ~b_1bar', 
            'MDME(522,1)=0 !Higgs(H+) decay into ~t_1 ~b_2bar', 
            'MDME(523,1)=0 !Higgs(H+) decay into ~t_2 ~b_2bar', 
            'MDME(524,1)=0 !Higgs(H+) decay into ~d_Lbar ~u_L', 
            'MDME(525,1)=0 !Higgs(H+) decay into ~s_Lbar ~c_L', 
            'MDME(526,1)=0 !Higgs(H+) decay into ~e_L+ ~nu_eL', 
            'MDME(527,1)=0 !Higgs(H+) decay into ~mu_L+ ~nu_muL', 
            'MDME(528,1)=0 !Higgs(H+) decay into ~tau_1+ ~nu_tauL', 
            'MDME(529,1)=0 !Higgs(H+) decay into ~tau_2+ ~nu_tauL'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters', 
            'pythiaMSSMmhmax'),
        pythiaMSSMmhmax = cms.vstring('RMSS(2)= 200. ! SU(2) gaugino mass ', 
            'RMSS(3)= 800. ! SU(3) (gluino) mass ', 
            'RMSS(4)= 200. ! higgsino mass parameter', 
            'RMSS(6)= 1000. ! left slepton mass', 
            'RMSS(7)= 1000. ! right slepton mass', 
            'RMSS(8)= 1000. ! right slepton mass', 
            'RMSS(9)= 1000. ! right squark mass', 
            'RMSS(10)= 1000. ! left sq mass for 3th gen/heaviest stop mass', 
            'RMSS(11)= 1000. ! right sbottom mass/lightest sbotoom mass', 
            'RMSS(12)= 1000. ! right stop mass/lightest stop mass', 
            'RMSS(13)= 1000. ! left stau mass', 
            'RMSS(14)= 1000. ! right stau mass', 
            'RMSS(15)= 2449. ! Ab', 
            'RMSS(16)= 2449. ! At', 
            'RMSS(17)= 2449. ! Atau')
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

