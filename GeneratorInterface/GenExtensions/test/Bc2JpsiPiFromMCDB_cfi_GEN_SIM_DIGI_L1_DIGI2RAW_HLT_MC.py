# Auto generated configuration file
# using: 
# Revision: 1.123 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Bc2JpsiPiFromMCDB_cfi -s GEN:ProducerSourceSequence,SIM,DIGI,L1,DIGI2RAW,HLT --eventcontent FEVTSIM --conditions FrontierConditions_GlobalTag,MC_31X_V2::All -n 10 --no_exe
import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/Digi_cff')
process.load('Configuration/StandardSequences/SimL1Emulator_cff')
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('HLTrigger/Configuration/HLT_1E31_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.MessageLogger.cerr.threshold = 'INFO'

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('Bc2JpsiPiFromMCDB_cfi nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("MCDBSource",
    articleID = cms.uint32(233),
    supportedProtocols = cms.untracked.vstring('rfio')
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.FEVTSIMEventContent.outputCommands,
    fileName = cms.untracked.string('Bc2JpsiPiFromMCDB_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
##process.GlobalTag.globaltag = 'MC_31X_V8::All'
process.GlobalTag.globaltag = 'MC_37Y_V0::All'
process.mumugenfilter = cms.EDFilter("MCParticlePairFilter",
    Status = cms.untracked.vint32(1, 1),
    MinPt = cms.untracked.vdouble(2.5, 2.5),
    MaxEta = cms.untracked.vdouble(2.5, 2.5),
    moduleLabel = cms.untracked.string('generator'),
    MinEta = cms.untracked.vdouble(-2.5, -2.5),
    ParticleCharge = cms.untracked.int32(-1),
    ParticleID1 = cms.untracked.vint32(13),
    ParticleID2 = cms.untracked.vint32(13)
)
process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(0.1235),
    eventsToPrint = cms.untracked.uint32(10),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    comEnergy = cms.double(10000.0),
    crossSection = cms.untracked.double(62),
    PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTJ(11)=3     ! Choice of the fragmentation function', 
            'MSTJ(22)=2     ! Decay those unstable particles', 
            'PARJ(71)=10 .  ! for which ctau  10 mm', 
            'MSTP(2)=1      ! which order running alphaS', 
            'MSTP(33)=0     ! no K factors in hard cross sections', 
            'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
            'MSTP(52)=2     ! work with LHAPDF', 
            'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default', 
            'MSTP(82)=4     ! Defines the multi-parton model', 
            'MSTU(21)=1     ! Check on possible errors during program execution', 
            'PARP(82)=1.8387   ! pt cutoff for multiparton interactions', 
            'PARP(89)=1960. ! sqrts for which PARP82 is set', 
            'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter', 
            'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter', 
            'PARP(90)=0.16  ! Multiple interactions: rescaling power', 
            'PARP(67)=2.5    ! amount of initial-state radiation', 
            'PARP(85)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(86)=1.0  ! gluon prod. mechanism in MI', 
            'PARP(62)=1.25   ! ', 
            'PARP(64)=0.2    ! ', 
            'MSTP(91)=1      !', 
            'PARP(91)=2.1   ! kt distribution', 
            'PARP(93)=15.0  ! '),
        processParameters = cms.vstring('MDME(997,2) = 0        ! PHASE SPACE', 
            'KFDP(997,1) = 211      ! pi+', 
            'KFDP(997,2) = 443      ! J/psi', 
            'KFDP(997,3) = 0        ! nada', 
            'KFDP(997,4) = 0        ! nada', 
            'KFDP(997,5) = 0        ! nada', 
            'PMAS(143,1) = 6.286', 
            'PMAS(143,4) = 0.138', 
            'MDME(858,1) = 0  ! J/psi->e+e-', 
            'MDME(859,1) = 1  ! J/psi->mumu', 
            'MDME(860,1) = 0', 
            'MDME(998,1) = 3', 
            'MDME(999,1) = 3', 
            'MDME(1000,1) = 3', 
            'MDME(1001,1) = 3', 
            'MDME(1002,1) = 3', 
            'MDME(1003,1) = 3', 
            'MDME(1004,1) = 3', 
            'MDME(1005,1) = 3', 
            'MDME(1006,1) = 3', 
            'MDME(1007,1) = 3', 
            'MDME(1008,1) = 3', 
            'MDME(1009,1) = 3', 
            'MDME(1010,1) = 3', 
            'MDME(1011,1) = 3', 
            'MDME(1012,1) = 3', 
            'MDME(1013,1) = 3', 
            'MDME(1014,1) = 3', 
            'MDME(1015,1) = 3', 
            'MDME(1016,1) = 3', 
            'MDME(1017,1) = 3', 
            'MDME(1018,1) = 3', 
            'MDME(1019,1) = 3', 
            'MDME(1020,1) = 3', 
            'MDME(1021,1) = 3', 
            'MDME(1022,1) = 3', 
            'MDME(1023,1) = 3', 
            'MDME(1024,1) = 3', 
            'MDME(1025,1) = 3', 
            'MDME(1026,1) = 3', 
            'MDME(1027,1) = 3', 
            'MDME(997,1) = 1        !  Bc -> pi J/Psi', 
            'MSTJ(22)=2   ! Do not decay unstable particles', 
            'PARJ(71)=10. ! with c*tau > cTauMin (in mm) in PYTHIA', 
            'MSTP(61)=0'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)
process.ProducerSourceSequence = cms.Sequence(process.generator*process.mumugenfilter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.out_step])
# special treatment in case of production filter sequence  
for path in process.paths: 
    getattr(process,path)._seq = process.ProducerSourceSequence*getattr(process,path)._seq
