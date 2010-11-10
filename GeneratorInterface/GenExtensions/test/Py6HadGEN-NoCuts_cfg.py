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
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('GeneratorInterface/Pythia6Interface/BCVEGPY_cfi.py nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("LHESource",
    fileNames = cms.untracked.vstring('file:Bc2JpsiPi.lhe')
##     firstEvent = cms.untracked.uint32(300001)
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('BCVEPY_onlyGEN-NoCuts.root'),
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
process.GlobalTag.globaltag = 'MC_31X_V8::All'
process.generator = cms.EDFilter("Pythia6HadronizerFilter",
    eventsToPrint = cms.untracked.uint32(10),
    pythiaHepMCVerbosity = cms.untracked.bool(True),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    comEnergy = cms.double(10000.0),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('MSTP(51)=10042', 
            'MSTP(52)=2', 
            'MSTP(61)=0             ! Hadronization of the initial protons', 
            'MDME(997,2) = 0        ! PHASE SPACE', 
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
            'MDME(997,1) = 2        !  Bc -> pi J/Psi', 
            'MSTJ(22)=2   ! Do not decay unstable particles', 
            'PARJ(71)=10. ! with c*tau > cTauMin (in mm) in PYTHIA'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters'),
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
            'PARP(93)=15.0  ! ')
    )
)
process.ProducerSourceSequence = cms.Sequence(process.generator)
process.generation_step = cms.Path(process.pgen)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step)
process.schedule.extend([process.endjob_step,process.out_step])

# special treatment in case of production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProducerSourceSequence*getattr(process,path)._seq

def customise(process):
	process.VtxSmeared.src = 'generator'
	process.output.outputCommands.append('keep *_source_*_*')
	process.output.outputCommands.append('keep *_generator_*_*')

	return process


# End of customisation function definition

process = customise(process)


