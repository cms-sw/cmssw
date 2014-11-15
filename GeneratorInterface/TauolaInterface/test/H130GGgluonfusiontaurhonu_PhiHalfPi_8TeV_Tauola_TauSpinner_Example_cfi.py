# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: H130GGgluonfusion_8TeV_tauola_cfi.py --conditions auto:startup -s GEN,VALIDATION:genvalid_all --datatier GEN --relval 1000000,20000 -n 1000 --eventcontent RAWSIM
import FWCore.ParameterSet.Config as cms

process = cms.Process('VALIDATION')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("GeneratorInterface.TauolaInterface.TauSpinner_cfi")
process.load("GeneratorInterface.TauolaInterface.TauSpinnerFilter_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('H130GGgluonfusion_8TeV_tauola_cfi.py nevts:1000'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('file:step1.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
						   generator = cms.PSet(initialSeed = cms.untracked.uint32(12345)),
						   TauSpinnerGen  = cms.PSet(initialSeed = cms.untracked.uint32(12345)),
						   TauSpinnerZHFilter = cms.PSet(initialSeed = cms.untracked.uint32(429842)),
						   VtxSmeared = cms.PSet(initialSeed = cms.untracked.uint32(275744))
						   )



# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.mix.playback = True
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')


process.generator = cms.EDFilter("Pythia6GeneratorFilter",
				 ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
	UseTauolaPolarization = cms.bool(False),
	InputCards = cms.PSet(
	mdtau = cms.int32(0),
	pjak2 = cms.int32(4),
	pjak1 = cms.int32(4)
	)
        ),
        parameterSets = cms.vstring('Tauola')
	),
				 maxEventsToPrint = cms.untracked.int32(3),
				 pythiaPylistVerbosity = cms.untracked.int32(1),
				 filterEfficiency = cms.untracked.double(1.0),
				 pythiaHepMCVerbosity = cms.untracked.bool(False),
				 comEnergy = cms.double(13000.0),
				 crossSection = cms.untracked.double(0.05),
				 UseExternalGenerators = cms.untracked.bool(True),
				 PythiaParameters = cms.PSet(
        pythiaUESettings = cms.vstring('MSTU(21)=1     ! Check on possible errors during program execution', 
				       'MSTJ(22)=2     ! Decay those unstable particles', 
				       'PARJ(71)=10 .  ! for which ctau  10 mm', 
				       'MSTP(33)=0     ! no K factors in hard cross sections', 
				       'MSTP(2)=1      ! which order running alphaS', 
				       'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)', 
				       'MSTP(52)=2     ! work with LHAPDF', 
				       'PARP(82)=1.921 ! pt cutoff for multiparton interactions', 
				       'PARP(89)=1800. ! sqrts for which PARP82 is set', 
				       'PARP(90)=0.227 ! Multiple interactions: rescaling power', 
				       'MSTP(95)=6     ! CR (color reconnection parameters)', 
				       'PARP(77)=1.016 ! CR', 
				       'PARP(78)=0.538 ! CR', 
				       'PARP(80)=0.1   ! Prob. colored parton from BBR', 
				       'PARP(83)=0.356 ! Multiple interactions: matter distribution parameter', 
				       'PARP(84)=0.651 ! Multiple interactions: matter distribution parameter', 
				       'PARP(62)=1.025 ! ISR cutoff', 
				       'MSTP(91)=1     ! Gaussian primordial kT', 
				       'PARP(93)=10.0  ! primordial kT-max', 
				       'MSTP(81)=21    ! multiple parton interactions 1 is Pythia default', 
				       'MSTP(82)=4     ! Defines the multi-parton model'),
        processParameters = cms.vstring('PMAS(25,1)=130.0      !mass of Higgs', 
					'MSEL=0                  ! user selection for process', 
					'MSUB(102)=1             !ggH', 
					'MSUB(123)=0             !ZZ fusion to H', 
					'MSUB(124)=0             !WW fusion to H', 
					'MSUB(24)=0              !ZH production', 
					'MSUB(26)=0              !WH production', 
					'MSUB(121)=0             !gg to ttH', 
					'MSUB(122)=0             !qq to ttH', 
					'MDME(210,1)=0           !Higgs decay into dd', 
					'MDME(211,1)=0           !Higgs decay into uu', 
					'MDME(212,1)=0           !Higgs decay into ss', 
					'MDME(213,1)=0           !Higgs decay into cc', 
					'MDME(214,1)=0           !Higgs decay into bb', 
					'MDME(215,1)=0           !Higgs decay into tt', 
					'MDME(216,1)=0           !Higgs decay into', 
					'MDME(217,1)=0           !Higgs decay into Higgs decay', 
					'MDME(218,1)=0           !Higgs decay into e nu e', 
					'MDME(219,1)=0           !Higgs decay into mu nu mu', 
					'MDME(220,1)=1           !Higgs decay into tau nu tau', 
					'MDME(221,1)=0           !Higgs decay into Higgs decay', 
					'MDME(222,1)=0           !Higgs decay into g g', 
					'MDME(223,1)=0           !Higgs decay into gam gam', 
					'MDME(224,1)=0           !Higgs decay into gam Z', 
					'MDME(225,1)=0           !Higgs decay into Z Z', 
					'MDME(226,1)=0           !Higgs decay into W W'),
        parameterSets = cms.vstring('pythiaUESettings', 
				    'processParameters')
	)
				 )


process.TauSpinnerGen.parameterSets=cms.vstring("HTSpinCorr")
process.TauSpinnerGen.HTSpinCorr = cms.vdouble(1.57)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen+process.TauSpinnerGen+process.TauSpinnerZHFilter)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.validation_step = cms.EndPath(process.genstepfilter+process.genvalid_all)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.validation_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

