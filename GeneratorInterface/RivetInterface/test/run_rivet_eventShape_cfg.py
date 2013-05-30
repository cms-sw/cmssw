# Auto generated configuration file
# using: 
# Revision: 1.381.2.18 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/MinBias_TuneZ2_7TeV_pythia6_cff.py -s GEN --datatier=GEN-SIM-RAW --conditions auto:mc --eventcontent RAWSIM --no_exec -n 10000 --python_filename=rivet_cfg.py --customise=Configuration/GenProduction/rivet_customize.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.RandomNumberGeneratorService.generator.initialSeed = 19999024
process.MessageLogger = cms.Service("MessageLogger", 
 cout = cms.untracked.PSet( 
  default = cms.untracked.PSet( ## kill all messages in the log 

   limit = cms.untracked.int32(0) 
  ), 
  FwkJob = cms.untracked.PSet( ## but FwkJob category - those unlimitted 

   limit = cms.untracked.int32(-1) 
  ) 
 ), 
 categories = cms.untracked.vstring('FwkJob'), 
 destinations = cms.untracked.vstring('cout') 
) 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('PYTHIA6-MinBias TuneZ2 at 7TeV'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/GenProduction/python/Attic/MinBias_TuneZ2_7TeV_pythia6_cff.py,v $')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('MinBias_TuneZ2_7TeV_pythia6_cff_py_GEN.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.0),
    crossSection = cms.untracked.double(71260000000.0),
    maxEventsToPrint = cms.untracked.int32(0),

				 
   PythiaParameters = cms.PSet(
       pythiaUESettings = cms.vstring('MSTU(21)=1     ! Check on possible errors during program execution',
                'MSTJ(22)=2     ! Decay those unstable particles', 
                'PARJ(71)=10.   ! for which ctau  10 mm', 
                'MSTP(33)=0     ! no K factors in hard cross sections', 
                'MSTP(2)=1      ! which order running alphaS', 
                'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
                'MSTP(52)=2     ! work with LHAPDF',

                'PARP(82)=1.832 ! pt cutoff for multiparton interactions', 
                'PARP(89)=1800. ! sqrts for which PARP82 is set', 
                'PARP(90)=0.275 ! Multiple interactions: rescaling power', 

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
				      
#            'PARP(82)=1.921 ! pt cutoff for multiparton interactions', 
#            'PARP(90)=0.227 ! Multiple interactions: rescaling power', 

        processParameters = cms.vstring('MSEL=1               ! QCD hight pT processes', 
            'CKIN(3)=50.          ! minimum pt hat for hard interactions', 
            'CKIN(4)=122220.         ! maximum pt hat for hard interactions',
            'MSTP(142)=2           ! Turns on the PYWEVT Pt reweighting routine'),
                             
           CSAParameters = cms.vstring(
                     'CSAMODE = 7     ! 7 towards a flat QCD spectrum',
                     'PTPOWER = 5.0   ! reweighting of the pt spectrum',
                ),

        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters', 'CSAParameters')
    )
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from Configuration.GenProduction.rivet_customize
from Configuration.GenProduction.rivet_customize import customise 

#call to customisation function customise imported from Configuration.GenProduction.rivet_customize
process = customise(process)

# End of customisation functions
#process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_EWK_11_021')
process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_SMP_12_022')
#process.rivetAnalyzer.AnalysisNames = cms.vstring('D0_2008_S7863608.aida')
process.rivetAnalyzer.OutputFile = cms.string('mcfile_flat.aida')
process.rivetAnalyzer.UseExternalWeight = cms.bool(True) 
