# Auto generated configuration file
# using: 
# Revision: 1.110 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/POMWIG_SingleDiffractiveDijetsPlus_10TeV_Pt_30_cff.py -s GEN:ProductionFilterSequence --conditions FrontierConditions_GlobalTag,IDEAL_30X::All --datatier GEN --eventcontent RAWSIM -n 1000 --customise=Configuration/GenProduction/Pomwig_custom.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('GEN')

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

process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.INFO.limit = -1

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('POMWIG SD plus Di-jets ptmin 30 GeV at 10TeV'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/GeneratorInterface/PomwigInterface/test/POMWIG_SingleDiffractiveDijetsPlus_10TeV_Pt_30_cff_py_GEN_IDEAL_Producer.py,v $')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

process.source = cms.Source("EmptySource")
 
herwig6Parameters = cms.PSet(
	comEnergy = cms.double(10000.0),
	useJimmy = cms.bool(False),
	doMPInteraction = cms.bool(False),

	herwigHepMCVerbosity = cms.untracked.bool(False),
	herwigVerbosity = cms.untracked.int32(1),
	printCards = cms.untracked.bool(True),
	maxEventsToPrint = cms.untracked.int32(2),

	crossSection = cms.untracked.double(-1.0),
	filterEfficiency = cms.untracked.double(1.0),

	emulatePythiaStatusCodes = cms.untracked.bool(True),
)

process.generator = cms.EDProducer("PomwigProducer",
    herwig6Parameters,
    HerwigParameters = cms.PSet(
        parameterSets = cms.vstring('SD1InclusiveJets'),
        SD1InclusiveJets = cms.vstring(
            'NSTRU      = 14         ! H1 Pomeron Fit B', 
            'Q2WWMN     = 1E-6       ! Minimum |t|', 
            'Q2WWMX     = 4.0        ! Maximum |t|', 
            'YWWMIN     = 1E-6       ! Minimum xi', 
            'YWWMAX     = 0.2        ! Maximum xi', 
            'IPROC      = 11500      ! Process PomP -> jets', 
            'PTMIN      = 30         ! 2->2 PT min', 
            'MODPDF(1)  = -1         ! Set MODPDF', 
            'MODPDF(2)  = 10150      ! Set MODPDF CTEQ61')
    ),
    diffTopology = cms.int32(1),
    survivalProbability = cms.double(0.05),
    h1fit = cms.int32(2)
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('/tmp/antoniov/POMWIG_SingleDiffractiveDijetsPlus_10TeV_Pt_30_cff_py_GEN_producer.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN'),
        filterName = cms.untracked.string('')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.GlobalTag.globaltag = 'IDEAL_30X::All'
#process.pomwigfilter = cms.EDFilter("PomwigFilter")
#process.ProductionFilterSequence = cms.Sequence(process.pomwigfilter)
process.herwig6filter = cms.EDFilter("Herwig6Filter")
process.ProductionFilterSequence = cms.Sequence(process.herwig6filter)

# Path and EndPath definitions
process.generation_step = cms.Path(process.generator*process.ProductionFilterSequence*process.pgen)
process.endjob_step = cms.Path(process.ProductionFilterSequence*process.endOfProcess)
process.out_step = cms.EndPath(process.ProductionFilterSequence*process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.endjob_step,process.out_step)

# special treatment in case of production filter sequence  
#for path in process.paths: 
#    getattr(process,path)._seq = process.ProductionFilterSequence*getattr(process,path)._seq


# Automatic addition of the customisation function

def customise(process):

	process.genParticles.abortOnUnknownPDGCode = False

	return(process)


# End of customisation function definition

process = customise(process)
