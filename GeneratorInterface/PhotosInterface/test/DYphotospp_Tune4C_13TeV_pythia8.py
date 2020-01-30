# A script to produce Z events using pythia8 and with FSR added by Photos++
# This script is to be used to compare with DYnorad_Tune4C_13TeV_pythia8.py to demonstrate/test
# the additions of FSR by Photos++

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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('Configuration/Generator/python/ThirteenTeV/WToMuNu_M_500_Tune4C_13TeV_pythia8_cfi.py nevts:100000'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('DYphotospp_Tune4C_13TeV_pythia8_cfi_py_GEN_VALIDATION.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
                                 ExternalDecays = cms.PSet(
	Photospp = cms.untracked.PSet(
	parameterSets = cms.vstring("setExponentiation","setInfraredCutOff","setMomentumConservationThreshold"),
	setExponentiation = cms.bool(True),
	setInfraredCutOff = cms.double(0.00011),
	setMomentumConservationThreshold = cms.double(20.0) # 0.5GeV
	),
	parameterSets = cms.vstring( "Photospp")
	),
				 
    UseExternalGenerators = cms.untracked.bool(True),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.0),
    crossSection = cms.untracked.double(0.2188),
    maxEventsToPrint = cms.untracked.int32(1),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('Main:timesAllowErrors = 10000', 
					'ParticleDecays:limitTau0 = on', 
					'ParticleDecays:tauMax = 10', 
					'Tune:ee 3', 
					'Tune:pp 5',
					'PartonLevel:FSR = off',
					'WeakSingleBoson:ffbar2gmZ = on', 
					'23:onMode = off', 
					'23:onIfAny = 13', 
					'23:mMin = 50.', 
					'23:mMax = 120.'
					),
        parameterSets = cms.vstring('processParameters')
    )
)

process.genParticles = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("generator:unsmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)
process.printTree1 = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint  = cms.untracked.int32(10)
)

process.ProductionFilterSequence = cms.Sequence(process.generator*process.genParticles*process.printTree1)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.validation_step = cms.EndPath(process.genstepfilter+process.genvalid_all)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.validation_step,process.endjob_step,process.RAWSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions
