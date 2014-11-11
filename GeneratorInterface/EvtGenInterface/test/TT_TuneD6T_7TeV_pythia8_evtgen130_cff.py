# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: TT_TuneD6T_7TeV_pythia6_evtgen_cff --conditions auto:startup -s GEN,VALIDATION:genvalid_all --datatier GEN --relval 1000000,20000 -n 30000 --eventcontent RAWSIM
import FWCore.ParameterSet.Config as cms

process = cms.Process('VALIDATION')

from GeneratorInterface.EvtGenInterface.EvtGenSetting_cff import *

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
process.load('GeneratorInterface.EvtGenInterface.test_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('TT_TuneD6T_7TeV_pythia8_evtgen_cff nevts:30000'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('Output_TT_TuneD6T_7TeV_pythia8_evtgen_cff_GEN_VALIDATION.root'),
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
        EvtGen130 = cms.untracked.PSet(
            decay_table = cms.string('DECAY_2010.DEC'),
            particle_property_file = cms.FileInPath('GeneratorInterface/EvtGenInterface/data/evt.pdl'),
            user_decay_files = cms.vstring('DECAY_2010.DEC'),
            list_forced_decays = cms.vstring(),
            operates_on_particles = cms.vint32(0)
        ),
        parameterSets = cms.vstring('EvtGen130')
    ),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(7000.0),
    crossSection = cms.untracked.double(91.32),
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('Top:gg2ttbar = on', 
                                        'Top:qqbar2ttbar = on'
					),
        parameterSets = cms.vstring('processParameters')
    )
)

process.generator.PythiaParameters.processParameters.extend(EvtGenExtraParticles)

process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.mywriter+process.pgen)
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
