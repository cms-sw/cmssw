# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Hadronizer_MgmMatchTune4C_13TeV_madgraph_pythia8_Tauola_taupinu_cff --conditions auto:run1_mc --filein das:/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN -s GEN,VALIDATION:genvalid_all --datatier GEN --relval 1000000,20000 -n 10 --eventcontent RAWSIM -n 10 --fileout file:step1.root
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
    input = cms.untracked.int32(10000)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/001CB45E-D0D4-E311-BFAC-001E4F32EA8A.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/0027BC57-FCD3-E311-AC82-0025904C5DE0.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/002E2248-C1D4-E311-8551-008CFA0514E0.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/004614E9-FDD3-E311-8585-0025904E9012.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/00554985-9BCE-E311-B256-00266CFFC550.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/006F4D36-D3D1-E311-93FA-00259021A4B2.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/0076FFB6-D2D4-E311-A2B2-001E4F32EA8A.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/00BB12D7-FBD3-E311-A814-0025904C6224.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/00BC6CA3-A3CD-E311-822B-02163E008DE1.root', 
        '/store/generator/Fall13wmLHE/WJetsToLNu_13TeV-madgraph/GEN/START62_V1-v1/00000/021E00D0-91CF-E311-8A92-7845C4FC3614.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('Hadronizer_MgmMatchTune4C_13TeV_madgraph_pythia8_Tauola_taupinu_cff nevts:10'),
    name = cms.untracked.string('Applications')
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

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            UseTauolaPolarization = cms.bool(True),
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak2 = cms.int32(3),
                pjak1 = cms.int32(3)
            ),
	    dmMatch = cms.untracked.double(0.5),
	    dolhe = cms.untracked.bool(True),
	    dolheBosonCorr = cms.untracked.bool(False),
	    ntries = cms.untracked.int32(20)
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.0),
    UseExternalGenerators = cms.untracked.bool(True),
    jetMatching = cms.untracked.PSet(
        MEMAIN_nqmatch = cms.int32(5),
        MEMAIN_showerkt = cms.double(0),
        MEMAIN_minjets = cms.int32(-1),
        MEMAIN_qcut = cms.double(-1),
        MEMAIN_excres = cms.string(''),
        MEMAIN_etaclmax = cms.double(-1),
        outTree_flag = cms.int32(0),
        scheme = cms.string('Madgraph'),
        MEMAIN_maxjets = cms.int32(-1),
        mode = cms.string('auto')
    ),
    PythiaParameters = cms.PSet(
        processParameters = cms.vstring('Main:timesAllowErrors = 10000', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tauMax = 10', 
            'Tune:ee 3', 
            'Tune:pp 5'),
        parameterSets = cms.vstring('processParameters')
    )
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

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
