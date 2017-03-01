# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Hadronizer_MgmMatchTune4C_13TeV_madgraph_pythia8_Tauola_taupinu_cff --conditions auto:run1_mc --filein das:/DYJetsToLL_M-50_13TeV-madgraph_v2/Fall13wmLHE-START62_V1-v1/GEN -s GEN,VALIDATION:genvalid_all --datatier GEN --relval 1000000,20000 -n 10 --eventcontent RAWSIM -n 10 --fileout file:step1.root
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
    fileNames = cms.untracked.vstring('/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/00035F04-A0C3-E311-AB8A-02163E00A054.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/0046FC1F-F9C8-E311-BADC-90B11C18C363.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/00B96A0B-A0C3-E311-9BCC-0025B3203750.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/020D792C-D8C9-E311-8ED0-002481D20912.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02233275-92C5-E311-9CAF-782BCB1CFD1C.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/022F6C36-FBC8-E311-8DFB-008CFA002ED8.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/0276432F-BCC3-E311-BCB1-003048678B8E.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02B46ECB-39C3-E311-A579-003048D3740E.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02CDBBE1-F8C8-E311-9CA8-7845C4F932D8.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02D65BCB-F7C8-E311-8A00-AC162DACC3F0.root')
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
			dolheBosonCorr = cms.untracked.bool(True),
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
