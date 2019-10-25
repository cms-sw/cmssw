# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/ThirteenTeV/Hadronizer_MgmMatchTune4C_13TeV_madgraph_pythia8_Tauola_cff.py --filein dbs:/DYJetsToLL_M-50_13TeV-madgraph_v2/Fall13wmLHE-START62_V1-v1/GEN --fileout file:SMP-Fall13-00008.root --mc --eventcontent RAWSIM --datatier GEN-SIM --conditions POSTLS162_V1::All --step GEN,SIM --magField 38T_PostLS1 --geometry Extended2015 --python_filename SMP-Fall13-00008_1_cfg.py --no_exec -n 93
import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/00035F04-A0C3-E311-AB8A-02163E00A054.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/0046FC1F-F9C8-E311-BADC-90B11C18C363.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/00B96A0B-A0C3-E311-9BCC-0025B3203750.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/020D792C-D8C9-E311-8ED0-002481D20912.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02233275-92C5-E311-9CAF-782BCB1CFD1C.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/022F6C36-FBC8-E311-8DFB-008CFA002ED8.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/0276432F-BCC3-E311-BCB1-003048678B8E.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02B46ECB-39C3-E311-A579-003048D3740E.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02CDBBE1-F8C8-E311-9CA8-7845C4F932D8.root', 
        '/store/generator/Fall13wmLHE/DYJetsToLL_M-50_13TeV-madgraph_v2/GEN/START62_V1-v1/00000/02D65BCB-F7C8-E311-8A00-AC162DACC3F0.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('Configuration/GenProduction/python/ThirteenTeV/Hadronizer_MgmMatchTune4C_13TeV_madgraph_pythia8_Tauola_cff.py nevts:93'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)


#AODSIM output definition

process.AODSIMoutput = cms.OutputModule("PoolOutputModule",
    compressionLevel = cms.untracked.int32(4),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    outputCommands = process.AODSIMEventContent.outputCommands,
    fileName = cms.untracked.string('Hadronizer_MgmMatchTune4C_ZJetsNoPSb_13TeV_madgraph_pythia8_Tauola.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('AODSIM')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)




# Output definition

#process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
#    SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('generation_step')
#    ),
#    dataset = cms.untracked.PSet(
#        dataTier = cms.untracked.string('GEN-SIM'),
#        filterName = cms.untracked.string('')
#    ),
#    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#    fileName = cms.untracked.string('file:Hadronizer_MgmMatchTune4C_ZJetsNoPSb_13TeV_madgraph_pythia8_Tauola.root'),
#    outputCommands = process.RAWSIMEventContent.outputCommands,
#    splitLevel = cms.untracked.int32(0)
#)
#
# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

process.generator = cms.EDFilter("Pythia8HadronizerFilter",
    ExternalDecays = cms.PSet(
        Tauola = cms.untracked.PSet(
            InputCards = cms.PSet(
                mdtau = cms.int32(0),
                pjak1 = cms.int32(0),
                pjak2 = cms.int32(0)
            ),
            UseTauolaPolarization = cms.bool(True)
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('processParameters'),
        processParameters = cms.vstring('Main:timesAllowErrors    = 10000', 
            'ParticleDecays:limitTau0 = on', 
            'ParticleDecays:tauMax = 10', 
            'Tune:ee 3', 
            'Tune:pp 5')
    ),
    nAttempts = cms.uint32(22),
    HepMCFilter = cms.PSet(
      filterName = cms.string('PartonShowerBsHepMCFilter'),
      filterParameters = cms.PSet(
        Particle_id = cms.int32(5), #pdg id of the particle to filter on. In this case b.
	exclude_status_id = cms.untracked.int32(23), #status id of the particles to filter on. In this case matrix element particles.
      ),
    ),
    UseExternalGenerators = cms.untracked.bool(True),
    comEnergy = cms.double(13000.0),
    filterEfficiency = cms.untracked.double(1.0),
    jetMatching = cms.untracked.PSet(
        MEMAIN_etaclmax = cms.double(-1),
        MEMAIN_excres = cms.string(''),
        MEMAIN_maxjets = cms.int32(-1),
        MEMAIN_minjets = cms.int32(-1),
        MEMAIN_nqmatch = cms.int32(5),
        MEMAIN_qcut = cms.double(-1),
        MEMAIN_showerkt = cms.double(0),
        mode = cms.string('auto'),
        outTree_flag = cms.int32(0),
        scheme = cms.string('Madgraph')
    ),
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
#process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.AODSIMoutput_step = cms.EndPath(process.AODSIMoutput)
#process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.endjob_step,process.AODSIMoutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


