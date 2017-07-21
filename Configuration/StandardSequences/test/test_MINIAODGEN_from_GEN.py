# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step1 --fileout file:out.root --filein /store/mc/RunIISummer15GS/DYToMuMu_M_50_TuneAZ_8TeV_pythia8/GEN-SIM/GenOnly_MCRUN2_71_V1-v3/100000/006A257F-D140-E711-93E2-0025901C1A92.root --mc --eventcontent MINIAODGEN --runUnscheduled --datatier MINIAODSIM --conditions 80X_mcRun2_asymptotic_2016_TrancheIV_v6 --step PATGEN --nThreads 1 --era Run2_2016 --python_filename test_MINIAODGEN_from_GEN.py --no_exec -n 5
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('PATGEN',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/RunIISummer15GS/DYToMuMu_M_50_TuneAZ_8TeV_pythia8/GEN-SIM/GenOnly_MCRUN2_71_V1-v3/100000/006A257F-D140-E711-93E2-0025901C1A92.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    allowUnscheduled = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step1 nevts:5'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.MINIAODGENoutput = cms.OutputModule("PoolOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(4),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('MINIAODSIM'),
        filterName = cms.untracked.string('')
    ),
    dropMetaData = cms.untracked.string('ALL'),
    eventAutoFlushCompressedSize = cms.untracked.int32(15728640),
    fastCloning = cms.untracked.bool(False),
    fileName = cms.untracked.string('file:out.root'),
    outputCommands = process.MINIAODGENEventContent.outputCommands,
    overrideInputFileSplitLevels = cms.untracked.bool(True)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_mcRun2_asymptotic_2016_TrancheIV_v6', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.MINIAODGENoutput_step = cms.EndPath(process.MINIAODGENoutput)

# Schedule definition
process.schedule = cms.Schedule(process.endjob_step,process.MINIAODGENoutput_step)

#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
process.load('Configuration.StandardSequences.PATMC_cff')
from FWCore.ParameterSet.Utilities import cleanUnscheduled
process=cleanUnscheduled(process)

# customisation of the process.

# Automatic addition of the customisation function from PhysicsTools.PatAlgos.slimming.miniAOD_tools
from PhysicsTools.PatAlgos.slimming.miniAOD_tools import miniAOD_customizeAllGEN 

#call to customisation function miniAOD_customizeAllGEN imported from PhysicsTools.PatAlgos.slimming.miniAOD_tools
process = miniAOD_customizeAllGEN(process)

# End of customisation functions
