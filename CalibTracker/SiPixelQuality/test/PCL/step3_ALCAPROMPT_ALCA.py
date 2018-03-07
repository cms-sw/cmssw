# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3_ALCAPROMPT --datatier ALCARECO --conditions 100X_dataRun2_Express_v2 -s ALCA:PromptCalibProdSiPixel --eventcontent ALCARECO -n -1 --filein file:SiPixelCalZeroBias.root --fileout file:step3.root --no_exe
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:SiPixelCalZeroBias.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3_ALCAPROMPT nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition


# Additional output definition
process.ALCARECOStreamPromptCalibProdSiPixel = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiPixel')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('ALCARECO'),
        filterName = cms.untracked.string('PromptCalibProdSiPixel')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('PromptCalibProdSiPixel.root'),
    outputCommands = cms.untracked.vstring(
        'drop *', 
        'keep *_ALCARECOPromptCalibProdSiPixel_*_*', 
        'keep *_siPixelStatusProducer_*_*'
    )
)

# Other statements
process.ALCARECOEventContent.outputCommands.extend(process.OutALCARECOPromptCalibProdSiPixel_noDrop.outputCommands)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_dataRun2_Express_v2', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiPixelOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiPixel)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiPixel,process.endjob_step,process.ALCARECOStreamPromptCalibProdSiPixelOutPath)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
