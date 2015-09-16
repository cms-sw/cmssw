# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: step3 --datatier ALCARECO --conditions auto:com10 -s ALCA:PromptCalibProdSiStripGains --eventcontent ALCARECO -n 100 --dasquery=file dataset=/MinimumBias/Run2012C-SiStripCalMinBias-v2/ALCARECO run=200190 --fileout file:step3.root --no_exec
import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCA')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

#process.ALCARECOCalibrationTracks.src = cms.InputTag("ALCARECOSiStripCalMinBias")
process.ALCARECOCalibrationTracks.src = cms.InputTag("generalTracks")

#process.ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = cms.vstring('pathALCARECOSiStripCalMinBias')
process.ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = cms.vstring('*')



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
         '/store/relval/CMSSW_7_5_0_pre4/RelValMinBias_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/0ECACE0E-EBF5-E411-9B86-0025905A6136.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition


# Additional output definition
process.ALCARECOStreamPromptCalibProdSiStripGains = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOPromptCalibProdSiStripGains')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_alcaBeamSpotProducer_*_*', 
        'keep *_MEtoEDMConvertSiStripGains_*_*'),
    fileName = cms.untracked.string('PromptCalibProdSiStripGains.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('PromptCalibProdSiStripGains'),
        dataTier = cms.untracked.string('ALCARECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MCRUN2_75_V1', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiStripGains)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiStripGains,process.endjob_step,process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath)

