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
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring('/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/00A42AC9-4ADF-E111-8749-5404A6388697.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/0CC85F9E-88DF-E111-904F-001D09F29321.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/14F9B88E-56DF-E111-B79B-001D09F24303.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/16E003CB-4ADF-E111-8883-BCAEC518FF41.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/1800C6E5-55DF-E111-AC90-003048F1C58C.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/2462FB44-4EDF-E111-8D3D-BCAEC518FF8A.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/268097BA-58DF-E111-93A0-0025901D6288.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/26F372C9-5FDF-E111-9418-001D09F242EF.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/287CFC74-59DF-E111-9175-5404A63886D4.root', 
        '/store/data/Run2012C/MinimumBias/ALCARECO/SiStripCalMinBias-v2/000/200/190/3866F4D2-4ADF-E111-8749-5404A63886A0.root')
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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiStripGains)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiStripGains,process.endjob_step,process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath)

