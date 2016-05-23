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
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.ALCARECOCalibrationTracks.src = cms.InputTag("ALCARECOSiStripCalMinBias")
process.ALCARECOCalibrationTracksAAG.src = cms.InputTag("ALCARECOSiStripCalMinBias")
#process.ALCARECOCalibrationTracks.src = cms.InputTag("generalTracks")    #for 2012 data

process.ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = cms.vstring('pathALCARECOSiStripCalMinBias')
#use the same trigger bit of SiStripCalMinBias because the FirstCollisionAfterAbortGap trigger is missing on 2015 data
process.ALCARECOCalMinBiasFilterForSiStripGainsAfterAbortGap.HLTPaths = cms.vstring('pathALCARECOSiStripCalMinBias')

#process.ALCARECOCalMinBiasFilterForSiStripGains.HLTPaths = cms.vstring('*')     #for 2012 data



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60007/869EE593-1FAB-E511-AF99-0025905A60B4.root",
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/0C35C6BF-D3AA-E511-9BC9-0CC47A4C8E16.root",
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/38B847F9-05AA-E511-AB78-00259074AE82.root",
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/D0BAD20B-09AB-E511-B073-0026189438F6.root",
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/DEFA8704-CCAA-E511-8203-0CC47A4D7634.root",
      "/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalMinBias-16Dec2015-v1/60009/FE24690A-2DAA-E511-A96A-00259074AE3E.root"
    ),
    #skipEvents = cms.untracked.uint32(9800),
)

# Uncomment to turn on verbosity output
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.threshold = cms.untracked.string('INFO')
#process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
#process.MessageLogger.debugModules = cms.untracked.vstring("*")
#process.MessageLogger.destinations = cms.untracked.vstring('cout')
#process.MessageLogger.cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))

#process.Tracer = cms.Service("Tracer")

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
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGains_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGainsAfterAbortGap_Output_cff import *

process.ALCARECOStreamPromptCalibProdSiStripGains = cms.OutputModule("PoolOutputModule",
    SelectEvents   = cms.untracked.PSet(
        SelectEvents = cms.vstring(
            'pathALCARECOPromptCalibProdSiStripGains',
            'pathALCARECOPromptCalibProdSiStripGainsAfterAbortGap')
                                       ),
    outputCommands = cms.untracked.vstring(
        'keep *_alcaBeamSpotProducer_*_*',
        'keep *_MEtoEDMConvertSiStripGains_*_*',
        'keep *_MEtoEDMConvertSiStripGainsAfterAbortGap_*_*'),
    fileName = cms.untracked.string('PromptCalibProdSiStripGains.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('PromptCalibProdSiStripGains'),
        dataTier = cms.untracked.string('ALCARECO')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880)
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath = cms.EndPath(process.ALCARECOStreamPromptCalibProdSiStripGains)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOPromptCalibProdSiStripGains,
                                process.pathALCARECOPromptCalibProdSiStripGainsAfterAbortGap,
                                process.endjob_step,
                                process.ALCARECOStreamPromptCalibProdSiStripGainsOutPath)

