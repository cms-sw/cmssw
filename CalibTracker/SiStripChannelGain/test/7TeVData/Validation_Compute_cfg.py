import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")

#this block is there to solve issue related to SiStripQualityRcd
process.load("CalibTracker.SiStripESProducers.SiStripQualityESProducer_cfi")
process.load("CalibTracker.SiStripESProducers.fake.SiStripDetVOffFakeESSource_cfi")
process.es_prefer_fakeSiStripDetVOff = cms.ESPrefer("SiStripDetVOffFakeESSource","siStripDetVOffFakeESSource")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('ERROR')  ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(XXX_FIRSTRUN_XXX),
    lastValue  = cms.uint64(XXX_LASTRUN_XXX)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.GlobalTag.globaltag = 'XXX_GT_XXX'

calibTreeList = cms.vstring()
XXX_CALIBTREE_XXX

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalibValidation.InputFiles          = calibTreeList 
process.SiStripCalibValidation.FirstSetOfConstants = cms.untracked.bool(False)
process.SiStripCalibValidation.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module


process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Validation_Tree.root')  
)

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMFileSaver_cfi")
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/Express/PCLTest/ALCAPROMPT'

process.p = cms.Path(process.SiStripCalibValidation * process.dqmSaver)
