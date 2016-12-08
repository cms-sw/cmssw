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

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'XXX_GT_XXX', '')

from FileList_cfg import calibTreeList

process.load("CalibTracker.SiStripChannelGain.computeGain_cff")
process.SiStripCalib.FirstSetOfConstants = cms.untracked.bool(False)
process.SiStripCalibValidation.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
process.SiStripCalibValidation.saveSummary         = cms.untracked.bool(True)


if(XXX_PCL_XXX):
   process.SiStripCalibValidation.AlgoMode = cms.untracked.string('PCL')
   process.SiStripCalibValidation.harvestingMode = cms.untracked.bool(True)
   process.source = cms.Source("PoolSource",
       secondaryFileNames = cms.untracked.vstring(),
       fileNames = calibTreeList,
       processingMode = cms.untracked.string('RunsAndLumis')
   )
else:
   process.SiStripCalibValidation.InputFiles          = calibTreeList
   process.source = cms.Source("EmptyIOVSource",
       timetype   = cms.string('runnumber'),
       interval   = cms.uint64(1),
       firstValue = cms.uint64(XXX_FIRSTRUN_XXX),
       lastValue  = cms.uint64(XXX_LASTRUN_XXX)
   )

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Validation_Tree.root')  
)

process.DQMStore = cms.Service("DQMStore")

if(XXX_PCL_XXX):
   process.load("DQMServices.Components.DQMFileSaver_cfi")
   process.dqmSaver.convention = 'Offline'
   process.dqmSaver.workflow = '/Express/PCLTest/ALCAPROMPT'

   process.EDMtoMEConvert = cms.EDAnalyzer("EDMtoMEConverter",
       Frequency = cms.untracked.int32(50),
       Name = cms.untracked.string('EDMtoMEConverter'),
       Verbosity = cms.untracked.int32(0),
       convertOnEndLumi = cms.untracked.bool(True),
       convertOnEndRun = cms.untracked.bool(True),
       lumiInputTag = cms.InputTag("MEtoEDMConvertSiStripGains","MEtoEDMConverterLumi"),
       runInputTag = cms.InputTag("MEtoEDMConvertSiStripGains","MEtoEDMConverterRun")
   )

   process.p = cms.Path(process.EDMtoMEConvert * process.SiStripCalibValidation * process.dqmSaver) 
else:
   process.p = cms.Path(process.SiStripCalibValidation)


#import PhysicsTools.PythonAnalysis.LumiList as LumiList
#process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/13TeV/DCSOnly/json_DCSONLY_Run2015B.txt').getVLuminosityBlockRange()
