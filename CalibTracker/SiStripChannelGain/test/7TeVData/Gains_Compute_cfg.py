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
process.SiStripCalib.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
process.SiStripCalib.saveSummary         = cms.untracked.bool(True)


if(XXX_PCL_XXX):
   process.SiStripCalib.AlgoMode = cms.untracked.string('PCL')
   process.SiStripCalib.harvestingMode = cms.untracked.bool(True)
   process.source = cms.Source("PoolSource",
       secondaryFileNames = cms.untracked.vstring(),
       fileNames = calibTreeList,
       processingMode = cms.untracked.string('RunsAndLumis')
   )
else:
   process.SiStripCalib.InputFiles          = calibTreeList
   process.source = cms.Source("EmptyIOVSource",
       timetype   = cms.string('runnumber'),
       interval   = cms.uint64(1),
       firstValue = cms.uint64(XXX_FIRSTRUN_XXX),
       lastValue  = cms.uint64(XXX_LASTRUN_XXX)
   )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:Gains_Sqlite.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripGainFromParticles')
    ))
)

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('Gains_Tree.root')  
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

   process.p = cms.Path(process.EDMtoMEConvert * process.SiStripCalib * process.dqmSaver) 


else:
   process.p = cms.Path(process.SiStripCalib)



#import PhysicsTools.PythonAnalysis.LumiList as LumiList
#process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/13TeV/DCSOnly/json_DCSONLY_Run2015B.txt').getVLuminosityBlockRange()

