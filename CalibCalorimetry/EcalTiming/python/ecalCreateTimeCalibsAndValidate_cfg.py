import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalCreateTimeCalibrations")

# Global Tag -- for original timing calibrations
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.GlobalTag.globaltag = 'GR_R_44_V5::All'

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
    categories = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('cout')
)

process.TFileService = cms.Service("TFileService", 
    fileName = cms.string("ecalCreateTimeCalibsAndValidate.root"),
    closeFileFast = cms.untracked.bool(True)
    )


process.createTimeCalibs = cms.EDAnalyzer("EcalCreateTimeCalibrations",
  OutputFileName = cms.string("file:converted1.root"),##Not Used
  OutputTimeCalibFileName = cms.string("EcalTimeCalibConstantsExample.xml"),##Name of output xml file with new constants
  OutputTimeOffsetFileName = cms.string("EcalTimeOffsetExample.xml"),##Name of output xml file with new constants
  FileNameStart = cms.string("ecalCreateTimeCalibs"),
  ZeroGlobalOffset = cms.bool(False), ## if true, disables global offset
  SubtractDBcalibs = cms.bool(True),
  BxIncludeExclude = cms.string("-1"),
  OrbitIncludeExclude = cms.string("-1"),
  TriggerBitIncludeExclude = cms.string("-1"),
  TechTriggerBitIncludeExclude = cms.string("-1"),
  LumiIncludeExclude = cms.string("-1"),
  RunIncludeExclude = cms.string("-1"),  
  AvgTimeMin = cms.double(-5),
  AvgTimeMax = cms.double(5),
  MinHitAmpEB = cms.double(26),  
  MinHitAmpEE = cms.double(47), 
  MaxSwissCross = cms.double(0.95),
  MaxHitTimeEB = cms.double(5),
  MinHitTimeEB = cms.double(-5),
  MaxHitTimeEE = cms.double(7),
  MinHitTimeEE = cms.double(-7),
  EventsUsedFractionNum = cms.double(1),
  EventsUsedFractionDen = cms.double(1),
  InputFileNames = cms.vstring("file:/data/jared/data/EcalTiming/DoubleElectron_Run2011A-ZElectron-PromptSkim-v4_RAW-RECO/rh-DoubleElectron_Run2011A-ZElectron-PromptSkim-v4_RAW-RECO-jul15.HADDED.root")
)

process.expressValidator = cms.EDAnalyzer("EcalTimeCalibrationValidator",
  InputFileNames = cms.vstring('file:/data/jared/data/EcalTiming/DoubleElectron_Run2011A-ZElectron-PromptSkim-v4_RAW-RECO/rh-DoubleElectron_Run2011A-ZElectron-PromptSkim-v4_RAW-RECO-jul15.HADDED.root'),
  OutputFileName = cms.string("file:/data/jared/EcalRecalibratedTimingTrees/converted-ZElectron-PromptSkim-v4_RAW-RECO-jul15ClosureTest.root"),
  CalibConstantXMLFileName = cms.string("EcalTimeCalibConstantsExample.xml"),
  CalibOffsetXMLFileName = cms.string("EcalTimeOffsetExample.xml"),
  ZeroGlobalOffset = cms.bool(False),## if true, does not use global shift
  MaxTreeEntriesToProcess = cms.untracked.int32(-1),
  RunIncludeExclude = cms.string("-1")
)

process.recalibratedTimeHists = cms.EDAnalyzer("EcalCreateTimeCalibrations",
  OutputFileName = cms.string("file:converted1.root"),##Not Used
  OutputTimeCalibFileName = cms.string("Junk.xml"),## Important - Don't overwrite previous xml file!
  OutputTimeOffsetFileName = cms.string("JunkOffset.xml"),## Important - Don't overwrite previous xml file!
  FileNameStart = cms.string("ecalCreateTimeCalibsRecalibrated"),
  ZeroGlobalOffset = cms.bool(True),
  SubtractDBcalibs = cms.bool(True),
  BxIncludeExclude = cms.string("-1"),
  OrbitIncludeExclude = cms.string("-1"),
  TriggerBitIncludeExclude = cms.string("-1"),
  TechTriggerBitIncludeExclude = cms.string("-1"),
  LumiIncludeExclude = cms.string("-1"),
  RunIncludeExclude = cms.string("-1"),
  AvgTimeMin = cms.double(-5),
  AvgTimeMax = cms.double(5),
  MinHitAmpEB = cms.double(26),  
  MinHitAmpEE = cms.double(47), 
  MaxSwissCross = cms.double(0.95),
  MaxHitTimeEB = cms.double(5),
  MinHitTimeEB = cms.double(-5),
  MaxHitTimeEE = cms.double(7),
  MinHitTimeEE = cms.double(-7),
  EventsUsedFractionNum = cms.double(1),
  EventsUsedFractionDen = cms.double(1),
  InputFileNames = cms.vstring("file:/data/jared/EcalRecalibratedTimingTrees/converted-ZElectron-PromptSkim-v4_RAW-RECO-jul15ClosureTest.root")
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
process.source = cms.Source("EmptySource",
       numberEventsInRun = cms.untracked.uint32(1),
       firstRun = cms.untracked.uint32(888888)
)
process.p = cms.Path(process.createTimeCalibs*process.expressValidator)
process.pEnd = cms.Path(process.recalibratedTimeHists)
process.schedule = cms.Schedule(process.p)
process.schedule.append(process.pEnd)
