import FWCore.ParameterSet.Config as cms

process = cms.Process("HitEff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

RunNumberBegin = 315252
RunNumberEnd = 315252
#RunNumberBegin = 325310
#RunNumberEnd = 325310

process.source = cms.Source("EmptyIOVSource",
  firstValue = cms.uint64(RunNumberBegin),
  lastValue = cms.uint64(RunNumberEnd),
  timetype = cms.string('runnumber'),
  interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

InputFilePath =  'root://cms-xrd-global.cern.ch///eos/cms/store/group/dpg_tracker_strip/comm_tracker/Strip/Calibration/calibrationtree/GR18/calibTree_'
InputFilePathEnd = '.root'

FileName1 = InputFilePath + str(RunNumberBegin) + InputFilePathEnd
#FileName2 = InputFilePath + str(RunNumberEnd) + InputFilePathEnd

process.SiStripHitEff = cms.EDAnalyzer("SiStripHitResolFromCalibTree",
    CalibTreeFilenames = cms.untracked.vstring(FileName1),
    Threshold         = cms.double(0.2),
    nModsMin          = cms.int32(25),
    doSummary         = cms.int32(0),
    #ResXSig           = cms.untracked.double(5),
    SinceAppendMode   = cms.bool(True),
    IOVMode           = cms.string('Run'),
    Record            = cms.string('SiStripBadStrip'),
    doStoreOnDB       = cms.bool(True),
    BadModulesFile    = cms.untracked.string("BadModules_input.txt"),   # default "" no input
    AutoIneffModTagging = cms.untracked.bool(True),   # default true, automatic limit for each layer to identify inefficient modules
    ClusterMatchingMethod  = cms.untracked.int32(4),     # default 0  case0,1,2,3,4
    ClusterTrajDist   = cms.untracked.double(64),   # default 64
    StripsApvEdge     = cms.untracked.double(10),   # default 10  
    UseOnlyHighPurityTracks = cms.untracked.bool(True), # default True
    SpaceBetweenTrains = cms.untracked.int32(25),   # default 25
    UseCommonMode     = cms.untracked.bool(False),  # default False
    ShowEndcapSides   = cms.untracked.bool(True),  # default True
    ShowRings         = cms.untracked.bool(True),  # default False
    ShowTOB6TEC9      = cms.untracked.bool(False),  # default False
    ShowOnlyGoodModules = cms.untracked.bool(False),  # default False
    TkMapMin          = cms.untracked.double(0.95), # default 0.90
    EffPlotMin        = cms.untracked.double(0.90), # default 0.90
    Title             = cms.string(' Hit Efficiency ')
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripHitEffBadModules')
    ))
)

UnitString = "strip unit"
#UnitString = "cm"

if UnitString == "cm":
	RootFileEnding = "_CM.root"
elif UnitString == "strip unit":
	RootFileEnding = "_StripUnit.root"
else:
 	print('ERROR: Unit must be cm or strip unit')	

RootFileBeginning = "SiStripHitEffHistos_run_"
RootFileName = RootFileBeginning + str(RunNumberBegin) + "_" + str(RunNumberEnd) + RootFileEnding

process.TFileService = cms.Service("TFileService",
        fileName = cms.string(RootFileName)  
)

process.allPath = cms.Path(process.SiStripHitEff)

