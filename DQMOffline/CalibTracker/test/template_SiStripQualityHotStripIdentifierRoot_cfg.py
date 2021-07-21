import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.load("DQMServices.Core.DQM_cfg")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(insertRun),
    lastValue = cms.uint64(insertRun),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
#process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRAFT_V3P::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT0831X_V1::All"

#to read offline info from the first step of the analysis
process.a = cms.ESSource("PoolDBESSource",
    appendToDataLabel = cms.string('testa'),
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
        tag = cms.string('SiStripHotAPVs')
    )),
    connect = cms.string('sqlite_file:dbfile.db')
)

#to read information of o2o and cabling
#process.b = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#    DBParameters = cms.PSet(
#        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
#    ),
#    timetype = cms.string('runnumber'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('SiStripFedCablingRcd'),
#        tag = cms.string('SiStripFedCabling_GR_21X_v2_hlt')
#    ), 
#        cms.PSet(
#            record = cms.string('SiStripBadChannelRcd'),
#            tag = cms.string('SiStripBadChannel_GR_21X_v2_hlt')
#        )),
#    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_21X_STRIP')
#)
#process.sistripconn = cms.ESProducer("SiStripConnectivity")
####
#to produce ESetup based on o2o and cabling and offline
process.MySSQ = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    PrintDebugOutput = cms.bool(False),
    UseEmptyRunInfo = cms.bool(False),
    appendToDataLabel = cms.string('test'),
    ReduceGranularity = cms.bool(True),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
       record = cms.string('SiStripBadFiberRcd'),
       tag = cms.string('testa')
    ))
)
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

#process.MySSQPrefer = cms.ESPrefer("PoolDBESSource","a")

#        cms.PSet(
#            record = cms.string('SiStripBadChannelRcd'),
#            tag = cms.string('')
#        ), 
#        cms.PSet(
#            record = cms.string('SiStripDetCablingRcd'),
#            tag = cms.string('')
#        ))
#)
###
#from CalibTracker.SiStripQuality.siStripQualityStatistics_cfi import siStripQualityStatistics
#process.reader = siStripQualityStatistics.clone(
#        #TkMapFileName = cms.untracked.string('TkMaps/TkMapBadComponents_offline.png'),
#        )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('SiStripBadStrip'),
      tag = cms.string('SiStripHotStrips')
    ))
)

process.prod = cms.EDFilter("SiStripQualityHotStripIdentifierRoot",
    OccupancyRootFile = cms.untracked.string('HotStripsOccupancy_insertRun.root'),
    WriteOccupancyRootFile = cms.untracked.bool(True), # Ouput File has a size of ~100MB. To suppress writing set parameter to 'False'
    UseInputDB = cms.untracked.bool(True), 
    dataLabel=cms.untracked.string('test'),
    OccupancyH_Xmax = cms.untracked.double(1.0),
    AlgoParameters = cms.PSet(
        AlgoName = cms.string('SiStripHotStripAlgorithmFromClusterOccupancy'),
        OccupancyHisto = cms.untracked.string('ClusterDigiPosition__det__'),
        NumberOfEvents = cms.untracked.uint32(0),
        ProbabilityThreshold = cms.untracked.double(1e-07),
        MinNumEntriesPerStrip = cms.untracked.uint32(20),
        MinNumEntries = cms.untracked.uint32(0),
        OccupancyThreshold = cms.untracked.double(0.0001),
        UseInputDB = cms.untracked.bool(True)
    ),
    SinceAppendMode = cms.bool(True),
    verbosity = cms.untracked.uint32(0),
    OccupancyH_Xmin = cms.untracked.double(-0.0005),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStrip'),
    rootDirPath = cms.untracked.string('Run insertRun/AlCaReco'),
    rootFilename = cms.untracked.string('insertCastorPath/insertDataset/insertDQMFile'),
    doStoreOnDB = cms.bool(True),
    OccupancyH_Nbin = cms.untracked.uint32(1001)
)

process.out = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.out)

