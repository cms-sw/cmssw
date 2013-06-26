import FWCore.ParameterSet.Config as cms



alcaSiStripQualityHarvester = cms.EDAnalyzer("SiStripQualityHotStripIdentifierRoot",
    OccupancyRootFile = cms.untracked.string(''),
    WriteOccupancyRootFile = cms.untracked.bool(False), # Ouput File has a size of ~100MB. To suppress writing set parameter to 'False'
    DQMHistoOutputFile = cms.untracked.string(''),
    WriteDQMHistoOutputFile = cms.untracked.bool(False),
    UseInputDB = cms.untracked.bool(True),
    dataLabel=cms.untracked.string('OnlineMasking'),
    OccupancyH_Xmax = cms.untracked.double(1.0),
    CalibrationThreshold = cms.untracked.uint32(10000), #FIXME: should be 10k
    AlgoParameters = cms.PSet(
        AlgoName = cms.string('SiStripBadAPVandHotStripAlgorithmFromClusterOccupancy'),
        OccupancyHisto = cms.untracked.string('ClusterDigiPosition__det__'),
        LowOccupancyThreshold  = cms.untracked.double(5),
        HighOccupancyThreshold = cms.untracked.double(5),
        AbsoluteLowThreshold   = cms.untracked.double(10),
        NumberIterations = cms.untracked.uint32(3),
        OccupancyThreshold = cms.untracked.double(0.002), #0.0001
        NumberOfEvents = cms.untracked.uint32(0),
        ProbabilityThreshold = cms.untracked.double(1e-07),
        MinNumEntriesPerStrip = cms.untracked.uint32(0),
        MinNumEntries = cms.untracked.uint32(0),
        UseInputDB = cms.untracked.bool(True)
    ),
    SinceAppendMode = cms.bool(True),
    verbosity = cms.untracked.uint32(0),
    OccupancyH_Xmin = cms.untracked.double(-0.0005),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStripRcd'),
    rootDirPath = cms.untracked.string('AlCaReco'),
    rootFilename = cms.untracked.string(''),
    doStoreOnDB = cms.bool(True),
    OccupancyH_Nbin = cms.untracked.uint32(1001),
    TimeFromEndRun = cms.untracked.bool(True)
)


#to produce ESetup based on o2o, cabling and RunInfo
onlineSiStripQualityProducer = cms.ESProducer("SiStripQualityESProducer",
    PrintDebug = cms.untracked.bool(True),
    PrintDebugOutput = cms.bool(False),
    UseEmptyRunInfo = cms.bool(False),
    appendToDataLabel = cms.string('OnlineMasking'),
    ReduceGranularity = cms.bool(True),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(
    cms.PSet(
       record = cms.string('SiStripBadChannelRcd'),
       tag = cms.string('')
    ),
    cms.PSet(
       record = cms.string('SiStripDetCablingRcd'),
       tag = cms.string('')
    ),
    cms.PSet(
       record = cms.string('RunInfoRcd'),
       tag = cms.string('')
   )
    )
)
