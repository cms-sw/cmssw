import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing()
options.register("isUnitTest",
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "are we running the unit test")
options.parseArguments()

process = cms.Process("HitEffHarvest")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')  

process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:DQM.root')
)

runNumber = 325172

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

process.hiteffharvest = cms.EDProducer("SiStripHitEfficiencyHarvester",
    Threshold           = cms.double(0.1),
    nModsMin            = cms.int32(5),
    AutoIneffModTagging = cms.untracked.bool(True),  # default true, automatic limit for each layer to identify inefficient modules
    Record              = cms.string('SiStripBadStrip'),
    doStoreOnDB         = cms.bool(True),
    ShowRings           = cms.untracked.bool(False),  # default False
    TkMapMin            = cms.untracked.double(0.90), # default 0.90
    EffPlotMin          = cms.untracked.double(0.90), # default 0.90
    Title               = cms.string(' Hit Efficiency - run {0:d}'.format(runNumber))
    )

process.load("DQM.SiStripCommon.TkHistoMap_cff")

process.allPath = cms.Path(process.hiteffharvest*process.DQMSaver)

if(options.isUnitTest):
    process.MessageLogger.cerr.enable = False
    process.MessageLogger.TkHistoMap = dict()
    process.MessageLogger.SiStripHitEfficiency = dict()  
    process.MessageLogger.SiStripHitEfficiencyHarvester = dict()
    process.MessageLogger.cout = cms.untracked.PSet(
        enable    = cms.untracked.bool(True),        
        threshold = cms.untracked.string("INFO"),
        enableStatistics = cms.untracked.bool(True),
        default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
        FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                       reportEvery = cms.untracked.int32(1)),
        TkHistoMap = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
        SiStripHitEfficiency = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
        SiStripHitEfficiencyHarvester =  cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )
else:
    process.MessageLogger = cms.Service(
        "MessageLogger",
        destinations = cms.untracked.vstring("log_hiteffharvest"),
        log_hiteffharvest = cms.untracked.PSet(
            threshold = cms.untracked.string("DEBUG"),
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            )
        ),
        debugModules = cms.untracked.vstring("hiteffharvest"),
        categories=cms.untracked.vstring("TkHistoMap", 
                                         "SiStripHitEfficiency:HitEff", 
                                         "SiStripHitEfficiency", 
                                         "SiStripHitEfficiencyHarvester")
    )

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('SiStripHitEffHistos_run{0:d}_NEW.root'.format(runNumber))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile_NEW.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripHitEffBadModules')
    ))
)
