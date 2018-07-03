import FWCore.ParameterSet.Config as cms
import string

process = cms.Process('RECODQM')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.verbosity = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
)
    
    # import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

# raw data source
process.source = cms.Source("PoolSource",
    # replace '*.root',',' with the source file you want to use
    fileNames = cms.untracked.vstring(
    *(
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/469D8C89-1477-E711-A6A4-02163E01190C.root',
     )
    ),
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi')

# pixel
process.load('RecoCTPPS.PixelLocal.ctppsPixelLocalTracks_cfi')

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")
process.ctppsDiamondDQMSource.excludeMultipleHits = cms.bool(True);

process.path = cms.Path(
    #process.ctppsRawToDigi *
    process.recoCTPPS *
    #process.ctppsDiamondRawToDigi *
    process.ctppsDiamondRecHits *
    process.ctppsDiamondLocalTracks *
    process.ctppsPixelLocalTracks *
    process.ctppsDQM
    )


process.end_path = cms.EndPath(
    process.dqmEnv +
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
