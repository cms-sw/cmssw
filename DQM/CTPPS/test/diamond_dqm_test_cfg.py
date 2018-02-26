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
    # replace 'myfile.root',',' with the source file you want to use
    fileNames = cms.untracked.vstring(
    *(
           '/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/02448DBC-0A77-E711-8240-02163E01A2ED.root',
#'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/106FE4C4-0D77-E711-8785-02163E01373C.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/469D8C89-1477-E711-A6A4-02163E01190C.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/4C729FDF-0C77-E711-A078-02163E014548.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/664E5FE8-0C77-E711-917D-02163E01469B.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/6A5C61DE-0C77-E711-A08C-02163E013720.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/6A76D2EC-0E77-E711-8903-02163E01A7A5.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/82937E4A-0C77-E711-8725-02163E01421E.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/8A8595B1-1177-E711-B1E8-02163E014761.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/A2BDBF69-1677-E711-83E2-02163E01A4ED.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/A4F9404C-0B77-E711-ADB0-02163E0134D7.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/A615970E-1177-E711-9AA8-02163E0139D9.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/ACBCD4BB-0F77-E711-9AB2-02163E014342.root',
'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/C2D92F59-0C77-E711-B3AD-02163E01A5B4.root',
#'/store/data/Run2017C/ZeroBias/AOD/PromptReco-v2/000/300/088/00000/E041BD4B-0B77-E711-A90A-02163E012140.root',

#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/1E2FB550-5190-E711-8010-02163E01A731.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/ECE07B13-5C90-E711-AA7C-02163E011DDC.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/AABBE824-5990-E711-BE37-02163E0144E2.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/562C1C92-5A90-E711-904D-02163E01A2C7.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/D23D5509-6390-E711-B82E-02163E014785.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/88377025-5990-E711-8E26-02163E01A4E6.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/FE7699DC-5990-E711-8DF5-02163E011D9E.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/600BB493-5A90-E711-84EE-02163E01A1EB.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/AA6E5A99-6090-E711-B9AA-02163E012B3A.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/F45AD155-5B90-E711-8ACE-02163E019C49.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/909B3BB3-5C90-E711-8649-02163E01A1EB.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/ACC8407D-5890-E711-A88A-02163E0134FB.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/A4DF7114-5E90-E711-9BD7-02163E011D9E.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/9EF2EBDC-5990-E711-BC87-02163E0138EC.root',
#'/store/data/Run2017D/ZeroBias/AOD/PromptReco-v1/000/302/159/00000/DE7748D6-5C90-E711-A008-02163E011B3C.root',

     )
    ),
    #inputCommands = cms.untracked.vstring(
       #'drop CTPPSPixelCluseredmDetSetVector_ctppsPixelClusters__RECO'
    #)
)


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
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
