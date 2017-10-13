import FWCore.ParameterSet.Config as cms
process = cms.Process("CTPPS")

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECODQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw data source
#process.source = cms.Source("NewEventStreamFileReader",
#    fileNames = cms.untracked.vstring(
#        '/store/t0streamer/Data/Physics/000/286/591/run286591_ls0521_streamPhysics_StorageManager.dat',
#    )
#)
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/001C958C-7FA9-E711-858F-02163E011A5F.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/00BA3B56-80A9-E711-8E7B-02163E01430D.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0219EE0A-C7A9-E711-9994-02163E0145B7.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/02362D0E-C7A9-E711-B61A-02163E014457.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/02412D74-83A9-E711-904C-02163E01A3F1.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/024405B7-74A9-E711-BC80-02163E019D96.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0417FFB5-79A9-E711-AEB4-02163E019D61.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/043E5522-C7A9-E711-96B9-02163E014360.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0653EC19-C7A9-E711-AF8F-02163E011918.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/08204734-6FA9-E711-8320-02163E01A2DD.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0A58360C-82A9-E711-A233-02163E013689.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0AA7DF3B-78A9-E711-A5EE-02163E019E85.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0C14A314-C7A9-E711-95BF-02163E013772.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0C61967B-89A9-E711-90E4-02163E0145A4.root',
'/store/data/Run2017E/ZeroBias/RAW/v1/000/304/447/00000/0C9104CF-8DA9-E711-B9E3-02163E019D0B.root',
#/store/t0streamer/Minidaq/A/000/303/982/run303982_ls0001_streamA_StorageManager.dat',
),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# rechits production
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondRecHits_cfi')

# local tracks fitter
process.load('RecoCTPPS.TotemRPLocal.ctppsDiamondLocalTracks_cfi')

#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.threshold = cms.double(1.5)
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.sigma = cms.double(0)
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.resolution = cms.double(0.025) # in mm
#process.ctppsDiamondLocalTracks.trackingAlgorithmParams.pixel_efficiency_function = cms.string("(TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4")

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:AOD.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_ctpps*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.ctppsDiamondRawToDigi *
    process.ctppsDiamondLocalReconstruction
)

process.outpath = cms.EndPath(process.output) 
