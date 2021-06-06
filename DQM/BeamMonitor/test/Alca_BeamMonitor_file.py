import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
         #"file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/07CBA04E-BEA0-AE4B-A9F1-3C6DF122B40E.root",
         #"file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0869AF69-CD0D-6F45-B1C7-FC1BF3AC7A01.root",
                    "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0874B9E6-DAA6-3148-8A8A-4A323D682591.root",
#                                                                   "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0E38963B-7E4A-9447-A56F-7E87903E2ED4.root",
                    "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0E497339-C393-ED4A-8FA6-E372183A841F.root",
                    "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0F68686E-FE48-7F42-AF13-6F9F058D7BB6.root",
        "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0874B9E6-DAA6-3148-8A8A-4A323D682591.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0E38963B-7E4A-9447-A56F-7E87903E2ED4.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0E497339-C393-ED4A-8FA6-E372183A841F.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/0F68686E-FE48-7F42-AF13-6F9F058D7BB6.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/11F4A099-8675-D248-B648-B39C866557DD.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/120196ED-8F8F-A044-8037-3B7416D9B273.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/1386E376-340A-964E-ADFC-F30D662B4DD8.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/159017AF-F91B-FC48-88FC-0BEF7C3C36F1.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/15A187C2-B285-B54D-9BDF-DB34CCCA9480.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/1789D57E-0D78-A342-BCDD-F3DC1193AF77.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/17F827A5-F84D-6D40-AB10-A1B11EA74CAC.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/1985F374-DBC9-5F4F-A52D-93FBB031FA5F.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/19DE7ADA-C714-9141-946B-311861E05DCE.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/1C7884AD-B58B-F441-959A-3EFC70EBECB0.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/1D4C6C74-E523-E84B-916F-5E9509837053.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/213103A9-2731-F740-B718-189C9D94750F.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/23A6C766-D39F-5F44-B017-D0CD2B42C398.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/26005441-0069-B24E-B040-91F00D129557.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/28CF7DC0-A45E-0248-938F-764FC31853B1.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/29B394F7-6729-3E44-9846-EBB63B2C4B88.root",
                                                                     "file:/eos/cms/store/data/Run2018D/JetHT/RAW-RECO/JetHTJetPlusHOFilter-12Nov2019_UL2018_rsb-v1/120000/2B9B72F2-06F4-A043-9BDE-5B9C6C454705.root",
                                                                                                                         
    
    )
    #    , duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamMonitor = cms.untracked.PSet(
        reportEvery = cms.untracked.int32(1), # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)


process.load("DQM.BeamMonitor.AlcaBeamMonitor_cff")


process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run2_data', '')
# you may need to set manually the GT in the line below
#process.GlobalTag.globaltag = '100X_upgrade2018_realistic_v10'


# DQM Live Environment
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'
process.dqmSaver.tag           = 'BeamMonitor'

process.dqmEnvPixelLess = process.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'



#import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
#process.offlineBeamSpotForDQM = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

# # summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32 (4),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(2)

    )

process.pp = cms.Path(process.alcaBeamMonitor+process.dqmSaver)
process.schedule = cms.Schedule(process.pp)
