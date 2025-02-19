import FWCore.ParameterSet.Config as cms

process = cms.Process("SynchronizeDCSO2O")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100000)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('timestamp'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripDetVOffRcd'),
        tag = cms.string('SiStripDetVOff_Fake_31X')
    ))
)

# process.load("MinimumBias_BeamCommissioning09_Jan29_ReReco_v2_RECO_cff")

# Select runs 124270 (Wed 16-12-09 02:47:00 + 36:00) 124275(04:00:00 + 01:43:00) 124277(06:39:00 + 20:00)
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124270:1-124270:9999','124275:1-124275:9999','124277:1-124277:9999')

# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring(
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0020/E8593279-0A0E-DF11-A36D-001A9281171E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0020/264B64FE-F10D-DF11-828B-0018F3D09644.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0019/C0E8E7B6-D30D-DF11-B949-001A92971BD8.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0018/FE0947DC-860D-DF11-9EBC-00261894390E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0018/CCD1EAD6-610D-DF11-88D3-001A92971B94.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0018/2EB16CF7-550D-DF11-A627-0018F3D096D2.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/EE20C722-2D0D-DF11-A4E4-0018F3D09678.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/E69E5703-2D0D-DF11-813B-00261894395F.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/DEB0E01F-2D0D-DF11-9DC7-00304867905A.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/CECEF3B8-310D-DF11-9B86-001A92971B5E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/ACCAB7D1-2F0D-DF11-802B-00304867900C.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/90B201B4-2D0D-DF11-AD1A-0018F3D0968E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/6A98ACE3-3D0D-DF11-A506-001A92971B08.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/6408DA20-2D0D-DF11-9FA8-00304867904E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/620B33EF-360D-DF11-A20D-001A92971B5E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/4E17EB0D-3B0D-DF11-A8AE-001A92810ABA.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/368FECBB-2B0D-DF11-B4CB-001A92971AEC.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/30B30B23-2D0D-DF11-8810-001BFCDBD166.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/2C894F21-2D0D-DF11-BFE2-001BFCDBD11E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/08CE8309-3B0D-DF11-B43D-0018F3D09690.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0017/001FFD22-2D0D-DF11-A91B-001BFCDBD19E.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/EA65409E-290D-DF11-BF76-0018F3D096BC.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/5ED2B19A-260D-DF11-9CB9-001BFCDBD1BC.root",
#     "/store/data/BeamCommissioning09/MinimumBias/RECO/Jan29ReReco-v2/0016/0ECF06A7-220D-DF11-98B8-001A92971B7C.root"
#     )
# )

# -------- #
# RAW data #
# -------- #
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/F615B99F-70EA-DE11-A289-001617C3B76E.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/F49F6BF2-6FEA-DE11-AA90-0019B9F730D2.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/EA9C82A1-70EA-DE11-9742-000423D33970.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/D6A379EF-6FEA-DE11-BC0E-001D09F25109.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/C27C3AA0-70EA-DE11-9CC7-001D09F24E39.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/AC221F16-6DEA-DE11-81A3-0019B9F705A3.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/9C52FBEE-6FEA-DE11-9ACC-001D09F2AF96.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/90D252A0-70EA-DE11-A9A0-001D09F27067.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/88963473-73EA-DE11-A598-003048D2BE08.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/862CE615-72EA-DE11-802E-001D09F25438.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/749FA331-74EA-DE11-AB5B-000423D6C8E6.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/74113B16-6DEA-DE11-999F-001D09F2A49C.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/5CDA95EE-6FEA-DE11-86C3-001D09F24600.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/5C9D00C8-72EA-DE11-81B3-000423D992A4.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/529DB9EE-6FEA-DE11-B2E0-001D09F295A1.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/32A22B18-6DEA-DE11-8059-001D09F28D54.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/22576A58-71EA-DE11-8C38-001D09F292D1.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/20051DA0-70EA-DE11-AAAF-001D09F244DE.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/0ECAC2CC-72EA-DE11-817D-001D09F2924F.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/06D89E1D-6DEA-DE11-80C9-000423D9863C.root",
    "/store/data/BeamCommissioning09/ZeroBiasB/RAW/v1/000/124/275/066A6E1B-6DEA-DE11-BBBC-001D09F23D1D.root"
    )
)

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

process.raw2digi_step = cms.Path(process.RawToDigi)

# process.GlobalTag.globaltag = 'GR09_R_35X_V2::All'
process.GlobalTag.globaltag = 'GR09_R_V6A::All'

process.es_prefer_DetVOff = cms.ESPrefer("PoolDBESSource", "poolDBESSource")

process.syncDCSO2O = cms.EDAnalyzer(
    'SyncDCSO2O',
    DigiProducersList = cms.VPSet(cms.PSet( DigiLabel = cms.string('ZeroSuppressed'), DigiProducer = cms.string('siStripDigis') ), 
                                  cms.PSet( DigiLabel = cms.string('VirginRaw'), DigiProducer = cms.string('siStripZeroSuppression') ), 
                                  cms.PSet( DigiLabel = cms.string('ProcessedRaw'), DigiProducer = cms.string('siStripZeroSuppression') ), 
                                  cms.PSet( DigiLabel = cms.string('ScopeMode'), DigiProducer = cms.string('siStripZeroSuppression') )
    )
)

# process.schedule = cms.Schedule(process.raw2digi_step)

process.p = cms.EndPath(process.siStripDigis+process.syncDCSO2O)
