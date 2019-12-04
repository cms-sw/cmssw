import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/FAF8A711-C297-DE11-A00E-001731AF66AF.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/FAB8A245-D996-DE11-A866-003048678A6A.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F8D6CB8E-CD96-DE11-A8D2-003048678FFA.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F8349AAD-C297-DE11-99E3-0030486792BA.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F8251242-DA96-DE11-B26B-003048678FC6.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F49C816C-D796-DE11-AC34-003048D15DDA.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F288C73C-C297-DE11-9F00-001731AF684D.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/F0BBFDE1-D496-DE11-8E1B-003048678FE6.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/EE1592A7-C297-DE11-8679-0030486792AC.root",
    "/store/data/CRAFT09/Cosmics/RAW-RECO/GR09_31X_V5P_CSCSkim_BFieldStudies-332_v4/0021/E2938851-CF96-DE11-8330-0017312B5651.root",
    ))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.StandAloneTest = cms.EDAnalyzer("StandAloneTest", Tracks = cms.InputTag(""))

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

### for cosmic rays (only use one)
process.load("TrackingTools.TrackRefitter.globalCosmicMuonTrajectories_cff")
process.TrackRefitter = process.globalCosmicMuons.clone()
process.TrackRefitter.Tracks = cms.InputTag("globalCosmicMuons")
process.StandAloneTest.Tracks = cms.InputTag("globalCosmicMuons")

### for collisions (only use one)
# process.load("TrackingTools.TrackRefitter.globalMuonTrajectories_cff")
# process.TrackRefitter = process.globalCosmicMuons.clone()
# process.TrackRefitter.Tracks = cms.InputTag("globalMuons")
# process.StandAloneTest.Tracks = cms.InputTag("globalMuons")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string("CRAFT0831X_V1::All")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

### for assigning a custom muon alignment
# process.MuonAlignment = cms.ESSource("PoolDBESSource",
#                                      process.CondDBSetup,
#                                      connect = cms.string("sqlite_file:customMuonAlignment.db"),
#                                      toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
#                                                        cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd"))))
# process.es_prefer_MuonAlignment = cms.ESPrefer("PoolDBESSource", "MuonAlignment")

### it is important to refit with zero weights ("infinite" APEs)
process.MuonAlignmentErrorsExtended = cms.ESSource("PoolDBESSource",
                                     process.CondDBSetup,
                                     connect = cms.string("sqlite_file:APE1000cm.db"),
                                     toGet = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                       cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))
process.es_prefer_MuonAlignmentErrorsExtended = cms.ESPrefer("PoolDBESSource", "MuonAlignmentErrorsExtended")

process.TFileService = cms.Service("TFileService", fileName = cms.string("standAloneTest.root"))
process.Path = cms.Path(process.TrackRefitter * process.StandAloneTest)
