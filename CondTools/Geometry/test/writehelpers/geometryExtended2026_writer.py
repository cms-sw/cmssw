import FWCore.ParameterSet.Config as cms

process = cms.Process("GeometryWriter")

process.load('CondCore.CondDB.CondDB_cfi')
#
# FIXME: the command "./createExtended2026Payloads.sh 113YV12" (i.e 113YV12 is a tag just for test) creates a problem related to:  
# 1) Tracker, if Configuration.Geometry.GeometryExtended2026D49_cff is used (Scenario2026D49 has to be set in DD4hep_GeometrySimPhase2_cff)  
# 2) GEM, if Configuration.Geometry.GeometryExtended2026D77_cff is used (Scenario2026D77 has to be set in DD4hep_GeometrySimPhase2_cff)  
# Please add the right Scenario (D49 or D77 or ..) also in geometryExtended2026_xmlwriter.py and in splitExtended2026Database.sh 
#
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('Configuration.StandardSequences.DD4hep_GeometrySimPhase2_cff')
process.load('Geometry.CaloEventSetup.CaloGeometry2026DBWriter_cfi')
process.load('CondTools.Geometry.HcalParametersWriter_cff')

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.XMLGeometryWriter = cms.EDAnalyzer("XMLGeometryBuilder",
                                           XMLFileName = cms.untracked.string("./geD49SingleBigFile.xml"),
                                           ZIP = cms.untracked.bool(True)
                                           )

process.TrackerGeometryWriter = cms.EDAnalyzer("PGeometricDetBuilder",fromDD4hep=cms.bool(False))
process.TrackerParametersWriter = cms.EDAnalyzer("PTrackerParametersDBBuilder",fromDD4hep=cms.bool(False))
process.TrackerAdditionalParametersPerDetWriter = cms.EDAnalyzer("PTrackerAdditionalParametersPerDetDBBuilder")

process.CaloGeometryWriter = cms.EDAnalyzer("PCaloGeometryBuilder",
                                            EcalE = cms.untracked.bool(False),
                                            EcalP = cms.untracked.bool(False),
                                            HGCal = cms.untracked.bool(False))

process.CSCGeometryWriter = cms.EDAnalyzer("CSCRecoIdealDBLoader",fromDD4Hep = cms.untracked.bool(False))

process.DTGeometryWriter = cms.EDAnalyzer("DTRecoIdealDBLoader",fromDD4Hep = cms.untracked.bool(False))

process.RPCGeometryWriter = cms.EDAnalyzer("RPCRecoIdealDBLoader",fromDD4Hep = cms.untracked.bool(False))

process.GEMGeometryWriter = cms.EDAnalyzer("GEMRecoIdealDBLoader")

process.ME0GeometryWriter = cms.EDAnalyzer("ME0RecoIdealDBLoader")

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'), tag = cms.string('XMLFILE_Geometry_TagXX_Extended2026D41_mc')),
                                                            cms.PSet(record = cms.string('IdealGeometryRecord'), tag = cms.string('TKRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PTrackerParametersRcd'), tag = cms.string('TKParameters_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PTrackerAdditionalParametersPerDetRcd'), tag = cms.string('TKAdditionalParametersPerDet_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('HcalParametersRcd'), tag = cms.string('HCALParameters_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PHGcalRcd'),         tag = cms.string('HGCALRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('HGcalParametersRcd'), tag = cms.string('HGCALParameters_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('CSCRecoGeometryRcd'), tag = cms.string('CSCRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'), tag = cms.string('CSCRECODIGI_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('DTRecoGeometryRcd'), tag = cms.string('DTRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('RPCRecoGeometryRcd'), tag = cms.string('RPCRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('GEMRecoGeometryRcd'), tag = cms.string('GEMRECO_Geometry_TagXX')),
                                                            cms.PSet(record = cms.string('ME0RecoGeometryRcd'), tag = cms.string('ME0RECO_Geometry_TagXX'))
                                                            )
                                          )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.p1 = cms.Path(process.XMLGeometryWriter+process.TrackerGeometryWriter+process.TrackerParametersWriter+process.TrackerAdditionalParametersPerDetWriter+process.CaloGeometryWriter+process.HcalParametersWriter+process.CSCGeometryWriter+process.DTGeometryWriter+process.RPCGeometryWriter+process.GEMGeometryWriter+process.ME0GeometryWriter)
