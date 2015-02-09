import os
import FWCore.ParameterSet.Config as cms

globalTag = os.getenv("ALIGNMENT_GLOBALTAG")
outputFile = os.getenv("ALIGNMENT_OUTPUTFILE")

process = cms.Process("CONVERT")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = globalTag

process.MuonGeometryDBConverter = cms.EDAnalyzer("MuonGeometryDBConverter",
                                                 input = cms.string("db"),
                                                 dtLabel = cms.string(""),
                                                 cscLabel = cms.string(""),
                                                 shiftErr = cms.double(1000.),
                                                 angleErr = cms.double(6.28),
                                                 getAPEs = cms.bool(True),
                                                 
                                                 output = cms.string("db")
                                                 )

process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:%s" % outputFile),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
                                                            cms.PSet(record = cms.string("DTAlignmentErrorExtendedRcd"), tag = cms.string("DTAlignmentErrorExtendedRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")),
                                                            cms.PSet(record = cms.string("CSCAlignmentErrorExtendedRcd"), tag = cms.string("CSCAlignmentErrorExtendedRcd"))))

process.Path = cms.Path(process.MuonGeometryDBConverter)
