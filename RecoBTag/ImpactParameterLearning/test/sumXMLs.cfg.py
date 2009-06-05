import FWCore.ParameterSet.Config as cms

process = cms.Process("sumXMLs")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
)
process.sumCalib = cms.EDFilter("SumHistoCalibration",
	
	 xmlfiles2d = cms.vstring("RecoBTag/ImpactParameterLearning/test/50_80/2d.xml", "RecoBTag/ImpactParameterLearning/test/80_120/2d.xml", "RecoBTag/ImpactParameterLearning/test/120_170/2d.xml", "RecoBTag/ImpactParameterLearning/test/170_230/2d.xml"),
	 xmlfiles3d = cms.vstring("RecoBTag/ImpactParameterLearning/test/50_80/3d.xml", "RecoBTag/ImpactParameterLearning/test/80_120/3d.xml", "RecoBTag/ImpactParameterLearning/test/120_170/3d.xml", "RecoBTag/ImpactParameterLearning/test/170_230/3d.xml"),
	 sum2D = cms.bool(True),
	 sum3D = cms.bool(True),
         writeToDB       = cms.bool(True),
         writeToRootXML  = cms.bool(False),
         writeToBinary   = cms.bool(False)
)



process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True),
    catalog = cms.untracked.string('file:mycatalog_new.xml'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:btagnew_test_startup.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('BTagTrackProbability2DRcd'),
        tag = cms.string('probBTagPDF2D_tag_mc')
    ), 
        cms.PSet(
            record = cms.string('BTagTrackProbability3DRcd'),
            tag = cms.string('probBTagPDF3D_tag_mc')
        ))
)

process.p = cms.Path(process.sumCalib)
