import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
#    lastRun = cms.untracked.uint32(1),
#    timetype = cms.string('runnumber'),
#    interval = cms.uint32(1),
    firstRun = cms.untracked.uint32(1)
)


process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )


process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "GR_R_70_V1::All"
process.GlobalTag.globaltag = 'FT_R_53_V21::All'

process.load("Configuration.StandardSequences.GeometryIdeal_cff")

process.QualityReader = cms.ESSource("PoolDBESSource",
#    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('')
    ),
    toGet = cms.VPSet(
		cms.PSet(
			record = cms.string("SiPixelLorentzAngleRcd"),
			#tag = cms.string("trivial_LorentzAngle")
			tag = cms.string("SiPixelLorentzAngle_v01")
		),
#		cms.PSet(
#			record = cms.string("SiPixelLorentzAngleSimRcd"),
#			tag = cms.string("trivial_LorentzAngle_Sim")
#		)
	),
    #connect = cms.string('sqlite_file:SiPixelLorentzAngle.db')
    #connect = cms.string('sqlite_file:prova_LA_2012_IOV7.db')
    connect = cms.string('sqlite_file:/afs/cern.ch/user/h/hidaspal/public/tracker/SiPixelLorentzAngle/pixelLorentzAngle_original_private_production_stuff/mytest/prova_LA_2012_IOV7.db')
)

process.es_prefer_QualityReader = cms.ESPrefer("PoolDBESSource","QualityReader")

process.LorentzAngleReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(False)
)

process.LorentzAngleSimReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(True)
                                             
)


#process.p = cms.Path(process.LorentzAngleReader*process.LorentzAngleSimReader)
process.p = cms.Path(process.LorentzAngleReader)


