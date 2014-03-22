import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
# firsRun defines the IOV used
process.source = cms.Source("EmptySource",
#    firstRun = cms.untracked.uint32(180000),
#    firstRun = cms.untracked.uint32(190000),
#    firstRun = cms.untracked.uint32(196000),
#    firstRun = cms.untracked.uint32(198000),
#    firstRun = cms.untracked.uint32(202000),
#    firstRun = cms.untracked.uint32(204000),
#    firstRun = cms.untracked.uint32(205000),
#    firstRun = cms.untracked.uint32(208686),
    firstRun = cms.untracked.uint32(1),
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

#process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_70_V1::All'
#process.GlobalTag.globaltag = 'FT_R_53_V18::All'
#process.GlobalTag.globaltag = 'GR_P_V43::All'

process.load("Configuration.Geometry.GeometryIdeal_cff")


# read from sqlite file 
## process.QualityReader = cms.ESSource("PoolDBESSource",
## #    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
##     DBParameters = cms.PSet(
##         messageLevel = cms.untracked.int32(0),
##         authenticationPath = cms.untracked.string('')
##     ),
##     toGet = cms.VPSet(
## 		cms.PSet(
## 			record = cms.string("SiPixelLorentzAngleRcd"),
## 			tag = cms.string("trivial_LorentzAngle")
## 		),
## 		cms.PSet(
## 			record = cms.string("SiPixelLorentzAngleSimRcd"),
## 			tag = cms.string("trivial_LorentzAngle_Sim")
## 		)
## 	),
##     connect = cms.string('sqlite_file:SiPixelLorentzAngle.db')
## )
## process.es_prefer_QualityReader = cms.ESPrefer("PoolDBESSource","QualityReader")

process.LorentzAngleReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(False)
)

process.LorentzAngleSimReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
    printDebug = cms.untracked.bool(False),
    useSimRcd = cms.bool(True)
                                             
)


process.p = cms.Path(process.LorentzAngleReader*process.LorentzAngleSimReader)
# process.p = cms.Path(process.LorentzAngleReader)

