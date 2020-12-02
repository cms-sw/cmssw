import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelCablingReader")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_design']

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    )
)


###### OUTPUT HISTOGRAM FILE NAME #######
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(10),
    firstRun = cms.untracked.uint32(1)
)

##### DATABASE CONNECTION INFO ######
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:SiPixelFedCablingMap_v21.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel = 1


###### TAGS TO READ ######
process.PoolDBESSourceForReader = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('SiPixelFedCablingMapRcd'),
            tag = cms.string('SiPixelFedCablingMap_v21')
       ))
)

###### PREFER ABOVE TAGS #######
process.esprefer_DBReaders = cms.ESPrefer("PoolDBESSource", "PoolDBESSourceForReader")

####### CABLING MAP READER ######
process.SiPixelFedCablingMapAnalyzer = cms.EDAnalyzer("SiPixelFedCablingMapAnalyzer")

process.p = cms.Path(
process.SiPixelFedCablingMapAnalyzer
)


