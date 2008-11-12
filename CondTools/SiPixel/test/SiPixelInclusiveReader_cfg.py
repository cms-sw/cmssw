import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelInclusiveReader")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("histo.root")
                                   )


process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_PIXEL_COMM_21X'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.CondDBCommon.DBParameters.messageLevel = 1

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(10),
    firstRun = cms.untracked.uint32(1)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(0)
)

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPixelFedCablingMapRcd'),
        tag = cms.string('SiPixelFedCablingMap_v14')
    ), 
        cms.PSet(
            record = cms.string('SiPixelLorentzAngleRcd'),
            tag = cms.string('SiPixelLorentzAngle_v01')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationOfflineRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_const')
        ), 
        cms.PSet(
            record = cms.string('SiPixelGainCalibrationForHLTRcd'),
            tag = cms.string('SiPixelGainCalibration_TBuffer_hlt_const')
        ))
)

process.prefer("PoolDBESSource")
process.SiPixelCondObjOfflineReader = cms.EDFilter("SiPixelCondObjOfflineReader",
    process.SiPixelGainCalibrationServiceParameters
)

process.SiPixelCondObjForHLTReader = cms.EDFilter("SiPixelCondObjForHLTReader",
    process.SiPixelGainCalibrationServiceParameters
)

process.SiPixelLorentzAngleReader = cms.EDFilter("SiPixelLorentzAngleReader")

process.SiPixelFedCablingMapAnalyzer = cms.EDAnalyzer("SiPixelFedCablingMapAnalyzer")

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.SiPixelCondObjOfflineReader*process.SiPixelLorentzAngleReader*process.SiPixelFedCablingMapAnalyzer*process.SiPixelCondObjForHLTReader)
#process.ep = cms.EndPath(process.print)


