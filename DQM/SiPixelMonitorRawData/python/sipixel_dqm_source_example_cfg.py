import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorRawDataProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")

process.load("CondTools.SiPixel.SiPixelCalibConfiguration_cfi")

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("IORawData.SiPixelInputSources.PixelSLinkDataInputSource_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRUZET3_V7.db"
process.GlobalTag.globaltag = "CRUZET3_V7::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/TAC/PIXEL/P5/PixelAlive_34_53901.root'),
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('debugmessages.txt')
)

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelRawDataErrorSource)
process.sipixelcalib_essource.toGet = cms.VPSet(cms.PSet(
                                                         record = cms.string('SiPixelCalibConfigurationRcd'),
				                         tag = cms.string('GainCalibration_2445')
				                        ),
					        cms.PSet(
                                                         record = cms.string('SiPixelFedCablingMapRcd'),
				                         tag = cms.string('SiPixelFedCablingMap_v10')
				                        )
						)
process.siPixelDigis.InputLabel = cms.untracked.string('source')
process.siPixelDigis.IncludeErrors = True
process.SiPixelRawDataErrorSource.saveFile = True
process.SiPixelRawDataErrorSource.isPIB = False
process.SiPixelRawDataErrorSource.slowDown = False
process.DQM.collectorHost = ''

