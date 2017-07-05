import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

process = cms.Process("SIPIXELDQM")
process.load("IORawData.SiPixelInputSources.PixelSLinkDataInputSource_cfi")

process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("CondTools.SiPixel.SiPixelCalibConfiguration_cfi")

process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")

process.load("CalibTracker.SiPixelGainCalibration.SiPixelCalibDigiProducer_cfi")

process.load("CalibTracker.SiPixelSCurveCalibration.SiPixelSCurveCalibrationAnalysis_cfi")

process.load("CalibTracker.SiPixelIsAliveCalibration.SiPixelIsAliveCalibration_cfi")

process.load("CalibTracker.SiPixelGainCalibration.SiPixelGainCalibrationAnalysis_cfi")

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")

process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.AdaptorConfig = cms.Service("AdaptorConfig")

process.sipixelEDAClient = DQMEDHarvester("SiPixelEDAClient",
    StaticUpdateFrequency = cms.untracked.int32(10),
    OutputFilePath = cms.untracked.string('.'),
)

process.preScaler = cms.EDFilter("Prescaler",
    prescaleFactor = cms.int32(1)
)

process.dqmEnv = cms.EDAnalyzer("DQMEventInfo",
    subSystemFolder = cms.untracked.string('Pixel'),
    eventInfoFolder = cms.untracked.string('EventInfo')
)

process.dqmSaver = cms.EDAnalyzer("DQMFileSaver",
    prescaleEvt = cms.untracked.int32(-1),
    producer = cms.untracked.string('DQM'),
    workflow = cms.untracked.string('/A/B/C'),
    prescaleLS = cms.untracked.int32(-1),
    saveAtJobEnd = cms.untracked.bool(False),
    fileName = cms.untracked.string('Pixel'),
    environment = cms.untracked.string('Online'),
    saveAtRunEnd = cms.untracked.bool(True),
    prescaleTime = cms.untracked.int32(-1),
    dirName = cms.untracked.string('.')
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.DigiReco = cms.Sequence(process.siPixelDigis)
process.CalibAnalysis = cms.Sequence(process.siPixelCalibDigis*process.siPixelSCurveAnalysis*process.siPixelGainCalibrationAnalysis*process.siPixelIsAliveCalibration)
process.RAWmonitor = cms.Sequence(process.SiPixelRawDataErrorSource)
process.DIGImonitor = cms.Sequence(process.SiPixelDigiSource)
process.DQMmodules = cms.Sequence(process.dqmEnv*process.dqmSaver)
process.p = cms.Path(process.DigiReco*process.CalibAnalysis*process.RAWmonitor*process.DIGImonitor*process.sipixelEDAClient*process.DQMmodules)
process.PixelSLinkDataInputSource.fileNames = ['rfio:/castor/cern.ch/cms/store/TAC/PIXEL/FPIX/HC-Z1/SCurve_35_340.dmp']
process.sipixelcalib_essource.toGet = cms.VPSet(cms.PSet(
    record = cms.string('SiPixelCalibConfigurationRcd'),
    tag = cms.string('SCurve_340')
), 
    cms.PSet(
        record = cms.string('SiPixelFedCablingMapRcd'),
        tag = cms.string('SiPixelFedCablingMap_v12')
    ))
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True
process.siPixelIsAliveCalibration.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelSCurveAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelGainCalibrationAnalysis.DetSetVectorSiPixelCalibDigiTag = 'siPixelCalibDigis'
process.siPixelIsAliveCalibration.saveFile = False
process.siPixelGainCalibrationAnalysis.saveFile = False
process.siPixelSCurveAnalysis.saveFile = False
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Online'
process.dqmSaver.producer = 'DQM'
process.dqmEnv.subSystemFolder = 'Pixel'
process.dqmSaver.saveByLumiSection = -1
process.dqmSaver.saveByRun = 1

