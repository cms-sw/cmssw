import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_50_V3::All"

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'rawDataCollector'
process.siPixelDigis.IncludeErrors = True
process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
                'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root'
	)
)

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
process.SiPixelDigiSource.outputFile = '/tmp/merkelp/Pixel_DQM_Digi.root'
#process.SiPixelDigiSource.modOn = True
#process.SiPixelDigiSource.twoDimOn = True
process.SiPixelDigiSource.hiRes = True
#process.SiPixelDigiSource.ladOn = False
#process.SiPixelDigiSource.layOn = True
#process.SiPixelDigiSource.phiOn = False
#process.SiPixelDigiSource.ringOn = False
#process.SiPixelDigiSource.bladeOn = False
#process.SiPixelDigiSource.diskOn = True
process.SiPixelDigiSource.reducedSet = False
process.SiPixelDigiSource.twoDimModOn = True 
process.SiPixelDigiSource.twoDimOnlyLayDisk = False 
process.DQM.collectorHost = ''

