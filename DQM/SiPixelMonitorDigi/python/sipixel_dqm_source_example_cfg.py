import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "FIRSTCOLL::All"

process.load("Configuration.StandardSequences.RawToDigi_cff")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/gpetrucc/scratch0/tracking-perf/tobonly/CMSSW_3_3_4/src/bit40-123151.root')
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.p1 = cms.Path(process.siPixelDigis*process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
#process.SiPixelDigiSource.isPIB = False
#process.SiPixelDigiSource.slowDown = False
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
process.SiPixelDigiSource.twoDimModOn = False 
process.SiPixelDigiSource.twoDimOnlyLayDisk = True 
process.DQM.collectorHost = ''

