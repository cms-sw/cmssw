import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorRecHitsProcess")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRUZET3_V7.db"
process.GlobalTag.globaltag = "CRUZET3_V7::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

process.p1 = cms.Path(process.siPixelRecHits*process.SiPixelRecHitSource)
process.SiPixelRecHitSource.saveFile = True
process.SiPixelRecHitSource.isPIB = False
process.SiPixelRecHitSource.slowDown = False
process.SiPixelRecHitSource.modOn = True
process.SiPixelRecHitSource.twoDimOn = True
process.SiPixelRecHitSource.ladOn = False
process.SiPixelRecHitSource.layOn = False
process.SiPixelRecHitSource.phiOn = False
process.SiPixelRecHitSource.bladeOn = False
process.SiPixelRecHitSource.ringOn = False
process.SiPixelRecHitSource.diskOn = False
process.DQM.collectorHost = ''

