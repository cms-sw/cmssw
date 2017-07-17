import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorRecHitsProcess")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRUZET3_V7.db"
#oldfprocess.GlobalTag.globaltag = "CRUZET3_V7::All"
process.GlobalTag.globaltag = "IDEAL_30X::All"
process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(
    #'/store/relval/2008/6/6/RelVal-RelValTTbar-1212531852-IDEAL_V1-2nd-02/0000/081018D5-EC33-DD11-A623-000423D6CA42.root')
    '/store/relval/CMSSW_3_1_0_pre3/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0001/3C8AABDF-FA0A-DE11-80A5-001D09F290BF.root')


)

process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.p1 = cms.Path(process.siPixelRecHits*process.SiPixelRecHitSource)
process.SiPixelRecHitSource.saveFile = True
process.SiPixelRecHitSource.isPIB = False
process.SiPixelRecHitSource.slowDown = False
process.SiPixelRecHitSource.modOn = True
process.SiPixelRecHitSource.twoDimOn = True
process.SiPixelRecHitSource.ladOn = True
process.SiPixelRecHitSource.layOn = True
process.SiPixelRecHitSource.phiOn = True
process.SiPixelRecHitSource.bladeOn = True
process.SiPixelRecHitSource.ringOn = True
process.SiPixelRecHitSource.diskOn = True
process.DQM.collectorHost = ''

