import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorClusterProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("CalibTracker.SiPixelESProducers.SiPixelFakeGainOfflineESSource_cfi")

#process.load("CondTools.SiPixel.SiPixelGainCalibrationService_cfi")

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")

process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect ="sqlite_file:/afs/cern.ch/user/m/malgeri/public/globtag/CRUZET3_V7.db"
#process.GlobalTag.globaltag = "CRUZET3_V7::All"
#process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
process.GlobalTag.globaltag = 'FIRSTCOLL::All'

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(-1)
    #input = cms.untracked.int32(5)
    input = cms.untracked.int32(18)
)
process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
   # fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/relval/2008/6/12/RelVal-RelValSingleMuPt10-ChainTest-02/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre5-RelVal-ChainTest-02-IDEAL_V1-unmerged/0000/6E294919-F83D-DD11-BBF0-000423D6B358.root')
    #fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/w/wehrlilu/DQM/CMSSW_3_3_3_DQM/src/BSCskim_123065_CAF.root')
    #BSC AND TRACKS
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/g/gpetrucc/scratch0/tracking-perf/tobonly/CMSSW_3_3_4/src/bit40-123151.root')
    #ALL TRACKS
    #fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/data/recotracks_123151.root')
    #MC
    #fileNames = cms.untracked.vstring('/store/mc/Summer09/MinBias/GEN-SIM-RECO/STARTUP3X_V8D_900GeV-v1/0002/FC5F59F1-05D7-DE11-8AA4-002618943874.root')
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)

#process.p1 = cms.Path(process.siPixelClusters*process.SiPixelClusterSource)
process.p1 = cms.Path(process.SiPixelClusterSource)
process.SiPixelClusterSource.saveFile = True
process.SiPixelClusterSource.isPIB = False
process.SiPixelClusterSource.slowDown = False
process.SiPixelClusterSource.modOn = True
process.SiPixelClusterSource.twoDimOn = True
process.SiPixelClusterSource.ladOn = True
process.SiPixelClusterSource.layOn = True
process.SiPixelClusterSource.phiOn = True
process.SiPixelClusterSource.ringOn = True
process.SiPixelClusterSource.bladeOn = True
process.SiPixelClusterSource.diskOn = True
process.SiPixelClusterSource.smileyOn = False
process.DQM.collectorHost = ''

