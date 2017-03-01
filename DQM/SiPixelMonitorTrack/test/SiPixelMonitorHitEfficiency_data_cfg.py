import FWCore.ParameterSet.Config as cms 

process = cms.Process("SiPixelMonitorHitEfficiency") 

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

###
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_31X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT09_R_V5::All"
process.prefer("GlobalTag")
###

# process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.siPixelDigis.InputLabel = 'source'
process.siPixelDigis.IncludeErrors = True

process.load("EventFilter.SiStripRawToDigi.SiStripRawToDigis_standard_cff")
process.siStripDigis.ProductLabel = 'source'

# process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")

process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
  
process.load("DQM.SiPixelMonitorRawData.SiPixelMonitorRawData_cfi")
process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
process.load("DQM.SiPixelMonitorCluster.SiPixelMonitorCluster_cfi")
process.load("DQM.SiPixelMonitorRecHit.SiPixelMonitorRecHit_cfi")
process.load("DQM.SiPixelMonitorTrack.SiPixelMonitorEfficiency_cfi")

#process.SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('newTrackCandidateMaker')
process.SiPixelHitEfficiencySource.trajectoryInput = cms.InputTag('TrackRefitterP5')
process.SiPixelHitEfficiencySource.debug = cms.untracked.bool(False)


#process.load("IORawData.SiPixelInputSources.PixelSLinkDataInputSource_cfi")
#process.PixelSLinkDataInputSource.fileNames = ['file:/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/PixelAlive_070106d.dmp']

process.source = cms.Source("PoolSource",
  #skipEvents = cms.untracked.uint32(7100), 
  fileNames = cms.untracked.vstring(

    ##RECO, superpointing
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/F0844CE5-49BB-DE11-9317-001A92810ACA.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/EACB27D8-49BB-DE11-BD19-003048678FE0.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/DC3B05D0-49BB-DE11-8323-001A928116E8.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/C8C779D4-49BB-DE11-8B57-0017319C97E0.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/C0712AE1-49BB-DE11-A299-001731AF685D.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/BE473CDD-49BB-DE11-8399-0018F3D09626.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/BA2C63DF-49BB-DE11-A5A6-001A928116F0.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/B46635D4-49BB-DE11-B5DB-00261894393A.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/B28BA2D0-49BB-DE11-8C54-0018F3D096BC.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/ACFF56DB-49BB-DE11-9AA6-001A92810ADE.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/AA7016D0-49BB-DE11-9C55-002354EF3BDF.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/A60A93DB-49BB-DE11-B52F-0018F3D096DA.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/A4461FD9-49BB-DE11-8D2C-002618943962.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/9ED7E9CE-49BB-DE11-977B-003048678FE4.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/9880E9DB-49BB-DE11-909F-001A92971BD6.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/90ECE3DC-49BB-DE11-804B-001A928116C8.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/88A1AADC-49BB-DE11-AA2D-0018F3D09624.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/80EAADD9-49BB-DE11-82B0-002618943962.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/6AD907DE-49BB-DE11-A488-001A92971B28.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/6A1230DC-49BB-DE11-9F7D-0018F3D09710.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/66A81BD7-49BB-DE11-929E-0018F3D09708.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/649B49DB-49BB-DE11-8F53-002354EF3BE0.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/5AE168DA-49BB-DE11-9980-002618943868.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/5A335ECF-49BB-DE11-970A-002618943950.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/484514DD-49BB-DE11-8D43-0018F3D09648.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/464197DD-49BB-DE11-A06C-0018F3D09670.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/442838DA-49BB-DE11-8D17-0026189438DD.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/3611A2D1-49BB-DE11-A5BF-003048679006.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/2C9F47D5-49BB-DE11-A047-003048678EE2.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0013/28AE71DC-49BB-DE11-8B83-0018F3D09624.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0003/183A7B78-CBB8-DE11-919F-003048678FF2.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0003/103C5003-AEB8-DE11-94D0-001A928116F0.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0003/0EBA6D6C-ABB8-DE11-8FCF-001731AF6A89.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0002/FED02BB9-A3B8-DE11-82CD-001731AF65F1.root",
"rfio:/castor/cern.ch/cms/store/data/CRAFT09/Cosmics/RAW-RECO/SuperPointing-CRAFT09_R_V4_CosmicsSeq_v1/0002/FCCA8A2D-A1B8-DE11-B07C-00304867BF9A.root"
    
    ),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10)
) 
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)
process.DQMStore = cms.Service("DQMStore",
  referenceFileName = cms.untracked.string(''),
  verbose = cms.untracked.int32(0)
)
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.AdaptorConfig = cms.Service("AdaptorConfig") 


##
## Load and Configure TrackRefitter
##
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitterP5.src = 'ctfWithMaterialTracksP5'
process.TrackRefitterP5.TrajectoryInEvent = True


process.siPixelLocalReco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siPixelRecHits) 
process.siStripLocalReco = cms.Sequence(process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
process.trackerLocalReco = cms.Sequence(process.siPixelLocalReco*process.siStripLocalReco)
#process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks) #*process.rstracks 
process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)

process.monitorTrack = cms.Sequence(process.SiPixelHitEfficiencySource)
process.monitors = cms.Sequence(process.SiPixelRawDataErrorSource*process.SiPixelDigiSource*process.SiPixelClusterSource*process.SiPixelRecHitSource*process.SiPixelHitEfficiencySource)

#event content analyzer
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.pathTrack = cms.Path(process.trackReconstruction*process.TrackRefitterP5*process.monitorTrack)
