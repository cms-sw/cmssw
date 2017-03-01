import FWCore.ParameterSet.Config as cms 

process = cms.Process("SiPixelMonitorTrackResiduals") 

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

###
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRUZET4_V4P::All"
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
process.load("DQM.SiPixelMonitorTrack.SiPixelMonitorTrack_cfi")

process.SiPixelTrackResidualSource.TrackCandidateProducer = cms.string('newTrackCandidateMaker')
process.SiPixelTrackResidualSource.trajectoryInput = cms.InputTag('TrackRefitterP5')


#process.load("IORawData.SiPixelInputSources.PixelSLinkDataInputSource_cfi")
#process.PixelSLinkDataInputSource.fileNames = ['file:/afs/cern.ch/cms/Tracker/Pixel/forward/ryd/PixelAlive_070106d.dmp']

process.source = cms.Source("PoolSource",
  #skipEvents = cms.untracked.uint32(7100), 
  fileNames = cms.untracked.vstring(

    ##RECO, superpointing

    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/34190739-739D-DD11-8E14-0019B9E4AD9A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/34C7E8B3-CC9D-DD11-87F1-001D0967DE90.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3664D2B2-CC9D-DD11-9DEA-001D0967D571.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3803B395-CC9D-DD11-BD79-001D0967DA76.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3AECD7B2-CC9D-DD11-8019-001D0968F779.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3C2551B3-CC9D-DD11-A104-001D0967C13A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3E968F06-CD9D-DD11-83CE-00145EDD7925.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/402738F8-CC9D-DD11-8668-0019B9E4F9CF.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/408F9E84-CC9D-DD11-9824-0019B9E71460.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4A571C76-CC9D-DD11-9025-0019B9E7C563.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4AFB9AAD-CC9D-DD11-98A1-001D0967DFC6.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4AFCBAB9-CC9D-DD11-B50F-001D0967CFCC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4C30B377-CC9D-DD11-A60A-001D0967D19D.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4E393F81-CC9D-DD11-A156-001D0967D5D5.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4EDF3174-CD9D-DD11-A17F-0019B9E4AF15.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/501C28EF-CC9D-DD11-B663-00145EDD72E5.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/50E3A82F-CC9D-DD11-A1FB-001D09690A04.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/58F75674-CC9D-DD11-981C-001D0967D60C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5AD7A697-CC9D-DD11-AA49-0019B9E4FDBB.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5CC5F76E-CD9D-DD11-8211-00145EDD7971.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5E5970B6-CC9D-DD11-A86F-001D0967DA85.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/621FACEC-CC9D-DD11-A88A-001D0967D32D.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/64B60FAA-CC9D-DD11-9073-001D0967DE90.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/66FC03E6-CC9D-DD11-B97F-0019B9E7DEB4.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6842220C-CD9D-DD11-8C78-0019B9E4FDED.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6C28E1C7-CC9D-DD11-A967-001D0967D670.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6E661676-CC9D-DD11-9B72-0019B9E4FE29.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6EDB95C7-CC9D-DD11-B110-0019B9E4F396.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6EF1B235-CC9D-DD11-BACF-0019B9E7EA35.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/706E3C78-CC9D-DD11-B916-001D0967D698.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/70A44ACA-CC9D-DD11-8AA4-0019B9E4FF87.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/72F2CD7D-CD9D-DD11-ACFE-0019B9E4B0E7.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7642247F-CC9D-DD11-B9D8-0019B9E50162.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/788C6371-CC9D-DD11-A1D0-001D0967D724.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7A777793-CC9D-DD11-9962-001D0968CA4C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7AB2807C-CC9D-DD11-999B-001D0967CFE5.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7C196FB6-CC9D-DD11-95F6-001D0967DCAB.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7CD4FAE7-CC9D-DD11-87A4-001D0968F765.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7E0FCA77-CC9D-DD11-977A-0019B9E4FC1C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/820CBAB3-CC9D-DD11-B3A4-001D0967DEA4.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/862927B7-CC9D-DD11-B37C-0019B9E713FC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/88534FB3-CC9D-DD11-84B6-001D0967DFAD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8AAF1481-CC9D-DD11-B3B9-0019B9E4FC76.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8C1847FC-CC9D-DD11-9801-001D0967C0CC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8E58DBB2-CC9D-DD11-8F30-001D0967DC38.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8E627483-CC9D-DD11-8484-0019B9E4F3F5.root'
 
    
    ),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10)
) 
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(100)
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
process.TrackRefitterP5.src = 'rsWithMaterialTracksP5'
process.TrackRefitterP5.TrajectoryInEvent = True


process.siPixelLocalReco = cms.Sequence(process.siPixelDigis*process.siPixelClusters*process.siPixelRecHits) 
process.siStripLocalReco = cms.Sequence(process.siStripDigis*process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
process.trackerLocalReco = cms.Sequence(process.siPixelLocalReco*process.siStripLocalReco)
#process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks) #*process.rstracks 
process.trackReconstruction = cms.Sequence(process.trackerLocalReco*process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks*process.rstracks)

process.monitorTrack = cms.Sequence(process.SiPixelTrackResidualSource)
process.monitors = cms.Sequence(process.SiPixelRawDataErrorSource*process.SiPixelDigiSource*process.SiPixelClusterSource*process.SiPixelRecHitSource*process.SiPixelTrackResidualSource)

#event content analyzer
process.dump = cms.EDAnalyzer('EventContentAnalyzer')

process.pathTrack = cms.Path(process.trackReconstruction*process.TrackRefitterP5*process.monitorTrack)
