import FWCore.ParameterSet.Config as cms 

process = cms.Process("SiPixelMonitorTrackResiduals") 

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.GlobalRuns.ForceZeroTeslaField_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

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

    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/04A0C588-CC9D-DD11-9DE8-0019B9E71357.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/04CCF512-CD9D-DD11-941E-001D0967DC42.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/04E3CEA5-CC9D-DD11-8541-0019B9E7CECC.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/0A8FBB7F-CC9D-DD11-B29B-0019B9E4AF83.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/0CF74C7F-CC9D-DD11-8E28-001D0967DC3D.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/0E53380D-CD9D-DD11-8586-0019B9E713ED.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/103F1DDB-CC9D-DD11-8197-001D0967DF0D.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/12CA452F-CD9D-DD11-BDCA-0019B9E7EA35.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/147B6D73-CC9D-DD11-A82F-001D0967D1BB.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/1C4420B3-CC9D-DD11-A102-001D0968F1FC.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/1ED9CAE4-CC9D-DD11-8B98-001D0967D305.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/207662BD-CC9D-DD11-9DF1-00145EDD775D.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/229CCEB5-CC9D-DD11-A6E3-0019B9E4FD8E.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/262FC672-CC9D-DD11-AC13-001D0967CFC7.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/28F4A2BF-CC9D-DD11-A9FB-001D0967C464.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/306FED88-CC9D-DD11-B7B6-001D0967D04E.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/325AF3B3-CC9D-DD11-924B-001D0967DF3F.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/32EE830C-CD9D-DD11-B2D3-0019B9E4FE1A.root',
    ###GOOD FILES START (WITH TRACKS)
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/34190739-739D-DD11-8E14-0019B9E4AD9A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/34C7E8B3-CC9D-DD11-87F1-001D0967DE90.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3664D2B2-CC9D-DD11-9DEA-001D0967D571.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3803B395-CC9D-DD11-BD79-001D0967DA76.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3A1E9A6E-CC9D-DD11-A867-0019B9E7130C.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3A90EF2D-CD9D-DD11-98DE-001D0968F256.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3AECD7B2-CC9D-DD11-8019-001D0968F779.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3C2551B3-CC9D-DD11-A104-001D0967C13A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/3E968F06-CD9D-DD11-83CE-00145EDD7925.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/402738F8-CC9D-DD11-8668-0019B9E4F9CF.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/408F9E84-CC9D-DD11-9824-0019B9E71460.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/42D1C7B2-CC9D-DD11-BC6F-001D0967DB7F.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4A571C76-CC9D-DD11-9025-0019B9E7C563.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4AFB9AAD-CC9D-DD11-98A1-001D0967DFC6.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4AFCBAB9-CC9D-DD11-B50F-001D0967CFCC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4C30B377-CC9D-DD11-A60A-001D0967D19D.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4E393F81-CC9D-DD11-A156-001D0967D5D5.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/4EDF3174-CD9D-DD11-A17F-0019B9E4AF15.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/501C28EF-CC9D-DD11-B663-00145EDD72E5.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/50E3A82F-CC9D-DD11-A1FB-001D09690A04.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/58E37AE1-CC9D-DD11-8859-001D0968F684.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/58F75674-CC9D-DD11-981C-001D0967D60C.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5AD7A697-CC9D-DD11-AA49-0019B9E4FDBB.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5CC5F76E-CD9D-DD11-8211-00145EDD7971.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/5E5970B6-CC9D-DD11-A86F-001D0967DA85.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/60DE217E-CC9D-DD11-8018-0019B9E71500.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/621FACEC-CC9D-DD11-A88A-001D0967D32D.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/62A2172B-CD9D-DD11-A706-001D0967D77E.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/64B60FAA-CC9D-DD11-9073-001D0967DE90.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/66FC03E6-CC9D-DD11-B97F-0019B9E7DEB4.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6842220C-CD9D-DD11-8C78-0019B9E4FDED.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6C28E1C7-CC9D-DD11-A967-001D0967D670.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6E661676-CC9D-DD11-9B72-0019B9E4FE29.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6EDB95C7-CC9D-DD11-B110-0019B9E4F396.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/6EF1B235-CC9D-DD11-BACF-0019B9E7EA35.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/706E3C78-CC9D-DD11-B916-001D0967D698.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/70A44ACA-CC9D-DD11-8AA4-0019B9E4FF87.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/72B085BD-CC9D-DD11-B7EC-001D0967C07B.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/72F2CD7D-CD9D-DD11-ACFE-0019B9E4B0E7.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7642247F-CC9D-DD11-B9D8-0019B9E50162.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/788C6371-CC9D-DD11-A1D0-001D0967D724.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7A777793-CC9D-DD11-9962-001D0968CA4C.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7AB2807C-CC9D-DD11-999B-001D0967CFE5.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7C196FB6-CC9D-DD11-95F6-001D0967DCAB.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7CD4FAE7-CC9D-DD11-87A4-001D0968F765.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/7E0FCA77-CC9D-DD11-977A-0019B9E4FC1C.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/820CBAB3-CC9D-DD11-B3A4-001D0967DEA4.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8440B9BF-CC9D-DD11-B160-001D0967D643.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/862927B7-CC9D-DD11-B37C-0019B9E713FC.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/88534FB3-CC9D-DD11-84B6-001D0967DFAD.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8AAF1481-CC9D-DD11-B3B9-0019B9E4FC76.root',
    #'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8C1847FC-CC9D-DD11-9801-001D0967C0CC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8E58DBB2-CC9D-DD11-8F30-001D0967DC38.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/8E627483-CC9D-DD11-8484-0019B9E4F3F5.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/9417007E-CC9D-DD11-91CD-001D0967DACB.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/94A7CC35-CC9D-DD11-9701-0019B9E4F3B8.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/94F79134-CC9D-DD11-BC3B-001D0967D319.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/96817033-CC9D-DD11-BB92-001D0967D026.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/96ABC954-CD9D-DD11-A799-0019B9E4F9B6.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/980098F3-CC9D-DD11-A4EB-001D0968F36E.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/987F1CB4-CC9D-DD11-9936-001D0967D0FD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/98C98EED-CC9D-DD11-8884-001D0967DF08.root',
    #DEFEKT'/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/9CE62A12-CD9D-DD11-96E5-001D0967D689.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/9EA26118-CD9D-DD11-96E2-001D0967DADF.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/9EAF01D5-CC9D-DD11-9504-001D0967DBF7.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/A0BE01C9-CC9D-DD11-9C4B-001D0967DAF3.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/A6DB12EF-CC9D-DD11-8118-0019B9E4FD5C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/AA11B194-CC9D-DD11-8148-001D0968F5AD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/AA2986B1-CC9D-DD11-A041-0019B9E4B0BA.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/AE7C0793-CC9D-DD11-892A-0019B9E4FD75.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/B04D4C80-CC9D-DD11-A665-001D0967DFFD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/B2817233-CC9D-DD11-8CDB-001D0967DC38.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/B6082384-CC9D-DD11-B568-0019B9E4F822.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/B8E76EC0-CC9D-DD11-9318-001D0967D48B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/BA6B882F-CD9D-DD11-84CB-001D0967C653.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/BC2B7D83-CC9D-DD11-9EF2-001D0967CF86.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/BC887F83-CC9D-DD11-BC53-0019B9E4FA2B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/BE0AEB74-739D-DD11-B680-001D0967DC7E.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/BE9CF173-CC9D-DD11-B36A-0019B9E4FD48.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C02F2F38-CC9D-DD11-B359-001D0967D571.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C0CB192F-CC9D-DD11-80BB-001D0967DF62.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C0F08EC1-CC9D-DD11-8B6D-001D0967CEBE.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C2A1F683-CC9D-DD11-A7D7-0019B9E7C446.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C2CED5C2-CC9D-DD11-BD84-0019B9E4FD57.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C43B9C88-CC9D-DD11-B695-001D0967D616.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C4B086DD-CC9D-DD11-B715-001D0968EBC1.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C8014737-CC9D-DD11-95B6-001D0968F1FC.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/C8B3F16A-CD9D-DD11-A491-0019B9E7CA12.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/CAD14282-CC9D-DD11-AD1D-0019B9E7CC01.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/CECCDFF9-CC9D-DD11-910C-001D0967CF8B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D07D58F1-CC9D-DD11-9752-001D0967DA17.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D0F5C0BB-CC9D-DD11-8935-001D0967DAA3.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D2489083-CC9D-DD11-BD94-0019B9E4FDF7.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D86F2EBD-CC9D-DD11-A59C-001D0967BE87.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D8A90188-CC9D-DD11-83CE-001D0967D11B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/D8CFEE88-CC9D-DD11-9280-0019B9E712AD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/E2C9B572-CC9D-DD11-B6FA-0019B9E4FCCB.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/E6DCBC9A-CC9D-DD11-8ECD-0019B9E4F9F2.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/E89FE589-CC9D-DD11-B9DA-0019B9E7DE64.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F26AC4F8-CC9D-DD11-895C-001D0967D25B.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F2C4BB7F-CC9D-DD11-914B-001D0967D99A.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F44A55D6-CC9D-DD11-88B2-001D0967DA12.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F4A8BC93-CC9D-DD11-83C7-001D0967D97C.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F6938C86-CC9D-DD11-B4CC-001D0967D55D.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/F845817E-CC9D-DD11-B57C-001D0967D0FD.root',
    '/store/data/Commissioning08/Cosmics/RECO/CRAFT_V1P_SuperPointing_v3/0000/FE79963E-CC9D-DD11-917A-0019B9E71497.root'
    
    
    ),
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10)
) 
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(3000)
)
process.DQMStore = cms.Service("DQMStore",
  referenceFileName = cms.untracked.string(''),
  verbose = cms.untracked.int32(0)
)
process.LockService = cms.Service("LockService", 
  labels = cms.untracked.vstring('source') 
)
process.ModuleWebRegistry = cms.Service("ModuleWebRegistry")

process.AdaptorConfig = cms.Service("AdaptorConfig") 


##
## Load and Configure TrackRefitter
##
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
#process.TrackRefitter.src = 'ALCARECOTkAlCosmicsCTF0T'
#process.TrackRefitterP5.src = 'ctfWithMaterialTracksP5'
#process.TrackRefitterP5.src = 'cosmictrackfinderP5'
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
#process.pathTrack = cms.Path(process.trackReconstruction*process.monitorTrack)
#process.pathTrack = cms.Path(process.monitorTrack)

#alcareco:
#process.pathTrack = cms.Path(process.offlineBeamSpot*process.TrackRefitterP5*process.monitorTrack) 
# process.pathStandard = cms.Path(process.RawToDigi*process.reconstruction*process.monitors) 
