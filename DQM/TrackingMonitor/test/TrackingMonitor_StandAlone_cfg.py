import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# DQMServices
#-------------------------------------------------
process = cms.Process("DQMTrackMon")

#-------------------------------------------------
# message logger
#-------------------------------------------------
process.MessageLogger = cms.Service("MessageLogger",
     default = cms.untracked.PSet( limit = cms.untracked.int32(500) )
)

#-------------------------------------------------
# Source  
#-------------------------------------------------
process.source = cms.Source("PoolSource",
#fileNames = cms.untracked.vstring('/store/data/Run2011A/Jet/RAW/v1/000/166/950/6EA38368-6696-E011-ACD3-0030487C8CB8.root')   
fileNames = cms.untracked.vstring(
    '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/10CF910B-E057-E011-9B13-000423D9A2AE.root',
	'/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/FA2A2CFC-EB57-E011-BF91-00304879EE3E.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/F8C8D3A7-D957-E011-85A4-003048F11CF0.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/DAACC74C-DF57-E011-A45D-001D09F2B30B.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/9A449B92-D757-E011-8FC7-003048F1183E.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/92FD48CA-F557-E011-959E-003048F1C832.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/888BEB25-0A58-E011-98B3-003048F024DC.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/7C7651A9-EC57-E011-B60D-00304879FBB2.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/6AF8632D-F057-E011-BD34-001D09F2516D.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/60819326-1658-E011-BB11-003048F1C420.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/560220D8-FC57-E011-9986-003048F024DC.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/4CB4C0AC-F357-E011-B294-003048F118E0.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/42CB3662-ED57-E011-A068-001617E30F58.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/32029E2D-E957-E011-8CAE-001D09F251CC.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/283A353B-E457-E011-AB13-0030487C8CBE.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/24196B2D-F057-E011-BB16-001D09F23D1D.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/20143BE5-E957-E011-AAF3-00304879EE3E.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/10CF910B-E057-E011-9B13-000423D9A2AE.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/312/08357011-F357-E011-9076-000423D98B6C.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/311/F46E958A-E357-E011-9FB1-0030487A18F2.root',
        '/store/data/Run2011A/Jet/RECO/PromptReco-v1/000/161/311/D63AC4DF-CF57-E011-942A-001D09F25460.root')
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))


#-------------------------------------------------
# MAGNETIC FIELD & RECO
#-------------------------------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR_R_43_V3::All"
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
#process.load("DQM.TrackingMonitor.TrackingMonitorAllTrackingSequences_cff")

process.load("DQM.TrackingMonitor.TrackingMonitor_cfi")

# InputTags for collections 
process.TrackMon.TrackProducer          = cms.InputTag("generalTracks")
process.TrackMon.SeedProducer  = cms.InputTag("newSeedFromTriplets")


process.TrackMon.ClusterLabels = cms.vstring('Pix','Strip','Tot')

process.TrackMon.TCProducer             = cms.InputTag("newTrackCandidateMaker")
process.TrackMon.beamSpot               = cms.InputTag("offlineBeamSpot")

# properties
process.TrackMon.AlgoName               = cms.string('GenTk')
process.TrackMon.OutputFileName 	= cms.string('TrackingMonitor.root')
process.TrackMon.OutputMEsInRootFile 	= cms.bool(True)

process.TrackMon.FolderName             = cms.string('Track/GlobalParameters')
process.TrackMon.BSFolderName           = cms.string('Track/BeamSpotParameters')
process.TrackMon.MeasurementState 		= cms.string('ImpactPoint')

# which plots to do
process.TrackMon.doTrackerSpecific 		= cms.bool(False)
process.TrackMon.doAllPlots 			= cms.bool(False)
process.TrackMon.doBeamSpotPlots 		= cms.bool(False)

process.TrackMon.doSeedParameterHistos   = cms.bool(True)
#process.TrackMon.doSeedNumberHisto          = cms.bool(True)
#process.TrackMon.doSeedVsClusterHisto       = cms.bool(True)
process.TrackMon.doTrackCandHistos          = cms.bool(True)

process.TrackMon.doSeedPTHisto          = cms.bool(False)

#-------------------------------------------------

# DQM Store 
#-------------------------------------------------


process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

#-------------------------------------------------
# Paths 
#-------------------------------------------------

# redo tracking (to get all the track builing and interative steps)

#process.preTracking = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits)
#process.tracking    = cms.Path(process.ckftracks_plus_pixelless)

#pixeltrackerlocalreco = cms.Path(process.siPixelClusters*process.siPixelRecHits)
#striptrackerlocalreco = cms.Path(process.siStripClusters*process.siStripMatchedRecHits)
#trackerlocalreco = cms.Path(process.pixeltrackerlocalreco*process.striptrackerlocalreco)
#process.digis = cms.Path(process.siPixelDigis*process.siStripDigis)
#process.preTracking = cms.Path(process.trackerlocalreco)
process.preTracking = cms.Path(process.siStripMatchedRecHits*process.siPixelRecHits)
process.tracking    = cms.Path(process.offlineBeamSpot*process.recopixelvertexing*process.ckftracks)

#process.tracking    = cms.Path(process.ckftracks)

# tracking monitor paths
#process.trackmon    = cms.Path(process.trkmon)
process.trackmon  = cms.Path( process.TrackMon )
# end path
process.outP    = cms.OutputModule("AsciiOutputModule")
process.ep      = cms.EndPath(process.outP)

process.trkmon = cms.Schedule(
#process.digis ,	
      process.preTracking,
     process.tracking
    , process.trackmon 
    , process.ep
)


