import FWCore.ParameterSet.Config as cms

process = cms.Process("clusterAnalysis")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/0017C8A5-7C3B-DF11-8BC1-001617E30CD4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/0695C757-7E3B-DF11-A20B-0030487C7392.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/069A8EFB-7C3B-DF11-88B0-0030487D0D3A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/08204E11-7F3B-DF11-8F59-001D09F27067.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/08532A3C-813B-DF11-8AF7-001D09F24498.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/0C923A5A-7E3B-DF11-B7C6-001D09F25401.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/0EFEFE59-7E3B-DF11-BEB0-0019B9F72BFF.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/12C7400E-7F3B-DF11-A59C-001D09F26509.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/1606997B-803B-DF11-8256-001D09F2932B.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/1AAE2EFC-7C3B-DF11-A749-000423D98800.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2002165A-7E3B-DF11-86F8-001D09F2906A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/26B42779-7C3B-DF11-A1F0-00304879EE3E.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2801F23C-813B-DF11-9A02-001D09F28D54.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2A8D7035-813B-DF11-AF82-001D09F241F0.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2C499810-7F3B-DF11-A767-001D09F28E80.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2C6BC636-813B-DF11-ADF6-001D09F25217.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2CDF8858-7E3B-DF11-B388-001D09F29538.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/2E0287C5-7F3B-DF11-BD19-001D09F26C5C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/306876F8-7C3B-DF11-AAD4-000423D98834.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/30F51036-813B-DF11-AE16-001D09F2983F.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/30F75C0D-7F3B-DF11-8744-001D09F2A49C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/34150412-7F3B-DF11-92E2-0019B9F709A4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/3E4B6459-7E3B-DF11-AB79-001D09F252F3.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/3E6CC286-7C3B-DF11-9DFB-000423D33970.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/401B4774-7C3B-DF11-8F30-001D09F232B9.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/463753F5-7C3B-DF11-A31D-000423D99896.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/46517C44-813B-DF11-AE3A-001D09F28755.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/48566458-7E3B-DF11-AE1B-001D09F2423B.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/4A2AC988-7C3B-DF11-BD17-001D09F2905B.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/4A2FEE57-7E3B-DF11-A307-0030487CD13A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/4AFC557F-7C3B-DF11-A027-001D09F251CC.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/4C1D325B-7E3B-DF11-BCF0-000423D174FE.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/4CB1F2F8-7C3B-DF11-89D3-0030487CD7C0.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/5076F6F9-7C3B-DF11-9FF1-000423D98B6C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/54447E11-7F3B-DF11-AA10-001D09F24EE3.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/5466A07B-803B-DF11-8AD7-001D09F2A690.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/547A67C4-7F3B-DF11-AAE2-001D09F24D8A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/54C6FDF8-7C3B-DF11-B184-000423D94A04.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/561C96F4-7C3B-DF11-9FC2-000423D985B0.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/602A8F2F-813B-DF11-855B-001D09F2B2CF.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/623AB45A-7E3B-DF11-BB05-001D09F253D4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/66757836-813B-DF11-8093-001D09F24047.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/66E05CC5-7F3B-DF11-972D-001D09F253D4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/6A2092C4-7F3B-DF11-8842-001D09F29849.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/6AD7DBC3-7F3B-DF11-9AC4-001D09F250AF.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/70235EC2-7F3B-DF11-84F2-001617DBD556.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/74C8DBC3-7F3B-DF11-96B7-001D09F23944.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/74DFBCA6-7C3B-DF11-B672-001617DBD556.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/76AE455A-7E3B-DF11-BC36-001D09F2438A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/78116D7B-803B-DF11-BB50-001D09F2525D.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/78B3A176-7C3B-DF11-87F7-0019B9F72BFF.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/7CB2E77E-803B-DF11-ADF5-001D09F28EC1.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/7E3C3E3C-813B-DF11-B4BB-001D09F29597.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/7E3FC40F-7F3B-DF11-B078-0019DB2F3F9A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/849F39C5-7F3B-DF11-A9B2-001D09F2447F.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/8606FF56-7E3B-DF11-A0FA-000423D94494.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/8824FBC5-7F3B-DF11-B690-001D09F2915A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/884A155A-7E3B-DF11-829D-0019B9F7312C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/884E2CFF-7C3B-DF11-AFA9-000423D174FE.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/8E82800F-7F3B-DF11-9866-001D09F251FE.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/92E84901-7D3B-DF11-BE60-000423D9A2AE.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/94E3607B-803B-DF11-B870-001D09F29533.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/960AC78B-7C3B-DF11-B41B-001D09F23D1D.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/9680125A-7E3B-DF11-962C-001D09F25109.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/9C426D7B-803B-DF11-8D00-001D09F2A49C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/9E855F7B-803B-DF11-A983-001D09F252DA.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/A00EE43D-813B-DF11-8291-001D09F2915A.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/A4A6155A-7E3B-DF11-956B-001D09F295FB.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/A83BA2C4-7F3B-DF11-AD5F-001D09F232B9.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/AA87FA56-7E3B-DF11-AB9A-000423D987FC.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/AC600714-7F3B-DF11-90E9-0019B9F70468.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/B0255B7D-803B-DF11-BF05-001D09F244DE.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/B638D357-7E3B-DF11-8FC1-000423D94C68.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/B675CD57-7E3B-DF11-A89A-0030487A1FEC.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/B6DE9F9B-7C3B-DF11-AC7C-001617E30F50.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/B8AC7E59-7E3B-DF11-8FB5-001D09F2924F.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/BC289F5E-7E3B-DF11-87EA-0030487A18F2.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/BC58F23C-813B-DF11-A670-0019B9F70607.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/C8255F59-7E3B-DF11-9AC5-001617E30D12.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/C84FA9F5-7C3B-DF11-9790-000423D9517C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/CC816C7B-803B-DF11-9509-001D09F29619.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/CCFBF311-7F3B-DF11-A260-000423D98920.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/D22E3B56-7E3B-DF11-B56F-000423D98A44.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/D8C3917B-803B-DF11-ADBD-001D09F25401.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/D8CC6A7B-803B-DF11-B063-001D09F2983F.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/DA81D93B-813B-DF11-912F-001D09F2512C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/DA8A07F5-7C3B-DF11-9E18-00151796CD80.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/DCD56F36-813B-DF11-8280-0019B9F705A3.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/DE2F32F9-7C3B-DF11-8697-000423D991D4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/DEE86C7A-803B-DF11-BC54-001D09F34488.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/E64659F4-7C3B-DF11-A846-000423D98BC4.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/E68B7A5A-7E3B-DF11-9DE0-0030487C8CB6.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/E8201B5A-7E3B-DF11-BFBC-001D09F25208.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/ECBEB132-813B-DF11-B48E-001617DBD230.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/ECF3C8C4-7F3B-DF11-A7DB-001D09F2441B.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/F0679901-7D3B-DF11-A3E7-000423D987FC.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/F206767B-803B-DF11-9AE6-001D09F24448.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/F6139510-7F3B-DF11-8E95-00304879EE3E.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/F645369E-7D3B-DF11-BB5B-0030487A3232.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/FCA7F078-803B-DF11-946B-001617E30E28.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/FEE17659-7E3B-DF11-9D8E-001D09F23A84.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/FEECADCA-7F3B-DF11-ABF3-001D09F2512C.root',
	'/store/express/Commissioning10/ExpressPhysics/FEVT/v7/000/132/420/FEF80757-7E3B-DF11-ADF0-000423D999CA.root',

)
)

# Conditions (Global Tag is used here):
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR10_E_V3::All'

#Geometry and field
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("TrackingTools.RecoGeometry.RecoGeometries_cff")

#tracker refitting -> trajectory
process.load('RecoTracker.TrackProducer.TrackRefitters_cff')
process.ttrhbwr.ComputeCoarseLocalPositionFromDisk = True
process.generalTracks = process.TrackRefitter.clone(
   src = cms.InputTag("generalTracks")
)
process.ctfPixelLess = process.TrackRefitter.clone(
   src = cms.InputTag("ctfPixelLess")
)
process.refit = cms.Sequence(process.generalTracks*process.ctfPixelLess*process.doAlldEdXEstimators)
## re_fitting
#process.load('Configuration/StandardSequences/Reconstruction_cff')
#process.refit = cms.Sequence(
#    process.siPixelRecHits * 
#    process.siStripMatchedRecHits *
#    process.ckftracks *
#    process.ctfTracksPixelLess
#)

#analysis
process.analysis = cms.EDAnalyzer('TrackerDpgAnalysis',
   ClustersLabel = cms.InputTag("siStripClusters"),
   PixelClustersLabel = cms.InputTag("siPixelClusters"),
   TracksLabel   = cms.VInputTag( cms.InputTag("generalTracks"), cms.InputTag("ctfPixelLess") ),
   vertexLabel   = cms.InputTag('offlinePrimaryVertices'),
   pixelVertexLabel = cms.InputTag('pixelVertices'),
   beamSpotLabel = cms.InputTag('offlineBeamSpot'),
   DeDx1Label    = cms.InputTag('dedxHarmonic2'),
   DeDx2Label    = cms.InputTag('dedxTruncated40'),
   DeDx3Label    = cms.InputTag('dedxMedian'),
   L1Label       = cms.InputTag('gtDigis'),
   HLTLabel      = cms.InputTag("TriggerResults"),
   InitalCounter = cms.uint32(1),
   keepOntrackClusters  = cms.untracked.bool(True),
   keepOfftrackClusters = cms.untracked.bool(True),
   keepPixelClusters    = cms.untracked.bool(True),
   keepPixelVertices    = cms.untracked.bool(True),
   keepMissingHits      = cms.untracked.bool(True),
   keepTracks           = cms.untracked.bool(True),
   keepVertices         = cms.untracked.bool(True),
   keepEvents           = cms.untracked.bool(True),
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('trackerDPG_Express_132420tkoff.root')
)

process.skimming = cms.EDFilter("PhysDecl",
  applyfilter = cms.untracked.bool(False),
  debugOn = cms.untracked.bool(False),
  HLTriggerResults = cms.InputTag("TriggerResults","","HLT")

)

process.p = cms.Path(process.skimming*process.refit*process.analysis)
#process.dump = cms.EDAnalyzer("EventContentAnalyzer")
#process.p = cms.Path(process.refit*process.dump)
