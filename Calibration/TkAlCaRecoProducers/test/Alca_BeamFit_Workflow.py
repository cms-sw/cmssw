import FWCore.ParameterSet.Config as cms

process = cms.Process("alcaBeamSpotWorkflow")

###################################################################
# initialize MessageLogger
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.AlcaBeamSpotProducer=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100)
                                   ),
    AlcaBeamSpotProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    enableStatistics = cms.untracked.bool(True)
)

process.load("Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducer_cff")
readFiles=['/store/data/Run2022C/JetMET/ALCARECO/TkAlJetHT-PromptReco-v1/000/357/482/00000/08365631-c05f-4584-b8a4-5cc7e23c1ac8.root']

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(readFiles)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

###################################################################
# standard includes
###################################################################
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')
process.load("Configuration.Geometry.GeometryRecoDB_cff")

###################################################################
# reco PV
###################################################################
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

###################################################################
# remove beam scraping events
###################################################################
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.20)
                                 )

###################################################################
# Primary Vertex
###################################################################
from RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi import offlinePrimaryVertices
process.offlinePrimaryVerticesFromTrks  = offlinePrimaryVertices.clone()
process.offlinePrimaryVerticesFromTrks.TrackLabel = cms.InputTag("ALCARECOTkAlJetHT")
process.offlinePrimaryVerticesFromTrks.vertexCollections.maxDistanceToBeam = 1
process.offlinePrimaryVerticesFromTrks.TkFilterParameters.maxNormalizedChi2 = 20
process.offlinePrimaryVerticesFromTrks.TkFilterParameters.minSiliconLayersWithHits = 5
process.offlinePrimaryVerticesFromTrks.TkFilterParameters.maxD0Significance = 5.0
process.offlinePrimaryVerticesFromTrks.TkFilterParameters.minPixelLayersWithHits = 2

###################################################################
# BeamSpot producer config
###################################################################
process.alcaBeamSpotProducer.BeamFitter.TrackCollection = 'ALCARECOTkAlJetHT'
process.alcaBeamSpotProducer.BeamFitter.MinimumTotalLayers = 6
process.alcaBeamSpotProducer.BeamFitter.MinimumPixelLayers = -1
process.alcaBeamSpotProducer.BeamFitter.MaximumNormChi2 = 10
process.alcaBeamSpotProducer.BeamFitter.MinimumInputTracks = 50
process.alcaBeamSpotProducer.BeamFitter.MinimumPt = 1.0
process.alcaBeamSpotProducer.BeamFitter.MaximumImpactParameter = 1.0
process.alcaBeamSpotProducer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.alcaBeamSpotProducer.BeamFitter.Debug = True

process.alcaBeamSpotProducer.PVFitter.VertexCollection = 'offlinePrimaryVerticesFromTrks'
process.alcaBeamSpotProducer.PVFitter.Apply3DFit = True
process.alcaBeamSpotProducer.PVFitter.minNrVerticesForFit = 10

###################################################################
# fit as function of lumi sections
###################################################################
process.alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.fitEveryNLumi = 1
process.alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.resetEveryNLumi = 1

###################################################################
# Output module
###################################################################
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string( 'AlcaBeamSpot.root' ),
                               outputCommands = cms.untracked.vstring("keep *"))


###################################################################
# paths and endpaths
###################################################################
process.e = cms.EndPath( process.out )

process.p = cms.Path(process.offlineBeamSpot +
                     # process.TrackRefitter + # in case of refit
                     process.offlinePrimaryVerticesFromTrks+
                     # process.noScraping +    # not needed in recent data
                     process.alcaBeamSpotProducer)
