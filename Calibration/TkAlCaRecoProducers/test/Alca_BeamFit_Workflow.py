import FWCore.ParameterSet.Config as cms

process = cms.Process("alcaBeamSpotWorkflow")
# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    default = cms.untracked.PSet(
       limit = cms.untracked.int32(0)
    ),
    AlcaBeamSpotProducer = cms.untracked.PSet(
        #reportEvery = cms.untracked.int32(100) # every 1000th only
	limit = cms.untracked.int32(0)
    )
)
#process.MessageLogger.cout.enableStatistics = cms.untracked.bool(True)

process.load("Calibration.TkAlCaRecoProducers.AlcaBeamSpotProducer_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0171/3A22D08D-456A-DF11-961F-001A92811734.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0171/083B6802-236C-DF11-8AC6-0026189437FE.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0170/E6D0589B-136A-DF11-9E90-002618943982.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0170/D098488B-276A-DF11-8069-003048678AF4.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/F466DD94-BF69-DF11-B9B8-00261894390A.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/F2A9245B-BC69-DF11-9C3F-0018F3D096E4.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/DED5E502-B969-DF11-AE8F-002618943964.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/563257AF-C169-DF11-865F-002618943907.root',
'/store/data/Commissioning10/MinimumBias/ALCARECO/TkAlMinBias-May27thReReco_v1/0166/3056B847-C069-DF11-B0B0-003048679010.root'

    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000) #1500
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# this is for filtering on L1 technical trigger bit
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND ( 40 OR 41 ) AND NOT (36 OR 37 OR 38 OR 39)')

##
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_38X_V9::All' #'GR_R_35X_V8::All'
process.load("Configuration.StandardSequences.Geometry_cff")


## reco PV
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = cms.InputTag("ALCARECOTkAlMinBias") 

#### remove beam scraping events
process.noScraping= cms.EDFilter("FilterOutScraping",
                                 applyfilter = cms.untracked.bool(True),
                                 debugOn = cms.untracked.bool(False), ## Or 'True' to get some per-event info
                                 numtrack = cms.untracked.uint32(10),
                                 thresh = cms.untracked.double(0.20)
)



################### Primary Vertex
process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 2
process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 20
process.offlinePrimaryVertices.TkFilterParameters.minSiliconLayersWithHits = 5
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelLayersWithHits = 1
process.offlinePrimaryVertices.TkClusParameters.TkGapClusParameters.zSeparation = 1


#######################
process.alcaBeamSpotProducer.BeamFitter.TrackCollection = 'ALCARECOTkAlMinBias'
process.alcaBeamSpotProducer.BeamFitter.MinimumTotalLayers = 6
process.alcaBeamSpotProducer.BeamFitter.MinimumPixelLayers = -1
process.alcaBeamSpotProducer.BeamFitter.MaximumNormChi2 = 10
process.alcaBeamSpotProducer.BeamFitter.MinimumInputTracks = 50
process.alcaBeamSpotProducer.BeamFitter.MinimumPt = 1.0
process.alcaBeamSpotProducer.BeamFitter.MaximumImpactParameter = 1.0
process.alcaBeamSpotProducer.BeamFitter.TrackAlgorithm =  cms.untracked.vstring()
#process.alcaBeamSpotProducer.BeamFitter.Debug = True

process.alcaBeamSpotProducer.PVFitter.Apply3DFit = True
process.alcaBeamSpotProducer.PVFitter.minNrVerticesForFit = 10 
#########################


# fit as function of lumi sections
process.alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.fitEveryNLumi = 1
process.alcaBeamSpotProducer.AlcaBeamSpotProducerParameters.resetEveryNLumi = 1

process.out = cms.OutputModule( "PoolOutputModule",
                                fileName = cms.untracked.string( 'AlcaBeamSpot.root' ),
                                outputCommands = cms.untracked.vstring("keep *")
                              )


process.e = cms.EndPath( process.out )

process.p = cms.Path(process.hltLevel1GTSeed +
                     process.offlineBeamSpot +
#                     process.TrackRefitter +
                     process.offlinePrimaryVertices+
#                     process.noScraping +
                     process.alcaBeamSpotProducer)
