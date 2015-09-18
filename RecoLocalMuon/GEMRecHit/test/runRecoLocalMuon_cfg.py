import FWCore.ParameterSet.Config as cms

process = cms.Process("MyMuonRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

# Process options :: 
# - wantSummary helps to understand which module crashes
# - skipEvent skips event in case a product was not found
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True), 
                                      # SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                    )



process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")                # recommended configuration
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')  # deprecated configuration
# Be careful here ot to load Configuration.StandardSequences.Reconstruction_cff
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


### Try to do RecoLocalMuon on all muon detectors ###
#####################################################
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
#process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')
process.load('RecoLocalMuon.GEMRecHit.me0LocalReco_cff')
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")

### me0Muon reco now
process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')


### Try to add also Tracker local reco ###
##########################################
# Digi2Raw and Raw2Digi for Tracker Dets:
process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = cms.InputTag('SiStripDigiToRaw')

# --------------------------
# - Sources of Information -
# -----------------------------------------------------------------
# Configuration/StandardSequences/python/Reconstruction_cff.py
# RecoTracker/Configuration/python/customiseForRunI.py
# RecoTracker/Configuration/python/RecoTrackerRunI_cff.py
# RecoTracker/IterativeTracking/python/RunI_iterativeTk_cff.py
# tgrIndex = process.globalreco.index(process.trackingGlobalReco)
# -----------------------------------------------------------------
# Twiki pages I should consider to read:
# https://twiki.cern.ch/twiki/bin/view/CMS/AndreaBocciTracking
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideIterativeTracking?redirectedfrom=CMS.SWGuideIterativeTracking
# -----------------------------------------------------------------

# --------------------------
# - Local Reco for Tracker -
# -----------------------------------------------------------------
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
# for Local Reco for Tracker in Run-I we need to load & redefine:
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff") 
# Run-II Local Reco
# process.pixeltrackerlocalreco = cms.Sequence(process.siPixelClustersPreSplitting*process.siPixelRecHitsPreSplitting)
# process.striptrackerlocalreco = cms.Sequence(process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
# process.trackerlocalreco = cms.Sequence(process.pixeltrackerlocalreco*process.striptrackerlocalreco*process.clusterSummaryProducer)
# Run-I Local Reco
process.pixeltrackerlocalreco = cms.Sequence(process.siPixelClusters*process.siPixelRecHits)
process.striptrackerlocalreco = cms.Sequence(process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
process.trackerlocalreco = cms.Sequence(process.pixeltrackerlocalreco*process.striptrackerlocalreco*process.clusterSummaryProducerNoSplitting)
# -----------------------------------------------------------------


# ---------------------------
# - Global Reco for Tracker -
# -----------------------------------------------------------------
# Run-2 configuration ... will not work because of Calo dependencies
# process.load("RecoTracker.Configuration.RecoTracker_cff")
# Run-1 configuration ... should work stand-alone
# process.load("RecoTracker.Configuration.RecoTrackerRunI_cff")
# taking iterTracking from this file, 
# removing all dEdX, EcalSeeds and trackExtrapolations for (B-)Jets
process.load("RecoTracker.IterativeTracking.RunI_iterativeTk_cff")
process.load("RecoTracker.CkfPattern.CkfTrackCandidates_cff")
process.ckftracks          = cms.Sequence(process.iterTracking)
process.trackingGlobalReco = cms.Sequence(process.ckftracks) 

# Now get rid of spurious reference to JetCore step
process.earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                             'lowPtTripletStepTracks',
                                             'pixelPairStepTracks',
                                             'detachedTripletStepTracks',
                                             'mixedTripletStepTracks',
                                             'pixelLessStepTracks',
                                             'tobTecStepTracks'
                                             ]

process.earlyGeneralTracks.inputClassifiers =["initialStepSelector",
                                              "lowPtTripletStepSelector",
                                              "pixelPairStepSelector",
                                              "detachedTripletStep",
                                              "mixedTripletStep",
                                              "pixelLessStepSelector",
                                              "tobTecStep"
                                              ]

# Now restore pixelVertices wherever was not possible with an ad-hoc RunI cfg
process.muonSeededTracksInOutClassifier.vertices = 'pixelVertices'
process.muonSeededTracksOutInClassifier.vertices = 'pixelVertices'
process.duplicateTrackClassifier.vertices = 'pixelVertices'
process.convStepSelector.vertices = 'pixelVertices'
# because of removal of dEdX, EcalSeeds and trackExtrapolations
# following processes are not called, therefore pixelVertices not restored
# process.muonSeededTracksOutInDisplacedClassifier.vertices = 'pixelVertices'
# process.duplicateDisplacedTrackClassifier.vertices = 'pixelVertices'
# process.pixelPairElectronSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = 'pixelVertices'
# process.ak4CaloJetsForTrk.srcPVs = 'pixelVertices'
# process.photonConvTrajSeedFromSingleLeg.primaryVerticesTag = 'pixelVertices'

# ... and finally turn off all possible references to CCC: this is
# done by switching off the Tight and Loose reftoPSet, rather than
# following all the places in which they are effectively used in
# release. The RunI-like tracking already uses CCCNone: this will
# be useful mainly for conversions.
process.SiStripClusterChargeCutTight.value = -1.
process.SiStripClusterChargeCutLoose.value = -1

# Defaults are ok here ...
# process.earlyMuons.TrackAssociatorParameters.useMuon      = cms.bool(True)
# process.earlyMuons.TrackAssociatorParameters.useHO        = cms.bool(False)
# process.earlyMuons.TrackAssociatorParameters.useEcal      = cms.bool(False)
# process.earlyMuons.TrackAssociatorParameters.useHcal      = cms.bool(False)
# process.earlyMuons.TrackAssociatorParameters.useCalo      = cms.bool(False)
# process.earlyMuons.TrackAssociatorParameters.usePreShower = cms.bool(False)

# Global Reco for Tracker, including BeamSpot, Vertices, special Muon Tracking
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
process.load("RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi")
# necessary for Run-II
# process.siPixelClusterShapeCachePreSplitting = process.siPixelClusterShapeCache.clone(
#     src = 'siPixelClustersPreSplitting'
# )
process.load("RecoVertex.Configuration.RecoVertex_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
process.load("RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi")

# redefine the process vertexreco
# check modification in RecoVertex/Configuration/python/RecoVertex_cff.py 
# the Run-II configuration has been commented out
# process.vertexreco = cms.Sequence(process.offlinePrimaryVertices*process.offlinePrimaryVerticesWithBS*process.generalV0Candidates*process.inclusiveVertexing)

# process.vertexreco.remove(process.caloTowerForTrk)
# process.vertexreco.remove(process.ak4CaloJetsForTrk)
# process.sortedPrimaryVertices.jets = ""


### Can we do also Muon Global Reco? ###
########################################
process.load("RecoMuon.Configuration.RecoMuon_cff")
# process.load("RecoVertex.Configuration.RecoVertex_cff")
# process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
# process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


# Fix DT and CSC Alignment #
############################
# does this work actually?
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
process = fixDTAlignmentConditions(process)
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
process = fixCSCAlignmentConditions(process)


# Skip Digi2Raw and Raw2Digi steps for Al Muon detectors #
##########################################################
process.gemRecHits.gemDigiLabel = cms.InputTag("simMuonGEMDigis","","GEMDIGI")
process.rpcRecHits.rpcDigiLabel = cms.InputTag('simMuonRPCDigis')
process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
process.dt1DRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")
process.dt1DCosmicRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")


# Explicit configuration of CSC for postls1 = run2 #
####################################################
process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
process.CSCIndexerESProducer.AlgoName = cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName = cms.string("CSCChannelMapperPostls1")
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
# process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
# process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

process.gemRecHits = cms.EDProducer("GEMRecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    # maskSource = cms.string('File'),
    # maskvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMMaskVec.dat'),
    # deadSource = cms.string('File'),
    # deadvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMDeadVec.dat')
)



### Input and Output Files
##########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_digi.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_local_reco.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('localreco_step', 'globalreco_step')
    )
)

### TO ACTIVATE LogTrace one NEEDS TO COMPILE IT WITH:
### -----------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"             
### Make sure that you first cleaned your CMSSW version:       
### --> scram b clean                                          
### before issuing the scram command above                     
### -----------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"
###############################################################
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# process.MessageLogger.categories.append("MuonIdentification")
# process.MessageLogger.categories.append("TrackAssociator")
# process.MessageLogger.debugModules = cms.untracked.vstring("*")
# process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
# process.MessageLogger.cout = cms.untracked.PSet(
#     threshold          = cms.untracked.string("DEBUG"),
#     default            = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
#     FwkReport          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     MuonIdentification = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     TrackAssociator    = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
# )




### Paths and Schedules
#######################
process.digi2raw_step   = cms.Path(process.siPixelRawData+process.SiStripDigiToRaw)
process.raw2digi_step   = cms.Path(process.siPixelDigis+process.siStripDigis) 
#process.localreco_step  = cms.Path(process.muonlocalreco+process.gemRecHits+process.me0RecHits+process.trackerlocalreco)
process.localreco_step  = cms.Path(process.muonlocalreco+process.gemRecHits+process.me0LocalReco+process.trackerlocalreco)

# Run-2 Global Reco Step:
# process.globalreco_step = cms.Path(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*
#                                    process.standalonemuontracking*process.iterTracking)#process.trackingGlobalReco*process.vertexreco)#*process.muonGlobalReco)
# Run-1 Global Reco Step: (no PreSplitting before iterTracking sequence)
process.globalreco_step = cms.Path(process.offlineBeamSpot*process.MeasurementTrackerEvent*process.siPixelClusterShapeCache*process.PixelLayerTriplets*process.recopixelvertexing*
                                   process.standalonemuontracking*process.trackingGlobalReco*process.vertexreco*process.me0MuonReco)#*process.muonGlobalReco)

process.endjob_step     = cms.Path(process.endOfProcess)
process.out_step        = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.digi2raw_step,
    process.raw2digi_step,
    process.localreco_step,
    process.globalreco_step,
    process.endjob_step,
    process.out_step
)
