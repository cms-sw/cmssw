import FWCore.ParameterSet.Config as cms

process = cms.Process("MyMuonRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
# process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')

### Try to do RecoLocalMuon on all muon detectors ###
#####################################################
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
# process.localreco = cms.Sequence(muonlocalreco)

### Try to add also Tracker local reco ###
##########################################
# Digi2Raw and Raw2Digi for Tracker Dets:
process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
process.siStripDigis.ProductLabel = cms.InputTag('SiStripDigiToRaw')

# Local Reco for Tracker
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")
process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
process.load("RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi")
process.siPixelClusterShapeCachePreSplitting = process.siPixelClusterShapeCache.clone(
    src = 'siPixelClustersPreSplitting'
)

# Global Reco for Tracker, including BeamSpot, Vertices, special Muon Tracking
process.load("RecoMuon.Configuration.RecoMuon_cff")
process.load("RecoVertex.Configuration.RecoVertex_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

### Can we do also Muon Global Reco? ###
########################################
# process.load("RecoMuon.Configuration.RecoMuon_cff")
# process.load("RecoVertex.Configuration.RecoVertex_cff")
# process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
# process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Fix DT and CSC Alignment #
############################
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




### Paths and Schedules
#######################
process.digi2raw_step   = cms.Path(process.siPixelRawData+process.SiStripDigiToRaw)
process.raw2digi_step   = cms.Path(process.siPixelDigis+process.siStripDigis) 
process.localreco_step  = cms.Path(process.muonlocalreco+process.gemRecHits+process.me0RecHits+process.trackerlocalreco)


# How to run the local and global tracker reco?
# I tried following these configuration files:
# 
# Configuration/StandardSequences/python/Reconstruction_cff.py
# |--> RecoTracker/Configuration/python/RecoTracker_cff.py
#      |--> RecoTracker/IterativeTracking/python/iterativeTk_cff.py
#           |--> RecoTracker/IterativeTracking/python/InitialStepPreSplitting_cff.py
#           |--> RecoTracker/IterativeTracking/python/InitialStep_cff.py
#           |--> ...

# Info: to factorize process.trackingGlobalReco: 
# trackingGlobalReco = ckftracks * trackExtrapolator
# ckfTracks          = iterTracking * electronSeedsSeq * doAlldEdXEstimators
# Therefore: to remove all calo dependent modules, run only iterTracking
# iterTracking itself contains also calo dependent module, so strip further
# iterTracking = cms.Sequence(InitialStepPreSplitting*
#                             InitialStep*
#                             DetachedTripletStep*
#                             LowPtTripletStep*
#                             PixelPairStep*
#                             MixedTripletStep*
#                             PixelLessStep*
#                             TobTecStep*
#                             JetCoreRegionalStep *
#                             earlyGeneralTracks*
#                             muonSeededStep*
#                             preDuplicateMergingGeneralTracks*
#                             generalTracksSequence*
#                             ConvStep*
#                             conversionStepTracks
#                             )


# Removing Calo Dependent stuff in iterative tracking
# process.InitialStepPreSplitting.remove(process.caloTowerForTrkPreSplitting)
# process.InitialStepPreSplitting.remove(process.ak4CaloJetsForTrkPreSplitting)
# process.InitialStepPreSplitting.remove(process.jetsForCoreTrackingPreSplitting)
# process.InitialStepPreSplitting.remove(process.siPixelClusters)
# process.InitialStepPreSplitting.remove(process.siPixelRecHits)
# process.InitialStepPreSplitting.remove(process.MeasurementTrackerEvent)
# process.InitialStepPreSplitting.remove(process.siPixelClusterShapeCache)

# Original globalreco step for Tracking and Muons only:
# process.globalreco_step = cms.Path(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*
#                                    process.standalonemuontracking*process.trackingGlobalReco*process.vertexreco)#*process.muonGlobalReco)

# My trial to have a non-calo-dependent trackingGlobalReco step:
# process.myTrackingGlobalReco = cms.Sequence(process.InitialStepPreSplitting*process.InitialStep*
#                             process.DetachedTripletStep*process.LowPtTripletStep*process.PixelPairStep*process.MixedTripletStep*
#                             process.PixelLessStep*process.TobTecStep*process.earlyGeneralTracks*
#                             process.muonSeededStep*process.preDuplicateMergingGeneralTracks*process.generalTracksSequence)

process.globalreco_step = cms.Path(process.offlineBeamSpot*process.MeasurementTrackerEventPreSplitting*process.siPixelClusterShapeCachePreSplitting*
                                   process.standalonemuontracking)#*process.myTrackingGlobalReco*process.vertexreco*process.muonGlobalReco)

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

