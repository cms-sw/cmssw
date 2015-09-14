import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMLocalRECO")

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
# process.load("EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi")
# process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")
# process.load("EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi")
# process.load("EventFilter.SiStripRawToDigi.SiStripDigis_cfi")
# can be skipped use instead:
# SLHCUpgradeSimulations/Geometry/python/recoFromSimDigis_cff.py
# process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
# process.siPixelClusters.src = 'simSiPixelDigis'
# process.siPixelClusters.MissCalibrate = False
# 
# process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
# process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'),
#                                                                      cms.InputTag('simSiStripDigis','ProcessedRaw'),
#                                                                      cms.InputTag('simSiStripDigis','ScopeMode'))
# 
# process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
# process.siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','ZeroSuppressed'),
#                                                           cms.InputTag('siStripZeroSuppression','VirginRaw'),
#                                                           cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
#                                                           cms.InputTag('siStripZeroSuppression','ScopeMode'))
# 
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.siPixelClusters.src = 'mix'
process.siPixelClusters.MissCalibrate = False

process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.siStripZeroSuppression.RawDigiProducersList = cms.VInputTag( cms.InputTag('mix','VirginRaw'),
                                                                     cms.InputTag('mix','ProcessedRaw'),
                                                                     cms.InputTag('mix','ScopeMode'))

process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
process.siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('mix','ZeroSuppressed'),
                                                          cms.InputTag('mix','VirginRaw'),
                                                          cms.InputTag('mix','ProcessedRaw'),
                                                          cms.InputTag('mix','ScopeMode'))

process.load("RecoTracker.Configuration.RecoTracker_cff")
process.load("TrackingTools.Configuration.TrackingTools_cff")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi")
process.load("RecoPixelVertexing.PixelLowPtUtilities.siPixelClusterShapeCache_cfi")
process.siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
    src = 'siPixelClustersPreSplitting'
    )
process.load("RecoLocalTracker.Configuration.RecoLocalTracker_cff")

### Can we do also Muon Global Reco? ###
########################################
process.load("RecoMuon.Configuration.RecoMuon_cff")
process.load("RecoVertex.Configuration.RecoVertex_cff")
process.load("RecoPixelVertexing.Configuration.RecoPixelVertexing_cff")
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")

#????
#process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
#process.load('Geometry.CommonDetUnit.globalTrackingGeometry_cfi')
#process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')
#process.load('Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff')

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
        SelectEvents = cms.vstring('rechit_step')
    )
)

### Paths and Schedules
#######################
# process.digi2raw_step = cms.Path(process.siPixelRawData+process.SiStripDigiToRaw)
# process.raw2digi_step = cms.Path(process.siPixelDigis+process.siStripDigis) 
process.rechit_step   = cms.Path(process.muonlocalreco+process.gemRecHits+process.me0RecHits)#+process.trackerlocalreco)
process.endjob_step   = cms.Path(process.endOfProcess)
process.out_step      = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    # process.digi2raw_step,
    # process.raw2digi_step,
    process.rechit_step,
    process.endjob_step,
    process.out_step
)

