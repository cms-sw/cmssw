# Auto generated configuration file
# using: 
# $Revision: 1.3 $
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
import FWCore.ParameterSet.Config as cms

process = cms.Process('EXPRESS')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/AlCaRecoStreams_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.3 $'),
    annotation = cms.untracked.string('promptReco nevts:-1'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    wantSummary = cms.untracked.bool(True)
)
# Input source
#process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring(
#'/store/data/Commissioning08/Cosmics/RAW/v1/000/069/578/085EFED4-E5AB-DD11-9ACA-001617C3B6FE.root')
#)

process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('/store/streamer/Data/Express/000/124/009/Data.00124009.0023.Express.storageManager.08.0000.dat'),
    cacheSize = cms.untracked.uint32(100000000)
)

#process.source.skipEvents = cms.untracked.uint32(290)

# Other statements
process.GlobalTag.globaltag = 'GR09_P_V8_34X::All'


#####################################################################################################
####
####  Top level replaces for handling strange scenarios of early collisions
####

## TRACKING:
## Skip events with HV off
process.newSeedFromTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
process.newSeedFromPairs.ClusterCheckPSet.MaxNumberOfCosmicClusters=20000
process.secTriplets.ClusterCheckPSet.MaxNumberOfPixelClusters=2000
process.fifthSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 20000
process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters=20000
process.thTripletsA.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000
process.thTripletsB.ClusterCheckPSet.MaxNumberOfPixelClusters = 5000

## local tracker strip reconstruction
process.OutOfTime.TOBlateBP=0.071
process.OutOfTime.TIBlateBP=0.036
        
###### FIXES TRIPLETS FOR LARGE BS DISPLACEMENT ######

### prevent bias in pixel vertex
process.pixelVertices.useBeamConstraint = False

### pixelTracks
#---- new parameters ----
process.pixelTracks.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.06)
process.pixelTracks.RegionFactoryPSet.RegionPSet.originHalfLength = cms.double(40.6)

### 0th step of iterative tracking
#---- new parameters ----
process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ   = cms.double(4.06)
process.newSeedFromTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 40.6

### 2nd step of iterative tracking
#---- new parameters ----
process.secTriplets.RegionFactoryPSet.RegionPSet.nSigmaZ  = cms.double(4.47)
process.secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 44.7

## Primary Vertex
process.offlinePrimaryVerticesWithBS.PVSelParameters.maxDistanceToBeam = 2
process.offlinePrimaryVerticesWithBS.TkFilterParameters.maxNormalizedChi2 = 20
process.offlinePrimaryVerticesWithBS.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVerticesWithBS.TkFilterParameters.minPixelLayersWithHits = 2
process.offlinePrimaryVerticesWithBS.TkFilterParameters.minSiliconLayersWithHits = 5
process.offlinePrimaryVerticesWithBS.TkClusParameters.TkGapClusParameters.zSeparation = 1
process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 2
process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 20
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelLayersWithHits = 2
process.offlinePrimaryVertices.TkFilterParameters.minSiliconLayersWithHits = 5
process.offlinePrimaryVertices.TkClusParameters.TkGapClusParameters.zSeparation = 1

## ECAL 
process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]

##Preshower
process.ecalPreshowerRecHit.ESBaseline = 0

##Preshower algo for data is different than for MC
process.ecalPreshowerRecHit.ESRecoAlgo = 1

## HCAL temporary fixes
process.hfreco.firstSample  = 3
process.hfreco.samplesToAdd = 4

## EGAMMA
process.photons.minSCEtBarrel = 5.
process.photons.minSCEtEndcap =5.
process.photonCore.minSCEt = 5.
process.conversionTrackCandidates.minSCEt = 5.
process.conversions.minSCEt =5.
process.trackerOnlyConversions.rCut = 2.
process.trackerOnlyConversions.vtxChi2 = 0.0005

## scaler beamspot
import RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi
process.offlineBeamSpot = RecoVertex.BeamSpotProducer.BeamSpotOnline_cfi.onlineBeamSpotProducer.clone()

###
###  end of top level replacements
###
###############################################################################################



# Path and EndPath definitions

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.dqmoffline_step = cms.Path(process.DQMOffline)

process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.endjob_step = cms.Path(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.pathALCARECOSiStripCalZeroBias,process.pathALCARECORpcCalHLT,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOTkAlMinBias,process.pathALCARECODQM,process.endjob_step)
