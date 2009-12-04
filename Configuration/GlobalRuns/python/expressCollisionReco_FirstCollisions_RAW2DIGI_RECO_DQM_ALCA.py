# Auto generated configuration file
# using: 
# $Revision: 1.3 $
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
import FWCore.ParameterSet.Config as cms

process = cms.Process('EXPRESS')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
#process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
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
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/Calo/RAW/v1/000/120/347/FE9E4A7E-90CE-DE11-893E-001D09F2447F.root')
)

# Other statements
process.GlobalTag.globaltag = 'GR09_P_V6::All'


#####################################################################################################
####
####  Top level replaces for handling strange scenarios of early collisions
####

## TRACKING:
process.globalPixelLessSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.5
process.pixelLessCkfTrajectoryFilter = process.ckfBaseTrajectoryFilter.clone(
    ComponentName = 'pixelLessCkfTrajectoryFilter',
    filterPset = process.ckfBaseTrajectoryFilter.filterPset.clone(minPt = 0.5)
    )
process.pixelLessCkfTrajectoryBuilder = process.GroupedCkfTrajectoryBuilder.clone(
    ComponentName = 'pixelLessCkfTrajectoryBuilder',
    trajectoryFilterName = 'pixelLessCkfTrajectoryFilter',
    )
process.ckfTrackCandidatesPixelLess.TrajectoryBuilder = 'pixelLessCkfTrajectoryBuilder'
process.globalPixelLessSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 40
process.globalPixelLessSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 5000
## Skip events with HV off
process.fourthPLSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 20000
process.fifthSeeds.ClusterCheckPSet.MaxNumberOfCosmicClusters = 5000
## Seeding: increase the region
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 100
process.fifthSeeds.RegionFactoryPSet.RegionPSet.originRadius     =  20
## Seeding: add TOB3 to the list, allow unmatched hits
process.fifthlayerpairs.TOB.useSimpleRphiHitsCleaner = cms.bool(False)
process.fifthlayerpairs.TOB.rphiRecHits = cms.InputTag("fifthStripRecHits","rphiRecHitUnmatched")
process.fifthlayerpairs.TOB.stereoRecHits = cms.InputTag("fifthStripRecHits","stereoRecHitUnmatched")
process.fifthlayerpairs.layerList += [ 'TOB1+TOB3', 'TOB2+TOB3' ]
#, 'TOB3+TOB4' ]
## Pattern recognition: lower the cut on the number of hits
process.fifthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 5
process.fifthCkfTrajectoryFilter.filterPset.maxLostHits = 3
process.fifthCkfTrajectoryFilter.filterPset.maxConsecLostHits = 1
process.fifthCkfInOutTrajectoryFilter.filterPset.minimumNumberOfHits = 3
process.fifthCkfInOutTrajectoryFilter.filterPset.maxLostHits = 3
process.fifthCkfInOutTrajectoryFilter.filterPset.maxConsecLostHits = 1
process.fifthCkfTrajectoryBuilder.minNrOfHitsForRebuild = 3
## Pattern recognition: enlarge a lot the search window, as the true momentum is very small while the tracking assumes p=5 GeV if B=0
#process.Chi2MeasurementEstimator.MaxChi2 = 200
process.Chi2MeasurementEstimator.nSigma  = 3
## Fitter-smoother: lower the cut on the number of hits
process.fifthRKTrajectorySmoother.minHits = 4
process.fifthRKTrajectoryFitter.minHits = 4
process.fifthFittingSmootherWithOutlierRejection.MinNumberOfHits = 5
## Fitter-smoother: loosen outlier rejection
process.fifthFittingSmootherWithOutlierRejection.BreakTrajWith2ConsecutiveMissing = False
process.fifthFittingSmootherWithOutlierRejection.EstimateCut = 1000
## Quality filter
process.tobtecStepLoose.minNumberLayers = 3
process.tobtecStepLoose.minNumber3DLayers = 0
process.tobtecStepLoose.maxNumberLostLayers = 4
process.tobtecStepLoose.dz_par1 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.dz_par2 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.d0_par1 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.d0_par2 = cms.vdouble(100.5, 4.0)
process.tobtecStepLoose.chi2n_par = cms.double(10.0)
process.tobtecStepLoose.keepAllTracks = False
process.tobtecStepTight = process.tobtecStepLoose.clone(
    keepAllTracks = True,
    qualityBit = cms.string('tight'),
    src = cms.InputTag("tobtecStepLoose"),
    minNumberLayers = 5,
    minNumber3DLayers = 0
    )
process.tobtecStep = process.tobtecStepLoose.clone(
    keepAllTracks = True,
    qualityBit = cms.string('highPurity'),
    src = cms.InputTag("tobtecStepTight"),
    minNumberLayers = 4,
    minNumber3DLayers = 2,
    )

## PV temporary fixes
process.offlinePrimaryVertices.PVSelParameters.maxDistanceToBeam = 10
process.offlinePrimaryVertices.TkFilterParameters.maxNormalizedChi2 = 500
process.offlinePrimaryVertices.TkFilterParameters.minSiliconHits = 5
process.offlinePrimaryVertices.TkFilterParameters.maxD0Significance = 100
process.offlinePrimaryVertices.TkFilterParameters.minPixelHits = -1
process.offlinePrimaryVertices.TkClusParameters.zSeparation = 10

## ECAL temporary fixes
process.load('RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi')
process.ecalLocalRecoSequence.replace(process.ecalGlobalUncalibRecHit,process.ecalFixedAlphaBetaFitUncalibRecHit)
process.ecalFixedAlphaBetaFitUncalibRecHit.alphaEB = 1.138
process.ecalFixedAlphaBetaFitUncalibRecHit.betaEB = 1.655
process.ecalFixedAlphaBetaFitUncalibRecHit.alphaEE = 1.890
process.ecalFixedAlphaBetaFitUncalibRecHit.betaEE = 1.400
process.ecalRecHit.EBuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalRecHit.EEuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalRecHit.ChannelStatusToBeExcluded = [ 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 78, 142 ]
process.ecalBarrelCosmicTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalEndcapCosmicTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
process.ecalBarrelTimingTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
process.ecalEndcapTimingTask.EcalUncalibratedRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'

## HCAL temporary fixes
process.hfreco.firstSample  = 3
process.hfreco.samplesToAdd = 4

## Beamspot temporary fix
from CondCore.DBCommon.CondDBSetup_cfi import *
process.firstCollBeamspot = cms.ESSource(
    "PoolDBESSource",CondDBSetup,
    connect = cms.string("frontier://PromptProd/CMS_COND_31X_BEAMSPOT"),
    toGet = cms.VPSet(cms.PSet(record = cms.string("BeamSpotObjectsRcd"),
                               tag = cms.string("firstcollisions"))
                      )
    )
process.es_prefer_firstCollBeamspot = cms.ESPrefer("PoolDBESSource","firstCollBeamspot")

###
###  end of top level replacements
###
###############################################################################################



# Path and EndPath definitions

process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_withPixellessTk)
process.dqmoffline_step = cms.Path(process.DQMOffline)

process.pathALCARECOSiStripCalZeroBias = cms.Path(process.seqALCARECOSiStripCalZeroBias*process.ALCARECOSiStripCalZeroBiasDQM)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOMuAlCalIsolatedMu = cms.Path(process.seqALCARECOMuAlCalIsolatedMu*process.ALCARECOMuAlCalIsolatedMuDQM*process.ALCARECODTCalibrationDQM)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias*process.ALCARECOTkAlMinBiasDQM)
process.pathALCARECODQM = cms.Path(process.MEtoEDMConverter)
process.endjob_step = cms.Path(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.pathALCARECOSiStripCalZeroBias,process.pathALCARECORpcCalHLT,process.pathALCARECOMuAlCalIsolatedMu,process.pathALCARECOTkAlMinBias,process.pathALCARECODQM,process.endjob_step)
