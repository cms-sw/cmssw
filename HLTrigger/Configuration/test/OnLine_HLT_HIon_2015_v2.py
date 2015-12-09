# /frozen/2015/HIon/v2.0/HLT/V1 (CMSSW_7_5_7_patch1)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLTHIon2015v2" )

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/frozen/2015/HIon/v2.0/HLT/V1')
)

process.HLTPSetInitialStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.05 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  minNumberOfHits = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 )
)
process.HLTPSetInitialStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterBase" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 6 ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "" ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  estimator = cms.string( "hltESPInitialStepChi2MeasurementEstimator36" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
process.HLTPSetDetachedStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.075 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 0.601 ),
  minNumberOfHits = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 )
)
process.HLTPSetDetachedStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedStepTrajectoryFilterBase" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 2 ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "" ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
process.HLTPSetPixelPairStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.05 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  minNumberOfHits = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 )
)
process.HLTPSetPixelPairStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilterBase" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 6 ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "" ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  estimator = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
process.HLTPSetMixedStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  seedPairPenalty = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.05 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 0 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHits = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 )
)
process.HLTPSetMixedStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForMixedStep" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMixedStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMixedStepTrajectoryFilterBase" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 3 ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "" ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  estimator = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
process.HLTPSetPixelLessStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 4 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.05 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 0 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  minNumberOfHits = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 )
)
process.HLTPSetPixelLessStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelLessStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelLessStepTrajectoryFilterBase" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 2 ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "" ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 4 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
process.transferSystem = cms.PSet( 
  destinations = cms.vstring( 'Tier0',
    'DQM',
    'ECAL',
    'EventDisplay',
    'Lustre',
    'None' ),
  transferModes = cms.vstring( 'default',
    'test',
    'emulator' ),
  streamA = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'Lustre' )
  ),
  streamCalibration = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamDQM = cms.PSet( 
    default = cms.vstring( 'DQM' ),
    test = cms.vstring( 'DQM',
      'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamDQMCalibration = cms.PSet( 
    default = cms.vstring( 'DQM' ),
    test = cms.vstring( 'DQM',
      'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamEcalCalibration = cms.PSet( 
    default = cms.vstring( 'ECAL' ),
    test = cms.vstring( 'ECAL' ),
    emulator = cms.vstring( 'None' )
  ),
  streamEventDisplay = cms.PSet( 
    default = cms.vstring( 'EventDisplay',
      'Tier0' ),
    test = cms.vstring( 'EventDisplay',
      'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamExpressCosmics = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'Lustre' )
  ),
  streamNanoDST = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamRPCMON = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  streamTrackerCalibration = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'None' )
  ),
  default = cms.PSet( 
    default = cms.vstring( 'Tier0' ),
    test = cms.vstring( 'Lustre' ),
    emulator = cms.vstring( 'Lustre' ),
    streamLookArea = cms.PSet(  )
  ),
  streamLookArea = cms.PSet( 
    default = cms.vstring( 'DQM' ),
    test = cms.vstring( 'DQM',
      'Lustre' ),
    emulator = cms.vstring( 'None' )
  )
)
process.HLTPSetTrajectoryBuilderForGsfElectrons = cms.PSet( 
  propagatorAlong = cms.string( "hltESPFwdElectronPropagator" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTrajectoryFilterForElectrons" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "hltESPBwdElectronPropagator" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator2000" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 90.0 )
)
process.HLTIter4PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTIter3PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTIter2PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.2 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetbJetRegionalTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 8 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetTrajectoryFilterL3 = cms.PSet( 
  minPt = cms.double( 0.5 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 1000000000 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetTrajectoryFilterForElectrons = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minPt = cms.double( 2.0 ),
  minHitsMinPt = cms.int32( -1 ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minimumNumberOfHits = cms.int32( 5 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetMuonCkfTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minimumNumberOfHits = cms.int32( 5 )
)
process.HLTPSetMuTrackJpsiTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 10.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 8 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetMuTrackJpsiEffTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 9 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetCkfTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetCkf3HitTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTIter4PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter4PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
process.HLTIter3PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter3PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTIter2PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter2PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter2ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTIter1PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter1PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter1ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetTrajectoryBuilderForElectrons = cms.PSet( 
  propagatorAlong = cms.string( "hltESPFwdElectronPropagator" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTrajectoryFilterForElectrons" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "hltESPBwdElectronPropagator" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 90.0 )
)
process.HLTPSetMuTrackJpsiTrajectoryBuilder = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuTrackJpsiTrajectoryFilter" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetMuTrackJpsiEffTrajectoryBuilder = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuTrackJpsiEffTrajectoryFilter" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetMuonCkfTrajectoryBuilderSeedHit = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryFilter" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "MuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( True ),
  deltaEta = cms.double( -1.0 ),
  deltaPhi = cms.double( -1.0 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetMuonCkfTrajectoryBuilder = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryFilter" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "MuonCkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  useSeedLayer = cms.bool( False ),
  deltaEta = cms.double( -1.0 ),
  deltaPhi = cms.double( -1.0 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  intermediateCleaning = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetPvClusterComparer = cms.PSet( 
  track_pt_min = cms.double( 2.5 ),
  track_pt_max = cms.double( 10.0 ),
  track_chi2_max = cms.double( 9999999.0 ),
  track_prob_min = cms.double( -1.0 )
)
process.HLTIter0PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTIter0PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetPvClusterComparerForBTag = cms.PSet( 
  track_pt_min = cms.double( 0.1 ),
  track_pt_max = cms.double( 20.0 ),
  track_chi2_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 )
)
process.HLTSeedFromConsecutiveHitsTripletOnlyCreator = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsTripletOnlyCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
process.HLTSeedFromConsecutiveHitsCreator = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterial" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "" )
)
process.HLTIter0HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 4 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTIter2HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter2HighPtTkMuPSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  MeasurementTrackerName = cms.string( "hltIter2HighPtTkMuESPMeasurementTracker" )
)
process.HLTIter2HighPtTkMuPSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 3 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 )
)
process.HLTPSetPvClusterComparerForIT = cms.PSet( 
  track_pt_min = cms.double( 1.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_chi2_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 )
)
process.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
process.HLTSiStripClusterChargeCutLoose = cms.PSet(  value = cms.double( 1620.0 ) )
process.HLTSiStripClusterChargeCutTight = cms.PSet(  value = cms.double( 1945.0 ) )
process.HLTSeedFromConsecutiveHitsCreatorIT = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
process.HLTSeedFromProtoTracks = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
process.HLTPSetMuonTrackingRegionBuilder8356 = cms.PSet( 
  Rescale_eta = cms.double( 3.0 ),
  Rescale_phi = cms.double( 3.0 ),
  Rescale_Dz = cms.double( 3.0 ),
  EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
  EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
  PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
  PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
  UseVertex = cms.bool( False ),
  Pt_fixed = cms.bool( False ),
  Z_fixed = cms.bool( True ),
  Phi_fixed = cms.bool( False ),
  Eta_fixed = cms.bool( False ),
  Pt_min = cms.double( 1.5 ),
  Phi_min = cms.double( 0.1 ),
  Eta_min = cms.double( 0.1 ),
  DeltaZ = cms.double( 15.9 ),
  DeltaR = cms.double( 0.2 ),
  DeltaEta = cms.double( 0.2 ),
  DeltaPhi = cms.double( 0.2 ),
  maxRegions = cms.int32( 2 ),
  precise = cms.bool( True ),
  OnDemand = cms.int32( -1 ),
  MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
  beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
  vertexCollection = cms.InputTag( "pixelVertices" ),
  input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.HLTPSetMuonTrackingRegionBuilder8356ForHI = cms.PSet( 
  Rescale_eta = cms.double( 3.0 ),
  Rescale_phi = cms.double( 3.0 ),
  Rescale_Dz = cms.double( 3.0 ),
  EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
  EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
  PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
  PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
  UseVertex = cms.bool( False ),
  Pt_fixed = cms.bool( False ),
  Z_fixed = cms.bool( True ),
  Phi_fixed = cms.bool( False ),
  Eta_fixed = cms.bool( False ),
  Pt_min = cms.double( 1.5 ),
  Phi_min = cms.double( 0.1 ),
  Eta_min = cms.double( 0.1 ),
  DeltaZ = cms.double( 15.9 ),
  DeltaR = cms.double( 0.2 ),
  DeltaEta = cms.double( 0.2 ),
  DeltaPhi = cms.double( 0.2 ),
  maxRegions = cms.int32( 2 ),
  precise = cms.bool( True ),
  OnDemand = cms.int32( -1 ),
  MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTrackerForHI" ),
  beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
  vertexCollection = cms.InputTag( "pixelVertices" ),
  input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.HLTPSetDetachedCkfTrajectoryBuilderForHI = cms.PSet( 
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryFilterForHI" ) ),
  maxCand = cms.int32( 2 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryFilterForHI" ) ),
  useSameTrajFilter = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 0.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  bestHitOnly = cms.bool( True )
)
process.HLTPSetDetachedCkfTrajectoryFilterForHI = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 6 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 1 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  constantValueForLostHitsFractionFilter = cms.double( 0.701 )
)
process.HLTPSetPixelPairCkfTrajectoryFilterForHI = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  maxNumberOfHits = cms.int32( 100 )
)
process.HLTPSetPixelPairCkfTrajectoryBuilderForHI = cms.PSet( 
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryFilterForHI" ) ),
  maxCand = cms.int32( 3 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9ForHI" ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryFilterForHI" ) ),
  useSameTrajFilter = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 ),
  bestHitOnly = cms.bool( True )
)
process.HLTSiStripClusterChargeCutForHI = cms.PSet(  value = cms.double( 2069.0 ) )
process.HLTPSetDetachedCkfTrajectoryFilterForHIGlobalPt8 = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 6 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 8.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 1 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  constantValueForLostHitsFractionFilter = cms.double( 0.701 )
)
process.HLTPSetDetachedCkfTrajectoryBuilderForHIGlobalPt8 = cms.PSet( 
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryFilterForHIGlobalPt8" ) ),
  maxCand = cms.int32( 2 ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator9" ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryFilterForHIGlobalPt8" ) ),
  useSameTrajFilter = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 0.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  bestHitOnly = cms.bool( True )
)
process.HLTPSetPixelPairCkfTrajectoryFilterForHIGlobalPt8 = cms.PSet( 
  minPt = cms.double( 8.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  maxNumberOfHits = cms.int32( 100 )
)
process.HLTPSetPixelPairCkfTrajectoryBuilderForHIGlobalPt8 = cms.PSet( 
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryFilterForHIGlobalPt8" ) ),
  maxCand = cms.int32( 3 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9ForHI" ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryFilterForHIGlobalPt8" ) ),
  useSameTrajFilter = cms.bool( True ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 ),
  bestHitOnly = cms.bool( True )
)
process.HLTPSetInitialCkfTrajectoryBuilderForHI = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialCkfTrajectoryFilterForHI" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  intermediateCleaning = cms.bool( False ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerForHI" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
process.HLTPSetInitialCkfTrajectoryFilterForHI = cms.PSet( 
  minimumNumberOfHits = cms.int32( 6 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 999 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minPt = cms.double( 0.9 )
)
process.streams = cms.PSet( 
  ALCALUMIPIXELS = cms.vstring( 'AlCaLumiPixels' ),
  ALCAPHISYM = cms.vstring( 'AlCaPhiSym' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcal' ),
  DQM = cms.vstring( 'OnlineMonitor' ),
  DQMCalibration = cms.vstring( 'TestEnablesEcalHcalDQM' ),
  DQMEventDisplay = cms.vstring( 'EventDisplay' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  HIExpress = cms.vstring( 'HIExpressPhysics' ),
  HIPhysicsHardProbes = cms.vstring( 'HIFlowCorr',
    'HIHardProbes',
    'HIHardProbesPeripheral',
    'HIPhoton40AndZ' ),
  HIPhysicsMinBiasUPC = cms.vstring( 'HIForward',
    'HIMinimumBias1',
    'HIMinimumBias2' ),
  HIPhysicsMuons = cms.vstring( 'HIEWQExo',
    'HIOniaCentral30L2L3',
    'HIOniaL1DoubleMu0',
    'HIOniaPeripheral30100',
    'HIOniaTnP' ),
  NanoDST = cms.vstring( 'L1Accept' ),
  RPCMON = cms.vstring( 'RPCMonitor' )
)
process.datasets = cms.PSet( 
  AlCaLumiPixels = cms.vstring( 'AlCa_LumiPixels_Random_v1',
    'AlCa_LumiPixels_ZeroBias_v2' ),
  AlCaPhiSym = cms.vstring( 'AlCa_EcalPhiSymForHI_v2' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration_v2' ),
  EventDisplay = cms.vstring( 'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
    'HLT_HISinglePhoton60_Eta3p1_v2' ),
  HIEWQExo = cms.vstring( 'HLT_HIL1DoubleMu10_v1',
    'HLT_HIL2Mu15_2HF0_v1',
    'HLT_HIL2Mu15_2HF_v1',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu20_2HF0_v1',
    'HLT_HIL2Mu20_2HF_v1',
    'HLT_HIL2Mu20_v1',
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIL3Mu15_2HF0_v1',
    'HLT_HIL3Mu15_2HF_v1',
    'HLT_HIL3Mu15_v1',
    'HLT_HIL3Mu20_2HF0_v1',
    'HLT_HIL3Mu20_2HF_v1',
    'HLT_HIL3Mu20_v1' ),
  HIExpressPhysics = cms.vstring( 'HLT_HICentralityVeto_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
    'HLT_HIFullTrack34_v2',
    'HLT_HIL1DoubleMu10_v1',
    'HLT_HIL1MinimumBiasHF1AND_v1',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
    'HLT_HIL2Mu20_2HF_v1',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton60_Eta1p5_v2',
    'HLT_HISinglePhoton60_Eta3p1_v2',
    'HLT_HIUCC020_v2',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
    'HLT_HIZeroBias_v1' ),
  HIFlowCorr = cms.vstring( 'HLT_HIQ2Bottom005_Centrality1030_v2',
    'HLT_HIQ2Bottom005_Centrality3050_v2',
    'HLT_HIQ2Bottom005_Centrality5070_v2',
    'HLT_HIQ2Top005_Centrality1030_v2',
    'HLT_HIQ2Top005_Centrality3050_v2',
    'HLT_HIQ2Top005_Centrality5070_v2',
    'HLT_HIUCC020_v2',
    'HLT_HIUCC100_v2' ),
  HIForward = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v1',
    'HLT_HIL1CastorMediumJetAK4CaloJet20_v2',
    'HLT_HIL1CastorMediumJet_v1',
    'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleEG2NotHF2_v1',
    'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleEG2NotZDCAND_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHF1_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHF2_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2',
    'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1',
    'HLT_HIUPCL1EG2NotZDCAND_v1',
    'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1MuOpenNotZDCAND_v1',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1',
    'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotMinimumBiasHF2_AND_v1',
    'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotZdcOR_BptxAND_v1',
    'HLT_HIUPCL1SingleEG2NotHF2_v1',
    'HLT_HIUPCL1SingleEG5NotHF2_v1',
    'HLT_HIUPCL1SingleMuOpenNotHF2_v1',
    'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1ZdcOR_BptxAND_v1',
    'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1ZdcXOR_BptxAND_v1',
    'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1' ),
  HIHardProbes = cms.vstring( 'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt50_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt70_v2',
    'HLT_HIFullTrack12_L1Centrality010_v2',
    'HLT_HIFullTrack12_L1Centrality30100_v2',
    'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2',
    'HLT_HIFullTrack18_L1Centrality010_v2',
    'HLT_HIFullTrack18_L1Centrality30100_v2',
    'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2',
    'HLT_HIFullTrack24_L1Centrality30100_v2',
    'HLT_HIFullTrack24_v2',
    'HLT_HIFullTrack34_L1Centrality30100_v2',
    'HLT_HIFullTrack34_v2',
    'HLT_HIFullTrack45_L1Centrality30100_v2',
    'HLT_HIFullTrack45_v2',
    'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2',
    'HLT_HIPuAK4CaloDJet60_Eta2p1_v2',
    'HLT_HIPuAK4CaloDJet80_Eta2p1_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2',
    'HLT_HIPuAK4CaloJet110_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2',
    'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton10_Eta1p5_v2',
    'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton10_Eta3p1_v2',
    'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton15_Eta1p5_v2',
    'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton15_Eta3p1_v2',
    'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton20_Eta1p5_v2',
    'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton20_Eta3p1_v2',
    'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton30_Eta1p5_v2',
    'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2' ),
  HIHardProbesPeripheral = cms.vstring( 'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2',
    'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2' ),
  HIMinimumBias1 = cms.vstring( 'HLT_HICentralityVeto_v1',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1',
    'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1MinimumBiasHF1OR_v1',
    'HLT_HIL1MinimumBiasHF2AND_v1',
    'HLT_HIL1MinimumBiasHF2OR_v1',
    'HLT_HIL1Tech5_BPTX_PlusOnly_v1',
    'HLT_HIL1Tech6_BPTX_MinusOnly_v1',
    'HLT_HIL1Tech7_NoBPTX_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIRandom_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBias_v1' ),
  HIMinimumBias2 = cms.vstring( 'HLT_HIL1MinimumBiasHF1AND_v1' ),
  HIOniaCentral30L2L3 = cms.vstring( 'HLT_HIL1DoubleMu0_Cent30_v1',
    'HLT_HIL2DoubleMu0_Cent30_NHitQ_v1',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
    'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1',
    'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1',
    'HLT_HIL3DoubleMu0_Cent30_v1' ),
  HIOniaL1DoubleMu0 = cms.vstring( 'HLT_HIL1DoubleMu0_2HF0_v1',
    'HLT_HIL1DoubleMu0_2HF_v1',
    'HLT_HIL1DoubleMu0_v1',
    'HLT_HIL2DoubleMu0_NHitQ_2HF0_v1',
    'HLT_HIL2DoubleMu0_NHitQ_2HF_v1',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
    'HLT_HIL3DoubleMu0_OS_m7to14_v1' ),
  HIOniaPeripheral30100 = cms.vstring( 'HLT_HIL1DoubleMu0_2HF0_Cent30100_v1',
    'HLT_HIL1DoubleMu0_2HF_Cent30100_v1',
    'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1',
    'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1' ),
  HIOniaTnP = cms.vstring( 'HLT_HIL2Mu3_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu3_NHitQ10_2HF_v1',
    'HLT_HIL2Mu5_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu5_NHitQ10_2HF_v1',
    'HLT_HIL2Mu7_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu7_NHitQ10_2HF_v1',
    'HLT_HIL3Mu3_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu3_NHitQ15_2HF_v1',
    'HLT_HIL3Mu5_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu5_NHitQ15_2HF_v1',
    'HLT_HIL3Mu7_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu7_NHitQ15_2HF_v1' ),
  HIPhoton40AndZ = cms.vstring( 'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2',
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2',
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2',
    'HLT_HISinglePhoton40_Eta1p5_v2',
    'HLT_HISinglePhoton40_Eta2p1_v2',
    'HLT_HISinglePhoton40_Eta3p1_v2',
    'HLT_HISinglePhoton50_Eta1p5_v2',
    'HLT_HISinglePhoton50_Eta3p1_v2',
    'HLT_HISinglePhoton60_Eta1p5_v2',
    'HLT_HISinglePhoton60_Eta3p1_v2' ),
  L1Accept = cms.vstring( 'DST_Physics_v1' ),
  OnlineMonitor = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v1',
    'HLT_HICentralityVeto_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt50_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt70_v2',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2',
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2',
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2',
    'HLT_HIFullTrack12_L1Centrality010_v2',
    'HLT_HIFullTrack12_L1Centrality30100_v2',
    'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2',
    'HLT_HIFullTrack18_L1Centrality010_v2',
    'HLT_HIFullTrack18_L1Centrality30100_v2',
    'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2',
    'HLT_HIFullTrack24_L1Centrality30100_v2',
    'HLT_HIFullTrack24_v2',
    'HLT_HIFullTrack34_L1Centrality30100_v2',
    'HLT_HIFullTrack34_v2',
    'HLT_HIFullTrack45_L1Centrality30100_v2',
    'HLT_HIFullTrack45_v2',
    'HLT_HIL1CastorMediumJetAK4CaloJet20_v2',
    'HLT_HIL1CastorMediumJet_v1',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1',
    'HLT_HIL1DoubleMu0_2HF0_Cent30100_v1',
    'HLT_HIL1DoubleMu0_2HF0_v1',
    'HLT_HIL1DoubleMu0_2HF_Cent30100_v1',
    'HLT_HIL1DoubleMu0_2HF_v1',
    'HLT_HIL1DoubleMu0_Cent30_v1',
    'HLT_HIL1DoubleMu0_v1',
    'HLT_HIL1DoubleMu10_v1',
    'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1',
    'HLT_HIL1MinimumBiasHF1AND_v1',
    'HLT_HIL1MinimumBiasHF1OR_v1',
    'HLT_HIL1MinimumBiasHF2AND_v1',
    'HLT_HIL1MinimumBiasHF2OR_v1',
    'HLT_HIL1Tech5_BPTX_PlusOnly_v1',
    'HLT_HIL1Tech6_BPTX_MinusOnly_v1',
    'HLT_HIL1Tech7_NoBPTX_v1',
    'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1',
    'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1',
    'HLT_HIL2DoubleMu0_Cent30_NHitQ_v1',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
    'HLT_HIL2DoubleMu0_NHitQ_2HF0_v1',
    'HLT_HIL2DoubleMu0_NHitQ_2HF_v1',
    'HLT_HIL2DoubleMu0_NHitQ_v2',
    'HLT_HIL2Mu15_2HF0_v1',
    'HLT_HIL2Mu15_2HF_v1',
    'HLT_HIL2Mu15_v2',
    'HLT_HIL2Mu20_2HF0_v1',
    'HLT_HIL2Mu20_2HF_v1',
    'HLT_HIL2Mu20_v1',
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
    'HLT_HIL2Mu3_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu3_NHitQ10_2HF_v1',
    'HLT_HIL2Mu5_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu5_NHitQ10_2HF_v1',
    'HLT_HIL2Mu7_NHitQ10_2HF0_v1',
    'HLT_HIL2Mu7_NHitQ10_2HF_v1',
    'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1',
    'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1',
    'HLT_HIL3DoubleMu0_Cent30_v1',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
    'HLT_HIL3DoubleMu0_OS_m7to14_v1',
    'HLT_HIL3Mu15_2HF0_v1',
    'HLT_HIL3Mu15_2HF_v1',
    'HLT_HIL3Mu15_v1',
    'HLT_HIL3Mu20_2HF0_v1',
    'HLT_HIL3Mu20_2HF_v1',
    'HLT_HIL3Mu20_v1',
    'HLT_HIL3Mu3_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu3_NHitQ15_2HF_v1',
    'HLT_HIL3Mu5_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu5_NHitQ15_2HF_v1',
    'HLT_HIL3Mu7_NHitQ15_2HF0_v1',
    'HLT_HIL3Mu7_NHitQ15_2HF_v1',
    'HLT_HIPhysics_v1',
    'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2',
    'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2',
    'HLT_HIPuAK4CaloDJet60_Eta2p1_v2',
    'HLT_HIPuAK4CaloDJet80_Eta2p1_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2',
    'HLT_HIPuAK4CaloJet110_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet40_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet60_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2',
    'HLT_HIPuAK4CaloJet80_Eta5p1_v2',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2',
    'HLT_HIQ2Bottom005_Centrality1030_v2',
    'HLT_HIQ2Bottom005_Centrality3050_v2',
    'HLT_HIQ2Bottom005_Centrality5070_v2',
    'HLT_HIQ2Top005_Centrality1030_v2',
    'HLT_HIQ2Top005_Centrality3050_v2',
    'HLT_HIQ2Top005_Centrality5070_v2',
    'HLT_HIRandom_v1',
    'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton10_Eta1p5_v2',
    'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton10_Eta3p1_v2',
    'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton15_Eta1p5_v2',
    'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton15_Eta3p1_v2',
    'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton20_Eta1p5_v2',
    'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton20_Eta3p1_v2',
    'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton30_Eta1p5_v2',
    'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton30_Eta3p1_v2',
    'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2',
    'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2',
    'HLT_HISinglePhoton40_Eta1p5_v2',
    'HLT_HISinglePhoton40_Eta2p1_v2',
    'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2',
    'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2',
    'HLT_HISinglePhoton40_Eta3p1_v2',
    'HLT_HISinglePhoton50_Eta1p5_v2',
    'HLT_HISinglePhoton50_Eta3p1_v2',
    'HLT_HISinglePhoton60_Eta1p5_v2',
    'HLT_HISinglePhoton60_Eta3p1_v2',
    'HLT_HIUCC020_v2',
    'HLT_HIUCC100_v2',
    'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleEG2NotHF2_v1',
    'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleEG2NotZDCAND_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHF1_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHF2_v1',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2',
    'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1',
    'HLT_HIUPCL1EG2NotZDCAND_v1',
    'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1MuOpenNotZDCAND_v1',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1',
    'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotMinimumBiasHF2_AND_v1',
    'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1NotZdcOR_BptxAND_v1',
    'HLT_HIUPCL1SingleEG2NotHF2_v1',
    'HLT_HIUPCL1SingleEG5NotHF2_v1',
    'HLT_HIUPCL1SingleMuOpenNotHF2_v1',
    'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1ZdcOR_BptxAND_v1',
    'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1',
    'HLT_HIUPCL1ZdcXOR_BptxAND_v1',
    'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1',
    'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1',
    'HLT_HIZeroBiasPixel_SingleTrack_v1',
    'HLT_HIZeroBias_v1' ),
  RPCMonitor = cms.vstring( 'AlCa_RPCMuonNoHitsForHI_v1',
    'AlCa_RPCMuonNoTriggersForHI_v1',
    'AlCa_RPCMuonNormalisationForHI_v1' ),
  TestEnablesEcalHcal = cms.vstring( 'HLT_EcalCalibration_v2',
    'HLT_HcalCalibration_v1' ),
  TestEnablesEcalHcalDQM = cms.vstring( 'HLT_EcalCalibration_v2',
    'HLT_HcalCalibration_v1' )
)

process.CSCChannelMapperESSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "CSCChannelMapperRecord" ),
    firstValid = cms.vuint32( 1 )
)
process.CSCINdexerESSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "CSCIndexerRecord" ),
    firstValid = cms.vuint32( 1 )
)
process.GlobalTag = cms.ESSource( "PoolDBESSource",
    snapshotTime = cms.string( "" ),
    globaltag = cms.string( "74X_dataRun2_HLT_v1" ),
    RefreshEachRun = cms.untracked.bool( True ),
    dbFormat = cms.untracked.int32( 0 ),
    toGet = cms.VPSet( 
    ),
    DBParameters = cms.PSet( 
      authenticationPath = cms.untracked.string( "." ),
      connectionRetrialTimeOut = cms.untracked.int32( 60 ),
      idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
      messageLevel = cms.untracked.int32( 0 ),
      enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
      enableConnectionSharing = cms.untracked.bool( True ),
      enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
      connectionTimeOut = cms.untracked.int32( 0 ),
      connectionRetrialPeriod = cms.untracked.int32( 10 )
    ),
    RefreshAlways = cms.untracked.bool( False ),
    connect = cms.string( "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_CONDITIONS" ),
    ReconnectEachRun = cms.untracked.bool( True ),
    RefreshOpenIOVs = cms.untracked.bool( False ),
    DumpStat = cms.untracked.bool( False )
)
process.HepPDTESSource = cms.ESSource( "HepPDTESSource",
    pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" )
)
process.eegeom = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalMappingRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
    fromDDD = cms.untracked.bool( False ),
    toGet = cms.untracked.vstring( 'GainWidths' )
)
process.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "JetTagComputerRecord" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    firstValid = cms.vuint32( 1 )
)

process.hltESPMixedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.PropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStep" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( 0.05 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPPixelPairTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelPairTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPPixelPairStepChi2MeasurementEstimator25 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 25.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" )
)
process.hltESPInitialStepChi2MeasurementEstimator36 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 36.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2MeasurementEstimator36" )
)
process.hltESPDetachedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.OppositePropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( 0.05 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.SimpleSecondaryVertex3TrkComputer = cms.ESProducer( "SimpleSecondaryVertexESProducer",
  minTracks = cms.uint32( 3 ),
  minVertices = cms.uint32( 1 ),
  use3d = cms.bool( True ),
  unBoost = cms.bool( False ),
  useSignificance = cms.bool( True )
)
process.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
)
process.hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
process.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKTrajectorySmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  Fitter = cms.string( "hltESPRKTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPRKTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  DoLorentz = cms.bool( True ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)
process.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFlexibleKFFittingSmoother" ),
  standardFitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  looperFitter = cms.string( "hltESPKFFittingSmootherForLoopers" )
)
process.PropagatorWithMaterialForLoopers = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForLoopers" ),
  Mass = cms.double( 0.1396 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 4.0 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPKFTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPKFFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForLoopers" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPChi2ChargeMeasurementEstimator9ForHI = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9ForHI" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutForHI" ) ),
  nSigma = cms.double( 3.0 )
)
process.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)
process.CSCChannelMapperESProducer = cms.ESProducer( "CSCChannelMapperESProducer",
  AlgoName = cms.string( "CSCChannelMapperPostls1" )
)
process.CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
  useRealWireGeometry = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  alignmentsLabel = cms.string( "" ),
  useGangedStripsInME1a = cms.bool( False ),
  debugV = cms.untracked.bool( False ),
  useOnlyWiresInME1a = cms.bool( False ),
  useDDD = cms.bool( False ),
  useCentreTIOffsets = cms.bool( False ),
  applyAlignment = cms.bool( True )
)
process.CSCIndexerESProducer = cms.ESProducer( "CSCIndexerESProducer",
  AlgoName = cms.string( "CSCIndexerPostls1" )
)
process.CSCObjectMapESProducer = cms.ESProducer( "CSCObjectMapESProducer",
  appendToDataLabel = cms.string( "" )
)
process.CaloGeometryBuilder = cms.ESProducer( "CaloGeometryBuilder",
  SelectedCalos = cms.vstring( 'HCAL',
    'ZDC',
    'EcalBarrel',
    'EcalEndcap',
    'EcalPreshower',
    'TOWER' )
)
process.CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
process.CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  appendToDataLabel = cms.string( "" ),
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
process.CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  applyAlignment = cms.bool( False ),
  hcalTopologyConstants = cms.PSet( 
    maxDepthHE = cms.int32( 3 ),
    maxDepthHB = cms.int32( 2 ),
    mode = cms.string( "HcalTopologyMode::LHC" )
  )
)
process.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par" )
)
process.DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.DTObjectMapESProducer = cms.ESProducer( "DTObjectMapESProducer",
  appendToDataLabel = cms.string( "" )
)
process.EcalBarrelGeometryFromDBEP = cms.ESProducer( "EcalBarrelGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalElectronicsMappingBuilder = cms.ESProducer( "EcalElectronicsMappingBuilder" )
process.EcalEndcapGeometryFromDBEP = cms.ESProducer( "EcalEndcapGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService" )
process.EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  applyAlignment = cms.bool( False ),
  hcalTopologyConstants = cms.PSet( 
    maxDepthHE = cms.int32( 3 ),
    maxDepthHB = cms.int32( 2 ),
    mode = cms.string( "HcalTopologyMode::LHC" )
  )
)
process.HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP",
  Exclude = cms.untracked.string( "" ),
  appendToDataLabel = cms.string( "" ),
  hcalTopologyConstants = cms.PSet( 
    maxDepthHE = cms.int32( 3 ),
    maxDepthHB = cms.int32( 2 ),
    mode = cms.string( "HcalTopologyMode::LHC" )
  )
)
process.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.MaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.ParametrizedMagneticFieldProducer = cms.ESProducer( "AutoParametrizedMagneticFieldProducer",
  version = cms.string( "Parabolic" ),
  valueOverride = cms.int32( -1 ),
  label = cms.untracked.string( "ParabolicMf" )
)
process.RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  useDDD = cms.untracked.bool( False ),
  compatibiltyWith11 = cms.untracked.bool( True )
)
process.SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  printDebug = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" ),
  APVGain = cms.VPSet( 
    cms.PSet(  Record = cms.string( "SiStripApvGainRcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    ),
    cms.PSet(  Record = cms.string( "SiStripApvGain2Rcd" ),
      NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" )
    )
  ),
  AutomaticNormalization = cms.bool( False )
)
process.SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
  PrintDebugOutput = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 ),
  UseEmptyRunInfo = cms.bool( False ),
  ReduceGranularity = cms.bool( False ),
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiStripDetVOffRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripDetCablingRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadChannelRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadFiberRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiStripBadModuleRcd" ),
      tag = cms.string( "" )
    )
  )
)
process.SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  PreFilter = cms.bool( False ),
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 )
)
process.SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
process.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "SteppingHelixPropagatorAny" )
)
process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False )
)
process.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
process.VolumeBasedMagneticFieldESProducer = cms.ESProducer( "VolumeBasedMagneticFieldESProducerFromDB",
  debugBuilder = cms.untracked.bool( False ),
  valueOverride = cms.int32( -1 ),
  label = cms.untracked.string( "" )
)
process.ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" ),
  SimpleMagneticField = cms.string( "" )
)
process.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
process.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
  dbstatusMask = cms.PSet( 
    kGood = cms.vstring( 'kOk' ),
    kProblematic = cms.vstring( 'kDAC',
      'kNoLaser',
      'kNoisy',
      'kNNoisy',
      'kNNNoisy',
      'kNNNNoisy',
      'kNNNNNoisy',
      'kFixedG6',
      'kFixedG1',
      'kFixedG0' ),
    kRecovered = cms.vstring(  ),
    kTime = cms.vstring(  ),
    kWeird = cms.vstring(  ),
    kBad = cms.vstring( 'kNonRespondingIsolated',
      'kDeadVFE',
      'kDeadFE',
      'kNoDataNoTP' )
  ),
  timeThresh = cms.double( 2.0 ),
  flagMask = cms.PSet( 
    kGood = cms.vstring( 'kGood' ),
    kProblematic = cms.vstring( 'kPoorReco',
      'kPoorCalib',
      'kNoisy',
      'kSaturated' ),
    kRecovered = cms.vstring( 'kLeadingEdgeRecovered',
      'kTowerRecovered' ),
    kTime = cms.vstring( 'kOutOfTime' ),
    kWeird = cms.vstring( 'kWeird',
      'kDiWeird' ),
    kBad = cms.vstring( 'kFaultyHardware',
      'kDead',
      'kKilled' )
  )
)
process.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  RecoveredRecHitBits = cms.vstring( 'TimingAddedBit',
    'TimingSubtractedBit' ),
  SeverityLevels = cms.VPSet( 
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' ),
      Level = cms.int32( 1 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HSCP_R1R2',
  'HSCP_FracLeader',
  'HSCP_OuterEnergy',
  'HSCP_ExpFit',
  'ADCSaturationBit',
  'HBHEIsolatedNoise',
  'AddedSimHcalNoise' ),
      ChannelStatus = cms.vstring( 'HcalCellExcludeFromHBHENoiseSummary' ),
      Level = cms.int32( 5 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
  'HBHEPulseShape',
  'HOBit',
  'HFInTimeWindow',
  'ZDCBit',
  'CalibrationBit',
  'TimingErrorBit',
  'HBHETriangleNoise',
  'HBHETS4TS5Noise' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 8 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HFLongShort',
  'HFPET',
  'HFS8S1Ratio',
  'HFDigiTime' ),
      ChannelStatus = cms.vstring(  ),
      Level = cms.int32( 11 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring( 'HBHEFlatNoise',
  'HBHESpikeNoise' ),
      ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' ),
      Level = cms.int32( 12 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellHot' ),
      Level = cms.int32( 15 )
    ),
    cms.PSet(  RecHitFlags = cms.vstring(  ),
      ChannelStatus = cms.vstring( 'HcalCellOff',
        'HcalCellDead' ),
      Level = cms.int32( 20 )
    )
  ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' )
)
process.hcal_db_producer = cms.ESProducer( "HcalDbProducer" )
process.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  categoryVariableName = cms.string( "vertexCategory" ),
  useTrackWeights = cms.bool( True ),
  useCategories = cms.bool( True ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  trackSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( -99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  calibrationRecords = cms.vstring( 'CombinedSVRecoVertex',
    'CombinedSVPseudoVertex',
    'CombinedSVNoVertex' ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  charmCut = cms.double( 1.5 ),
  vertexFlip = cms.bool( False ),
  minimumTrackWeight = cms.double( 0.5 ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackMultiplicityMin = cms.uint32( 3 ),
  trackPseudoSelection = cms.PSet( 
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    maxDistToAxis = cms.double( 0.07 ),
    sip2dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( 2.0 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  trackSort = cms.string( "sip2dSig" ),
  trackFlip = cms.bool( False )
)
process.hltCombinedSecondaryVertexV2 = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  categoryVariableName = cms.string( "vertexCategory" ),
  useTrackWeights = cms.bool( True ),
  useCategories = cms.bool( True ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  trackSelection = cms.PSet( 
    b_pT = cms.double( 0.3684 ),
    max_pT = cms.double( 500.0 ),
    useVariableJTA = cms.bool( False ),
    maxDecayLen = cms.double( 5.0 ),
    sip3dValMin = cms.double( -99999.9 ),
    max_pT_dRcut = cms.double( 0.1 ),
    a_pT = cms.double( 0.005263 ),
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    a_dR = cms.double( -0.001053 ),
    maxDistToAxis = cms.double( 0.07 ),
    ptMin = cms.double( 0.0 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip2dValMax = cms.double( 99999.9 ),
    max_pT_trackPTcut = cms.double( 3.0 ),
    sip2dValMin = cms.double( -99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip3dSigMin = cms.double( -99999.9 ),
    min_pT = cms.double( 120.0 ),
    min_pT_dRcut = cms.double( 0.5 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( -99999.9 ),
    b_dR = cms.double( 0.6263 )
  ),
  calibrationRecords = cms.vstring( 'CombinedSVIVFV2RecoVertex',
    'CombinedSVIVFV2PseudoVertex',
    'CombinedSVIVFV2NoVertex' ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  charmCut = cms.double( 1.5 ),
  vertexFlip = cms.bool( False ),
  minimumTrackWeight = cms.double( 0.5 ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackMultiplicityMin = cms.uint32( 3 ),
  trackPseudoSelection = cms.PSet( 
    b_pT = cms.double( 0.3684 ),
    max_pT = cms.double( 500.0 ),
    useVariableJTA = cms.bool( False ),
    maxDecayLen = cms.double( 5.0 ),
    sip3dValMin = cms.double( -99999.9 ),
    max_pT_dRcut = cms.double( 0.1 ),
    a_pT = cms.double( 0.005263 ),
    totalHitsMin = cms.uint32( 0 ),
    jetDeltaRMax = cms.double( 0.3 ),
    a_dR = cms.double( -0.001053 ),
    maxDistToAxis = cms.double( 0.07 ),
    ptMin = cms.double( 0.0 ),
    qualityClass = cms.string( "any" ),
    pixelHitsMin = cms.uint32( 0 ),
    sip2dValMax = cms.double( 99999.9 ),
    max_pT_trackPTcut = cms.double( 3.0 ),
    sip2dValMin = cms.double( -99999.9 ),
    normChi2Max = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    sip3dSigMin = cms.double( -99999.9 ),
    min_pT = cms.double( 120.0 ),
    min_pT_dRcut = cms.double( 0.5 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip2dSigMin = cms.double( 2.0 ),
    b_dR = cms.double( 0.6263 )
  ),
  trackSort = cms.string( "sip2dSig" ),
  trackFlip = cms.bool( False )
)
process.hltDisplacedDijethltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  maxImpactParameterSig = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.1 ),
  useSignedImpactParameterSig = cms.bool( True ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  nthTrack = cms.int32( -1 )
)
process.hltDisplacedDijethltESPTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
  b_pT = cms.double( 0.3684 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( 0.05 ),
  a_dR = cms.double( -0.001053 ),
  min_pT = cms.double( 120.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  max_pT = cms.double( 500.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  max_pT_dRcut = cms.double( 0.1 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  maximumDecayLength = cms.double( 999999.0 ),
  nthTrack = cms.int32( 1 ),
  useSignedImpactParameterSig = cms.bool( False )
)
process.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
process.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)
process.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  nSigma = cms.double( 3.0 )
)
process.hltESPChi2ChargeMeasurementEstimator2000 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 2000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator2000" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  nSigma = cms.double( 3.0 )
)
process.hltESPChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  nSigma = cms.double( 3.0 )
)
process.hltESPChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  nSigma = cms.double( 3.0 )
)
process.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" )
)
process.hltESPChi2MeasurementEstimator30 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator30" )
)
process.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" )
)
process.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" )
)
process.hltESPDisplacedDijethltPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  maxImpactParameterSig = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.1 ),
  useSignedImpactParameterSig = cms.bool( True ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  nthTrack = cms.int32( -1 )
)
process.hltESPDisplacedDijethltPromptTrackCountingESProducerLong = cms.ESProducer( "PromptTrackCountingESProducer",
  maxImpactParameterSig = cms.double( 999999.0 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.2 ),
  useSignedImpactParameterSig = cms.bool( True ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  nthTrack = cms.int32( -1 )
)
process.hltESPDisplacedDijethltTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
  b_pT = cms.double( 0.3684 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( 0.05 ),
  a_dR = cms.double( -0.001053 ),
  min_pT = cms.double( 120.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  max_pT = cms.double( 500.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  max_pT_dRcut = cms.double( 0.1 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  maximumDecayLength = cms.double( 999999.0 ),
  nthTrack = cms.int32( 1 ),
  useSignedImpactParameterSig = cms.bool( False )
)
process.hltESPDisplacedDijethltTrackCounting2D2ndLong = cms.ESProducer( "TrackCountingESProducer",
  b_pT = cms.double( 0.3684 ),
  deltaR = cms.double( -1.0 ),
  minimumImpactParameter = cms.double( 0.2 ),
  a_dR = cms.double( -0.001053 ),
  min_pT = cms.double( 120.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  max_pT = cms.double( 500.0 ),
  impactParameterType = cms.int32( 1 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  max_pT_dRcut = cms.double( 0.1 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  maximumDecayLength = cms.double( 999999.0 ),
  nthTrack = cms.int32( 2 ),
  useSignedImpactParameterSig = cms.bool( True )
)
process.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPEcalTrigTowerConstituentsMapBuilder = cms.ESProducer( "EcalTrigTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/EcalMapping/data/EndCap_TTMap.txt" )
)
process.hltESPElectronMaterialEffects = cms.ESProducer( "GsfMaterialEffectsESProducer",
  BetheHeitlerParametrization = cms.string( "BetheHeitler_cdfmom_nC6_O5.par" ),
  EnergyLossUpdator = cms.string( "GsfBetheHeitlerUpdator" ),
  ComponentName = cms.string( "hltESPElectronMaterialEffects" ),
  MultipleScatteringUpdator = cms.string( "MultipleScatteringUpdator" ),
  Mass = cms.double( 5.11E-4 ),
  BetheHeitlerCorrection = cms.int32( 2 )
)
process.hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorAny" )
)
process.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" )
)
process.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
process.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" )
)
process.hltESPGlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
process.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPGsfTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPGsfTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  ComponentName = cms.string( "hltESPGsfElectronFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  RejectTracks = cms.bool( True )
)
process.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" )
)
process.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ErrorRescaling = cms.double( 100.0 ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" )
)
process.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  RejectTracks = cms.bool( True )
)
process.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" )
)
process.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  DistanceMeasure = cms.string( "KullbackLeibler" )
)
process.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  UseStripStripQualityDB = cms.bool( True ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TID = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 9999 ),
      maxBad = cms.uint32( 9999 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  SiStripQualityLabel = cms.string( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True )
)
process.hltESPMeasurementTrackerForHI = cms.ESProducer( "MeasurementTrackerESProducer",
  UseStripStripQualityDB = cms.bool( True ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TID = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TEC = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    ),
    TIB = cms.PSet( 
      maxConsecutiveBad = cms.uint32( 2 ),
      maxBad = cms.uint32( 4 )
    )
  ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  ComponentName = cms.string( "hltESPMeasurementTrackerForHI" ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  UsePixelModuleQualityDB = cms.bool( True ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  HitMatcher = cms.string( "StandardMatcher" ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  SiStripQualityLabel = cms.string( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  MaskBadAPVFibers = cms.bool( True )
)
process.hltESPMuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
process.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
)
process.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
  useLAAlignmentOffsets = cms.bool( False ),
  DoCosmics = cms.bool( False ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  size_cutY = cms.double( 3.0 ),
  size_cutX = cms.double( 3.0 ),
  useLAWidthFromDB = cms.bool( False ),
  inflate_errors = cms.bool( False ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  IrradiationBiasCorrection = cms.bool( False )
)
process.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
process.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagator" )
)
process.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAny" )
)
process.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" )
)
process.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
process.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorAlong" )
)
process.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  NoErrorPropagation = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  returnTangentPlane = cms.bool( True ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
)
process.hltESPStripCPEfromTrackAngle = cms.ESProducer( "StripCPEESProducer",
  ComponentType = cms.string( "StripCPEfromTrackAngle" ),
  ComponentName = cms.string( "hltESPStripCPEfromTrackAngle" ),
  parameters = cms.PSet( 
    mLC_P2 = cms.double( 0.3 ),
    mLC_P1 = cms.double( 0.618 ),
    mLC_P0 = cms.double( -0.326 ),
    useLegacyError = cms.bool( False ),
    mTEC_P1 = cms.double( 0.471 ),
    mTEC_P0 = cms.double( -1.885 ),
    mTOB_P0 = cms.double( -1.026 ),
    mTOB_P1 = cms.double( 0.253 ),
    mTIB_P0 = cms.double( -0.742 ),
    mTIB_P1 = cms.double( 0.202 ),
    mTID_P0 = cms.double( -1.427 ),
    mTID_P1 = cms.double( 0.433 ),
    maxChgOneMIP = cms.double( 6000.0 )
  )
)
process.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)
process.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" )
)
process.hltESPTrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  appendToDataLabel = cms.string( "" ),
  trackerGeometryLabel = cms.untracked.string( "" )
)
process.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
process.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
process.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
process.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
process.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  SimpleMagneticField = cms.string( "ParabolicMf" )
)
process.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
process.siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
process.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer( "SiStripBackPlaneCorrectionDepESProducer",
  LatencyRecord = cms.PSet( 
    record = cms.string( "SiStripLatencyRcd" ),
    label = cms.untracked.string( "" )
  ),
  BackPlaneCorrectionDeconvMode = cms.PSet( 
    record = cms.string( "SiStripBackPlaneCorrectionRcd" ),
    label = cms.untracked.string( "deconvolution" )
  ),
  BackPlaneCorrectionPeakMode = cms.PSet( 
    record = cms.string( "SiStripBackPlaneCorrectionRcd" ),
    label = cms.untracked.string( "peak" )
  )
)
process.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
  LatencyRecord = cms.PSet( 
    record = cms.string( "SiStripLatencyRcd" ),
    label = cms.untracked.string( "" )
  ),
  LorentzAngleDeconvMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "deconvolution" )
  ),
  LorentzAnglePeakMode = cms.PSet( 
    record = cms.string( "SiStripLorentzAngleRcd" ),
    label = cms.untracked.string( "peak" )
  )
)
process.sistripconn = cms.ESProducer( "SiStripConnectivity" )
process.trackerTopology = cms.ESProducer( "TrackerTopologyEP",
  appendToDataLabel = cms.string( "" )
)

process.FastTimerService = cms.Service( "FastTimerService",
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    useRealTimeClock = cms.untracked.bool( True ),
    enableTimingModules = cms.untracked.bool( True ),
    enableDQM = cms.untracked.bool( True ),
    enableDQMbyModule = cms.untracked.bool( False ),
    enableTimingExclusive = cms.untracked.bool( True ),
    skipFirstPath = cms.untracked.bool( False ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmTimeRange = cms.untracked.double( 1000.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    enableDQMSummary = cms.untracked.bool( True ),
    enableTimingSummary = cms.untracked.bool( True ),
    enableDQMbyPathTotal = cms.untracked.bool( True ),
    enableTimingPaths = cms.untracked.bool( True ),
    enableDQMbyPathExclusive = cms.untracked.bool( False ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    enableDQMbyPathActive = cms.untracked.bool( False ),
    enableDQMbyPathDetails = cms.untracked.bool( False ),
    enableDQMbyPathOverhead = cms.untracked.bool( False ),
    enableDQMbyPathCounters = cms.untracked.bool( True ),
    enableDQMbyModuleType = cms.untracked.bool( False )
)
process.MessageLogger = cms.Service( "MessageLogger",
    suppressInfo = cms.untracked.vstring(  ),
    debugs = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    suppressDebug = cms.untracked.vstring(  ),
    cout = cms.untracked.PSet(  placeholder = cms.untracked.bool( True ) ),
    cerr_stats = cms.untracked.PSet( 
      threshold = cms.untracked.string( "WARNING" ),
      output = cms.untracked.string( "cerr" ),
      optionalPSet = cms.untracked.bool( True )
    ),
    warnings = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    statistics = cms.untracked.vstring( 'cerr' ),
    cerr = cms.untracked.PSet( 
      INFO = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      noTimeStamps = cms.untracked.bool( False ),
      FwkReport = cms.untracked.PSet( 
        reportEvery = cms.untracked.int32( 1 ),
        limit = cms.untracked.int32( 0 )
      ),
      default = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) ),
      Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkSummary = cms.untracked.PSet( 
        reportEvery = cms.untracked.int32( 1 ),
        limit = cms.untracked.int32( 10000000 )
      ),
      threshold = cms.untracked.string( "INFO" ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    FrameworkJobReport = cms.untracked.PSet( 
      default = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      FwkJob = cms.untracked.PSet(  limit = cms.untracked.int32( 10000000 ) )
    ),
    suppressWarning = cms.untracked.vstring( 'hltOnlineBeamSpot',
      'hltCtf3HitL1SeededWithMaterialTracks',
      'hltL3MuonsOIState',
      'hltPixelTracksForHighMult',
      'hltHITPixelTracksHE',
      'hltHITPixelTracksHB',
      'hltCtfL1SeededWithMaterialTracks',
      'hltRegionalTracksForL3MuonIsolation',
      'hltSiPixelClusters',
      'hltActivityStartUpElectronPixelSeeds',
      'hltLightPFTracks',
      'hltPixelVertices3DbbPhi',
      'hltL3MuonsIOHit',
      'hltPixelTracks',
      'hltSiPixelDigis',
      'hltL3MuonsOIHit',
      'hltL1SeededElectronGsfTracks',
      'hltL1SeededStartUpElectronPixelSeeds',
      'hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV',
      'hltCtfActivityWithMaterialTracks' ),
    errors = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    fwkJobReports = cms.untracked.vstring( 'FrameworkJobReport' ),
    debugModules = cms.untracked.vstring(  ),
    infos = cms.untracked.PSet( 
      threshold = cms.untracked.string( "INFO" ),
      Root_NoDictionary = cms.untracked.PSet(  limit = cms.untracked.int32( 0 ) ),
      placeholder = cms.untracked.bool( True ),
      suppressInfo = cms.untracked.vstring(  ),
      suppressWarning = cms.untracked.vstring(  ),
      suppressDebug = cms.untracked.vstring(  ),
      suppressError = cms.untracked.vstring(  )
    ),
    categories = cms.untracked.vstring( 'FwkJob',
      'FwkReport',
      'FwkSummary',
      'Root_NoDictionary' ),
    destinations = cms.untracked.vstring( 'warnings',
      'errors',
      'infos',
      'debugs',
      'cout',
      'cerr' ),
    threshold = cms.untracked.string( "INFO" ),
    suppressError = cms.untracked.vstring( 'hltOnlineBeamSpot',
      'hltL3MuonCandidates',
      'hltL3TkTracksFromL2OIState',
      'hltPFJetCtfWithMaterialTracks',
      'hltL3TkTracksFromL2IOHit',
      'hltL3TkTracksFromL2OIHit' )
)

process.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
)
process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32( 813 ),
    Verbosity = cms.untracked.int32( 0 ),
    UnpackBxInEvent = cms.int32( 5 ),
    ActiveBoardsMask = cms.uint32( 0xffff ),
    DaqGtInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltCaloStage1Digis = cms.EDProducer( "L1TRawToDigi",
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    CTP7 = cms.untracked.bool( False ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    Setup = cms.string( "stage1::CaloSetup" ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    FWId = cms.uint32( 4294967295 ),
    debug = cms.untracked.bool( False ),
    FedIds = cms.vint32( 1352 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    FWOverride = cms.bool( False )
)
process.hltCaloStage1LegacyFormatDigis = cms.EDProducer( "L1TCaloUpgradeToGCTConverter",
    InputHFCountsCollection = cms.InputTag( 'hltCaloStage1Digis','HFBitCounts' ),
    InputHFSumsCollection = cms.InputTag( 'hltCaloStage1Digis','HFRingSums' ),
    bxMin = cms.int32( 0 ),
    bxMax = cms.int32( 0 ),
    InputCollection = cms.InputTag( "hltCaloStage1Digis" ),
    InputIsoTauCollection = cms.InputTag( 'hltCaloStage1Digis','isoTaus' ),
    InputRlxTauCollection = cms.InputTag( 'hltCaloStage1Digis','rlxTaus' )
)
process.hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    TechnicalTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( False ),
    EmulateBxInEvent = cms.int32( 1 ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    RecordLength = cms.vint32( 3, 0 ),
    TechnicalTriggersUnmasked = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    TechnicalTriggersVetoUnmasked = cms.bool( True ),
    AlternativeNrBxBoardEvm = cms.uint32( 0 ),
    TechnicalTriggersInputTags = cms.VInputTag( 'simBscDigis' ),
    CastorInputTag = cms.InputTag( "castorL1Digis" ),
    GctInputTag = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    BstLengthBytes = cms.int32( -1 )
)
process.hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    tauJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','tauJets' ),
    etHadSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    isoTauJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','isoTauJets' ),
    etTotalSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    centralBxOnly = cms.bool( True ),
    centralJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','cenJets' ),
    etMissSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    hfRingEtSumsSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    produceMuonParticles = cms.bool( True ),
    forwardJetSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','forJets' ),
    ignoreHtMiss = cms.bool( False ),
    htMissSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" ),
    produceCaloParticles = cms.bool( True ),
    muonSource = cms.InputTag( "hltGtDigis" ),
    isolatedEmSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','isoEm' ),
    nonIsolatedEmSource = cms.InputTag( 'hltCaloStage1LegacyFormatDigis','nonIsoEm' ),
    hfRingBitCountsSource = cms.InputTag( "hltCaloStage1LegacyFormatDigis" )
)
process.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    maxZ = cms.double( 40.0 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    changeToCMSCoordinates = cms.bool( False ),
    setSigmaZ = cms.double( 0.0 ),
    maxRadius = cms.double( 2.0 )
)
process.hltPreDSTPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1sL1MinimumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet40Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalDigis = cms.EDProducer( "EcalRawToDigi",
    orderedDCCIdList = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 ),
    FedLabel = cms.InputTag( "listfeds" ),
    eventPut = cms.bool( True ),
    srpUnpacking = cms.bool( True ),
    syncCheck = cms.bool( True ),
    headerUnpacking = cms.bool( True ),
    feUnpacking = cms.bool( True ),
    orderedFedList = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    tccUnpacking = cms.bool( True ),
    numbTriggerTSamples = cms.int32( 1 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    numbXtalTSamples = cms.int32( 10 ),
    feIdCheck = cms.bool( True ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    silentMode = cms.untracked.bool( True ),
    DoRegional = cms.bool( False ),
    forceToKeepFRData = cms.bool( False ),
    memUnpacking = cms.bool( True )
)
process.hltEcalUncalibRecHit50nsMultiFit = cms.EDProducer( "EcalUncalibRecHitProducer",
    EEdigiCollection = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    EBdigiCollection = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    algo = cms.string( "EcalUncalibRecHitWorkerMultiFit" ),
    algoPSet = cms.PSet( 
      outOfTimeThresholdGain61pEB = cms.double( 5.0 ),
      EBtimeFitParameters = cms.vdouble( -2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621 ),
      activeBXs = cms.vint32( -4, -2, 0, 2, 4 ),
      amplitudeThresholdEE = cms.double( 10.0 ),
      EBtimeConstantTerm = cms.double( 0.6 ),
      EEtimeFitLimits_Lower = cms.double( 0.2 ),
      outOfTimeThresholdGain61pEE = cms.double( 1000.0 ),
      ebSpikeThreshold = cms.double( 1.042 ),
      EBtimeNconst = cms.double( 28.5 ),
      ampErrorCalculation = cms.bool( False ),
      kPoorRecoFlagEB = cms.bool( True ),
      EBtimeFitLimits_Lower = cms.double( 0.2 ),
      kPoorRecoFlagEE = cms.bool( False ),
      chi2ThreshEB_ = cms.double( 65.0 ),
      EEtimeFitParameters = cms.vdouble( -2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277 ),
      useLumiInfoRunHeader = cms.bool( False ),
      outOfTimeThresholdGain12mEE = cms.double( 1000.0 ),
      outOfTimeThresholdGain12mEB = cms.double( 5.0 ),
      EEtimeFitLimits_Upper = cms.double( 1.4 ),
      prefitMaxChiSqEB = cms.double( 100.0 ),
      EEamplitudeFitParameters = cms.vdouble( 1.89, 1.4 ),
      prefitMaxChiSqEE = cms.double( 10.0 ),
      EBamplitudeFitParameters = cms.vdouble( 1.138, 1.652 ),
      EBtimeFitLimits_Upper = cms.double( 1.4 ),
      timealgo = cms.string( "None" ),
      amplitudeThresholdEB = cms.double( 10.0 ),
      outOfTimeThresholdGain12pEE = cms.double( 1000.0 ),
      outOfTimeThresholdGain12pEB = cms.double( 5.0 ),
      EEtimeNconst = cms.double( 31.8 ),
      outOfTimeThresholdGain61mEB = cms.double( 5.0 ),
      outOfTimeThresholdGain61mEE = cms.double( 1000.0 ),
      EEtimeConstantTerm = cms.double( 1.0 ),
      chi2ThreshEE_ = cms.double( 50.0 ),
      doPrefitEE = cms.bool( False ),
      doPrefitEB = cms.bool( False )
    )
)
process.hltEcalDetIdToBeRecovered = cms.EDProducer( "EcalDetIdToBeRecoveredProducer",
    ebIntegrityChIdErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityChIdErrors' ),
    ebDetIdToBeRecovered = cms.string( "ebDetId" ),
    integrityTTIdErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityTTIdErrors' ),
    eeIntegrityGainErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityGainErrors' ),
    ebFEToBeRecovered = cms.string( "ebFE" ),
    ebIntegrityGainErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityGainErrors' ),
    eeDetIdToBeRecovered = cms.string( "eeDetId" ),
    eeIntegrityGainSwitchErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityGainSwitchErrors' ),
    eeIntegrityChIdErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityChIdErrors' ),
    ebIntegrityGainSwitchErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityGainSwitchErrors' ),
    ebSrFlagCollection = cms.InputTag( "hltEcalDigis" ),
    eeSrFlagCollection = cms.InputTag( "hltEcalDigis" ),
    integrityBlockSizeErrors = cms.InputTag( 'hltEcalDigis','EcalIntegrityBlockSizeErrors' ),
    eeFEToBeRecovered = cms.string( "eeFE" )
)
process.hltEcalRecHit50nsMultiFit = cms.EDProducer( "EcalRecHitProducer",
    recoverEEVFE = cms.bool( False ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    recoverEBIsolatedChannels = cms.bool( False ),
    recoverEBVFE = cms.bool( False ),
    laserCorrection = cms.bool( True ),
    EBLaserMIN = cms.double( 0.5 ),
    killDeadChannels = cms.bool( True ),
    dbStatusToBeExcludedEB = cms.vint32( 14, 78, 142 ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHit50nsMultiFit','EcalUncalibRecHitsEE' ),
    EBLaserMAX = cms.double( 3.0 ),
    EELaserMIN = cms.double( 0.5 ),
    ebFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebFE' ),
    EELaserMAX = cms.double( 8.0 ),
    recoverEEIsolatedChannels = cms.bool( False ),
    eeDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeDetId' ),
    recoverEBFE = cms.bool( True ),
    algo = cms.string( "EcalRecHitWorkerSimple" ),
    ebDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebDetId' ),
    singleChannelRecoveryThreshold = cms.double( 8.0 ),
    ChannelStatusToBeExcluded = cms.vstring(  ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    singleChannelRecoveryMethod = cms.string( "NeuralNetworks" ),
    recoverEEFE = cms.bool( True ),
    triggerPrimitiveDigiCollection = cms.InputTag( 'hltEcalDigis','EcalTriggerPrimitives' ),
    dbStatusToBeExcludedEE = cms.vint32( 14, 78, 142 ),
    flagsMapDBReco = cms.PSet( 
      kGood = cms.vstring( 'kOk',
        'kDAC',
        'kNoLaser',
        'kNoisy' ),
      kNeighboursRecovered = cms.vstring( 'kFixedG0',
        'kNonRespondingIsolated',
        'kDeadVFE' ),
      kDead = cms.vstring( 'kNoDataNoTP' ),
      kNoisy = cms.vstring( 'kNNoisy',
        'kFixedG6',
        'kFixedG1' ),
      kTowerRecovered = cms.vstring( 'kDeadFE' )
    ),
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHit50nsMultiFit','EcalUncalibRecHitsEB' ),
    algoRecover = cms.string( "EcalRecHitWorkerRecover" ),
    eeFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeFE' ),
    cleaningConfig = cms.PSet( 
      e6e2thresh = cms.double( 0.04 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 ),
      cThreshold_endcap = cms.double( 15.0 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      cThreshold_double = cms.double( 10.0 )
    ),
    logWarningEtThreshold_EB_FE = cms.double( 50.0 ),
    logWarningEtThreshold_EE_FE = cms.double( 50.0 )
)
process.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    ExpectedOrbitMessageTime = cms.untracked.int32( -1 ),
    FilterDataQuality = cms.bool( True ),
    silent = cms.untracked.bool( True ),
    HcalFirstFED = cms.untracked.int32( 700 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ComplainEmptyData = cms.untracked.bool( False ),
    ElectronicsMap = cms.string( "" ),
    UnpackCalib = cms.untracked.bool( True ),
    FEDs = cms.untracked.vint32(  ),
    UnpackerMode = cms.untracked.int32( 0 ),
    UnpackTTP = cms.untracked.bool( False ),
    lastSample = cms.int32( 9 ),
    UnpackZDC = cms.untracked.bool( True ),
    firstSample = cms.int32( 0 )
)
process.hltHbherecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 9.27638, -2.05585, 9.27638, -2.05585, 9.27638, -2.05585 ),
    pedestalSubtractionType = cms.int32( 1 ),
    respCorrM3 = cms.double( 0.95 ),
    timeSlewParsType = cms.int32( 3 ),
    digiTimeFromDB = cms.bool( True ),
    mcOOTCorrectionName = cms.string( "" ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    mcOOTCorrectionCategory = cms.string( "MC" ),
    dataOOTCorrectionName = cms.string( "" ),
    puCorrMethod = cms.int32( 0 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    digistat = cms.PSet(  ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    setSaturationFlags = cms.bool( False ),
    setNegativeFlags = cms.bool( False ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HBHE" ),
    dataOOTCorrectionCategory = cms.string( "Data" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    pulseJitter = cms.double( 1.0 ),
    chargeMax = cms.double( 6.0 ),
    timeMin = cms.double( -15.0 ),
    ts4chi2 = cms.double( 15.0 ),
    ts345chi2 = cms.double( 100.0 ),
    applyPulseJitter = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
    applyTimeConstraint = cms.bool( True ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    ts4Min = cms.double( 5.0 ),
    pulseShapeParameters = cms.PSet(  ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    applyTimeSlew = cms.bool( True ),
    meanTime = cms.double( -2.5 ),
    flagParameters = cms.PSet( 
      nominalPedestal = cms.double( 3.0 ),
      hitMultiplicityThreshold = cms.int32( 17 ),
      hitEnergyMinimum = cms.double( 1.0 ),
      pulseShapeParameterSets = cms.VPSet( 
        cms.PSet(  pulseShapeParameters = cms.vdouble( 0.0, 100.0, -50.0, 0.0, -15.0, 0.15 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 100.0, 2000.0, -50.0, 0.0, -5.0, 0.05 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( 2000.0, 1000000.0, -50.0, 0.0, 95.0, 0.0 )        ),
        cms.PSet(  pulseShapeParameters = cms.vdouble( -1000000.0, 1000000.0, 45.0, 0.1, 1000000.0, 0.0 )        )
      )
    ),
    fitTimes = cms.int32( 1 ),
    timeMax = cms.double( 10.0 ),
    timeSigma = cms.double( 5.0 ),
    pedSigma = cms.double( 0.5 ),
    meanPed = cms.double( 0.0 ),
    ts3chi2 = cms.double( 5.0 ),
    hscpParameters = cms.PSet( 
      slopeMax = cms.double( -0.6 ),
      r1Max = cms.double( 1.0 ),
      r1Min = cms.double( 0.15 ),
      TimingEnergyThreshold = cms.double( 30.0 ),
      slopeMin = cms.double( -1.5 ),
      outerMin = cms.double( 0.0 ),
      outerMax = cms.double( 0.1 ),
      fracLeaderMin = cms.double( 0.4 ),
      r2Min = cms.double( 0.1 ),
      r2Max = cms.double( 0.5 ),
      fracLeaderMax = cms.double( 0.7 )
    )
)
process.hltHfrecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 9.27638, -2.05585, 9.27638, -2.05585, 9.27638, -2.05585 ),
    pedestalSubtractionType = cms.int32( 1 ),
    respCorrM3 = cms.double( 0.95 ),
    timeSlewParsType = cms.int32( 3 ),
    digiTimeFromDB = cms.bool( True ),
    mcOOTCorrectionName = cms.string( "" ),
    S9S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 24 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 2 ),
    mcOOTCorrectionCategory = cms.string( "MC" ),
    dataOOTCorrectionName = cms.string( "" ),
    puCorrMethod = cms.int32( 0 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet( 
      hflongEthresh = cms.double( 40.0 ),
      hflongMinWindowTime = cms.vdouble( -10.0 ),
      hfshortEthresh = cms.double( 40.0 ),
      hflongMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMaxWindowTime = cms.vdouble( 10.0 ),
      hfshortMinWindowTime = cms.vdouble( -12.0 )
    ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 1 ),
    digistat = cms.PSet( 
      HFdigiflagFirstSample = cms.int32( 1 ),
      HFdigiflagMinEthreshold = cms.double( 40.0 ),
      HFdigiflagSamplesToAdd = cms.int32( 3 ),
      HFdigiflagExpectedPeak = cms.int32( 2 ),
      HFdigiflagCoef = cms.vdouble( 0.93, -0.012667, -0.38275 )
    ),
    hfTimingTrustParameters = cms.PSet( 
      hfTimingTrustLevel2 = cms.int32( 4 ),
      hfTimingTrustLevel1 = cms.int32( 1 )
    ),
    PETstat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_R_29 = cms.vdouble( 0.8 ),
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      flagsToSkip = cms.int32( 0 ),
      short_R = cms.vdouble( 0.8 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      long_R = cms.vdouble( 0.98 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    setSaturationFlags = cms.bool( False ),
    setNegativeFlags = cms.bool( False ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet( 
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      flagsToSkip = cms.int32( 16 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    correctForPhaseContainment = cms.bool( False ),
    correctForTimeslew = cms.bool( False ),
    setNoiseFlags = cms.bool( True ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HF" ),
    dataOOTCorrectionCategory = cms.string( "Data" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 2 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    pulseJitter = cms.double( 1.0 ),
    chargeMax = cms.double( 6.0 ),
    timeMin = cms.double( -15.0 ),
    ts4chi2 = cms.double( 15.0 ),
    ts345chi2 = cms.double( 100.0 ),
    applyPulseJitter = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
    applyTimeConstraint = cms.bool( True ),
    timingshapedcutsParameters = cms.PSet(  ),
    ts4Min = cms.double( 5.0 ),
    pulseShapeParameters = cms.PSet(  ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    applyTimeSlew = cms.bool( True ),
    meanTime = cms.double( -2.5 ),
    flagParameters = cms.PSet(  ),
    fitTimes = cms.int32( 1 ),
    timeMax = cms.double( 10.0 ),
    timeSigma = cms.double( 5.0 ),
    pedSigma = cms.double( 0.5 ),
    meanPed = cms.double( 0.0 ),
    ts3chi2 = cms.double( 5.0 ),
    hscpParameters = cms.PSet(  )
)
process.hltHorecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 9.27638, -2.05585, 9.27638, -2.05585, 9.27638, -2.05585 ),
    pedestalSubtractionType = cms.int32( 1 ),
    respCorrM3 = cms.double( 0.95 ),
    timeSlewParsType = cms.int32( 3 ),
    digiTimeFromDB = cms.bool( True ),
    mcOOTCorrectionName = cms.string( "" ),
    S9S1stat = cms.PSet(  ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    tsFromDB = cms.bool( True ),
    samplesToAdd = cms.int32( 4 ),
    mcOOTCorrectionCategory = cms.string( "MC" ),
    dataOOTCorrectionName = cms.string( "" ),
    puCorrMethod = cms.int32( 0 ),
    correctionPhaseNS = cms.double( 13.0 ),
    HFInWindowStat = cms.PSet(  ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    setHSCPFlags = cms.bool( False ),
    firstAuxTS = cms.int32( 4 ),
    digistat = cms.PSet(  ),
    hfTimingTrustParameters = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    setSaturationFlags = cms.bool( False ),
    setNegativeFlags = cms.bool( False ),
    useLeakCorrection = cms.bool( False ),
    setTimingTrustFlags = cms.bool( False ),
    S8S1stat = cms.PSet(  ),
    correctForPhaseContainment = cms.bool( True ),
    correctForTimeslew = cms.bool( True ),
    setNoiseFlags = cms.bool( False ),
    correctTiming = cms.bool( False ),
    setPulseShapeFlags = cms.bool( False ),
    Subdetector = cms.string( "HO" ),
    dataOOTCorrectionCategory = cms.string( "Data" ),
    dropZSmarkedPassed = cms.bool( True ),
    recoParamsFromDB = cms.bool( True ),
    firstSample = cms.int32( 4 ),
    setTimingShapedCutsFlags = cms.bool( False ),
    pulseJitter = cms.double( 1.0 ),
    chargeMax = cms.double( 6.0 ),
    timeMin = cms.double( -15.0 ),
    ts4chi2 = cms.double( 15.0 ),
    ts345chi2 = cms.double( 100.0 ),
    applyPulseJitter = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
    applyTimeConstraint = cms.bool( True ),
    timingshapedcutsParameters = cms.PSet(  ),
    ts4Min = cms.double( 5.0 ),
    pulseShapeParameters = cms.PSet(  ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    applyTimeSlew = cms.bool( True ),
    meanTime = cms.double( -2.5 ),
    flagParameters = cms.PSet(  ),
    fitTimes = cms.int32( 1 ),
    timeMax = cms.double( 10.0 ),
    timeSigma = cms.double( 5.0 ),
    pedSigma = cms.double( 0.5 ),
    meanPed = cms.double( 0.0 ),
    ts3chi2 = cms.double( 5.0 ),
    hscpParameters = cms.PSet(  )
)
process.hltTowerMakerHcalMethod050nsMultiFitForAll = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfrecoMethod0" ),
    AllowMissingInputs = cms.bool( False ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "hltHbherecoMethod0" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHorecoMethod0" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( False ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHit50nsMultiFit:EcalRecHitsEB','hltEcalRecHit50nsMultiFit:EcalRecHitsEE' ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltPuAK4CaloJets50nsMultiFit = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 1 ),
    doAreaFastjet = cms.bool( True ),
    voronoiRfact = cms.double( -0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.5 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.4 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 10.0 ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetPtMin = cms.double( 10.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( True ),
    inputEMin = cms.double( 0.0 ),
    useMassDropTagger = cms.bool( False ),
    muMin = cms.double( -1.0 ),
    subtractorName = cms.string( "MultipleAlgoIterator" ),
    muCut = cms.double( -1.0 ),
    subjetPtMin = cms.double( -1.0 ),
    useTrimming = cms.bool( False ),
    muMax = cms.double( -1.0 ),
    yMin = cms.double( -1.0 ),
    useFiltering = cms.bool( False ),
    rFilt = cms.double( -1.0 ),
    yMax = cms.double( -1.0 ),
    zcut = cms.double( -1.0 ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    dRMin = cms.double( -1.0 ),
    nFilt = cms.int32( -1 ),
    usePruning = cms.bool( False ),
    maxDepth = cms.int32( -1 ),
    yCut = cms.double( -1.0 ),
    DzTrVtxMax = cms.double( 0.0 ),
    UseOnlyOnePV = cms.bool( False ),
    rcut_factor = cms.double( -1.0 ),
    sumRecHits = cms.bool( False ),
    trimPtFracMin = cms.double( -1.0 ),
    dRMax = cms.double( -1.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    useCMSBoostedTauSeedingAlgorithm = cms.bool( False )
)
process.hltPuAK4CaloJetsIDPassed50nsMultiFit = cms.EDProducer( "HLTCaloJetIDProducer",
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( 2 ),
    min_EMF = cms.double( 1.0E-6 ),
    jetsInput = cms.InputTag( "hltPuAK4CaloJets50nsMultiFit" ),
    JetIDParams = cms.PSet( 
      useRecHits = cms.bool( True ),
      hbheRecHitsColl = cms.InputTag( "hltHbherecoMethod0" ),
      hoRecHitsColl = cms.InputTag( "hltHorecoMethod0" ),
      hfRecHitsColl = cms.InputTag( "hltHfrecoMethod0" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' )
    ),
    max_EMF = cms.double( 999.0 )
)
process.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 = cms.EDProducer( "FixedGridRhoProducerFastjet",
    gridSpacing = cms.double( 0.55 ),
    maxRapidity = cms.double( 5.0 ),
    pfCandidatesTag = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" )
)
process.hltAK4CaloRelativeCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    algorithm = cms.string( "AK4CaloHLT" ),
    level = cms.string( "L2Relative" )
)
process.hltAK4CaloAbsoluteCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    algorithm = cms.string( "AK4CaloHLT" ),
    level = cms.string( "L3Absolute" )
)
process.hltPuAK4CaloCorrector = cms.EDProducer( "ChainedJetCorrectorProducer",
    correctors = cms.VInputTag( 'hltAK4CaloRelativeCorrector','hltAK4CaloAbsoluteCorrector' )
)
process.hltPuAK4CaloJetsCorrected50nsMultiFit = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJets50nsMultiFit" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
process.hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsIDPassed50nsMultiFit" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
process.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltSiStripRawToDigi = cms.EDProducer( "SiStripRawToDigiModule",
    UseDaqRegister = cms.bool( False ),
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    MarkModulesOnMissingFeds = cms.bool( True ),
    UnpackCommonModeValues = cms.bool( False ),
    AppendedBytes = cms.int32( 0 ),
    UseFedKey = cms.bool( False ),
    LegacyUnpacker = cms.bool( False ),
    ErrorThreshold = cms.uint32( 7174 ),
    TriggerFedId = cms.int32( 0 ),
    DoAPVEmulatorCheck = cms.bool( False ),
    UnpackBadChannels = cms.bool( False ),
    DoAllCorruptBufferChecks = cms.bool( False )
)
process.hltSiStripZeroSuppression = cms.EDProducer( "SiStripZeroSuppression",
    fixCM = cms.bool( False ),
    DigisToMergeVR = cms.InputTag( 'hltSiStripRawToDigi','VirginRaw' ),
    produceCalculatedBaseline = cms.bool( False ),
    produceBaselinePoints = cms.bool( False ),
    RawDigiProducersList = cms.VInputTag( 'hltSiStripRawToDigi:VirginRaw','hltSiStripRawToDigi:ProcessedRaw','hltSiStripRawToDigi:ScopeMode' ),
    storeInZScollBadAPV = cms.bool( True ),
    mergeCollections = cms.bool( False ),
    Algorithms = cms.PSet( 
      Fraction = cms.double( 0.2 ),
      slopeY = cms.int32( 4 ),
      slopeX = cms.int32( 3 ),
      PedestalSubtractionFedMode = cms.bool( False ),
      CutToAvoidSignal = cms.double( 2.0 ),
      minStripsToFit = cms.uint32( 4 ),
      consecThreshold = cms.uint32( 5 ),
      hitStripThreshold = cms.uint32( 40 ),
      Deviation = cms.uint32( 25 ),
      CommonModeNoiseSubtractionMode = cms.string( "IteratedMedian" ),
      filteredBaselineDerivativeSumSquare = cms.double( 30.0 ),
      ApplyBaselineCleaner = cms.bool( True ),
      doAPVRestore = cms.bool( True ),
      TruncateInSuppressor = cms.bool( True ),
      restoreThreshold = cms.double( 0.5 ),
      APVInspectMode = cms.string( "BaselineFollower" ),
      ForceNoRestore = cms.bool( False ),
      useRealMeanCM = cms.bool( False ),
      ApplyBaselineRejection = cms.bool( True ),
      DeltaCMThreshold = cms.uint32( 20 ),
      nSigmaNoiseDerTh = cms.uint32( 4 ),
      nSaturatedStrip = cms.uint32( 2 ),
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      useCMMeanMap = cms.bool( False ),
      APVRestoreMode = cms.string( "BaselineFollower" ),
      distortionThreshold = cms.uint32( 20 ),
      filteredBaselineMax = cms.double( 6.0 ),
      Iterations = cms.int32( 3 ),
      CleaningSequence = cms.uint32( 1 ),
      nSmooth = cms.uint32( 9 ),
      SelfSelectRestoreAlgo = cms.bool( False ),
      MeanCM = cms.int32( 0 )
    ),
    DigisToMergeZS = cms.InputTag( 'hltSiStripRawToDigi','ZeroSuppressed' ),
    storeCM = cms.bool( True ),
    produceRawDigis = cms.bool( True )
)
process.hltSiStripDigiToZSRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "ZERO_SUPPRESSED" ),
    UseWrongDigiType = cms.bool( False ),
    UseFedKey = cms.bool( False ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" )
)
process.hltSiStripRawDigiToVirginRaw = cms.EDProducer( "SiStripDigiToRawModule",
    InputDigiLabel = cms.string( "VirginRaw" ),
    FedReadoutMode = cms.string( "VIRGIN_RAW" ),
    UseWrongDigiType = cms.bool( False ),
    UseFedKey = cms.bool( False ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" )
)
process.virginRawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripRawDigiToVirginRaw' )
)
process.rawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripDigiToZSRaw','source','rawDataCollector' )
)
process.hltL1sL1SingleS1Jet28BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet60Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleJet44BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet44_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet80Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleS1Jet56BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet56_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet100Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet110Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet110Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 110.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet120Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet120Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 120.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleS1Jet64BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet64_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet150Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet150Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 150.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleS1Jet16Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet40Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet28Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet60Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet44Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet44_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet80Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPuAK4CaloJet100Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet16Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet40Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet28Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet60Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet44Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet44_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIPuAK4CaloJet80Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPuAK4CaloJet100Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPuAK4CaloJet80Jet35Eta1p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet80Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 35.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet80Jet35Eta0p7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet80Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 35.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet100Jet35Eta1p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet100Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet100Jet35Eta0p7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet100Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloJet804545Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTriplePuAK4CaloJet45Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 45.0 ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIPuAK4CaloDJet60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltEta2CaloJetsEta2p1ForJets = cms.EDFilter( "CaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    cut = cms.string( "abs(eta)<2.1" )
)
process.hltReduceJetMultEta2p1Forjets = cms.EDFilter( "LargestEtCaloJetSelector",
    maxNumber = cms.uint32( 3 ),
    filter = cms.bool( False ),
    src = cms.InputTag( "hltEta2CaloJetsEta2p1ForJets" )
)
process.hltJets4bTaggerCaloJet60Eta2p1Forjets = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
    etMin = cms.double( 60.0 )
)
process.hltHIJetsForCoreTracking = cms.EDFilter( "CandPtrSelector",
    src = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    cut = cms.string( "pt > 100 && abs(eta) < 2.4" )
)
process.hltHISiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    UseQualityInfo = cms.bool( False ),
    UsePilotBlade = cms.bool( False ),
    UsePhase1 = cms.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    IncludeErrors = cms.bool( False ),
    ErrorList = cms.vint32(  ),
    Regions = cms.PSet(  ),
    Timing = cms.untracked.bool( False ),
    UserErrorList = cms.vint32(  )
)
process.hltHISiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
    src = cms.InputTag( "hltHISiPixelDigis" ),
    ChannelThreshold = cms.int32( 1000 ),
    maxNumberOfClusters = cms.int32( -1 ),
    VCaltoElectronGain = cms.int32( 65 ),
    MissCalibrate = cms.untracked.bool( True ),
    SplitClusters = cms.bool( False ),
    VCaltoElectronOffset = cms.int32( -414 ),
    payloadType = cms.string( "HLT" ),
    SeedThreshold = cms.int32( 1000 ),
    ClusterThreshold = cms.double( 4000.0 )
)
process.hltHISiPixelClustersCache = cms.EDProducer( "SiPixelClusterShapeCacheProducer",
    src = cms.InputTag( "hltHISiPixelClusters" ),
    onDemand = cms.bool( False )
)
process.hltHISiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltHISiPixelClusters" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
process.hltHIPixelClusterVertices = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 30.0 ),
    zStep = cms.double( 0.1 ),
    minZ = cms.double( -30.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHits" )
)
process.hltHIPixelLayerTriplets = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2+BPix3',
      'BPix1+BPix2+FPix1_pos',
      'BPix1+BPix2+FPix1_neg',
      'BPix1+FPix1_pos+FPix2_pos',
      'BPix1+FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHits" ),
      hitErrorRZ = cms.double( 0.0036 )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHits" ),
      hitErrorRZ = cms.double( 0.006 )
    ),
    TIB = cms.PSet(  )
)
process.hltHIPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" )
    ),
    passLabel = cms.string( "Pixel triplet tracks for vertexing" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.7 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        directionYCoord = cms.double( 1.0 ),
        useFoundVertices = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 3.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "none" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCache" )
        )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerTriplets" )
    )
)
process.hltHIPixelMedianVertex = cms.EDProducer( "HIPixelMedianVtxProducer",
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    FitThreshold = cms.int32( 5 ),
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    PtMin = cms.double( 0.075 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
process.hltHISelectedProtoTracks = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    maxD0Significance = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertex" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 )
)
process.hltHIPixelAdaptiveVertex = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 0.1 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      )
    ),
    verbose = cms.untracked.bool( False ),
    TkFilterParameters = cms.PSet( 
      maxD0Significance = cms.double( 3.0 ),
      minPt = cms.double( 0.0 ),
      maxNormalizedChi2 = cms.double( 5.0 ),
      minSiliconLayersWithHits = cms.int32( 0 ),
      minPixelLayersWithHits = cms.int32( 2 ),
      trackQuality = cms.string( "any" ),
      numTracksThreshold = cms.int32( 2 ),
      algorithm = cms.string( "filterWithThreshold" )
    ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltHISelectedProtoTracks" ),
    TkClusParameters = cms.PSet( 
      algorithm = cms.string( "gap" ),
      TkGapClusParameters = cms.PSet(  zSeparation = cms.double( 1.0 ) )
    )
)
process.hltHIBestAdaptiveVertex = cms.EDFilter( "HIBestVertexSelection",
    maxNumber = cms.uint32( 1 ),
    src = cms.InputTag( "hltHIPixelAdaptiveVertex" )
)
process.hltHISelectedVertex = cms.EDProducer( "HIBestVertexProducer",
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" )
)
process.hltHISiPixelClustersAfterSplitting = cms.EDProducer( "JetCoreClusterSplitter",
    verbose = cms.bool( False ),
    chargeFractionMin = cms.double( 2.0 ),
    forceXError = cms.double( 100.0 ),
    vertices = cms.InputTag( "hltHISelectedVertex" ),
    chargePerUnit = cms.double( 2000.0 ),
    centralMIPCharge = cms.double( 26000.0 ),
    forceYError = cms.double( 150.0 ),
    pixelClusters = cms.InputTag( "hltHISiPixelClusters" ),
    ptMin = cms.double( 100.0 ),
    deltaRmax = cms.double( 0.1 ),
    cores = cms.InputTag( "hltHIJetsForCoreTracking" ),
    fractionalWidth = cms.double( 0.4 ),
    pixelCPE = cms.string( "hltESPPixelCPEGeneric" )
)
process.hltHISiPixelClustersCacheAfterSplitting = cms.EDProducer( "SiPixelClusterShapeCacheProducer",
    src = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    onDemand = cms.bool( False )
)
process.hltHISiPixelRecHitsAfterSplitting = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
process.hltHIPixelClusterVerticesAfterSplitting = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 30.0 ),
    zStep = cms.double( 0.1 ),
    minZ = cms.double( -30.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHitsAfterSplitting" )
)
process.hltHIPixelLayerTripletsAfterSplitting = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2+BPix3',
      'BPix1+BPix2+FPix1_pos',
      'BPix1+BPix2+FPix1_neg',
      'BPix1+FPix1_pos+FPix2_pos',
      'BPix1+FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.0036 )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.006 )
    ),
    TIB = cms.PSet(  )
)
process.hltHIPixel3ProtoTracksAfterSplitting = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 1.0 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHitsAfterSplitting" )
    ),
    passLabel = cms.string( "Pixel triplet tracks for vertexing" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.7 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        directionYCoord = cms.double( 1.0 ),
        useFoundVertices = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 3.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHitsAfterSplitting" ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVerticesAfterSplitting" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "none" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCache" )
        )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerTripletsAfterSplitting" )
    )
)
process.hltHIPixelMedianVertexAfterSplitting = cms.EDProducer( "HIPixelMedianVtxProducer",
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    FitThreshold = cms.int32( 5 ),
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracksAfterSplitting" ),
    PtMin = cms.double( 0.075 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
process.hltHISelectedProtoTracksAfterSplitting = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracksAfterSplitting" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    maxD0Significance = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertexAfterSplitting" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 )
)
process.hltHIPixelAdaptiveVertexAfterSplitting = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 0.1 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      )
    ),
    verbose = cms.untracked.bool( False ),
    TkFilterParameters = cms.PSet( 
      maxD0Significance = cms.double( 3.0 ),
      minPt = cms.double( 0.0 ),
      maxNormalizedChi2 = cms.double( 5.0 ),
      minSiliconLayersWithHits = cms.int32( 0 ),
      minPixelLayersWithHits = cms.int32( 2 ),
      trackQuality = cms.string( "any" ),
      numTracksThreshold = cms.int32( 2 ),
      algorithm = cms.string( "filterWithThreshold" )
    ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltHISelectedProtoTracksAfterSplitting" ),
    TkClusParameters = cms.PSet( 
      algorithm = cms.string( "gap" ),
      TkGapClusParameters = cms.PSet(  zSeparation = cms.double( 1.0 ) )
    )
)
process.hltHIBestAdaptiveVertexAfterSplitting = cms.EDFilter( "HIBestVertexSelection",
    maxNumber = cms.uint32( 1 ),
    src = cms.InputTag( "hltHIPixelAdaptiveVertexAfterSplitting" )
)
process.hltHISelectedVertexAfterSplitting = cms.EDProducer( "HIBestVertexProducer",
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" )
)
process.hltHITrackingSiStripRawToClustersFacilityZeroSuppression = cms.EDProducer( "SiStripClusterizer",
    DigiProducersList = cms.VInputTag( 'hltSiStripRawToDigi:ZeroSuppressed','hltSiStripZeroSuppression:VirginRaw','hltSiStripZeroSuppression:ProcessedRaw','hltSiStripZeroSuppression:ScopeMode' ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      MaxSequentialHoles = cms.uint32( 0 ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      SeedThreshold = cms.double( 3.0 ),
      RemoveApvShots = cms.bool( True ),
      ClusterThreshold = cms.double( 5.0 )
    )
)
process.hltHISiStripClustersZeroSuppression = cms.EDProducer( "MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(  ),
    stripClusterProducer = cms.string( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    pixelClusterProducer = cms.string( "hltHISiPixelClustersAfterSplitting" ),
    switchOffPixelsIfEmpty = cms.bool( True ),
    inactiveStripDetectorLabels = cms.VInputTag(  ),
    skipClusters = cms.InputTag( "" ),
    measurementTracker = cms.string( "hltESPMeasurementTrackerForHI" )
)
process.hltHIPixel3PrimTracksForjets = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIPixelTrackFilter" ),
      ptMin = cms.double( 0.9 ),
      tipMax = cms.double( 0.0 ),
      useClusterShape = cms.bool( False ),
      VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
      nSigmaTipMaxTolerance = cms.double( 6.0 ),
      nSigmaLipMaxTolerance = cms.double( 0.0 ),
      lipMax = cms.double( 0.3 ),
      clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "CandidateSeededTrackingRegionsProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.1 ),
        ptMin = cms.double( 0.9 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFoundVertices = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 0.2 ),
        sigmaZVertex = cms.double( 3.0 ),
        deltaEta = cms.double( 0.5 ),
        deltaPhi = cms.double( 0.5 ),
        input = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
        maxNVertices = cms.int32( 100 ),
        measurementTrackerName = cms.string( "hltHISiStripClustersZeroSuppression" ),
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 1000000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "none" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
        ),
        extraHitRZtolerance = cms.double( 0.037 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerTripletsAfterSplitting" )
    )
)
process.hltHIPixelTrackSeedsForjets = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    usePV = cms.bool( False ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      forceKinematicWithRegionDirection = cms.bool( False ),
      magneticField = cms.string( "ParabolicMf" ),
      SeedMomentumForBOFF = cms.double( 5.0 ),
      OriginTransverseErrorMultiplier = cms.double( 1.0 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MinOneOverPtError = cms.double( 1.0 ),
      propagator = cms.string( "PropagatorWithMaterialForHI" )
    ),
    InputVertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltHIPixel3PrimTracksForjets" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltHIPrimTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelTrackSeedsForjets" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "none" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialCkfTrajectoryBuilderForHI" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIPrimTrackCandidatesForjets" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
    Fitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "initialStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( False ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter0TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIGlobalPrimTracksForjets" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiInitialStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiInitialStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter4" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'dxyperdxyerror',
      'dzperdzerror',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIter1ClustersRefRemovalForjets = cms.EDProducer( "HITrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    trajectories = cms.InputTag( "hltHIGlobalPrimTracksForjets" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    stripClusters = cms.InputTag( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    overrideTrkQuals = cms.InputTag( 'hltHIIter0TrackSelectionForjets','hiInitialStep' ),
    pixelClusters = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    Strip = cms.PSet( 
      maxChi2 = cms.double( 9.0 ),
      maxSize = cms.uint32( 2 )
    ),
    TrackQuality = cms.string( "highPurity" ),
    clusterLessSolution = cms.bool( True )
)
process.hltHIIter1MaskedMeasurementTrackerEventForjets = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter1ClustersRefRemovalForjets" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
process.hltHIDetachedPixelLayerTripletsForjets = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2+BPix3',
      'BPix1+BPix2+FPix1_pos',
      'BPix1+BPix2+FPix1_neg',
      'BPix1+FPix1_pos+FPix2_pos',
      'BPix1+FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.0036 ),
      skipClusters = cms.InputTag( "hltHIIter1ClustersRefRemovalForjets" )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.006 ),
      skipClusters = cms.InputTag( "hltHIIter1ClustersRefRemovalForjets" )
    ),
    TIB = cms.PSet(  )
)
process.hltHIDetachedPixelTracksForjets = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIPixelTrackFilter" ),
      ptMin = cms.double( 0.95 ),
      tipMax = cms.double( 1.0 ),
      useClusterShape = cms.bool( False ),
      VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      nSigmaLipMaxTolerance = cms.double( 0.0 ),
      lipMax = cms.double( 1.0 ),
      clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    passLabel = cms.string( "Pixel detached tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "CandidateSeededTrackingRegionsProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.5 ),
        ptMin = cms.double( 0.9 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFoundVertices = cms.bool( True ),
        nSigmaZ = cms.double( 4.0 ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 0.5 ),
        sigmaZVertex = cms.double( 4.0 ),
        deltaEta = cms.double( 0.5 ),
        deltaPhi = cms.double( 0.5 ),
        input = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
        maxNVertices = cms.int32( 100 ),
        measurementTrackerName = cms.string( "hltHISiStripClustersZeroSuppression" ),
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 1000000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.0 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
        ),
        extraHitRZtolerance = cms.double( 0.0 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.InputTag( "hltHIDetachedPixelLayerTripletsForjets" )
    )
)
process.hltHIDetachedPixelTrackSeedsForjets = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    usePV = cms.bool( False ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      forceKinematicWithRegionDirection = cms.bool( False ),
      magneticField = cms.string( "ParabolicMf" ),
      SeedMomentumForBOFF = cms.double( 5.0 ),
      OriginTransverseErrorMultiplier = cms.double( 1.0 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MinOneOverPtError = cms.double( 1.0 ),
      propagator = cms.string( "PropagatorWithMaterialForHI" )
    ),
    InputVertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltHIDetachedPixelTracksForjets" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltHIDetachedTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIDetachedPixelTrackSeedsForjets" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter1MaskedMeasurementTrackerEventForjets" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( True ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryBuilderForHI" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIDetachedGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIDetachedTrackCandidatesForjets" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter1MaskedMeasurementTrackerEventForjets" ),
    Fitter = cms.string( "hltESPFlexibleKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "detachedTripletStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( False ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter1TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIDetachedGlobalPrimTracksForjets" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.2 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiDetachedTripletStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.09 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiDetachedTripletStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter7" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIter2ClustersRefRemovalForjets = cms.EDProducer( "HITrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    trajectories = cms.InputTag( "hltHIDetachedGlobalPrimTracksForjets" ),
    oldClusterRemovalInfo = cms.InputTag( "hltHIIter1ClustersRefRemovalForjets" ),
    stripClusters = cms.InputTag( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    overrideTrkQuals = cms.InputTag( 'hltHIIter1TrackSelectionForjets','hiDetachedTripletStep' ),
    pixelClusters = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    Strip = cms.PSet( 
      maxChi2 = cms.double( 9.0 ),
      maxSize = cms.uint32( 2 )
    ),
    TrackQuality = cms.string( "highPurity" ),
    clusterLessSolution = cms.bool( True )
)
process.hltHIIter2MaskedMeasurementTrackerEventForjets = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter2ClustersRefRemovalForjets" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
process.hltHIPixelLayerPairsForjets = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2',
      'BPix1+BPix3',
      'BPix2+BPix3',
      'BPix1+FPix1_pos',
      'BPix1+FPix1_neg',
      'BPix2+FPix1_pos',
      'BPix2+FPix1_neg',
      'FPix1_pos+FPix2_pos',
      'FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.0036 ),
      skipClusters = cms.InputTag( "hltHIIter2ClustersRefRemovalForjets" )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.006 ),
      skipClusters = cms.InputTag( "hltHIIter2ClustersRefRemovalForjets" )
    ),
    TIB = cms.PSet(  )
)
process.hltHIPixelPairSeedsForjets = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "CandidateSeededTrackingRegionsProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.005 ),
        ptMin = cms.double( 1.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( False ),
        sigmaZVertex = cms.double( 4.0 ),
        fixedError = cms.double( 0.2 ),
        useFoundVertices = cms.bool( True ),
        useFakeVertices = cms.bool( False ),
        nSigmaZ = cms.double( 4.0 ),
        deltaEta = cms.double( 0.5 ),
        deltaPhi = cms.double( 0.5 ),
        input = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
        maxNVertices = cms.int32( 100 ),
        measurementTrackerName = cms.string( "hltHISiStripClustersZeroSuppression" ),
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    SeedComparitorPSet = cms.PSet( 
      ComponentName = cms.string( "PixelClusterShapeSeedComparitor" ),
      FilterAtHelixStage = cms.bool( True ),
      FilterPixelHits = cms.bool( True ),
      FilterStripHits = cms.bool( False ),
      ClusterShapeHitFilterName = cms.string( "ClusterShapeHitFilter" ),
      ClusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000000 ),
      doClusterCheck = cms.bool( True ),
      ClusterCollectionLabel = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
      MaxNumberOfPixelClusters = cms.uint32( 500000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 5000000 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 5000000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerPairsForjets" )
    ),
    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromConsecutiveHitsCreatorIT" ) )
)
process.hltHIPixelPairTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelPairSeedsForjets" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter2MaskedMeasurementTrackerEventForjets" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( True ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryBuilderForHI" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIPixelPairGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIPixelPairTrackCandidatesForjets" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter2MaskedMeasurementTrackerEventForjets" ),
    Fitter = cms.string( "hltESPFlexibleKFFittingSmoother" ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "pixelPairStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( True ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter2TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIPixelPairGlobalPrimTracksForjets" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.58 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiPixelPairStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( 0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiPixelPairStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter6" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'dxyperdxyerror',
      'dzperdzerror',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIterTrackingMergedHighPurityForjets = cms.EDProducer( "TrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    writeOnlyTrkQuals = cms.bool( False ),
    MinPT = cms.double( 0.05 ),
    allowFirstHitShare = cms.bool( True ),
    copyExtras = cms.untracked.bool( False ),
    Epsilon = cms.double( -0.001 ),
    selectedTrackQuals = cms.VInputTag( 'hltHIIter0TrackSelectionForjets:hiInitialStep','hltHIIter1TrackSelectionForjets:hiDetachedTripletStep','hltHIIter2TrackSelectionForjets:hiPixelPairStep' ),
    indivShareFrac = cms.vdouble( 1.0, 1.0, 1.0 ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    copyMVA = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( True ),
        tLists = cms.vint32( 0, 1, 2 )
      )
    ),
    MinFound = cms.int32( 3 ),
    hasSelector = cms.vint32( 1, 1, 1 ),
    TrackProducers = cms.VInputTag( 'hltHIGlobalPrimTracksForjets','hltHIDetachedGlobalPrimTracksForjets','hltHIPixelPairGlobalPrimTracksForjets' ),
    LostHitPenalty = cms.double( 20.0 ),
    newQuality = cms.string( "confirmed" )
)
process.hltHIIterTrackingMergedTightForjets = cms.EDProducer( "TrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    writeOnlyTrkQuals = cms.bool( False ),
    MinPT = cms.double( 0.05 ),
    allowFirstHitShare = cms.bool( True ),
    copyExtras = cms.untracked.bool( False ),
    Epsilon = cms.double( -0.001 ),
    selectedTrackQuals = cms.VInputTag( 'hltHIIter0TrackSelectionForjets:hiInitialStepTight','hltHIIter1TrackSelectionForjets:hiDetachedTripletStepTight','hltHIIter2TrackSelectionForjets:hiPixelPairStepTight' ),
    indivShareFrac = cms.vdouble( 1.0, 1.0, 1.0 ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    copyMVA = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( True ),
        tLists = cms.vint32( 0, 1, 2 )
      )
    ),
    MinFound = cms.int32( 3 ),
    hasSelector = cms.vint32( 1, 1, 1 ),
    TrackProducers = cms.VInputTag( 'hltHIGlobalPrimTracksForjets','hltHIDetachedGlobalPrimTracksForjets','hltHIPixelPairGlobalPrimTracksForjets' ),
    LostHitPenalty = cms.double( 20.0 ),
    newQuality = cms.string( "confirmed" )
)
process.hltHIFullTrackCandsForDmesonjets = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    particleType = cms.string( "pi+" )
)
process.hltHIFullTrackFilterForDmesonjets = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 0 ),
    MinPt = cms.double( 2.5 ),
    MaxVz = cms.double( 9999.0 ),
    MaxEta = cms.double( 99999.0 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForDmesonjets" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 999.0 )
)
process.hltTktkVtxForDmesonjetsCaloJet60 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonjets" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonjets" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.17 ),
    MinPtPair = cms.double( 7.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.57 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonjetsCaloJet60 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonjets" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonjetsCaloJet60" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.95 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 2.5 )
)
process.hltPreHIPuAK4CaloDJet80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltJets4bTaggerCaloJet80Eta2p1Forjets = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
    etMin = cms.double( 80.0 )
)
process.hltTktkVtxForDmesonjetsCaloJet80 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonjets" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonjets" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.17 ),
    MinPtPair = cms.double( 7.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.57 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonjetsCaloJet80 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonjets" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonjetsCaloJet80" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.95 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 2.5 )
)
process.hltPreHIPuAK4CaloBJetCSV60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIVerticesL3 = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  maxDistanceToBeam = cms.double( 1.0 ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "" )
      ),
      cms.PSet(  maxDistanceToBeam = cms.double( 1.0 ),
        useBeamConstraint = cms.bool( True ),
        minNdof = cms.double( 0.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" ),
        label = cms.string( "WithBS" )
      )
    ),
    verbose = cms.untracked.bool( False ),
    TkFilterParameters = cms.PSet( 
      maxNormalizedChi2 = cms.double( 20.0 ),
      minPt = cms.double( 0.0 ),
      algorithm = cms.string( "filter" ),
      maxD0Significance = cms.double( 999.0 ),
      trackQuality = cms.string( "any" ),
      minPixelLayersWithHits = cms.int32( 2 ),
      minSiliconLayersWithHits = cms.int32( 5 )
    ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    TkClusParameters = cms.PSet( 
      TkDAClusParameters = cms.PSet( 
        d0CutOff = cms.double( 999.0 ),
        Tmin = cms.double( 4.0 ),
        dzCutOff = cms.double( 4.0 ),
        coolingFactor = cms.double( 0.6 ),
        use_vdt = cms.untracked.bool( True ),
        vertexSize = cms.double( 0.15 )
      ),
      algorithm = cms.string( "DA_vect" )
    )
)
process.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    tracks = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    useAssigned = cms.bool( False ),
    coneSize = cms.double( 0.4 ),
    pvSrc = cms.InputTag( "" )
)
process.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 = cms.EDProducer( "TrackIPProducer",
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    primaryVertex = cms.InputTag( 'hltHIVerticesL3','WithBS' ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    computeGhostTrack = cms.bool( False ),
    ghostTrackPriorDeltaR = cms.double( 0.03 ),
    jetTracks = cms.InputTag( "hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1" ),
    jetDirectionUsingGhostTrack = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    jetDirectionUsingTracks = cms.bool( False ),
    computeProbabilities = cms.bool( False ),
    useTrackQuality = cms.bool( False ),
    maximumChiSquared = cms.double( 20.0 )
)
process.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 = cms.EDProducer( "SecondaryVertexProducer",
    extSVDeltaRToJet = cms.double( 0.3 ),
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    vertexReco = cms.PSet( 
      primcut = cms.double( 1.8 ),
      seccut = cms.double( 6.0 ),
      smoothing = cms.bool( False ),
      weightthreshold = cms.double( 0.001 ),
      minweight = cms.double( 0.5 ),
      finder = cms.string( "avr" )
    ),
    vertexSelection = cms.PSet(  sortCriterium = cms.string( "dist3dError" ) ),
    constraint = cms.string( "BeamSpot" ),
    trackIPTagInfos = cms.InputTag( "hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1" ),
    vertexCuts = cms.PSet( 
      distSig3dMax = cms.double( 99999.9 ),
      fracPV = cms.double( 0.65 ),
      distVal2dMax = cms.double( 2.5 ),
      useTrackWeights = cms.bool( True ),
      maxDeltaRToJetAxis = cms.double( 0.5 ),
      v0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
      distSig2dMin = cms.double( 3.0 ),
      multiplicityMin = cms.uint32( 2 ),
      distVal2dMin = cms.double( 0.01 ),
      distSig2dMax = cms.double( 99999.9 ),
      distVal3dMax = cms.double( 99999.9 ),
      minimumTrackWeight = cms.double( 0.5 ),
      distVal3dMin = cms.double( -99999.9 ),
      massMax = cms.double( 6.5 ),
      distSig3dMin = cms.double( -99999.9 )
    ),
    useExternalSV = cms.bool( False ),
    minimumTrackWeight = cms.double( 0.5 ),
    usePVError = cms.bool( True ),
    trackSelection = cms.PSet( 
      totalHitsMin = cms.uint32( 2 ),
      jetDeltaRMax = cms.double( 0.3 ),
      qualityClass = cms.string( "any" ),
      pixelHitsMin = cms.uint32( 2 ),
      sip3dSigMin = cms.double( -99999.9 ),
      sip3dSigMax = cms.double( 99999.9 ),
      normChi2Max = cms.double( 99999.9 ),
      maxDistToAxis = cms.double( 0.2 ),
      sip2dValMax = cms.double( 99999.9 ),
      maxDecayLen = cms.double( 99999.9 ),
      ptMin = cms.double( 1.0 ),
      sip2dSigMax = cms.double( 99999.9 ),
      sip2dSigMin = cms.double( -99999.9 ),
      sip3dValMax = cms.double( 99999.9 ),
      sip2dValMin = cms.double( -99999.9 ),
      sip3dValMin = cms.double( -99999.9 )
    ),
    trackSort = cms.string( "sip3dSig" ),
    extSVCollection = cms.InputTag( "secondaryVertices" )
)
process.hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltCombinedSecondaryVertex" ),
    tagInfos = cms.VInputTag( 'hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1','hltL3SecondaryVertexTagInfosCaloJet60Eta2p1' )
)
process.hltBLifetimeL3FilterCSVCaloJet60Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltPreHIPuAK4CaloBJetCSV80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    tracks = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    useAssigned = cms.bool( False ),
    coneSize = cms.double( 0.4 ),
    pvSrc = cms.InputTag( "" )
)
process.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 = cms.EDProducer( "TrackIPProducer",
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    primaryVertex = cms.InputTag( 'hltHIVerticesL3','WithBS' ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    computeGhostTrack = cms.bool( False ),
    ghostTrackPriorDeltaR = cms.double( 0.03 ),
    jetTracks = cms.InputTag( "hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1" ),
    jetDirectionUsingGhostTrack = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    jetDirectionUsingTracks = cms.bool( False ),
    computeProbabilities = cms.bool( False ),
    useTrackQuality = cms.bool( False ),
    maximumChiSquared = cms.double( 20.0 )
)
process.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 = cms.EDProducer( "SecondaryVertexProducer",
    extSVDeltaRToJet = cms.double( 0.3 ),
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    vertexReco = cms.PSet( 
      primcut = cms.double( 1.8 ),
      seccut = cms.double( 6.0 ),
      smoothing = cms.bool( False ),
      weightthreshold = cms.double( 0.001 ),
      minweight = cms.double( 0.5 ),
      finder = cms.string( "avr" )
    ),
    vertexSelection = cms.PSet(  sortCriterium = cms.string( "dist3dError" ) ),
    constraint = cms.string( "BeamSpot" ),
    trackIPTagInfos = cms.InputTag( "hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1" ),
    vertexCuts = cms.PSet( 
      distSig3dMax = cms.double( 99999.9 ),
      fracPV = cms.double( 0.65 ),
      distVal2dMax = cms.double( 2.5 ),
      useTrackWeights = cms.bool( True ),
      maxDeltaRToJetAxis = cms.double( 0.5 ),
      v0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
      distSig2dMin = cms.double( 3.0 ),
      multiplicityMin = cms.uint32( 2 ),
      distVal2dMin = cms.double( 0.01 ),
      distSig2dMax = cms.double( 99999.9 ),
      distVal3dMax = cms.double( 99999.9 ),
      minimumTrackWeight = cms.double( 0.5 ),
      distVal3dMin = cms.double( -99999.9 ),
      massMax = cms.double( 6.5 ),
      distSig3dMin = cms.double( -99999.9 )
    ),
    useExternalSV = cms.bool( False ),
    minimumTrackWeight = cms.double( 0.5 ),
    usePVError = cms.bool( True ),
    trackSelection = cms.PSet( 
      totalHitsMin = cms.uint32( 2 ),
      jetDeltaRMax = cms.double( 0.3 ),
      qualityClass = cms.string( "any" ),
      pixelHitsMin = cms.uint32( 2 ),
      sip3dSigMin = cms.double( -99999.9 ),
      sip3dSigMax = cms.double( 99999.9 ),
      normChi2Max = cms.double( 99999.9 ),
      maxDistToAxis = cms.double( 0.2 ),
      sip2dValMax = cms.double( 99999.9 ),
      maxDecayLen = cms.double( 99999.9 ),
      ptMin = cms.double( 1.0 ),
      sip2dSigMax = cms.double( 99999.9 ),
      sip2dSigMin = cms.double( -99999.9 ),
      sip3dValMax = cms.double( 99999.9 ),
      sip2dValMin = cms.double( -99999.9 ),
      sip3dValMin = cms.double( -99999.9 )
    ),
    trackSort = cms.string( "sip3dSig" ),
    extSVCollection = cms.InputTag( "secondaryVertices" )
)
process.hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltCombinedSecondaryVertex" ),
    tagInfos = cms.VInputTag( 'hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1','hltL3SecondaryVertexTagInfosCaloJet80Eta2p1' )
)
process.hltBLifetimeL3FilterCSVCaloJet80Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltPreHIPuAK4CaloBJetSSV60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "SimpleSecondaryVertex3TrkComputer" ),
    tagInfos = cms.VInputTag( 'hltL3SecondaryVertexTagInfosCaloJet60Eta2p1' )
)
process.hltBLifetimeL3FilterSSVCaloJet60Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    MinTag = cms.double( 0.01 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltPreHIPuAK4CaloBJetSSV80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "SimpleSecondaryVertex3TrkComputer" ),
    tagInfos = cms.VInputTag( 'hltL3SecondaryVertexTagInfosCaloJet80Eta2p1' )
)
process.hltBLifetimeL3FilterSSVCaloJet80Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    MinTag = cms.double( 0.01 ),
    MaxTag = cms.double( 99999.0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPixel3PrimTracksForGlobalPt8 = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIPixelTrackFilter" ),
      ptMin = cms.double( 8.0 ),
      tipMax = cms.double( 0.0 ),
      useClusterShape = cms.bool( False ),
      VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
      nSigmaTipMaxTolerance = cms.double( 6.0 ),
      nSigmaLipMaxTolerance = cms.double( 0.0 ),
      lipMax = cms.double( 0.3 ),
      clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalTrackingRegionWithVerticesProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.1 ),
        ptMin = cms.double( 4.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFoundVertices = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 0.2 ),
        sigmaZVertex = cms.double( 3.0 ),
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 1000000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "none" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
        ),
        extraHitRZtolerance = cms.double( 0.037 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerTripletsAfterSplitting" )
    )
)
process.hltHIPixelTrackSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    usePV = cms.bool( False ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      forceKinematicWithRegionDirection = cms.bool( False ),
      magneticField = cms.string( "ParabolicMf" ),
      SeedMomentumForBOFF = cms.double( 5.0 ),
      OriginTransverseErrorMultiplier = cms.double( 1.0 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MinOneOverPtError = cms.double( 1.0 ),
      propagator = cms.string( "PropagatorWithMaterialForHI" )
    ),
    InputVertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltHIPixel3PrimTracksForGlobalPt8" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltHIPrimTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelTrackSeedsForGlobalPt8" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "none" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialCkfTrajectoryBuilderForHI" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIPrimTrackCandidatesForGlobalPt8" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
    Fitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "initialStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( False ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter0TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIGlobalPrimTracksForGlobalPt8" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiInitialStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiInitialStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiInitialStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter4" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'dxyperdxyerror',
      'dzperdzerror',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIter1ClustersRefRemovalForGlobalPt8 = cms.EDProducer( "HITrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    trajectories = cms.InputTag( "hltHIGlobalPrimTracksForGlobalPt8" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    stripClusters = cms.InputTag( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    overrideTrkQuals = cms.InputTag( 'hltHIIter0TrackSelectionForGlobalPt8','hiInitialStep' ),
    pixelClusters = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    Strip = cms.PSet( 
      maxChi2 = cms.double( 9.0 ),
      maxSize = cms.uint32( 2 )
    ),
    TrackQuality = cms.string( "highPurity" ),
    clusterLessSolution = cms.bool( True )
)
process.hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8 = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter1ClustersRefRemovalForGlobalPt8" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
process.hltHIDetachedPixelLayerTripletsForGlobalPt8 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2+BPix3',
      'BPix1+BPix2+FPix1_pos',
      'BPix1+BPix2+FPix1_neg',
      'BPix1+FPix1_pos+FPix2_pos',
      'BPix1+FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.0036 ),
      skipClusters = cms.InputTag( "hltHIIter1ClustersRefRemovalForGlobalPt8" )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.006 ),
      skipClusters = cms.InputTag( "hltHIIter1ClustersRefRemovalForGlobalPt8" )
    ),
    TIB = cms.PSet(  )
)
process.hltHIDetachedPixelTracksForGlobalPt8 = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( True ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIPixelTrackFilter" ),
      ptMin = cms.double( 8.0 ),
      tipMax = cms.double( 1.0 ),
      useClusterShape = cms.bool( False ),
      VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      nSigmaLipMaxTolerance = cms.double( 0.0 ),
      lipMax = cms.double( 1.0 ),
      clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    passLabel = cms.string( "Pixel detached tracks with vertex constraint" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalTrackingRegionWithVerticesProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.5 ),
        ptMin = cms.double( 4.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFoundVertices = cms.bool( True ),
        nSigmaZ = cms.double( 4.0 ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 0.5 ),
        sigmaZVertex = cms.double( 4.0 ),
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "TrackCleaner" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 1000000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.0 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
        ),
        extraHitRZtolerance = cms.double( 0.0 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.InputTag( "hltHIDetachedPixelLayerTripletsForGlobalPt8" )
    )
)
process.hltHIDetachedPixelTrackSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    useEventsWithNoVertex = cms.bool( True ),
    originHalfLength = cms.double( 1.0E9 ),
    useProtoTrackKinematics = cms.bool( False ),
    usePV = cms.bool( False ),
    SeedCreatorPSet = cms.PSet( 
      ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
      forceKinematicWithRegionDirection = cms.bool( False ),
      magneticField = cms.string( "ParabolicMf" ),
      SeedMomentumForBOFF = cms.double( 5.0 ),
      OriginTransverseErrorMultiplier = cms.double( 1.0 ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MinOneOverPtError = cms.double( 1.0 ),
      propagator = cms.string( "PropagatorWithMaterialForHI" )
    ),
    InputVertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    InputCollection = cms.InputTag( "hltHIDetachedPixelTracksForGlobalPt8" ),
    originRadius = cms.double( 1.0E9 )
)
process.hltHIDetachedTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIDetachedPixelTrackSeedsForGlobalPt8" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( True ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedCkfTrajectoryBuilderForHIGlobalPt8" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIDetachedGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIDetachedTrackCandidatesForGlobalPt8" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8" ),
    Fitter = cms.string( "hltESPFlexibleKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "detachedTripletStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( False ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter1TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIDetachedGlobalPrimTracksForGlobalPt8" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.2 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiDetachedTripletStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( False ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.09 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiDetachedTripletStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiDetachedTripletStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter7" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIter2ClustersRefRemovalForGlobalPt8 = cms.EDProducer( "HITrackClusterRemover",
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    trajectories = cms.InputTag( "hltHIDetachedGlobalPrimTracksForGlobalPt8" ),
    oldClusterRemovalInfo = cms.InputTag( "hltHIIter1ClustersRefRemovalForGlobalPt8" ),
    stripClusters = cms.InputTag( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    overrideTrkQuals = cms.InputTag( 'hltHIIter1TrackSelectionForGlobalPt8','hiDetachedTripletStep' ),
    pixelClusters = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    Common = cms.PSet(  maxChi2 = cms.double( 9.0 ) ),
    Strip = cms.PSet( 
      maxChi2 = cms.double( 9.0 ),
      maxSize = cms.uint32( 2 )
    ),
    TrackQuality = cms.string( "highPurity" ),
    clusterLessSolution = cms.bool( True )
)
process.hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8 = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter2ClustersRefRemovalForGlobalPt8" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
process.hltHIPixelLayerPairsForGlobalPt8 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2',
      'BPix1+BPix3',
      'BPix2+BPix3',
      'BPix1+FPix1_pos',
      'BPix1+FPix1_neg',
      'BPix2+FPix1_pos',
      'BPix2+FPix1_neg',
      'FPix1_pos+FPix2_pos',
      'FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.0036 ),
      skipClusters = cms.InputTag( "hltHIIter2ClustersRefRemovalForGlobalPt8" )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHitsAfterSplitting" ),
      hitErrorRZ = cms.double( 0.006 ),
      skipClusters = cms.InputTag( "hltHIIter2ClustersRefRemovalForGlobalPt8" )
    ),
    TIB = cms.PSet(  )
)
process.hltHIPixelPairSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalTrackingRegionWithVerticesProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.005 ),
        ptMin = cms.double( 4.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( False ),
        sigmaZVertex = cms.double( 4.0 ),
        fixedError = cms.double( 0.2 ),
        useFoundVertices = cms.bool( True ),
        useFakeVertices = cms.bool( False ),
        nSigmaZ = cms.double( 4.0 ),
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" )
      )
    ),
    SeedComparitorPSet = cms.PSet( 
      ComponentName = cms.string( "PixelClusterShapeSeedComparitor" ),
      FilterAtHelixStage = cms.bool( True ),
      FilterPixelHits = cms.bool( True ),
      FilterStripHits = cms.bool( False ),
      ClusterShapeHitFilterName = cms.string( "ClusterShapeHitFilter" ),
      ClusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCacheAfterSplitting" )
    ),
    ClusterCheckPSet = cms.PSet( 
      PixelClusterCollectionLabel = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
      MaxNumberOfCosmicClusters = cms.uint32( 50000000 ),
      doClusterCheck = cms.bool( True ),
      ClusterCollectionLabel = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
      MaxNumberOfPixelClusters = cms.uint32( 500000 )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      maxElement = cms.uint32( 5000000 ),
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      GeneratorPSet = cms.PSet( 
        maxElement = cms.uint32( 5000000 ),
        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerPairsForGlobalPt8" )
    ),
    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromConsecutiveHitsCreatorIT" ) )
)
process.hltHIPixelPairTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltHIPixelPairSeedsForGlobalPt8" ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialForHI" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOppositeForHI" ),
      numberMeasurementsForFit = cms.int32( 4 )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8" ),
    cleanTrajectoryAfterInOut = cms.bool( True ),
    useHitsSplitting = cms.bool( True ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( True ),
    maxNSeeds = cms.uint32( 500000 ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairCkfTrajectoryBuilderForHIGlobalPt8" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" )
)
process.hltHIPixelPairGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIPixelPairTrackCandidatesForGlobalPt8" ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8" ),
    Fitter = cms.string( "hltESPFlexibleKFFittingSmoother" ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "pixelPairStep" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    GeometricInnerState = cms.bool( True ),
    useSimpleMF = cms.bool( True ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
)
process.hltHIIter2TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
    src = cms.InputTag( "hltHIPixelPairGlobalPrimTracksForGlobalPt8" ),
    trackSelectors = cms.VPSet( 
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "loose" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( False ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 0.4, 4.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.2 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -1.0 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStepLoose" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 9999.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 0.4, 4.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "tight" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        dz_par2 = cms.vdouble( 5.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        nSigmaZ = cms.double( 9999.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.075 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( -0.58 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStepTight" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.25 ),
        preFilterName = cms.string( "hiPixelPairStepLoose" ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 5.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        minHitsToBypassChecks = cms.uint32( 999 )
      ),
      cms.PSet(  max_d0 = cms.double( 100.0 ),
        minNumber3DLayers = cms.uint32( 0 ),
        max_lostHitFraction = cms.double( 1.0 ),
        applyAbsCutsIfNoPV = cms.bool( False ),
        qualityBit = cms.string( "highPurity" ),
        minNumberLayers = cms.uint32( 0 ),
        useMVA = cms.bool( True ),
        nSigmaZ = cms.double( 9999.0 ),
        dz_par2 = cms.vdouble( 3.0, 0.0 ),
        applyAdaptedPVCuts = cms.bool( True ),
        min_eta = cms.double( -9999.0 ),
        dz_par1 = cms.vdouble( 9999.0, 0.0 ),
        copyTrajectories = cms.untracked.bool( True ),
        vtxNumber = cms.int32( -1 ),
        keepAllTracks = cms.bool( False ),
        maxNumberLostLayers = cms.uint32( 999 ),
        max_relpterr = cms.double( 0.05 ),
        copyExtras = cms.untracked.bool( True ),
        minMVA = cms.double( 0.77 ),
        vertexCut = cms.string( "" ),
        max_z0 = cms.double( 100.0 ),
        min_nhits = cms.uint32( 8 ),
        name = cms.string( "hiPixelPairStep" ),
        max_minMissHitOutOrIn = cms.int32( 99 ),
        chi2n_no1Dmod_par = cms.double( 0.15 ),
        res_par = cms.vdouble( 99999.0, 99999.0 ),
        chi2n_par = cms.double( 0.3 ),
        max_eta = cms.double( 9999.0 ),
        d0_par2 = cms.vdouble( 3.0, 0.0 ),
        d0_par1 = cms.vdouble( 9999.0, 0.0 ),
        preFilterName = cms.string( "hiPixelPairStepTight" ),
        minHitsToBypassChecks = cms.uint32( 999 )
      )
    ),
    GBRForestLabel = cms.string( "HIMVASelectorIter6" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    GBRForestVars = cms.vstring( 'chi2perdofperlayer',
      'dxyperdxyerror',
      'dzperdzerror',
      'nhits',
      'nlayers',
      'eta' ),
    useVtxError = cms.bool( True ),
    useAnyMVA = cms.bool( True ),
    useVertices = cms.bool( True )
)
process.hltHIIterTrackingMergedHighPurityForGlobalPt8 = cms.EDProducer( "TrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    writeOnlyTrkQuals = cms.bool( False ),
    MinPT = cms.double( 0.05 ),
    allowFirstHitShare = cms.bool( True ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    selectedTrackQuals = cms.VInputTag( 'hltHIIter0TrackSelectionForGlobalPt8:hiInitialStep','hltHIIter1TrackSelectionForGlobalPt8:hiDetachedTripletStep','hltHIIter2TrackSelectionForGlobalPt8:hiPixelPairStep' ),
    indivShareFrac = cms.vdouble( 1.0, 1.0, 1.0 ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    copyMVA = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( True ),
        tLists = cms.vint32( 0, 1, 2 )
      )
    ),
    MinFound = cms.int32( 3 ),
    hasSelector = cms.vint32( 1, 1, 1 ),
    TrackProducers = cms.VInputTag( 'hltHIGlobalPrimTracksForGlobalPt8','hltHIDetachedGlobalPrimTracksForGlobalPt8','hltHIPixelPairGlobalPrimTracksForGlobalPt8' ),
    LostHitPenalty = cms.double( 20.0 ),
    newQuality = cms.string( "confirmed" )
)
process.hltHIIterTrackingMergedTightForGlobalPt8 = cms.EDProducer( "TrackListMerger",
    ShareFrac = cms.double( 0.19 ),
    writeOnlyTrkQuals = cms.bool( False ),
    MinPT = cms.double( 0.05 ),
    allowFirstHitShare = cms.bool( True ),
    copyExtras = cms.untracked.bool( True ),
    Epsilon = cms.double( -0.001 ),
    selectedTrackQuals = cms.VInputTag( 'hltHIIter0TrackSelectionForGlobalPt8:hiInitialStepTight','hltHIIter1TrackSelectionForGlobalPt8:hiDetachedTripletStepTight','hltHIIter2TrackSelectionForGlobalPt8:hiPixelPairStepTight' ),
    indivShareFrac = cms.vdouble( 1.0, 1.0, 1.0 ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    copyMVA = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( True ),
        tLists = cms.vint32( 0, 1, 2 )
      )
    ),
    MinFound = cms.int32( 3 ),
    hasSelector = cms.vint32( 1, 1, 1 ),
    TrackProducers = cms.VInputTag( 'hltHIGlobalPrimTracksForGlobalPt8','hltHIDetachedGlobalPrimTracksForGlobalPt8','hltHIPixelPairGlobalPrimTracksForGlobalPt8' ),
    LostHitPenalty = cms.double( 20.0 ),
    newQuality = cms.string( "confirmed" )
)
process.hltHIFullTrackCandsForDmesonGlobalPt8 = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIIterTrackingMergedTightForGlobalPt8" ),
    particleType = cms.string( "pi+" )
)
process.hltHIFullTrackFilterForDmesonGlobalPt8 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 0 ),
    MinPt = cms.double( 0.0 ),
    MaxVz = cms.double( 9999.0 ),
    MaxEta = cms.double( 99999.0 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 999.0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt20 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 20.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp20 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt20" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltL1sL1Centralityext30100MinimumumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext30_100_MinimumumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt20Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Centralityext50100MinimumumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( " L1_Centrality_ext50_100_MinimumumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt20Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet16BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt30 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 30.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp30 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt30" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt30Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt30Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt40 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt40 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 40.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp40 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt40" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt40Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt40Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleS1Jet32BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet32_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt50 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 50.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp50 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt50" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt60 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt60 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 60.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp60 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt60" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltL1sL1SingleS1Jet52BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet52_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt70 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTktkVtxForDmesonGlobal8Dpt70 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
    Src = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    massParticle1 = cms.double( 0.1396 ),
    PreviousCandTag = cms.InputTag( "hltHIFullTrackFilterForDmesonGlobalPt8" ),
    massParticle2 = cms.double( 0.4937 ),
    ChargeOpt = cms.int32( -1 ),
    MaxEta = cms.double( 2.5 ),
    MaxInvMass = cms.double( 2.27 ),
    MinPtPair = cms.double( 70.0 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinInvMass = cms.double( 1.47 ),
    MinPt = cms.double( 0.0 )
)
process.hltTktkFilterForDmesonGlobal8Dp70 = cms.EDFilter( "HLTDisplacedtktkFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinVtxProbability = cms.double( 0.0 ),
    MaxLxySignificance = cms.double( 0.0 ),
    TrackTag = cms.InputTag( "hltHIFullTrackCandsForDmesonGlobalPt8" ),
    DisplacedVertexTag = cms.InputTag( "hltTktkVtxForDmesonGlobal8Dpt70" ),
    MaxNormalisedChi2 = cms.double( 999.0 ),
    FastAccept = cms.bool( False ),
    MinCosinePointingAngle = cms.double( 0.8 ),
    triggerTypeDaughters = cms.int32( 91 ),
    MinLxySignificance = cms.double( 1.0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt60Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt60Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Centralityext010MinimumumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext0_10_MinimumumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIDmesonHITrackingGlobalDpt20Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt30Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDmesonHITrackingGlobalDpt40Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton10Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltIslandBasicClusters50nsMultiFitHI = cms.EDProducer( "IslandClusterProducer",
    endcapHits = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      T0_endcPresh = cms.double( 1.2 ),
      W0 = cms.double( 4.2 ),
      X0 = cms.double( 0.89 )
    ),
    barrelShapeAssociation = cms.string( "islandBarrelShapeAssoc" ),
    endcapShapeAssociation = cms.string( "islandEndcapShapeAssoc" ),
    barrelHits = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    clustershapecollectionEE = cms.string( "islandEndcapShape" ),
    clustershapecollectionEB = cms.string( "islandBarrelShape" ),
    VerbosityLevel = cms.string( "ERROR" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" )
)
process.hltHiIslandSuperClusters50nsMultiFitHI = cms.EDProducer( "HiSuperClusterProducer",
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClustersHI" ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClusters50nsMultiFitHI" ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    endcapBCEnergyThreshold = cms.double( 0.0 ),
    VerbosityLevel = cms.string( "ERROR" ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    barrelEtaSearchRoad = cms.double( 0.07 ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClustersHI" ),
    barrelBCEnergyThreshold = cms.double( 0.0 ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClusters50nsMultiFitHI" )
)
process.hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    superClusterAlgo = cms.string( "Island" ),
    etThresh = cms.double( 0.0 ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClusters50nsMultiFitHI','islandBarrelSuperClustersHI' ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet( 
      fEtaVect = cms.vdouble( 0.993, 0.0, 0.00546, 1.165, -0.180844, 0.040312 ),
      fBremVect = cms.vdouble( -0.773799, 2.73438, -1.07235, 0.986821, -0.0101822, 3.06744E-4, 1.00595, -0.0495958, 0.00451986, 1.00595, -0.0495958, 0.00451986 ),
      fBremThVect = cms.vdouble( 1.2, 1.2 ),
      fEtEtaVect = cms.vdouble( 0.9497, 0.006985, 1.03754, -0.0142667, -0.0233993, 0.0, 0.0, 0.908915, 0.0137322, 16.9602, -29.3093, 19.8976, -5.92666, 0.654571 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      minR9Barrel = cms.double( 0.94 ),
      minR9Endcap = cms.double( 0.95 ),
      maxR9 = cms.double( 1.5 )
    ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' )
)
process.hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string( "" ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    superClusterAlgo = cms.string( "Island" ),
    etThresh = cms.double( 0.0 ),
    rawSuperClusterProducer = cms.InputTag( 'hltHiIslandSuperClusters50nsMultiFitHI','islandEndcapSuperClustersHI' ),
    applyEnergyCorrection = cms.bool( True ),
    isl_fCorrPset = cms.PSet( 
      fEtaVect = cms.vdouble( 0.993, 0.0, 0.00546, 1.165, -0.180844, 0.040312 ),
      fBremVect = cms.vdouble( -0.773799, 2.73438, -1.07235, 0.986821, -0.0101822, 3.06744E-4, 1.00595, -0.0495958, 0.00451986, 1.00595, -0.0495958, 0.00451986 ),
      fBremThVect = cms.vdouble( 1.2, 1.2 ),
      fEtEtaVect = cms.vdouble( 0.9497, 0.006985, 1.03754, -0.0142667, -0.0233993, 0.0, 0.0, 0.908915, 0.0137322, 16.9602, -29.3093, 19.8976, -5.92666, 0.654571 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 ),
      minR9Barrel = cms.double( 0.94 ),
      minR9Endcap = cms.double( 0.95 ),
      maxR9 = cms.double( 1.5 )
    ),
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' )
)
process.hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI = cms.EDProducer( "HiSpikeCleaner",
    originalSuperClusterProducer = cms.InputTag( "hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI" ),
    recHitProducerEndcap = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    TimingCut = cms.untracked.double( 9999999.0 ),
    swissCutThr = cms.untracked.double( 0.95 ),
    recHitProducerBarrel = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    etCut = cms.double( 8.0 ),
    outputColl = cms.string( "" )
)
process.hltRecoHIEcalWithCleaningCandidate50nsMultiFit = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( "hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI" ),
    scHybridBarrelProducer = cms.InputTag( "hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI" ),
    recoEcalCandidateCollection = cms.string( "" )
)
process.hltHIPhoton10Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton15Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton20Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton20Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1SingleEG7BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton30Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton30Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( False ),
    MinPt = cms.double( 30.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1SingleEG21BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton40Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton40Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton50Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton50Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 50.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1SingleEG30BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG30_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton60Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton60Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltL1sL1SingleEG3Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG3_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton10Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton15Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton20Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG7Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton30Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG21Centext50100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_Centrality_ext50_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton40Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG3Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG3_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton10Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton15Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton20Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG7Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton30Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleEG21Centext30100BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHISinglePhoton40Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton40Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton40Eta2p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton10Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton10Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton15Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton15Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton20Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton20Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton30Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton30Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 30.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton40Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton40Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton50Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton50Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 50.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton60Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPhoton60Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltPreHISinglePhoton10Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton15Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton20Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton30Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton40Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton10Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton15Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton20Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton30Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHISinglePhoton40Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIDoublePhoton15Eta1p5Mass501000 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( True ),
    lowerMassCut = cms.double( 50.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
process.hltPreHIDoublePhoton15Eta1p5Mass501000R9HECut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIEgammaR9ID50nsMultiFit = cms.EDProducer( "EgammaHLTR9IDProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    ecalRechitEE = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' )
)
process.hltHIEgammaR9IDDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.6 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.6 ),
    lessThan = cms.bool( False ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltHIEgammaHoverE50nsMultiFit = cms.EDProducer( "EgammaHLTBcHcalIsolationProducersRegional",
    caloTowerProducer = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" ),
    effectiveAreaBarrel = cms.double( 0.105 ),
    outerCone = cms.double( 0.14 ),
    innerCone = cms.double( 0.0 ),
    useSingleTower = cms.bool( False ),
    rhoProducer = cms.InputTag( "hltFixedGridRhoFastjetAllCaloForMuons" ),
    depth = cms.int32( -1 ),
    doRhoCorrection = cms.bool( False ),
    effectiveAreaEndcap = cms.double( 0.17 ),
    recoEcalCandidateProducer = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    rhoMax = cms.double( 9.9999999E7 ),
    etMin = cms.double( 0.0 ),
    rhoScale = cms.double( 1.0 ),
    doEtSum = cms.bool( False )
)
process.hltHIEgammaHOverEDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.25 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( 0.25 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 1 ),
    isoTag = cms.InputTag( "hltHIEgammaHoverE50nsMultiFit" ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltPreHIDoublePhoton15Eta2p1Mass501000R9Cut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton15Eta2p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltHIDoublePhoton15Eta2p1GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( False ),
    lowerMassCut = cms.double( 50.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p150nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
process.hltHIEgammaR9IDDoublePhoton15Eta2p150nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.4 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.4 ),
    lessThan = cms.bool( False ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p150nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltPreHIDoublePhoton15Eta2p5Mass501000R9SigmaHECut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
process.hltHIDoublePhoton15Eta2p5GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( False ),
    lowerMassCut = cms.double( 50.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
process.hltHIEgammaR9IDDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.5 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.4 ),
    lessThan = cms.bool( False ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer = cms.EDProducer( "EgammaHLTClusterShapeProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    ecalRechitEE = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    isIeta = cms.bool( True )
)
process.hltHIEgammaSigmaIEtaIEtaDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.045 ),
    thrOverEEE = cms.double( -1.0 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.02 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( 'hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer','sigmaIEtaIEta5x5' ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltHIEgammaHOverEDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    doIsolated = cms.bool( True ),
    thrOverE2EE = cms.double( -1.0 ),
    L1NonIsoCand = cms.InputTag( "" ),
    saveTags = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    L1IsoCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    thrOverEEB = cms.double( 0.3 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    useEt = cms.bool( False ),
    ncandcut = cms.int32( 2 ),
    isoTag = cms.InputTag( "hltHIEgammaHoverE50nsMultiFit" ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" ),
    nonIsoTag = cms.InputTag( "" )
)
process.hltL1sL1SingleMu3MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet40Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3MinBiasFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
    useStandardFEDid = cms.bool( True ),
    maxFEDid = cms.untracked.int32( 779 ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    minFEDid = cms.untracked.int32( 770 ),
    dataType = cms.string( "DDU" ),
    readOutParameters = cms.PSet( 
      debug = cms.untracked.bool( False ),
      rosParameters = cms.PSet( 
        writeSC = cms.untracked.bool( True ),
        readingDDU = cms.untracked.bool( True ),
        performDataIntegrityMonitor = cms.untracked.bool( False ),
        readDDUIDfromDDU = cms.untracked.bool( True ),
        debug = cms.untracked.bool( False ),
        localDAQ = cms.untracked.bool( False )
      ),
      localDAQ = cms.untracked.bool( False ),
      performDataIntegrityMonitor = cms.untracked.bool( False )
    ),
    dqmOnly = cms.bool( False )
)
process.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
    recAlgoConfig = cms.PSet( 
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      minTime = cms.double( -3.0 ),
      stepTwoFromDigi = cms.bool( False ),
      doVdriftCorr = cms.bool( True ),
      debug = cms.untracked.bool( False ),
      maxTime = cms.double( 420.0 ),
      tTrigModeConfig = cms.PSet( 
        vPropWire = cms.double( 24.4 ),
        doTOFCorrection = cms.bool( True ),
        tofCorrType = cms.int32( 0 ),
        wirePropCorrType = cms.int32( 0 ),
        tTrigLabel = cms.string( "" ),
        doWirePropCorrection = cms.bool( True ),
        doT0Correction = cms.bool( True ),
        debug = cms.untracked.bool( False )
      ),
      useUncertDB = cms.bool( True )
    ),
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" ),
    recAlgo = cms.string( "DTLinearDriftFromDBAlgo" )
)
process.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    Reco4DAlgoConfig = cms.PSet( 
      segmCleanerMode = cms.int32( 2 ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      recAlgoConfig = cms.PSet( 
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        minTime = cms.double( -3.0 ),
        stepTwoFromDigi = cms.bool( False ),
        doVdriftCorr = cms.bool( True ),
        debug = cms.untracked.bool( False ),
        maxTime = cms.double( 420.0 ),
        tTrigModeConfig = cms.PSet( 
          vPropWire = cms.double( 24.4 ),
          doTOFCorrection = cms.bool( True ),
          tofCorrType = cms.int32( 0 ),
          wirePropCorrType = cms.int32( 0 ),
          tTrigLabel = cms.string( "" ),
          doWirePropCorrection = cms.bool( True ),
          doT0Correction = cms.bool( True ),
          debug = cms.untracked.bool( False )
        ),
        useUncertDB = cms.bool( True )
      ),
      nSharedHitsMax = cms.int32( 2 ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      Reco2DAlgoConfig = cms.PSet( 
        segmCleanerMode = cms.int32( 2 ),
        recAlgoConfig = cms.PSet( 
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          minTime = cms.double( -3.0 ),
          stepTwoFromDigi = cms.bool( False ),
          doVdriftCorr = cms.bool( True ),
          debug = cms.untracked.bool( False ),
          maxTime = cms.double( 420.0 ),
          tTrigModeConfig = cms.PSet( 
            vPropWire = cms.double( 24.4 ),
            doTOFCorrection = cms.bool( True ),
            tofCorrType = cms.int32( 0 ),
            wirePropCorrType = cms.int32( 0 ),
            tTrigLabel = cms.string( "" ),
            doWirePropCorrection = cms.bool( True ),
            doT0Correction = cms.bool( True ),
            debug = cms.untracked.bool( False )
          ),
          useUncertDB = cms.bool( True )
        ),
        nSharedHitsMax = cms.int32( 2 ),
        AlphaMaxPhi = cms.double( 1.0 ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        MaxAllowedHits = cms.uint32( 50 ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        AlphaMaxTheta = cms.double( 0.9 ),
        debug = cms.untracked.bool( False ),
        recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
        nUnSharedHitsMin = cms.int32( 2 ),
        performT0SegCorrection = cms.bool( False ),
        perform_delta_rejecting = cms.bool( False )
      ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      debug = cms.untracked.bool( False ),
      recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
      nUnSharedHitsMin = cms.int32( 2 ),
      AllDTRecHits = cms.bool( True ),
      performT0SegCorrection = cms.bool( False ),
      perform_delta_rejecting = cms.bool( False )
    )
)
process.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool( False ),
    SuppressZeroLCT = cms.untracked.bool( True ),
    UseExaminer = cms.bool( True ),
    Debug = cms.untracked.bool( False ),
    ErrorMask = cms.uint32( 0x0 ),
    InputObjects = cms.InputTag( "rawDataCollector" ),
    ExaminerMask = cms.uint32( 0x1febf3f6 ),
    runDQM = cms.untracked.bool( False ),
    UnpackStatusDigis = cms.bool( False ),
    VisualFEDInspect = cms.untracked.bool( False ),
    FormatedEventDump = cms.untracked.bool( False ),
    UseFormatStatus = cms.bool( True ),
    UseSelectiveUnpacking = cms.bool( True ),
    VisualFEDShort = cms.untracked.bool( False )
)
process.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    XTasymmetry_ME1b = cms.double( 0.0 ),
    XTasymmetry_ME1a = cms.double( 0.0 ),
    ConstSyst_ME1a = cms.double( 0.022 ),
    ConstSyst_ME1b = cms.double( 0.007 ),
    XTasymmetry_ME41 = cms.double( 0.0 ),
    CSCStripxtalksOffset = cms.double( 0.03 ),
    CSCUseCalibrations = cms.bool( True ),
    CSCUseTimingCorrections = cms.bool( True ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    XTasymmetry_ME22 = cms.double( 0.0 ),
    UseFivePoleFit = cms.bool( True ),
    XTasymmetry_ME21 = cms.double( 0.0 ),
    ConstSyst_ME21 = cms.double( 0.0 ),
    CSCDebug = cms.untracked.bool( False ),
    ConstSyst_ME22 = cms.double( 0.0 ),
    CSCUseGasGainCorrections = cms.bool( False ),
    XTasymmetry_ME31 = cms.double( 0.0 ),
    readBadChambers = cms.bool( True ),
    NoiseLevel_ME13 = cms.double( 8.0 ),
    NoiseLevel_ME12 = cms.double( 9.0 ),
    NoiseLevel_ME32 = cms.double( 9.0 ),
    NoiseLevel_ME31 = cms.double( 9.0 ),
    XTasymmetry_ME32 = cms.double( 0.0 ),
    ConstSyst_ME41 = cms.double( 0.0 ),
    CSCStripClusterSize = cms.untracked.int32( 3 ),
    CSCStripClusterChargeCut = cms.double( 25.0 ),
    CSCStripPeakThreshold = cms.double( 10.0 ),
    readBadChannels = cms.bool( False ),
    UseParabolaFit = cms.bool( False ),
    XTasymmetry_ME13 = cms.double( 0.0 ),
    XTasymmetry_ME12 = cms.double( 0.0 ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    ConstSyst_ME12 = cms.double( 0.0 ),
    ConstSyst_ME13 = cms.double( 0.0 ),
    ConstSyst_ME32 = cms.double( 0.0 ),
    ConstSyst_ME31 = cms.double( 0.0 ),
    UseAverageTime = cms.bool( False ),
    NoiseLevel_ME1a = cms.double( 7.0 ),
    NoiseLevel_ME1b = cms.double( 8.0 ),
    CSCWireClusterDeltaT = cms.int32( 1 ),
    CSCUseStaticPedestals = cms.bool( False ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    CSCstripWireDeltaTime = cms.int32( 8 ),
    NoiseLevel_ME21 = cms.double( 9.0 ),
    NoiseLevel_ME22 = cms.double( 9.0 ),
    NoiseLevel_ME41 = cms.double( 9.0 )
)
process.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_psets = cms.VPSet( 
      cms.PSet(  chamber_types = cms.vstring( 'ME1/a',
  'ME1/b',
  'ME1/2',
  'ME1/3',
  'ME2/1',
  'ME2/2',
  'ME3/1',
  'ME3/2',
  'ME4/1',
  'ME4/2' ),
        algo_name = cms.string( "CSCSegAlgoST" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 ),
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 20 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.double( 1.5 ),
            maxRecHitsInCluster = cms.int32( 24 ),
            dPhiFineMax = cms.double( 0.025 ),
            preClusteringUseChaining = cms.bool( True ),
            ForceCovariance = cms.bool( False ),
            hitDropLimit6Hits = cms.double( 0.3333 ),
            NormChi2Cut2D = cms.double( 20.0 ),
            BPMinImprovement = cms.double( 10000.0 ),
            Covariance = cms.double( 0.0 ),
            tanPhiMax = cms.double( 0.5 ),
            SeedBig = cms.double( 0.0015 ),
            onlyBestSegment = cms.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            SeedSmall = cms.double( 2.0E-4 ),
            curvePenalty = cms.double( 2.0 ),
            dXclusBoxMax = cms.double( 4.0 ),
            BrutePruning = cms.bool( True ),
            curvePenaltyThreshold = cms.double( 0.85 ),
            CorrectTheErrors = cms.bool( True ),
            hitDropLimit4Hits = cms.double( 0.6 ),
            useShowering = cms.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            tanThetaMax = cms.double( 1.2 ),
            NormChi2Cut3D = cms.double( 10.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            ForceCovarianceAll = cms.bool( False ),
            yweightPenaltyThreshold = cms.double( 1.0 ),
            prePrunLimit = cms.double( 3.17 ),
            hitDropLimit5Hits = cms.double( 0.8 ),
            preClustering = cms.bool( True ),
            prePrun = cms.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.bool( True ),
            dYclusBoxMax = cms.double( 8.0 )
          )
        )
      )
    ),
    algo_type = cms.int32( 1 )
)
process.hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    doSynchro = cms.bool( False )
)
process.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    deadSource = cms.string( "File" ),
    maskSource = cms.string( "File" )
)
process.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
    SMB_21 = cms.vdouble( 1.043, -0.124, 0.0, 0.183, 0.0, 0.0 ),
    SMB_20 = cms.vdouble( 1.011, -0.052, 0.0, 0.188, 0.0, 0.0 ),
    SMB_22 = cms.vdouble( 1.474, -0.758, 0.0, 0.185, 0.0, 0.0 ),
    OL_2213 = cms.vdouble( 0.117, 0.0, 0.0, 0.044, 0.0, 0.0 ),
    SME_11 = cms.vdouble( 3.295, -1.527, 0.112, 0.378, 0.02, 0.0 ),
    SME_13 = cms.vdouble( -1.286, 1.711, 0.0, 0.356, 0.0, 0.0 ),
    SME_12 = cms.vdouble( 0.102, 0.599, 0.0, 0.38, 0.0, 0.0 ),
    DT_34_2_scale = cms.vdouble( -11.901897, 0.0 ),
    OL_1213_0_scale = cms.vdouble( -4.488158, 0.0 ),
    OL_1222_0_scale = cms.vdouble( -5.810449, 0.0 ),
    DT_13 = cms.vdouble( 0.315, 0.068, -0.127, 0.051, -0.002, 0.0 ),
    DT_12 = cms.vdouble( 0.183, 0.054, -0.087, 0.028, 0.002, 0.0 ),
    DT_14 = cms.vdouble( 0.359, 0.052, -0.107, 0.072, -0.004, 0.0 ),
    CSC_13_3_scale = cms.vdouble( -1.701268, 0.0 ),
    DT_24_2_scale = cms.vdouble( -6.63094, 0.0 ),
    CSC_23 = cms.vdouble( -0.081, 0.113, -0.029, 0.015, 0.008, 0.0 ),
    CSC_24 = cms.vdouble( 0.004, 0.021, -0.002, 0.053, 0.0, 0.0 ),
    OL_2222 = cms.vdouble( 0.107, 0.0, 0.0, 0.04, 0.0, 0.0 ),
    DT_14_2_scale = cms.vdouble( -4.808546, 0.0 ),
    SMB_10 = cms.vdouble( 1.387, -0.038, 0.0, 0.19, 0.0, 0.0 ),
    SMB_11 = cms.vdouble( 1.247, 0.72, -0.802, 0.229, -0.075, 0.0 ),
    SMB_12 = cms.vdouble( 2.128, -0.956, 0.0, 0.199, 0.0, 0.0 ),
    SME_21 = cms.vdouble( -0.529, 1.194, -0.358, 0.472, 0.086, 0.0 ),
    SME_22 = cms.vdouble( -1.207, 1.491, -0.251, 0.189, 0.243, 0.0 ),
    DT_13_2_scale = cms.vdouble( -4.257687, 0.0 ),
    CSC_34 = cms.vdouble( 0.062, -0.067, 0.019, 0.021, 0.003, 0.0 ),
    SME_22_0_scale = cms.vdouble( -3.457901, 0.0 ),
    DT_24_1_scale = cms.vdouble( -7.490909, 0.0 ),
    OL_1232_0_scale = cms.vdouble( -5.964634, 0.0 ),
    DT_23_1_scale = cms.vdouble( -5.320346, 0.0 ),
    SME_13_0_scale = cms.vdouble( 0.104905, 0.0 ),
    SMB_22_0_scale = cms.vdouble( 1.346681, 0.0 ),
    CSC_12_1_scale = cms.vdouble( -6.434242, 0.0 ),
    DT_34 = cms.vdouble( 0.044, 0.004, -0.013, 0.029, 0.003, 0.0 ),
    SME_32 = cms.vdouble( -0.901, 1.333, -0.47, 0.41, 0.073, 0.0 ),
    SME_31 = cms.vdouble( -1.594, 1.482, -0.317, 0.487, 0.097, 0.0 ),
    CSC_13_2_scale = cms.vdouble( -6.077936, 0.0 ),
    crackEtas = cms.vdouble( 0.2, 1.6, 1.7 ),
    SME_11_0_scale = cms.vdouble( 1.325085, 0.0 ),
    SMB_20_0_scale = cms.vdouble( 1.486168, 0.0 ),
    DT_13_1_scale = cms.vdouble( -4.520923, 0.0 ),
    CSC_24_1_scale = cms.vdouble( -6.055701, 0.0 ),
    CSC_01_1_scale = cms.vdouble( -1.915329, 0.0 ),
    DT_23 = cms.vdouble( 0.13, 0.023, -0.057, 0.028, 0.004, 0.0 ),
    DT_24 = cms.vdouble( 0.176, 0.014, -0.051, 0.051, 0.003, 0.0 ),
    SMB_12_0_scale = cms.vdouble( 2.283221, 0.0 ),
    deltaPhiSearchWindow = cms.double( 0.25 ),
    SMB_30_0_scale = cms.vdouble( -3.629838, 0.0 ),
    SME_42 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    SME_41 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    deltaEtaSearchWindow = cms.double( 0.2 ),
    CSC_12_2_scale = cms.vdouble( -1.63622, 0.0 ),
    DT_34_1_scale = cms.vdouble( -13.783765, 0.0 ),
    CSC_34_1_scale = cms.vdouble( -11.520507, 0.0 ),
    OL_2213_0_scale = cms.vdouble( -7.239789, 0.0 ),
    SMB_32_0_scale = cms.vdouble( -3.054156, 0.0 ),
    CSC_12_3_scale = cms.vdouble( -1.63622, 0.0 ),
    deltaEtaCrackSearchWindow = cms.double( 0.25 ),
    SME_21_0_scale = cms.vdouble( -0.040862, 0.0 ),
    OL_1232 = cms.vdouble( 0.184, 0.0, 0.0, 0.066, 0.0, 0.0 ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
    SMB_10_0_scale = cms.vdouble( 2.448566, 0.0 ),
    EnableDTMeasurement = cms.bool( True ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    CSC_23_2_scale = cms.vdouble( -6.079917, 0.0 ),
    scaleDT = cms.bool( True ),
    DT_12_2_scale = cms.vdouble( -3.518165, 0.0 ),
    OL_1222 = cms.vdouble( 0.848, -0.591, 0.0, 0.062, 0.0, 0.0 ),
    CSC_23_1_scale = cms.vdouble( -19.084285, 0.0 ),
    OL_1213 = cms.vdouble( 0.96, -0.737, 0.0, 0.052, 0.0, 0.0 ),
    CSC_02 = cms.vdouble( 0.612, -0.207, 0.0, 0.067, -0.001, 0.0 ),
    CSC_03 = cms.vdouble( 0.787, -0.338, 0.029, 0.101, -0.008, 0.0 ),
    CSC_01 = cms.vdouble( 0.166, 0.0, 0.0, 0.031, 0.0, 0.0 ),
    SMB_32 = cms.vdouble( 0.67, -0.327, 0.0, 0.22, 0.0, 0.0 ),
    SMB_30 = cms.vdouble( 0.505, -0.022, 0.0, 0.215, 0.0, 0.0 ),
    SMB_31 = cms.vdouble( 0.549, -0.145, 0.0, 0.207, 0.0, 0.0 ),
    crackWindow = cms.double( 0.04 ),
    CSC_14_3_scale = cms.vdouble( -1.969563, 0.0 ),
    SMB_31_0_scale = cms.vdouble( -3.323768, 0.0 ),
    DT_12_1_scale = cms.vdouble( -3.692398, 0.0 ),
    SMB_21_0_scale = cms.vdouble( 1.58384, 0.0 ),
    DT_23_2_scale = cms.vdouble( -5.117625, 0.0 ),
    SME_12_0_scale = cms.vdouble( 2.279181, 0.0 ),
    DT_14_1_scale = cms.vdouble( -5.644816, 0.0 ),
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    SMB_11_0_scale = cms.vdouble( 2.56363, 0.0 ),
    EnableCSCMeasurement = cms.bool( True ),
    CSC_14 = cms.vdouble( 0.606, -0.181, -0.002, 0.111, -0.003, 0.0 ),
    OL_2222_0_scale = cms.vdouble( -7.667231, 0.0 ),
    CSC_13 = cms.vdouble( 0.901, -1.302, 0.533, 0.045, 0.005, 0.0 ),
    CSC_12 = cms.vdouble( -0.161, 0.254, -0.047, 0.042, -0.007, 0.0 )
)
process.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL1extraParticles" ),
    L1MaxEta = cms.double( 2.5 ),
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2OfflineMuonSeeds" ),
    L1MinPt = cms.double( 0.0 ),
    L1MinQuality = cms.uint32( 1 ),
    GMTReadoutCollection = cms.InputTag( "hltGtDigis" ),
    UseUnassociatedL1 = cms.bool( False ),
    UseOfflineSeed = cms.untracked.bool( True ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" )
)
process.hltL2Muons = cms.EDProducer( "L2MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    SeedTransformerParameters = cms.PSet( 
      Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      NMinRecHits = cms.uint32( 2 ),
      UseSubRecHits = cms.bool( False ),
      Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      RescaleError = cms.double( 100.0 )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      DoRefit = cms.bool( False ),
      SeedPropagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      FilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        FitDirection = cms.string( "insideOut" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 1000.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 0 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        NMinRecHits = cms.uint32( 2 ),
        UseSubRecHits = cms.bool( False ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        RescaleError = cms.double( 100.0 )
      ),
      DoBackwardFilter = cms.bool( True ),
      SeedPosition = cms.string( "in" ),
      BWFilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        FitDirection = cms.string( "outsideIn" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 100.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          Granularity = cms.int32( 0 ),
          ExcludeRPCFromFit = cms.bool( False ),
          UseInvalidHits = cms.bool( True ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False )
    ),
    DoSeedRefit = cms.bool( False ),
    TrackLoaderParameters = cms.PSet( 
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      DoSmoothing = cms.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
    ),
    MuonTrajectoryBuilder = cms.string( "Exhaustive" )
)
process.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL2Mu3N10HitQL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3MinBiasFiltered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltSinglePuAK4CaloJet40Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltL1sL1SingleMu3SingleCenJet28 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleCenJet28" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3CenJet28Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3SingleCenJet28" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu3N10HitQL2FilteredWithJet28 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3CenJet28Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu3SingleCenJet40 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleCenJet40" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3CenJet40Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3SingleCenJet40" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu3N10HitQL2FilteredWithJet40 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3CenJet40Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet100Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSinglePuAK4CaloJet100Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHIL2Mu3Eta2p5HIPhoton10Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL2Mu3Eta2p5HIPhoton15Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL2Mu3Eta2p5HIPhoton20Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1SingleMu3SingleEG12 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleEG12" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3Eta2p5HIPhoton30Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3EG12Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3SingleEG12" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu3N10HitQL2FilteredWithEG12 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3EG12Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu3SingleEG20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleEG20" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3Eta2p5HIPhoton40Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3EG20Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3SingleEG20" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu3N10HitQL2FilteredWithEG20 = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3EG20Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIUCC100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltTowerMakerForHf = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    MomHBDepth = cms.double( 0.2 ),
    UseEtEBTreshold = cms.bool( False ),
    hfInput = cms.InputTag( "hltHfrecoMethod0" ),
    AllowMissingInputs = cms.bool( True ),
    MomEEDepth = cms.double( 0.0 ),
    EESumThreshold = cms.double( 0.45 ),
    HBGrid = cms.vdouble(  ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    HBThreshold = cms.double( 0.7 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseEcalRecoveredHits = cms.bool( True ),
    MomConstrMethod = cms.int32( 1 ),
    MomHEDepth = cms.double( 0.4 ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    UseSymEBTreshold = cms.bool( False ),
    EEWeights = cms.vdouble(  ),
    EEWeight = cms.double( 1.0E-99 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HEDGrid = cms.vdouble(  ),
    EBWeight = cms.double( 1.0E-99 ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    HESWeight = cms.double( 1.0 ),
    HESThreshold = cms.double( 0.8 ),
    hbheInput = cms.InputTag( "" ),
    HF2Weight = cms.double( 1.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HcalAcceptSeverityLevel = cms.uint32( 11 ),
    EEThreshold = cms.double( 0.3 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "" ),
    HF1Threshold = cms.double( 0.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kProblematic',
      'kRecovered',
      'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    EBThreshold = cms.double( 0.07 ),
    UseRejectedHitsOnly = cms.bool( False ),
    UseHcalRecoveredHits = cms.bool( True ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HOThreshold0 = cms.double( 3.5 ),
    ecalInputs = cms.VInputTag(  ),
    UseRejectedRecoveredHcalHits = cms.bool( True ),
    MomEBDepth = cms.double( 0.3 ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  )
)
process.hltMetForHf = cms.EDProducer( "CaloMETProducer",
    alias = cms.string( "RawCaloMET" ),
    calculateSignificance = cms.bool( False ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False ),
    src = cms.InputTag( "hltTowerMakerForHf" )
)
process.hltGlobalSumETHfFilter4470 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 4470.0 ),
    Max = cms.double( 6400.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPixelActivityFilter40000 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 40000 )
)
process.hltPreHIUCC020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilter4680 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 4680.0 ),
    Max = cms.double( 6400.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltPixelActivityFilter60000 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 60000 )
)
process.hltL1sMinimumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIQ2Bottom005Centrality1030 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilterCentrality1030 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 1200.0 ),
    Max = cms.double( 3380.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltEvtPlaneProducer = cms.EDProducer( "EvtPlaneProducer",
    maxet = cms.double( -1.0 ),
    caloCentRefWidth = cms.double( 5.0 ),
    dzerr = cms.double( 10.0 ),
    centralityVariable = cms.string( "HFtowers" ),
    NumFlatBins = cms.int32( 40 ),
    minpt = cms.double( 0.3 ),
    caloCentRef = cms.double( 80.0 ),
    chi2 = cms.double( 40.0 ),
    minet = cms.double( -1.0 ),
    trackTag = cms.InputTag( "hiGeneralTracksDummy" ),
    centralityBinTag = cms.InputTag( 'centralityBin','HFtowersDummy' ),
    FlatOrder = cms.int32( 9 ),
    maxpt = cms.double( 3.0 ),
    minvtx = cms.double( -9999.0 ),
    caloTag = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" ),
    vertexTag = cms.InputTag( "hiSelectedVertexDummy" ),
    castorTag = cms.InputTag( "CastorTowerRecoDummy" ),
    maxvtx = cms.double( 9999.0 ),
    CentBinCompression = cms.int32( 5 ),
    loadDB = cms.bool( False ),
    nonDefaultGlauberModel = cms.string( "" )
)
process.hltEvtPlaneFilterB005Cent1030 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
process.hltPreHIQ2Top005Centrality1030 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEvtPlaneFilterT005Cent1030 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.145 )
)
process.hltPreHIQ2Bottom005Centrality3050 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilterCentrality3050 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 400.0 ),
    Max = cms.double( 1500.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltEvtPlaneFilterB005Cent3050 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
process.hltPreHIQ2Top005Centrality3050 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEvtPlaneFilterT005Cent3050 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.183 )
)
process.hltPreHIQ2Bottom005Centrality5070 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltGlobalSumETHfFilterCentrality5070 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( False ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 60.0 ),
    Max = cms.double( 600.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
process.hltEvtPlaneFilterB005Cent5070 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
process.hltPreHIQ2Top005Centrality5070 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEvtPlaneFilterT005Cent5070 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.223 )
)
process.hltPreHIFullTrack12L1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIFullTrackSelectedTracks = cms.EDProducer( "AnalyticalTrackSelector",
    max_d0 = cms.double( 100.0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    max_lostHitFraction = cms.double( 1.0 ),
    applyAbsCutsIfNoPV = cms.bool( False ),
    qualityBit = cms.string( "highPurity" ),
    minNumberLayers = cms.uint32( 0 ),
    chi2n_par = cms.double( 9999.0 ),
    useVtxError = cms.bool( True ),
    nSigmaZ = cms.double( 100.0 ),
    dz_par2 = cms.vdouble( 3.5, 0.0 ),
    applyAdaptedPVCuts = cms.bool( True ),
    min_eta = cms.double( -9999.0 ),
    dz_par1 = cms.vdouble( 9999.0, 0.0 ),
    copyTrajectories = cms.untracked.bool( True ),
    vtxNumber = cms.int32( -1 ),
    max_d0NoPV = cms.double( 100.0 ),
    keepAllTracks = cms.bool( False ),
    maxNumberLostLayers = cms.uint32( 999 ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    max_relpterr = cms.double( 0.1 ),
    copyExtras = cms.untracked.bool( True ),
    max_z0NoPV = cms.double( 100.0 ),
    vertexCut = cms.string( "" ),
    max_z0 = cms.double( 100.0 ),
    useVertices = cms.bool( True ),
    min_nhits = cms.uint32( 10 ),
    src = cms.InputTag( "hltHIIterTrackingMergedHighPurityForGlobalPt8" ),
    max_minMissHitOutOrIn = cms.int32( 99 ),
    chi2n_no1Dmod_par = cms.double( 0.25 ),
    vertices = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    max_eta = cms.double( 9999.0 ),
    d0_par2 = cms.vdouble( 3.5, 0.0 ),
    d0_par1 = cms.vdouble( 9999.0, 0.0 ),
    res_par = cms.vdouble( 9999.0, 9999.0 ),
    minHitsToBypassChecks = cms.uint32( 20 )
)
process.hltHIFullTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIFullTrackSelectedTracks" ),
    particleType = cms.string( "pi+" )
)
process.hltHIFullTrackFilter12 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 12.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 1.05 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack12L1Centrality010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sSingleTrack12Centrality30100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack12_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack12L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack18L1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIFullTrackFilter18 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 18.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 1.05 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack18L1Centrality010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack18L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sSingleTrack16 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack16_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack24 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIFullTrackFilter24 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 24.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 1.05 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltL1sSingleTrack16Centrality30100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack16_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack24L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sSingleTrack24 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack24_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack34 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIFullTrackFilter34 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 34.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 1.05 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltL1sSingleTrack24Centrality30100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack24_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIFullTrack34L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIFullTrack45 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIFullTrackFilter45 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
    saveTags = cms.bool( True ),
    MinTrks = cms.int32( 1 ),
    MinPt = cms.double( 45.0 ),
    MaxVz = cms.double( 15.0 ),
    MaxEta = cms.double( 1.05 ),
    trackCollection = cms.InputTag( "hltHIFullTrackCandsForHITrackTrigger" ),
    vertexCollection = cms.InputTag( "hltHISelectedVertex" ),
    MaxPt = cms.double( 10000.0 ),
    MinSep = cms.double( 0.2 )
)
process.hltPreHIFullTrack45L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleMu0BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1DoubleMu0MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu02HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0MinBiasL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1DoubleMu0HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu02HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1DoubleMu10BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu10_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu10L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu10BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIL2DoubleMu0NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0L1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL2DoubleMu0NHitQ2HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0NHitQ2HFFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0MinBiasL1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL2DoubleMu0NHitQ2HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0NHitQ2HF0Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0HFTower0Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL2Mu3NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2Mu3N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3MinBiasFiltered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu3HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu3NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu3HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu3N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu3HFTower0Filtered" ),
    MinPt = cms.double( 3.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu3NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
process.hltHISiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    DoAPVEmulatorCheck = cms.bool( False ),
    Algorithms = cms.PSet( 
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      CommonModeNoiseSubtractionMode = cms.string( "IteratedMedian" ),
      PedestalSubtractionFedMode = cms.bool( False ),
      TruncateInSuppressor = cms.bool( True ),
      doAPVRestore = cms.bool( True ),
      useCMMeanMap = cms.bool( False ),
      CutToAvoidSignal = cms.double( 2.0 ),
      Fraction = cms.double( 0.2 ),
      minStripsToFit = cms.uint32( 4 ),
      consecThreshold = cms.uint32( 5 ),
      hitStripThreshold = cms.uint32( 40 ),
      Deviation = cms.uint32( 25 ),
      restoreThreshold = cms.double( 0.5 ),
      APVInspectMode = cms.string( "BaselineFollower" ),
      ForceNoRestore = cms.bool( False ),
      useRealMeanCM = cms.bool( False ),
      DeltaCMThreshold = cms.uint32( 20 ),
      nSigmaNoiseDerTh = cms.uint32( 4 ),
      nSaturatedStrip = cms.uint32( 2 ),
      APVRestoreMode = cms.string( "BaselineFollower" ),
      distortionThreshold = cms.uint32( 20 ),
      Iterations = cms.int32( 3 ),
      nSmooth = cms.uint32( 9 ),
      SelfSelectRestoreAlgo = cms.bool( False ),
      MeanCM = cms.int32( 0 ),
      CleaningSequence = cms.uint32( 1 ),
      slopeX = cms.int32( 3 ),
      slopeY = cms.int32( 4 ),
      ApplyBaselineRejection = cms.bool( True ),
      filteredBaselineMax = cms.double( 6.0 ),
      filteredBaselineDerivativeSumSquare = cms.double( 30.0 ),
      ApplyBaselineCleaner = cms.bool( True )
    ),
    Clusterizer = cms.PSet( 
      ChannelThreshold = cms.double( 2.0 ),
      MaxSequentialBad = cms.uint32( 1 ),
      MaxSequentialHoles = cms.uint32( 0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      MaxAdjacentBad = cms.uint32( 0 ),
      QualityLabel = cms.string( "" ),
      SeedThreshold = cms.double( 3.0 ),
      ClusterThreshold = cms.double( 5.0 ),
      setDetId = cms.bool( True ),
      RemoveApvShots = cms.bool( True ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    onDemand = cms.bool( True )
)
process.hltHISiStripClusters = cms.EDProducer( "MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(  ),
    stripClusterProducer = cms.string( "hltHISiStripRawToClustersFacility" ),
    pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
    switchOffPixelsIfEmpty = cms.bool( True ),
    inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
    skipClusters = cms.InputTag( "" ),
    measurementTracker = cms.string( "hltESPMeasurementTrackerForHI" )
)
process.hltHIL3TrajSeedOIState = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      propagatorCompatibleName = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
      option = cms.uint32( 3 ),
      maxChi2 = cms.double( 40.0 ),
      errorMatrixPset = cms.PSet( 
        atIP = cms.bool( True ),
        action = cms.string( "use" ),
        errorMatrixValuesPSet = cms.PSet( 
          pf3_V12 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V13 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V11 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V14 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V15 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
          pf3_V33 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V44 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
          pf3_V22 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V23 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V45 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V55 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
          ),
          pf3_V34 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V35 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V25 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          ),
          pf3_V24 = cms.PSet( 
            action = cms.string( "scale" ),
            values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
          )
        )
      ),
      propagatorName = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
      manySeeds = cms.bool( False ),
      copyMuonRecHit = cms.bool( False ),
      ComponentName = cms.string( "TSGForRoadSearch" ),
      MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSteppingHelixPropagatorOpposite',
        'hltESPSteppingHelixPropagatorAlong' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet(  ),
    PtCut = cms.double( 1.0 )
)
process.hltHIL3TrackCandidateFromL2OIState = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltHIL3TrajSeedOIState" ),
    reverseTrajectories = cms.bool( True ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryBuilder" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltHIL3TkTracksFromL2OIState = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIState" ),
    SimpleMagneticField = cms.string( "" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "hltIterX" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    GeometricInnerState = cms.bool( True ),
    useSimpleMF = cms.bool( False ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltHIL3MuonsOIState = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.001 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltHIL3TkTracksFromL2OIState" ),
      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      tkTrajMaxChi2 = cms.double( 9999.0 ),
      tkTrajMaxDXYBeamSpot = cms.double( 0.2 ),
      tkTrajVertex = cms.InputTag( "pixelVertices" ),
      tkTrajUseVertex = cms.bool( False ),
      MuonTrackingRegionBuilder = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonTrackingRegionBuilder8356" ) )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL3TrajSeedOIHit = cms.EDProducer( "TSGFromL2Muon",
    TkSeedGenerator = cms.PSet( 
      PSetNames = cms.vstring( 'skipTSG',
        'iterativeTSG' ),
      L3TkCollectionA = cms.InputTag( "hltHIL3MuonsOIState" ),
      iterativeTSG = cms.PSet( 
        ErrorRescaling = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "unused" ),
        MaxChi2 = cms.double( 40.0 ),
        errorMatrixPset = cms.PSet( 
          atIP = cms.bool( True ),
          action = cms.string( "use" ),
          errorMatrixValuesPSet = cms.PSet( 
            pf3_V12 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V13 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V11 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V14 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V15 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            yAxis = cms.vdouble( 0.0, 1.0, 1.4, 10.0 ),
            pf3_V33 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            zAxis = cms.vdouble( -3.14159, 3.14159 ),
            pf3_V44 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            xAxis = cms.vdouble( 0.0, 13.0, 30.0, 70.0, 1000.0 ),
            pf3_V22 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V23 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V45 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V55 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 3.0, 3.0, 3.0, 5.0, 4.0, 5.0, 10.0, 7.0, 10.0, 10.0, 10.0, 10.0 )
            ),
            pf3_V34 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V35 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V25 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            ),
            pf3_V24 = cms.PSet( 
              action = cms.string( "scale" ),
              values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 )
            )
          )
        ),
        UpdateState = cms.bool( True ),
        MeasurementTrackerName = cms.string( "hltESPMeasurementTrackerForHI" ),
        SelectState = cms.bool( False ),
        SigmaZ = cms.double( 25.0 ),
        ResetMethod = cms.string( "matrix" ),
        ComponentName = cms.string( "TSGFromPropagation" ),
        UseVertexState = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
        MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" )
      ),
      skipTSG = cms.PSet(  ),
      ComponentName = cms.string( "DualByL2TSG" )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'PropagatorWithMaterial',
        'hltESPSmartPropagatorAnyOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    MuonTrackingRegionBuilder = cms.PSet(  ),
    PCut = cms.double( 2.5 ),
    TrackerSeedCleaner = cms.PSet( 
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      directionCleaner = cms.bool( True )
    ),
    PtCut = cms.double( 1.0 )
)
process.hltHIL3TrackCandidateFromL2OIHit = cms.EDProducer( "CkfTrajectoryMaker",
    src = cms.InputTag( "hltHIL3TrajSeedOIHit" ),
    reverseTrajectories = cms.bool( True ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" ),
    cleanTrajectoryAfterInOut = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    doSeedingRegionRebuilding = cms.bool( False ),
    trackCandidateAlso = cms.bool( True ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryBuilder" ) ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    TrajectoryBuilder = cms.string( "" ),
    maxNSeeds = cms.uint32( 100000 )
)
process.hltHIL3TkTracksFromL2OIHit = cms.EDProducer( "TrackProducer",
    src = cms.InputTag( "hltHIL3TrackCandidateFromL2OIHit" ),
    SimpleMagneticField = cms.string( "" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltHISiStripClusters" ),
    Fitter = cms.string( "hltESPKFFittingSmoother" ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTracker = cms.string( "" ),
    AlgorithmName = cms.string( "hltIterX" ),
    alias = cms.untracked.string( "" ),
    NavigationSchool = cms.string( "" ),
    TrajectoryInEvent = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    GeometricInnerState = cms.bool( True ),
    useSimpleMF = cms.bool( False ),
    Propagator = cms.string( "PropagatorWithMaterial" )
)
process.hltHIL3MuonsOIHit = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      ScaleTECyFactor = cms.double( -1.0 ),
      GlbRefitterParameters = cms.PSet( 
        TrackerSkipSection = cms.int32( -1 ),
        DoPredictionsOnly = cms.bool( False ),
        PropDirForCosmics = cms.bool( False ),
        HitThreshold = cms.int32( 1 ),
        MuonHitsOption = cms.int32( 1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        TrackerSkipSystem = cms.int32( -1 ),
        DYTthrs = cms.vint32( 30, 15 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      RefitRPCHits = cms.bool( True ),
      PCut = cms.double( 2.5 ),
      TrackTransformer = cms.PSet( 
        DoPredictionsOnly = cms.bool( False ),
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        LocChi2Cut = cms.double( 0.001 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_3 = cms.double( 7.0 ),
        Quality_2 = cms.double( 15.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        Quality_1 = cms.double( 20.0 )
      ),
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      tkTrajLabel = cms.InputTag( "hltHIL3TkTracksFromL2OIHit" ),
      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      tkTrajMaxChi2 = cms.double( 9999.0 ),
      tkTrajMaxDXYBeamSpot = cms.double( 0.2 ),
      tkTrajVertex = cms.InputTag( "pixelVertices" ),
      tkTrajUseVertex = cms.bool( False ),
      MuonTrackingRegionBuilder = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonTrackingRegionBuilder8356" ) )
    ),
    TrackLoaderParameters = cms.PSet( 
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      SmoothTkTrack = cms.untracked.bool( False ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( False ),
      DoSmoothing = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltHIL3TkFromL2OICombination = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
process.hltHIL3TrajectorySeed = cms.EDProducer( "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag( 'hltHIL3TrajSeedOIState','hltHIL3TrajSeedOIHit' )
)
process.hltHIL3TrackCandidateFromL2 = cms.EDProducer( "L3TrackCandCombiner",
    labels = cms.VInputTag( 'hltHIL3TrackCandidateFromL2OIHit','hltHIL3TrackCandidateFromL2OIState' )
)
process.hltHIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
process.hltHIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
process.hltHIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputLinksObjects = cms.InputTag( "hltHIL3MuonsLinksCombination" ),
    InputObjects = cms.InputTag( "hltHIL3Muons" ),
    MuonPtOption = cms.string( "Tracker" )
)
process.hltHISingleMu3NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu3N10HitQ2HFL2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 3.0 )
)
process.hltPreHIL3Mu3NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleMu3NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu3N10HitQ2HF0L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 3.0 )
)
process.hltL1sL1SingleMu5MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu5NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu5MinBiasFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu5MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu5N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu5MinBiasFiltered" ),
    MinPt = cms.double( 5.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu5HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu5NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu5HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu5HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu5N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu5HFTower0Filtered" ),
    MinPt = cms.double( 5.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu5NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleMu5NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu5N10HitQ2HFL2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 5.0 )
)
process.hltPreHIL3Mu5NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleMu5NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu5N10HitQ2HF0L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 5.0 )
)
process.hltL1sL1SingleMu7MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7_MinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu7NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu7MinBiasFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu7N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu7MinBiasFiltered" ),
    MinPt = cms.double( 7.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu7HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu7NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu7HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu7N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu7HFTower0Filtered" ),
    MinPt = cms.double( 7.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 10 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu7NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleMu7NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu7N10HitQ2HFL2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 7.0 )
)
process.hltPreHIL3Mu7NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHISingleMu7NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL2Mu7N10HitQ2HF0L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 15 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 7.0 )
)
process.hltL1sL1SingleMu12BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu12Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu12BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu15L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12Filtered" ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu12MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_MinimumBiasHF1_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu12MinBiasFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu12MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu152HFFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12MinBiasFiltered" ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu12HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu12HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu12HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu15N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12HFTower0Filtered" ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu15L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12Filtered" ),
    MinPt = cms.double( 12.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHISingleMu15L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu15L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 15.0 )
)
process.hltPreHIL3Mu152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu152HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12MinBiasFiltered" ),
    MinPt = cms.double( 12.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHISingleMu152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu152HFL2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 15.0 )
)
process.hltPreHIL3Mu152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu152HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu12HFTower0Filtered" ),
    MinPt = cms.double( 12.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHISingleMu152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu152HF0L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 15.0 )
)
process.hltL1sL1SingleMu16BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu16Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu16BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu20L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16Filtered" ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu16MinBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_MinimumBiasHF1_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu202HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu16MinBiasFiltered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu16MinBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu202HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16MinBiasFiltered" ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltL1sL1SingleMu16HFTower0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_HFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL2Mu202HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL1SingleMu16HFTower0Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( False ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu16HFTower0" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltHIL2Mu202HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16HFTower0Filtered" ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltPreHIL3Mu20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu20L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16Filtered" ),
    MinPt = cms.double( 16.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHIL3SingleMu20L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu20L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 20.0 )
)
process.hltPreHIL3Mu202HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu202HFL2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16MinBiasFiltered" ),
    MinPt = cms.double( 16.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHISingleMu202HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu202HFL2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 20.0 )
)
process.hltPreHIL3Mu202HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL3Mu202HF0L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIL1SingleMu16HFTower0Filtered" ),
    MinPt = cms.double( 16.0 ),
    MinN = cms.int32( 1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHISingleMu202HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
    MaxNormalizedChi2 = cms.double( 20.0 ),
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltHIL3Mu202HF0L2Filtered" ),
    MinNmuonHits = cms.int32( 0 ),
    MinN = cms.int32( 1 ),
    MinTrackPt = cms.double( 0.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxDXYBeamSpot = cms.double( 0.1 ),
    MinNhits = cms.int32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MaxDz = cms.double( 9999.0 ),
    MaxPtDifference = cms.double( 9999.0 ),
    MaxDr = cms.double( 2.0 ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    MinDXYBeamSpot = cms.double( -1.0 ),
    MinDr = cms.double( -1.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    InputLinks = cms.InputTag( "" ),
    MinPt = cms.double( 20.0 )
)
process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30to100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND_Centrality_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu02HFCent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0MinBiasCent30to100L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30to100" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sL1DoubleMu0HFTower0Centrality30to100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_HFplusANDminusTH0_Centrliatiy_ext30_100_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu02HF0Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0HFTower0Cent30to100L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0HFTower0Centrality30to100" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIL2DoubleMu02HFCent30100NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu02HFcent30100NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0MinBiasCent30to100L1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND_Centrality_ext0_30_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1DoubleMu0Cent30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDoubleMu0MinBiasCent30L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIL2DoubleMu02HF0Cent30100NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu02HF0cent30100NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0HFTower0Cent30to100L1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL2DoubleMu0Cent30OSNHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0cent30OSNHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0MinBiasCent30L1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL2DoubleMu0Cent30NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIL2DoubleMu0cent30NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MinNchambers = cms.int32( 2 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0MinBiasCent30L1Filtered" ),
    MinAngle = cms.double( -999.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtMin = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinPtMax = cms.double( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxAngle = cms.double( 2.5 ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.double( 0.0 ),
    MaxDr = cms.double( 100.0 ),
    MinAcop = cms.double( -1.0 ),
    MinNstations = cms.int32( 0 ),
    MinNhits = cms.int32( 1 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MinInvMass = cms.double( 1.6 )
)
process.hltPreHIL3DoubleMu0Cent30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonOpenCentrality30L2Filtered = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0MinBiasCent30L1Filtered" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHIDimuonOpenCentrality30L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( 0 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonOpenCentrality30L2Filtered" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 300.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 0.0 )
)
process.hltPreHIL3DoubleMu0Cent30OSm2p5to4p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonOpenCentrality30OSm2p5to4p5L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonOpenCentrality30L2Filtered" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 4.5 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 2.5 )
)
process.hltPreHIL3DoubleMu0Cent30OSm7to14 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonOpenCentrality30OSm7to14L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonOpenCentrality30L2Filtered" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 14.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 7.0 )
)
process.hltPreHIL3DoubleMu0OSm2p5to4p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonOpenL2FilteredNoMBHFgated = cms.EDFilter( "HLTMuonL2PreFilter",
    saveTags = cms.bool( True ),
    MaxDr = cms.double( 9999.0 ),
    CutOnChambers = cms.bool( False ),
    PreviousCandTag = cms.InputTag( "hltHIDoubleMu0L1Filtered" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SeedMapTag = cms.InputTag( "hltL2Muons" ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.vint32( 0 ),
    MinDxySig = cms.double( -1.0 ),
    MinNchambers = cms.vint32( 0 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MaxDz = cms.double( 9999.0 ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MinDr = cms.double( -1.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinNstations = cms.vint32( 0 )
)
process.hltHIDimuonOpenOSm2p5to4p5L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonOpenL2FilteredNoMBHFgated" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 4.5 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 2.5 )
)
process.hltPreHIL3DoubleMu0OSm7to14 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIDimuonOpenOSm7to14L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
    saveTags = cms.bool( True ),
    ChargeOpt = cms.int32( -1 ),
    MaxPtMin = cms.vdouble( 1.0E125 ),
    FastAccept = cms.bool( False ),
    CandTag = cms.InputTag( "hltHIL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltHIDimuonOpenL2FilteredNoMBHFgated" ),
    MaxPtBalance = cms.double( 999999.0 ),
    MaxPtPair = cms.vdouble( 1.0E125 ),
    MaxAcop = cms.double( 999.0 ),
    MinPtMin = cms.vdouble( 0.0 ),
    MaxInvMass = cms.vdouble( 14.0 ),
    MinPtMax = cms.vdouble( 0.0 ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    MaxDz = cms.double( 9999.0 ),
    MinPtPair = cms.vdouble( 0.0 ),
    MaxDr = cms.double( 20.0 ),
    MinAcop = cms.double( -999.0 ),
    MaxDCAMuMu = cms.double( 999.0 ),
    MinNhits = cms.int32( 0 ),
    NSigmaPt = cms.double( 0.0 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxEta = cms.double( 2.5 ),
    MaxRapidityPair = cms.double( 999999.0 ),
    CutCowboys = cms.bool( False ),
    MinInvMass = cms.vdouble( 7.0 )
)
process.hltL1sL1MuOpenNotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1SingleMuOpenNotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenNotHF2L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1MuOpenNotMinimumBiasHF2AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCSingleMuNotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelLayerTripletsForUPC = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'BPix1+BPix2+BPix3',
      'BPix1+BPix2+FPix1_pos',
      'BPix1+BPix2+FPix1_neg',
      'BPix1+FPix1_pos+FPix2_pos',
      'BPix1+FPix1_neg+FPix2_neg' ),
    MTOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTID = cms.PSet(  ),
    FPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHits" ),
      hitErrorRZ = cms.double( 0.0036 )
    ),
    MTEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    BPix = cms.PSet( 
      useErrorsFromParam = cms.bool( True ),
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      HitProducer = cms.string( "hltHISiPixelRecHits" ),
      hitErrorRZ = cms.double( 0.006 )
    ),
    TIB = cms.PSet(  )
)
process.hltPixelTracksForUPC = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 ),
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      ptMin = cms.double( 0.1 ),
      tipMax = cms.double( 1.0 )
    ),
    passLabel = cms.string( "" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      fixImpactParameter = cms.double( 0.0 )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.1 ),
        originHalfLength = cms.double( 24.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        maxElement = cms.uint32( 100000 ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useMultScattering = cms.bool( True ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCache" )
        ),
        extraHitRZtolerance = cms.double( 0.06 ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" )
      ),
      SeedingLayers = cms.InputTag( "hltPixelLayerTripletsForUPC" )
    )
)
process.hltPixelCandsForUPC = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForUPC" ),
    particleType = cms.string( "pi+" )
)
process.hltSinglePixelTrackForUPC = cms.EDFilter( "CandViewCountFilter",
    src = cms.InputTag( "hltPixelCandsForUPC" ),
    minNumber = cms.uint32( 1 )
)
process.hltL1sL1DoubleMuOpenNotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenNotHF2L1Filtered2 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenNotMinimumBiasHF2AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCDoubleMuNotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1EG2NotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_EG2_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1SingleEG2NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCSingleEG2NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleEG2NotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG2_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleEG2NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCDoubleEG2NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1EG5NotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_EG5_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1SingleEG5NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCSingleEG5NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleMuOpenNotMinimumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotMinimumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotHF1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenL1Filtered2 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenNotMinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCDoubleMuNotHF1PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleEG2NotZDCAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG2_NotZdc_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleEG2NotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1DoubleEG2NotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1DoubleMuOpenNotZDCAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotZdc_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenL1Filtered3 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpenNotZDCAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1EG2NotZDCAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2_NotZDC_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1EG2NotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCEG2NotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1MuOpenNotZDCAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_NotZdc_AND_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1MuOpenNotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1MuOpenL1Filtered4 = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1MuOpenNotZDCAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCL1MuOpenNotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1NotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotHFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1NotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1NotHFplusANDminusTH0BptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotHFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1DoubleMuOpenNotHFplusANDminustTH0BptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpenNotHFplusANDminusTH0_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1DoubleMuOpenTH0L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sHLTHIUPCL1DoubleMuOpenNotHFplusANDminustTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1CastorMediumJet = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_CastorMediumJet_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1CastorMediumJet = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1CastorMediumJetAK4CaloJet20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPuAK4CaloJetsUPC = cms.EDProducer( "FastjetJetProducer",
    Active_Area_Repeats = cms.int32( 1 ),
    doAreaFastjet = cms.bool( True ),
    voronoiRfact = cms.double( -0.9 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    doAreaDiskApprox = cms.bool( False ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    jetType = cms.string( "CaloJet" ),
    minSeed = cms.uint32( 14327 ),
    Ghost_EtaMax = cms.double( 6.5 ),
    doRhoFastjet = cms.bool( False ),
    jetAlgorithm = cms.string( "AntiKt" ),
    nSigmaPU = cms.double( 1.0 ),
    GhostArea = cms.double( 0.01 ),
    Rho_EtaMax = cms.double( 4.4 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    useDeterministicSeed = cms.bool( True ),
    doPVCorrection = cms.bool( False ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    rParam = cms.double( 0.4 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    doOutputJets = cms.bool( True ),
    src = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" ),
    inputEtMin = cms.double( 0.3 ),
    puPtMin = cms.double( 8.0 ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetPtMin = cms.double( 10.0 ),
    radiusPU = cms.double( 0.5 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    doPUOffsetCorr = cms.bool( True ),
    inputEMin = cms.double( 0.0 ),
    useMassDropTagger = cms.bool( False ),
    muMin = cms.double( -1.0 ),
    subtractorName = cms.string( "MultipleAlgoIterator" ),
    muCut = cms.double( -1.0 ),
    subjetPtMin = cms.double( -1.0 ),
    useTrimming = cms.bool( False ),
    muMax = cms.double( -1.0 ),
    yMin = cms.double( -1.0 ),
    useFiltering = cms.bool( False ),
    rFilt = cms.double( -1.0 ),
    yMax = cms.double( -1.0 ),
    zcut = cms.double( -1.0 ),
    MinVtxNdof = cms.int32( 5 ),
    MaxVtxZ = cms.double( 15.0 ),
    UseOnlyVertexTracks = cms.bool( False ),
    dRMin = cms.double( -1.0 ),
    nFilt = cms.int32( -1 ),
    usePruning = cms.bool( False ),
    maxDepth = cms.int32( -1 ),
    yCut = cms.double( -1.0 ),
    DzTrVtxMax = cms.double( 0.0 ),
    UseOnlyOnePV = cms.bool( False ),
    rcut_factor = cms.double( -1.0 ),
    sumRecHits = cms.bool( False ),
    trimPtFracMin = cms.double( -1.0 ),
    dRMax = cms.double( -1.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    useCMSBoostedTauSeedingAlgorithm = cms.bool( False )
)
process.hltPuAK4CaloJetsUPCIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( 2 ),
    min_EMF = cms.double( 1.0E-6 ),
    jetsInput = cms.InputTag( "hltPuAK4CaloJetsUPC" ),
    JetIDParams = cms.PSet( 
      useRecHits = cms.bool( True ),
      hbheRecHitsColl = cms.InputTag( "hltHbherecoMethod0" ),
      hoRecHitsColl = cms.InputTag( "hltHorecoMethod0" ),
      hfRecHitsColl = cms.InputTag( "hltHfrecoMethod0" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' )
    ),
    max_EMF = cms.double( 999.0 )
)
process.hltPuAK4CaloJetsUPCCorrected = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsUPC" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
process.hltPuAK4CaloJetsUPCCorrectedIDPassed = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsUPCIDPassed" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
process.hltSinglePuAK4CaloJet20Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsUPCCorrectedIDPassed" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
process.hltPreHICastorMediumJetPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1CastorMediumJetFiltered0UPC = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1CastorMediumJet" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltL1sHLTHIUPCL1NotMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotMinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1NotMinimumBiasHF2AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1NotMinimumBiasHF2ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1ZdcORBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZdcOR_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1ZdcORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1ZdcORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1ZdcXORBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZdcXOR_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1ZdcXORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1ZdcXORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sHLTHIUPCL1NotZdcORBptxAND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotZdcOR_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIUPCL1NotZdcORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIUPCL1NotZdcORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1ZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHICentralityVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPixelActivityFilterCentralityVeto = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000 ),
    saveTags = cms.bool( False ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 3 )
)
process.hltL1sL1Tech5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "5" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPreHIL1Tech5BPTXPlusOnly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Tech6 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "6" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPreHIL1Tech6BPTXMinusOnly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sTech7 = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "7" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( True )
)
process.hltPreHIL1Tech7NoBPTX = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sMinimumBiasHF1OR = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_OR" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1MinimumBiasHF1OR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sMinimumBiasHF2OR = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF2_OR" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1MinimumBiasHF2OR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sMinimumBiasHF2AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF2_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1MinimumBiasHF2AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1MinimumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHIPixel3ProtoTracksForHITrackTrigger = cms.EDProducer( "PixelTrackProducer",
    useFilterWithES = cms.bool( False ),
    FilterPSet = cms.PSet( 
      chi2 = cms.double( 1000.0 ),
      ComponentName = cms.string( "HIProtoTrackFilter" ),
      ptMin = cms.double( 0.4 ),
      tipMax = cms.double( 1.0 ),
      doVariablePtMin = cms.bool( True ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" )
    ),
    passLabel = cms.string( "Pixel triplet tracks for vertexing" ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
    ),
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "HITrackingRegionForPrimaryVtxProducer" ),
      RegionPSet = cms.PSet( 
        precise = cms.bool( True ),
        originRadius = cms.double( 0.2 ),
        ptMin = cms.double( 0.4 ),
        directionXCoord = cms.double( 1.0 ),
        directionZCoord = cms.double( 0.0 ),
        directionYCoord = cms.double( 1.0 ),
        useFoundVertices = cms.bool( True ),
        doVariablePtMin = cms.bool( True ),
        nSigmaZ = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useFixedError = cms.bool( True ),
        fixedError = cms.double( 3.0 ),
        sigmaZVertex = cms.double( 3.0 ),
        siPixelRecHits = cms.InputTag( "hltHISiPixelRecHits" ),
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" )
      )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      GeneratorPSet = cms.PSet( 
        useBending = cms.bool( True ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRPhitolerance = cms.double( 0.032 ),
        useMultScattering = cms.bool( True ),
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        extraHitRZtolerance = cms.double( 0.037 ),
        maxElement = cms.uint32( 100000 ),
        SeedComparitorPSet = cms.PSet( 
          ComponentName = cms.string( "none" ),
          clusterShapeCacheSrc = cms.InputTag( "hltHISiPixelClustersCache" )
        )
      ),
      SeedingLayers = cms.InputTag( "hltHIPixelLayerTriplets" )
    )
)
process.hltPixelCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIPixel3ProtoTracksForHITrackTrigger" ),
    particleType = cms.string( "pi+" )
)
process.hltHISinglePixelTrackFilter = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCandsForHITrackTrigger" ),
    saveTags = cms.bool( False ),
    MinTrks = cms.uint32( 1 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 1.0 )
)
process.hltPreHIZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1sL1Centralityext70100MinimumumBiasHF1AND = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext70_100_MinimumumBiasHF1_AND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreHIL1Centralityext70100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1Centralityext70100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1Centralityext50100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1Centralityext50100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1Centralityext30100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIL1Centralityext30100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
process.hltPreHIRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
process.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 1024 )
)
process.hltPreHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 ),
    FilterSummary = cms.untracked.bool( False )
)
process.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 1024 )
)
process.hltPreAlCaEcalPhiSymForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltEcalPreshowerDigis = cms.EDProducer( "ESRawToDigi",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    debugMode = cms.untracked.bool( False ),
    InstanceES = cms.string( "" ),
    ESdigiCollection = cms.string( "" ),
    LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" )
)
process.hltEcalPreshowerRecHit = cms.EDProducer( "ESRecHitProducer",
    ESRecoAlgo = cms.int32( 0 ),
    ESrechitCollection = cms.string( "EcalRecHitsES" ),
    algo = cms.string( "ESRecHitWorker" ),
    ESdigiCollection = cms.InputTag( "hltEcalPreshowerDigis" )
)
process.hltEcal50nsMultifitPhiSymFilter = cms.EDFilter( "HLTEcalPhiSymFilter",
    ampCut_endcapM = cms.vdouble( 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0 ),
    phiSymBarrelDigiCollection = cms.string( "phiSymEcalDigisEB" ),
    phiSymEndcapDigiCollection = cms.string( "phiSymEcalDigisEE" ),
    barrelDigiCollection = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    barrelUncalibHitCollection = cms.InputTag( 'hltEcalUncalibRecHit50nsMultiFit','EcalUncalibRecHitsEB' ),
    statusThreshold = cms.uint32( 3 ),
    useRecoFlag = cms.bool( False ),
    endcapUncalibHitCollection = cms.InputTag( 'hltEcalUncalibRecHit50nsMultiFit','EcalUncalibRecHitsEE' ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    ampCut_barrelM = cms.vdouble( 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0 ),
    endcapDigiCollection = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    barrelHitCollection = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    ampCut_endcapP = cms.vdouble( 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0 ),
    ampCut_barrelP = cms.vdouble( 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0 )
)
process.hltL1sAlCaRPCForHI = cms.EDFilter( "HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen OR L1_SingleMu12_BptxAND" ),
    saveTags = cms.bool( True ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" ),
    L1UseL1TriggerObjectMaps = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1NrBxInEvent = cms.int32( 3 ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1TechTriggerSeeding = cms.bool( False )
)
process.hltPreAlCaRPCMuonNoTriggersForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCMuonNoTriggersL1Filtered0ForHI = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sAlCaRPCForHI" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    SelectQualities = cms.vint32( 6 ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreAlCaRPCMuonNoHitsForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCPointProducer = cms.EDProducer( "RPCPointProducer",
    cscSegments = cms.InputTag( "hltCscSegments" ),
    rangestripsRB4 = cms.untracked.double( 4.0 ),
    TrackTransformer = cms.PSet(  ),
    tracks = cms.InputTag( "NotUsed" ),
    rangestrips = cms.untracked.double( 4.0 ),
    incltrack = cms.untracked.bool( False ),
    MinCosAng = cms.untracked.double( 0.95 ),
    MaxDrb4 = cms.untracked.double( 150.0 ),
    inclcsc = cms.untracked.bool( True ),
    dt4DSegments = cms.InputTag( "hltDt4DSegments" ),
    ExtrapolatedRegion = cms.untracked.double( 0.5 ),
    incldt = cms.untracked.bool( True ),
    debug = cms.untracked.bool( False ),
    MaxD = cms.untracked.double( 80.0 )
)
process.hltRPCFilter = cms.EDFilter( "HLTRPCFilter",
    rangestrips = cms.untracked.double( 1.0 ),
    rpcDTPoints = cms.InputTag( 'hltRPCPointProducer','RPCDTExtrapolatedPoints' ),
    rpcRecHits = cms.InputTag( "hltRpcRecHits" ),
    rpcCSCPoints = cms.InputTag( 'hltRPCPointProducer','RPCCSCExtrapolatedPoints' )
)
process.hltPreAlCaRPCMuonNormalisationForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltRPCMuonNormaL1Filtered0ForHI = cms.EDFilter( "HLTMuonL1Filter",
    saveTags = cms.bool( True ),
    CSCTFtag = cms.InputTag( "unused" ),
    PreviousCandTag = cms.InputTag( "hltL1sAlCaRPCForHI" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    ExcludeSingleSegmentCSC = cms.bool( False )
)
process.hltPreAlCaLumiPixelsRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFEDSelectorLumiPixels = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 )
)
process.hltPreAlCaLumiPixelsZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023, 1024 )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltPreAnalyzerEndpath = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltL1GtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    PrintVerbosity = cms.untracked.int32( 10 ),
    UseL1GlobalTriggerRecord = cms.bool( False ),
    PrintOutput = cms.untracked.int32( 3 ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 ),
    serviceBy = cms.untracked.string( "never" ),
    resetBy = cms.untracked.string( "never" ),
    reportBy = cms.untracked.string( "job" ),
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.hltPreHIPhysicsMuonsOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPhysicsHardProbesOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIPhysicsMinBiasUPCOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_HIPuAK4CaloJet40_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet60_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet80_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet110_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2',
      'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2',
      'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2',
      'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2',
      'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2',
      'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2',
      'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2',
      'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2',
      'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2',
      'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2',
      'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2',
      'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2',
      'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2',
      'HLT_HIPuAK4CaloDJet60_Eta2p1_v2',
      'HLT_HIPuAK4CaloDJet80_Eta2p1_v2',
      'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2',
      'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
      'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2',
      'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt30_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt40_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt50_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt70_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2',
      'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1',
      'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1',
      'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1',
      'HLT_HISinglePhoton10_Eta1p5_v2',
      'HLT_HISinglePhoton15_Eta1p5_v2',
      'HLT_HISinglePhoton20_Eta1p5_v2',
      'HLT_HISinglePhoton30_Eta1p5_v2',
      'HLT_HISinglePhoton40_Eta1p5_v2',
      'HLT_HISinglePhoton50_Eta1p5_v2',
      'HLT_HISinglePhoton60_Eta1p5_v2',
      'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2',
      'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2',
      'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2',
      'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2',
      'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2',
      'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2',
      'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2',
      'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2',
      'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2',
      'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2',
      'HLT_HISinglePhoton40_Eta2p1_v2',
      'HLT_HISinglePhoton10_Eta3p1_v2',
      'HLT_HISinglePhoton15_Eta3p1_v2',
      'HLT_HISinglePhoton20_Eta3p1_v2',
      'HLT_HISinglePhoton30_Eta3p1_v2',
      'HLT_HISinglePhoton40_Eta3p1_v2',
      'HLT_HISinglePhoton50_Eta3p1_v2',
      'HLT_HISinglePhoton60_Eta3p1_v2',
      'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2',
      'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2',
      'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2',
      'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2',
      'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2',
      'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2',
      'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2',
      'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2',
      'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2',
      'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2',
      'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2',
      'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
      'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2',
      'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2',
      'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
      'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
      'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
      'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
      'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2',
      'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2',
      'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2',
      'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2',
      'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2',
      'HLT_HIUCC100_v2',
      'HLT_HIUCC020_v2',
      'HLT_HIQ2Bottom005_Centrality1030_v2',
      'HLT_HIQ2Top005_Centrality1030_v2',
      'HLT_HIQ2Bottom005_Centrality3050_v2',
      'HLT_HIQ2Top005_Centrality3050_v2',
      'HLT_HIQ2Bottom005_Centrality5070_v2',
      'HLT_HIQ2Top005_Centrality5070_v2',
      'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2',
      'HLT_HIFullTrack12_L1Centrality010_v2',
      'HLT_HIFullTrack12_L1Centrality30100_v2',
      'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2',
      'HLT_HIFullTrack18_L1Centrality010_v2',
      'HLT_HIFullTrack18_L1Centrality30100_v2',
      'HLT_HIFullTrack24_v2',
      'HLT_HIFullTrack24_L1Centrality30100_v2',
      'HLT_HIFullTrack34_v2',
      'HLT_HIFullTrack34_L1Centrality30100_v2',
      'HLT_HIFullTrack45_v2',
      'HLT_HIFullTrack45_L1Centrality30100_v2',
      'HLT_HIL1DoubleMu0_v1',
      'HLT_HIL1DoubleMu0_2HF_v1',
      'HLT_HIL1DoubleMu0_2HF0_v1',
      'HLT_HIL1DoubleMu10_v1',
      'HLT_HIL2DoubleMu0_NHitQ_v2',
      'HLT_HIL2DoubleMu0_NHitQ_2HF_v1',
      'HLT_HIL2DoubleMu0_NHitQ_2HF0_v1',
      'HLT_HIL2Mu3_NHitQ10_2HF_v1',
      'HLT_HIL2Mu3_NHitQ10_2HF0_v1',
      'HLT_HIL3Mu3_NHitQ15_2HF_v1',
      'HLT_HIL3Mu3_NHitQ15_2HF0_v1',
      'HLT_HIL2Mu5_NHitQ10_2HF_v1',
      'HLT_HIL2Mu5_NHitQ10_2HF0_v1',
      'HLT_HIL3Mu5_NHitQ15_2HF_v1',
      'HLT_HIL3Mu5_NHitQ15_2HF0_v1',
      'HLT_HIL2Mu7_NHitQ10_2HF_v1',
      'HLT_HIL2Mu7_NHitQ10_2HF0_v1',
      'HLT_HIL3Mu7_NHitQ15_2HF_v1',
      'HLT_HIL3Mu7_NHitQ15_2HF0_v1',
      'HLT_HIL2Mu15_v2',
      'HLT_HIL2Mu15_2HF_v1',
      'HLT_HIL2Mu15_2HF0_v1',
      'HLT_HIL3Mu15_v1',
      'HLT_HIL3Mu15_2HF_v1',
      'HLT_HIL3Mu15_2HF0_v1',
      'HLT_HIL2Mu20_v1',
      'HLT_HIL2Mu20_2HF_v1',
      'HLT_HIL2Mu20_2HF0_v1',
      'HLT_HIL3Mu20_v1',
      'HLT_HIL3Mu20_2HF_v1',
      'HLT_HIL3Mu20_2HF0_v1',
      'HLT_HIL1DoubleMu0_2HF_Cent30100_v1',
      'HLT_HIL1DoubleMu0_2HF0_Cent30100_v1',
      'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1',
      'HLT_HIL1DoubleMu0_Cent30_v1',
      'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1',
      'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
      'HLT_HIL2DoubleMu0_Cent30_NHitQ_v1',
      'HLT_HIL3DoubleMu0_Cent30_v1',
      'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1',
      'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1',
      'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
      'HLT_HIL3DoubleMu0_OS_m7to14_v1',
      'HLT_HIUPCL1SingleMuOpenNotHF2_v1',
      'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1',
      'HLT_HIUPCL1DoubleMuOpenNotHF2_v1',
      'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
      'HLT_HIUPCL1SingleEG2NotHF2_v1',
      'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1',
      'HLT_HIUPCL1DoubleEG2NotHF2_v1',
      'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1',
      'HLT_HIUPCL1SingleEG5NotHF2_v1',
      'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1',
      'HLT_HIUPCL1DoubleMuOpenNotHF1_v1',
      'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1',
      'HLT_HIUPCL1DoubleEG2NotZDCAND_v1',
      'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1',
      'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1EG2NotZDCAND_v1',
      'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1MuOpenNotZDCAND_v1',
      'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1',
      'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2',
      'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
      'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2',
      'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
      'HLT_HIL1CastorMediumJet_v1',
      'HLT_HIL1CastorMediumJetAK4CaloJet20_v2',
      'HLT_HICastorMediumJetPixel_SingleTrack_v1',
      'HLT_HIUPCL1NotMinimumBiasHF2_AND_v1',
      'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1ZdcOR_BptxAND_v1',
      'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1ZdcXOR_BptxAND_v1',
      'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1',
      'HLT_HIUPCL1NotZdcOR_BptxAND_v1',
      'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1',
      'HLT_HIZeroBias_v1',
      'HLT_HICentralityVeto_v1',
      'HLT_HIL1Tech5_BPTX_PlusOnly_v1',
      'HLT_HIL1Tech6_BPTX_MinusOnly_v1',
      'HLT_HIL1Tech7_NoBPTX_v1',
      'HLT_HIL1MinimumBiasHF1OR_v1',
      'HLT_HIL1MinimumBiasHF2OR_v1',
      'HLT_HIL1MinimumBiasHF1AND_v1',
      'HLT_HIL1MinimumBiasHF2AND_v1',
      'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1',
      'HLT_HIZeroBiasPixel_SingleTrack_v1',
      'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1',
      'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
      'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1',
      'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
      'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1',
      'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
      'HLT_HIPhysics_v1',
      'HLT_HIRandom_v1' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreDQMCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMEventDisplayOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreDQMEventDisplayOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
      'HLT_HISinglePhoton60_Eta3p1_v2' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreRPCMONOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreEcalCalibrationOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCAPHISYMOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreALCALUMIPIXELSOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIExpressOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)
process.hltPreHIExpressOutputSmart = cms.EDFilter( "TriggerResultsFilter",
    l1tIgnoreMask = cms.bool( False ),
    l1tResults = cms.InputTag( "hltGtDigis" ),
    l1techIgnorePrescales = cms.bool( False ),
    hltResults = cms.InputTag( "TriggerResults" ),
    triggerConditions = cms.vstring( 'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
      'HLT_HIPuAK4CaloJet120_Eta5p1_v2 / 6',
      'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2 / 8',
      'HLT_HIDmesonHITrackingGlobal_Dpt20_v2 / 2',
      'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
      'HLT_HISinglePhoton60_Eta1p5_v2 / 5',
      'HLT_HISinglePhoton60_Eta3p1_v2 / 20',
      'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2 / 2',
      'HLT_HIUCC020_v2 / 8',
      'HLT_HIFullTrack34_v2 / 5',
      'HLT_HIL1DoubleMu10_v1',
      'HLT_HIL2Mu20_2HF_v1 / 2',
      'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1 / 50',
      'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1 / 8',
      'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1 / 6',
      'HLT_HIZeroBias_v1',
      'HLT_HICentralityVeto_v1',
      'HLT_HIL1MinimumBiasHF1AND_v1 / 30',
      'HLT_HIRandom_v1' ),
    throw = cms.bool( True ),
    daqPartitions = cms.uint32( 1 )
)
process.hltPreNanoDSTOutput = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    offset = cms.uint32( 0 )
)

process.hltOutputHIPhysicsMuons = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHIPhysicsMuons.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIL1DoubleMu0_2HF0_Cent30100_v1',
  'HLT_HIL1DoubleMu0_2HF0_v1',
  'HLT_HIL1DoubleMu0_2HF_Cent30100_v1',
  'HLT_HIL1DoubleMu0_2HF_v1',
  'HLT_HIL1DoubleMu0_Cent30_v1',
  'HLT_HIL1DoubleMu0_v1',
  'HLT_HIL1DoubleMu10_v1',
  'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1',
  'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1',
  'HLT_HIL2DoubleMu0_Cent30_NHitQ_v1',
  'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
  'HLT_HIL2DoubleMu0_NHitQ_2HF0_v1',
  'HLT_HIL2DoubleMu0_NHitQ_2HF_v1',
  'HLT_HIL2DoubleMu0_NHitQ_v2',
  'HLT_HIL2Mu15_2HF0_v1',
  'HLT_HIL2Mu15_2HF_v1',
  'HLT_HIL2Mu15_v2',
  'HLT_HIL2Mu20_2HF0_v1',
  'HLT_HIL2Mu20_2HF_v1',
  'HLT_HIL2Mu20_v1',
  'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
  'HLT_HIL2Mu3_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu3_NHitQ10_2HF_v1',
  'HLT_HIL2Mu5_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu5_NHitQ10_2HF_v1',
  'HLT_HIL2Mu7_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu7_NHitQ10_2HF_v1',
  'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1',
  'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1',
  'HLT_HIL3DoubleMu0_Cent30_v1',
  'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
  'HLT_HIL3DoubleMu0_OS_m7to14_v1',
  'HLT_HIL3Mu15_2HF0_v1',
  'HLT_HIL3Mu15_2HF_v1',
  'HLT_HIL3Mu15_v1',
  'HLT_HIL3Mu20_2HF0_v1',
  'HLT_HIL3Mu20_2HF_v1',
  'HLT_HIL3Mu20_v1',
  'HLT_HIL3Mu3_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu3_NHitQ15_2HF_v1',
  'HLT_HIL3Mu5_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu5_NHitQ15_2HF_v1',
  'HLT_HIL3Mu7_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu7_NHitQ15_2HF_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHIPhysicsHardProbes = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHIPhysicsHardProbes.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt50_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt70_v2',
  'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
  'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2',
  'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2',
  'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2',
  'HLT_HIFullTrack12_L1Centrality010_v2',
  'HLT_HIFullTrack12_L1Centrality30100_v2',
  'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2',
  'HLT_HIFullTrack18_L1Centrality010_v2',
  'HLT_HIFullTrack18_L1Centrality30100_v2',
  'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2',
  'HLT_HIFullTrack24_L1Centrality30100_v2',
  'HLT_HIFullTrack24_v2',
  'HLT_HIFullTrack34_L1Centrality30100_v2',
  'HLT_HIFullTrack34_v2',
  'HLT_HIFullTrack45_L1Centrality30100_v2',
  'HLT_HIFullTrack45_v2',
  'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2',
  'HLT_HIPuAK4CaloDJet60_Eta2p1_v2',
  'HLT_HIPuAK4CaloDJet80_Eta2p1_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2',
  'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2',
  'HLT_HIPuAK4CaloJet110_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2',
  'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2',
  'HLT_HIQ2Bottom005_Centrality1030_v2',
  'HLT_HIQ2Bottom005_Centrality3050_v2',
  'HLT_HIQ2Bottom005_Centrality5070_v2',
  'HLT_HIQ2Top005_Centrality1030_v2',
  'HLT_HIQ2Top005_Centrality3050_v2',
  'HLT_HIQ2Top005_Centrality5070_v2',
  'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton10_Eta1p5_v2',
  'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton10_Eta3p1_v2',
  'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton15_Eta1p5_v2',
  'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton15_Eta3p1_v2',
  'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton20_Eta1p5_v2',
  'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton20_Eta3p1_v2',
  'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton30_Eta1p5_v2',
  'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton40_Eta1p5_v2',
  'HLT_HISinglePhoton40_Eta2p1_v2',
  'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton40_Eta3p1_v2',
  'HLT_HISinglePhoton50_Eta1p5_v2',
  'HLT_HISinglePhoton50_Eta3p1_v2',
  'HLT_HISinglePhoton60_Eta1p5_v2',
  'HLT_HISinglePhoton60_Eta3p1_v2',
  'HLT_HIUCC020_v2',
  'HLT_HIUCC100_v2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHIPhysicsMinBiasUPC = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHIPhysicsMinBiasUPC.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v1',
  'HLT_HICentralityVeto_v1',
  'HLT_HIL1CastorMediumJetAK4CaloJet20_v2',
  'HLT_HIL1CastorMediumJet_v1',
  'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1',
  'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1',
  'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1',
  'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1MinimumBiasHF1AND_v1',
  'HLT_HIL1MinimumBiasHF1OR_v1',
  'HLT_HIL1MinimumBiasHF2AND_v1',
  'HLT_HIL1MinimumBiasHF2OR_v1',
  'HLT_HIL1Tech5_BPTX_PlusOnly_v1',
  'HLT_HIL1Tech6_BPTX_MinusOnly_v1',
  'HLT_HIL1Tech7_NoBPTX_v1',
  'HLT_HIPhysics_v1',
  'HLT_HIRandom_v1',
  'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1',
  'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleEG2NotHF2_v1',
  'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleEG2NotZDCAND_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHF1_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHF2_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
  'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2',
  'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1',
  'HLT_HIUPCL1EG2NotZDCAND_v1',
  'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1MuOpenNotZDCAND_v1',
  'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
  'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2',
  'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1',
  'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotMinimumBiasHF2_AND_v1',
  'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotZdcOR_BptxAND_v1',
  'HLT_HIUPCL1SingleEG2NotHF2_v1',
  'HLT_HIUPCL1SingleEG5NotHF2_v1',
  'HLT_HIUPCL1SingleMuOpenNotHF2_v1',
  'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1ZdcOR_BptxAND_v1',
  'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1ZdcXOR_BptxAND_v1',
  'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1',
  'HLT_HIZeroBiasPixel_SingleTrack_v1',
  'HLT_HIZeroBias_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v1',
  'HLT_HICentralityVeto_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt30_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt40_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt50_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt70_v2',
  'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
  'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2',
  'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2',
  'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2',
  'HLT_HIFullTrack12_L1Centrality010_v2',
  'HLT_HIFullTrack12_L1Centrality30100_v2',
  'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2',
  'HLT_HIFullTrack18_L1Centrality010_v2',
  'HLT_HIFullTrack18_L1Centrality30100_v2',
  'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2',
  'HLT_HIFullTrack24_L1Centrality30100_v2',
  'HLT_HIFullTrack24_v2',
  'HLT_HIFullTrack34_L1Centrality30100_v2',
  'HLT_HIFullTrack34_v2',
  'HLT_HIFullTrack45_L1Centrality30100_v2',
  'HLT_HIFullTrack45_v2',
  'HLT_HIL1CastorMediumJetAK4CaloJet20_v2',
  'HLT_HIL1CastorMediumJet_v1',
  'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1',
  'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1',
  'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1',
  'HLT_HIL1DoubleMu0_2HF0_Cent30100_v1',
  'HLT_HIL1DoubleMu0_2HF0_v1',
  'HLT_HIL1DoubleMu0_2HF_Cent30100_v1',
  'HLT_HIL1DoubleMu0_2HF_v1',
  'HLT_HIL1DoubleMu0_Cent30_v1',
  'HLT_HIL1DoubleMu0_v1',
  'HLT_HIL1DoubleMu10_v1',
  'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1',
  'HLT_HIL1MinimumBiasHF1AND_v1',
  'HLT_HIL1MinimumBiasHF1OR_v1',
  'HLT_HIL1MinimumBiasHF2AND_v1',
  'HLT_HIL1MinimumBiasHF2OR_v1',
  'HLT_HIL1Tech5_BPTX_PlusOnly_v1',
  'HLT_HIL1Tech6_BPTX_MinusOnly_v1',
  'HLT_HIL1Tech7_NoBPTX_v1',
  'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1',
  'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1',
  'HLT_HIL2DoubleMu0_Cent30_NHitQ_v1',
  'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
  'HLT_HIL2DoubleMu0_NHitQ_2HF0_v1',
  'HLT_HIL2DoubleMu0_NHitQ_2HF_v1',
  'HLT_HIL2DoubleMu0_NHitQ_v2',
  'HLT_HIL2Mu15_2HF0_v1',
  'HLT_HIL2Mu15_2HF_v1',
  'HLT_HIL2Mu15_v2',
  'HLT_HIL2Mu20_2HF0_v1',
  'HLT_HIL2Mu20_2HF_v1',
  'HLT_HIL2Mu20_v1',
  'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2',
  'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2',
  'HLT_HIL2Mu3_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu3_NHitQ10_2HF_v1',
  'HLT_HIL2Mu5_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu5_NHitQ10_2HF_v1',
  'HLT_HIL2Mu7_NHitQ10_2HF0_v1',
  'HLT_HIL2Mu7_NHitQ10_2HF_v1',
  'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1',
  'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1',
  'HLT_HIL3DoubleMu0_Cent30_v1',
  'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
  'HLT_HIL3DoubleMu0_OS_m7to14_v1',
  'HLT_HIL3Mu15_2HF0_v1',
  'HLT_HIL3Mu15_2HF_v1',
  'HLT_HIL3Mu15_v1',
  'HLT_HIL3Mu20_2HF0_v1',
  'HLT_HIL3Mu20_2HF_v1',
  'HLT_HIL3Mu20_v1',
  'HLT_HIL3Mu3_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu3_NHitQ15_2HF_v1',
  'HLT_HIL3Mu5_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu5_NHitQ15_2HF_v1',
  'HLT_HIL3Mu7_NHitQ15_2HF0_v1',
  'HLT_HIL3Mu7_NHitQ15_2HF_v1',
  'HLT_HIPhysics_v1',
  'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2',
  'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2',
  'HLT_HIPuAK4CaloDJet60_Eta2p1_v2',
  'HLT_HIPuAK4CaloDJet80_Eta2p1_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2',
  'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2',
  'HLT_HIPuAK4CaloJet110_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet40_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet60_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2',
  'HLT_HIPuAK4CaloJet80_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2',
  'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2',
  'HLT_HIQ2Bottom005_Centrality1030_v2',
  'HLT_HIQ2Bottom005_Centrality3050_v2',
  'HLT_HIQ2Bottom005_Centrality5070_v2',
  'HLT_HIQ2Top005_Centrality1030_v2',
  'HLT_HIQ2Top005_Centrality3050_v2',
  'HLT_HIQ2Top005_Centrality5070_v2',
  'HLT_HIRandom_v1',
  'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton10_Eta1p5_v2',
  'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton10_Eta3p1_v2',
  'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton15_Eta1p5_v2',
  'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton15_Eta3p1_v2',
  'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton20_Eta1p5_v2',
  'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton20_Eta3p1_v2',
  'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton30_Eta1p5_v2',
  'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton30_Eta3p1_v2',
  'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2',
  'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2',
  'HLT_HISinglePhoton40_Eta1p5_v2',
  'HLT_HISinglePhoton40_Eta2p1_v2',
  'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2',
  'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2',
  'HLT_HISinglePhoton40_Eta3p1_v2',
  'HLT_HISinglePhoton50_Eta1p5_v2',
  'HLT_HISinglePhoton50_Eta3p1_v2',
  'HLT_HISinglePhoton60_Eta1p5_v2',
  'HLT_HISinglePhoton60_Eta3p1_v2',
  'HLT_HIUCC020_v2',
  'HLT_HIUCC100_v2',
  'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1',
  'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleEG2NotHF2_v1',
  'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleEG2NotZDCAND_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHF1_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHF2_v1',
  'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
  'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2',
  'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1',
  'HLT_HIUPCL1EG2NotZDCAND_v1',
  'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1MuOpenNotZDCAND_v1',
  'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2',
  'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2',
  'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1',
  'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotMinimumBiasHF2_AND_v1',
  'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1NotZdcOR_BptxAND_v1',
  'HLT_HIUPCL1SingleEG2NotHF2_v1',
  'HLT_HIUPCL1SingleEG5NotHF2_v1',
  'HLT_HIUPCL1SingleMuOpenNotHF2_v1',
  'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1ZdcOR_BptxAND_v1',
  'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1',
  'HLT_HIUPCL1ZdcXOR_BptxAND_v1',
  'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1',
  'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1',
  'HLT_HIZeroBiasPixel_SingleTrack_v1',
  'HLT_HIZeroBias_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltCombinedSecondaryVertexBJetTagsCalo_*_*',
      'keep *_hltCombinedSecondaryVertexBJetTagsPF_*_*',
      'keep *_hltIter2Merged_*_*',
      'keep *_hltL1GtObjectMap_*_*',
      'keep *_hltL3NoFiltersNoVtxMuonCandidates_*_*',
      'keep *_hltOnlineBeamSpot_*_*',
      'keep *_hltPFJetForBtag_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltSelector8CentralJetsL1FastJet_*_*',
      'keep *_hltSiPixelClusters_*_*',
      'keep *_hltSiStripRawToClustersFacility_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQMCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration_v2',
  'HLT_HcalCalibration_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQMEventDisplay = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMEventDisplay.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HIPuAK4CaloJet150_Eta5p1_v2',
  'HLT_HISinglePhoton60_Eta3p1_v2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputRPCMON = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputRPCMON.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_RPCMuonNoHitsForHI_v1',
  'AlCa_RPCMuonNoTriggersForHI_v1',
  'AlCa_RPCMuonNormalisationForHI_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltCscSegments_*_*',
      'keep *_hltDt4DSegments_*_*',
      'keep *_hltMuonCSCDigis_MuonCSCStripDigi_*',
      'keep *_hltMuonCSCDigis_MuonCSCWireDigi_*',
      'keep *_hltMuonDTDigis_*_*',
      'keep *_hltMuonRPCDigis_*_*',
      'keep *_hltRpcRecHits_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep L1MuGMTCands_hltGtDigis_*_*',
      'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration_v2',
  'HLT_HcalCalibration_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputEcalCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputEcalCalibration.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_EcalCalibration_v2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *_hlt*_*_*',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputALCAPHISYM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPHISYM.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_EcalPhiSymForHI_v2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalPhiSymFilter_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALUMIPIXELS = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALUMIPIXELS.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'AlCa_LumiPixels_Random_v1',
  'AlCa_LumiPixels_ZeroBias_v2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorLumiPixels_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputHIExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHIExpress.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'HLT_HICentralityVeto_v1',
  'HLT_HIDmesonHITrackingGlobal_Dpt20_v2',
  'HLT_HIDmesonHITrackingGlobal_Dpt60_v2',
  'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2',
  'HLT_HIFullTrack34_v2',
  'HLT_HIL1DoubleMu10_v1',
  'HLT_HIL1MinimumBiasHF1AND_v1',
  'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1',
  'HLT_HIL2Mu20_2HF_v1',
  'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1',
  'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2',
  'HLT_HIPuAK4CaloJet100_Eta5p1_v2',
  'HLT_HIPuAK4CaloJet120_Eta5p1_v2',
  'HLT_HIRandom_v1',
  'HLT_HISinglePhoton60_Eta1p5_v2',
  'HLT_HISinglePhoton60_Eta3p1_v2',
  'HLT_HIUCC020_v2',
  'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1',
  'HLT_HIZeroBias_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltL1GtObjectMap_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_source_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputNanoDST = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputNanoDST.root" ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'DST_Physics_v1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelector_*_*',
      'keep L1GlobalTriggerReadoutRecord_hltGtDigis_*_*',
      'keep L1MuGMTReadoutCollection_hltGtDigis_*_*',
      'keep edmTriggerResults_*_*_*' )
)

process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtDigis + process.hltCaloStage1Digis + process.hltCaloStage1LegacyFormatDigis + process.hltL1GtObjectMap + process.hltL1extraParticles )
process.HLTBeamSpot = cms.Sequence( process.hltScalersRawToDigi + process.hltOnlineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )
process.HLTDoFullUnpackingEgammaEcalWithoutPreshower50nsMultiFitSequence = cms.Sequence( process.hltEcalDigis + process.hltEcalUncalibRecHit50nsMultiFit + process.hltEcalDetIdToBeRecovered + process.hltEcalRecHit50nsMultiFit )
process.HLTDoLocalHcalMethod0Sequence = cms.Sequence( process.hltHcalDigis + process.hltHbherecoMethod0 + process.hltHfrecoMethod0 + process.hltHorecoMethod0 )
process.HLTDoCaloHcalMethod050nsMultiFitSequence = cms.Sequence( process.HLTDoFullUnpackingEgammaEcalWithoutPreshower50nsMultiFitSequence + process.HLTDoLocalHcalMethod0Sequence + process.hltTowerMakerHcalMethod050nsMultiFitForAll )
process.HLTPuAK4CaloJetsReconstruction50nsMultiFitSequence = cms.Sequence( process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltPuAK4CaloJets50nsMultiFit + process.hltPuAK4CaloJetsIDPassed50nsMultiFit )
process.HLTPuAK4CaloCorrectorProducersSequence = cms.Sequence( process.hltAK4CaloRelativeCorrector + process.hltAK4CaloAbsoluteCorrector + process.hltPuAK4CaloCorrector )
process.HLTPuAK4CaloJetsCorrection50nsMultiFitSequence = cms.Sequence( process.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 + process.HLTPuAK4CaloCorrectorProducersSequence + process.hltPuAK4CaloJetsCorrected50nsMultiFit + process.hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit )
process.HLTPuAK4CaloJets50nsMultiFitSequence = cms.Sequence( process.HLTPuAK4CaloJetsReconstruction50nsMultiFitSequence + process.HLTPuAK4CaloJetsCorrection50nsMultiFitSequence )
process.HLTDoHIStripZeroSuppression = cms.Sequence( process.hltSiStripRawToDigi + process.hltSiStripZeroSuppression + process.hltSiStripDigiToZSRaw + process.hltSiStripRawDigiToVirginRaw + process.virginRawDataRepacker + process.rawDataRepacker )
process.HLTDoHILocalPixelSequence = cms.Sequence( process.hltHISiPixelDigis + process.hltHISiPixelClusters + process.hltHISiPixelClustersCache + process.hltHISiPixelRecHits )
process.HLTHIRecopixelvetexingSequence = cms.Sequence( process.hltHIPixelClusterVertices + process.hltHIPixelLayerTriplets + process.hltHIPixel3ProtoTracks + process.hltHIPixelMedianVertex + process.hltHISelectedProtoTracks + process.hltHIPixelAdaptiveVertex + process.hltHIBestAdaptiveVertex + process.hltHISelectedVertex )
process.HLTDoHILocalPixelSequenceAfterSplitting = cms.Sequence( process.hltHISiPixelClustersAfterSplitting + process.hltHISiPixelClustersCacheAfterSplitting + process.hltHISiPixelRecHitsAfterSplitting )
process.HLTHIRecopixelvetexingSequenceAfterSplitting = cms.Sequence( process.hltHIPixelClusterVerticesAfterSplitting + process.hltHIPixelLayerTripletsAfterSplitting + process.hltHIPixel3ProtoTracksAfterSplitting + process.hltHIPixelMedianVertexAfterSplitting + process.hltHISelectedProtoTracksAfterSplitting + process.hltHIPixelAdaptiveVertexAfterSplitting + process.hltHIBestAdaptiveVertexAfterSplitting + process.hltHISelectedVertexAfterSplitting )
process.HLTHIPixelClusterSplitting = cms.Sequence( process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltHIJetsForCoreTracking + process.HLTDoHILocalPixelSequence + process.HLTHIRecopixelvetexingSequence + process.HLTDoHILocalPixelSequenceAfterSplitting + process.HLTHIRecopixelvetexingSequenceAfterSplitting )
process.HLTDoHITrackingLocalStripSequenceZeroSuppression = cms.Sequence( process.hltSiStripRawToDigi + process.hltSiStripZeroSuppression + process.hltHITrackingSiStripRawToClustersFacilityZeroSuppression + process.hltHISiStripClustersZeroSuppression )
process.HLTHIIterativeTrackingIteration0Forjets = cms.Sequence( process.hltHIPixel3PrimTracksForjets + process.hltHIPixelTrackSeedsForjets + process.hltHIPrimTrackCandidatesForjets + process.hltHIGlobalPrimTracksForjets + process.hltHIIter0TrackSelectionForjets )
process.HLTHIIterativeTrackingIteration1Forjets = cms.Sequence( process.hltHIIter1ClustersRefRemovalForjets + process.hltHIIter1MaskedMeasurementTrackerEventForjets + process.hltHIDetachedPixelLayerTripletsForjets + process.hltHIDetachedPixelTracksForjets + process.hltHIDetachedPixelTrackSeedsForjets + process.hltHIDetachedTrackCandidatesForjets + process.hltHIDetachedGlobalPrimTracksForjets + process.hltHIIter1TrackSelectionForjets )
process.HLTHIIterativeTrackingIteration2Forjets = cms.Sequence( process.hltHIIter2ClustersRefRemovalForjets + process.hltHIIter2MaskedMeasurementTrackerEventForjets + process.hltHIPixelLayerPairsForjets + process.hltHIPixelPairSeedsForjets + process.hltHIPixelPairTrackCandidatesForjets + process.hltHIPixelPairGlobalPrimTracksForjets + process.hltHIIter2TrackSelectionForjets )
process.HLTHIIterativeTrackingForJets = cms.Sequence( process.HLTHIIterativeTrackingIteration0Forjets + process.HLTHIIterativeTrackingIteration1Forjets + process.HLTHIIterativeTrackingIteration2Forjets + process.hltHIIterTrackingMergedHighPurityForjets + process.hltHIIterTrackingMergedTightForjets )
process.HLTDoHIStripZeroSuppressionRepacker = cms.Sequence( process.hltSiStripDigiToZSRaw + process.hltSiStripRawDigiToVirginRaw + process.virginRawDataRepacker + process.rawDataRepacker )
process.HLTBtagCSVSequenceL3CaloJet60Eta2p1 = cms.Sequence( process.hltHIVerticesL3 + process.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 + process.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 + process.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 + process.hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1 )
process.HLTBtagCSVSequenceL3CaloJet80Eta2p1 = cms.Sequence( process.hltHIVerticesL3 + process.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 + process.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 + process.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 + process.hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1 )
process.HLTBtagSSVSequenceL3CaloJet60Eta2p1 = cms.Sequence( process.hltHIVerticesL3 + process.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 + process.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 + process.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 + process.hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1 )
process.HLTBtagSSVSequenceL3CaloJet80Eta2p1 = cms.Sequence( process.hltHIVerticesL3 + process.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 + process.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 + process.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 + process.hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1 )
process.HLTHIIterativeTrackingIteration0ForGlobalPt8 = cms.Sequence( process.hltHIPixel3PrimTracksForGlobalPt8 + process.hltHIPixelTrackSeedsForGlobalPt8 + process.hltHIPrimTrackCandidatesForGlobalPt8 + process.hltHIGlobalPrimTracksForGlobalPt8 + process.hltHIIter0TrackSelectionForGlobalPt8 )
process.HLTHIIterativeTrackingIteration1ForGlobalPt8 = cms.Sequence( process.hltHIIter1ClustersRefRemovalForGlobalPt8 + process.hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8 + process.hltHIDetachedPixelLayerTripletsForGlobalPt8 + process.hltHIDetachedPixelTracksForGlobalPt8 + process.hltHIDetachedPixelTrackSeedsForGlobalPt8 + process.hltHIDetachedTrackCandidatesForGlobalPt8 + process.hltHIDetachedGlobalPrimTracksForGlobalPt8 + process.hltHIIter1TrackSelectionForGlobalPt8 )
process.HLTHIIterativeTrackingIteration2ForGlobalPt8 = cms.Sequence( process.hltHIIter2ClustersRefRemovalForGlobalPt8 + process.hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8 + process.hltHIPixelLayerPairsForGlobalPt8 + process.hltHIPixelPairSeedsForGlobalPt8 + process.hltHIPixelPairTrackCandidatesForGlobalPt8 + process.hltHIPixelPairGlobalPrimTracksForGlobalPt8 + process.hltHIIter2TrackSelectionForGlobalPt8 )
process.HLTHIIterativeTrackingForGlobalPt8 = cms.Sequence( process.HLTHIIterativeTrackingIteration0ForGlobalPt8 + process.HLTHIIterativeTrackingIteration1ForGlobalPt8 + process.HLTHIIterativeTrackingIteration2ForGlobalPt8 + process.hltHIIterTrackingMergedHighPurityForGlobalPt8 + process.hltHIIterTrackingMergedTightForGlobalPt8 )
process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence = cms.Sequence( process.hltIslandBasicClusters50nsMultiFitHI + process.hltHiIslandSuperClusters50nsMultiFitHI + process.hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI + process.hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI + process.hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI + process.hltRecoHIEcalWithCleaningCandidate50nsMultiFit )
process.HLTMuonLocalRecoSequence = cms.Sequence( process.hltMuonDTDigis + process.hltDt1DRecHits + process.hltDt4DSegments + process.hltMuonCSCDigis + process.hltCsc2DRecHits + process.hltCscSegments + process.hltMuonRPCDigis + process.hltRpcRecHits )
process.HLTL2muonrecoNocandSequence = cms.Sequence( process.HLTMuonLocalRecoSequence + process.hltL2OfflineMuonSeeds + process.hltL2MuonSeeds + process.hltL2Muons )
process.HLTL2muonrecoSequence = cms.Sequence( process.HLTL2muonrecoNocandSequence + process.hltL2MuonCandidates )
process.HLTDoLocalHfSequence = cms.Sequence( process.hltHcalDigis + process.hltHfrecoMethod0 + process.hltTowerMakerForHf )
process.HLTRecoMETHfSequence = cms.Sequence( process.HLTDoLocalHfSequence + process.hltMetForHf )
process.HLTDoHILocalPixelClustersSequence = cms.Sequence( process.hltHISiPixelDigis + process.hltHISiPixelClusters )
process.HLTDoHILocalStripSequence = cms.Sequence( process.hltSiStripExcludedFEDListProducer + process.hltHISiStripRawToClustersFacility + process.hltHISiStripClusters )
process.HLTHIL3muonTkCandidateSequence = cms.Sequence( process.HLTDoHILocalPixelSequence + process.HLTDoHILocalStripSequence + process.hltHIL3TrajSeedOIState + process.hltHIL3TrackCandidateFromL2OIState + process.hltHIL3TkTracksFromL2OIState + process.hltHIL3MuonsOIState + process.hltHIL3TrajSeedOIHit + process.hltHIL3TrackCandidateFromL2OIHit + process.hltHIL3TkTracksFromL2OIHit + process.hltHIL3MuonsOIHit + process.hltHIL3TkFromL2OICombination + process.hltHIL3TrajectorySeed + process.hltHIL3TrackCandidateFromL2 )
process.HLTHIL3muonrecoNocandSequence = cms.Sequence( process.HLTHIL3muonTkCandidateSequence + process.hltHIL3MuonsLinksCombination + process.hltHIL3Muons )
process.HLTHIL3muonrecoSequence = cms.Sequence( process.HLTHIL3muonrecoNocandSequence + process.hltHIL3MuonCandidates )
process.HLTRecopixelvertexingSequenceForUPC = cms.Sequence( process.hltPixelLayerTripletsForUPC + process.hltPixelTracksForUPC )
process.HLTPuAK4CaloJetsUPCReconstructionSequence = cms.Sequence( process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltPuAK4CaloJetsUPC + process.hltPuAK4CaloJetsUPCIDPassed )
process.HLTPuAK4CaloJetsUPCCorrectionSequence = cms.Sequence( process.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 + process.HLTPuAK4CaloCorrectorProducersSequence + process.hltPuAK4CaloJetsUPCCorrected + process.hltPuAK4CaloJetsUPCCorrectedIDPassed )
process.HLTPuAK4CaloJetsUPCSequence = cms.Sequence( process.HLTPuAK4CaloJetsUPCReconstructionSequence + process.HLTPuAK4CaloJetsUPCCorrectionSequence )
process.HLTPixelTrackingForHITrackTrigger = cms.Sequence( process.hltHIPixelClusterVertices + process.hltHIPixelLayerTriplets + process.hltHIPixel3ProtoTracksForHITrackTrigger + process.hltPixelCandsForHITrackTrigger )
process.HLTBeginSequenceRandom = cms.Sequence( process.hltRandomEventsFilter + process.hltGtDigis )
process.HLTBeginSequenceCalibration = cms.Sequence( process.hltCalibrationEventsFilter + process.hltGtDigis )
process.HLTDoFullUnpackingEgammaEcal50nsMultiFitSequence = cms.Sequence( process.hltEcalDigis + process.hltEcalPreshowerDigis + process.hltEcalUncalibRecHit50nsMultiFit + process.hltEcalDetIdToBeRecovered + process.hltEcalRecHit50nsMultiFit + process.hltEcalPreshowerRecHit )

process.HLTriggerFirstPath = cms.Path( process.hltGetConditions + process.hltGetRaw + process.hltBoolFalse )
process.DST_Physics_v1 = cms.Path( process.HLTBeginSequence + process.hltPreDSTPhysics + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet40_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHIPuAK4CaloJet40Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet60_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28BptxAND + process.hltPreHIPuAK4CaloJet60Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloJet80Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet100_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet56BptxAND + process.hltPreHIPuAK4CaloJet100Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet110_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet56BptxAND + process.hltPreHIPuAK4CaloJet110Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet110Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet120_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet56BptxAND + process.hltPreHIPuAK4CaloJet120Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet120Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet150_Eta5p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet64BptxAND + process.hltPreHIPuAK4CaloJet150Eta5p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet150Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet16Centext30100BptxAND + process.hltPreHIPuAK4CaloJet40Eta5p1Cent30100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28Centext30100BptxAND + process.hltPreHIPuAK4CaloJet60Eta5p1Cent30100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext30100BptxAND + process.hltPreHIPuAK4CaloJet80Eta5p1Cent30100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext30100BptxAND + process.hltPreHIPuAK4CaloJet100Eta5p1Cent30100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet16Centext50100BptxAND + process.hltPreHIPuAK4CaloJet40Eta5p1Cent50100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28Centext50100BptxAND + process.hltPreHIPuAK4CaloJet60Eta5p1Cent50100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext50100BptxAND + process.hltPreHIPuAK4CaloJet80Eta5p1Cent50100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext50100BptxAND + process.hltPreHIPuAK4CaloJet100Eta5p1Cent50100 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloJet80Jet35Eta1p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta1p150nsMultiFit + process.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloJet80Jet35Eta0p7 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta0p750nsMultiFit + process.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet56BptxAND + process.hltPreHIPuAK4CaloJet100Jet35Eta1p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta1p150nsMultiFit + process.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet56BptxAND + process.hltPreHIPuAK4CaloJet100Jet35Eta0p7 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta0p750nsMultiFit + process.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloJet804545Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltTriplePuAK4CaloJet45Eta2p150nsMultiFit + process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPuAK4CaloDJet60_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28BptxAND + process.hltPreHIPuAK4CaloDJet60Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet60Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.hltHIFullTrackCandsForDmesonjets + process.hltHIFullTrackFilterForDmesonjets + process.hltTktkVtxForDmesonjetsCaloJet60 + process.hltTktkFilterForDmesonjetsCaloJet60 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIPuAK4CaloDJet80_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloDJet80Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet80Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.hltHIFullTrackCandsForDmesonjets + process.hltHIFullTrackFilterForDmesonjets + process.hltTktkVtxForDmesonjetsCaloJet80 + process.hltTktkFilterForDmesonjetsCaloJet80 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28BptxAND + process.hltPreHIPuAK4CaloBJetCSV60Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet60Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.HLTBtagCSVSequenceL3CaloJet60Eta2p1 + process.hltBLifetimeL3FilterCSVCaloJet60Eta2p1 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloBJetCSV80Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet80Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.HLTBtagCSVSequenceL3CaloJet80Eta2p1 + process.hltBLifetimeL3FilterCSVCaloJet80Eta2p1 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28BptxAND + process.hltPreHIPuAK4CaloBJetSSV60Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet60Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.HLTBtagSSVSequenceL3CaloJet60Eta2p1 + process.hltBLifetimeL3FilterSSVCaloJet60Eta2p1 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIPuAK4CaloBJetSSV80Eta2p1 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + process.hltEta2CaloJetsEta2p1ForJets + process.hltReduceJetMultEta2p1Forjets + process.hltJets4bTaggerCaloJet80Eta2p1Forjets + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForJets + process.HLTBtagSSVSequenceL3CaloJet80Eta2p1 + process.hltBLifetimeL3FilterSSVCaloJet80Eta2p1 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt20_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt20 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt20 + process.hltTktkFilterForDmesonGlobal8Dp20 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext30100MinimumumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt20Cent30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt20 + process.hltTktkFilterForDmesonGlobal8Dp20 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext50100MinimumumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt20Cent50100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt20 + process.hltTktkFilterForDmesonGlobal8Dp20 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt30_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet16BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt30 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt30 + process.hltTktkFilterForDmesonGlobal8Dp30 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet16Centext30100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt30Cent30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt30 + process.hltTktkFilterForDmesonGlobal8Dp30 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet16Centext50100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt30Cent50100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt30 + process.hltTktkFilterForDmesonGlobal8Dp30 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt40_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt40 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt40 + process.hltTktkFilterForDmesonGlobal8Dp40 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28Centext30100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt40Cent30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt40 + process.hltTktkFilterForDmesonGlobal8Dp40 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet28Centext50100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt40Cent50100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt40 + process.hltTktkFilterForDmesonGlobal8Dp40 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt50_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet32BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt50 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt50 + process.hltTktkFilterForDmesonGlobal8Dp50 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt60_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleJet44BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt60 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt60 + process.hltTktkFilterForDmesonGlobal8Dp60 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt70_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet52BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt70 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt70 + process.hltTktkFilterForDmesonGlobal8Dp70 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext30100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt60Cent30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt60 + process.hltTktkFilterForDmesonGlobal8Dp60 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleS1Jet44Centext50100BptxAND + process.hltPreHIDmesonHITrackingGlobalDpt60Cent50100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt60 + process.hltTktkFilterForDmesonGlobal8Dp60 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt20Cent010 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt20 + process.hltTktkFilterForDmesonGlobal8Dp20 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt30Cent010 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt30 + process.hltTktkFilterForDmesonGlobal8Dp30 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIDmesonHITrackingGlobalDpt40Cent010 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackCandsForDmesonGlobalPt8 + process.hltHIFullTrackFilterForDmesonGlobalPt8 + process.hltTktkVtxForDmesonGlobal8Dpt40 + process.hltTktkFilterForDmesonGlobal8Dp40 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton10Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton15Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton20Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7BptxAND + process.hltPreHISinglePhoton30Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHISinglePhoton40Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton50_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHISinglePhoton50Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton50Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton60_Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG30BptxAND + process.hltPreHISinglePhoton60Eta1p5 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton60Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton10Eta1p5Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton15Eta1p5Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton20Eta1p5Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7Centext50100BptxAND + process.hltPreHISinglePhoton30Eta1p5Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21Centext50100BptxAND + process.hltPreHISinglePhoton40Eta1p5Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton10Eta1p5Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton15Eta1p5Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton20Eta1p5Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7Centext30100BptxAND + process.hltPreHISinglePhoton30Eta1p5Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21Centext30100BptxAND + process.hltPreHISinglePhoton40Eta1p5Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHISinglePhoton40Eta2p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton10Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton15Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreHISinglePhoton20Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7BptxAND + process.hltPreHISinglePhoton30Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHISinglePhoton40Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton50_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHISinglePhoton50Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton50Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton60_Eta3p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG30BptxAND + process.hltPreHISinglePhoton60Eta3p1 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton60Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton10Eta3p1Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton15Eta3p1Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext50100BptxAND + process.hltPreHISinglePhoton20Eta3p1Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7Centext50100BptxAND + process.hltPreHISinglePhoton30Eta3p1Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21Centext50100BptxAND + process.hltPreHISinglePhoton40Eta3p1Cent50100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton10Eta3p1Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton15Eta3p1Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG3Centext30100BptxAND + process.hltPreHISinglePhoton20Eta3p1Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG7Centext30100BptxAND + process.hltPreHISinglePhoton30Eta3p1Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21Centext30100BptxAND + process.hltPreHISinglePhoton40Eta3p1Cent30100 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta3p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHIDoublePhoton15Eta1p5Mass501000 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIDoublePhoton15Eta1p550nsMultiFit + process.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHIDoublePhoton15Eta1p5Mass501000R9HECut + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIDoublePhoton15Eta1p550nsMultiFit + process.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter + process.hltHIEgammaR9ID50nsMultiFit + process.hltHIEgammaR9IDDoublePhoton15Eta1p550nsMultiFit + process.hltHIEgammaHoverE50nsMultiFit + process.hltHIEgammaHOverEDoublePhoton15Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHIDoublePhoton15Eta2p1Mass501000R9Cut + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIDoublePhoton15Eta2p150nsMultiFit + process.hltHIDoublePhoton15Eta2p1GlobalMass501000Filter + process.hltHIEgammaR9ID50nsMultiFit + process.hltHIEgammaR9IDDoublePhoton15Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG21BptxAND + process.hltPreHIDoublePhoton15Eta2p5Mass501000R9SigmaHECut + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIDoublePhoton15Eta2p550nsMultiFit + process.hltHIDoublePhoton15Eta2p5GlobalMass501000Filter + process.hltHIEgammaR9ID50nsMultiFit + process.hltHIEgammaR9IDDoublePhoton15Eta2p550nsMultiFit + process.hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer + process.hltHIEgammaSigmaIEtaIEtaDoublePhoton15Eta2p550nsMultiFit + process.hltHIEgammaHoverE50nsMultiFit + process.hltHIEgammaHOverEDoublePhoton15Eta2p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet40Eta2p1 + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2Filtered + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet40Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3SingleCenJet28 + process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet60Eta2p1 + process.hltHIL1SingleMu3CenJet28Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2FilteredWithJet28 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3SingleCenJet40 + process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet80Eta2p1 + process.hltHIL1SingleMu3CenJet40Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2FilteredWithJet40 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3SingleCenJet40 + process.hltPreHIL2Mu3Eta2p5PuAK4CaloJet100Eta2p1 + process.hltHIL1SingleMu3CenJet40Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2FilteredWithJet40 + process.HLTPuAK4CaloJets50nsMultiFitSequence + process.hltSinglePuAK4CaloJet100Eta2p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL2Mu3Eta2p5HIPhoton10Eta1p5 + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2Filtered + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton10Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL2Mu3Eta2p5HIPhoton15Eta1p5 + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2Filtered + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton15Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL2Mu3Eta2p5HIPhoton20Eta1p5 + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2Filtered + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton20Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3SingleEG12 + process.hltPreHIL2Mu3Eta2p5HIPhoton30Eta1p5 + process.hltHIL1SingleMu3EG12Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2FilteredWithEG12 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton30Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3SingleEG20 + process.hltPreHIL2Mu3Eta2p5HIPhoton40Eta1p5 + process.hltHIL1SingleMu3EG20Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQL2FilteredWithEG20 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + process.hltHIPhoton40Eta1p550nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUCC100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIUCC100 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilter4470 + process.HLTDoHILocalPixelClustersSequence + process.hltPixelActivityFilter40000 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUCC020_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIUCC020 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilter4680 + process.HLTDoHILocalPixelClustersSequence + process.hltPixelActivityFilter60000 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Bottom005_Centrality1030_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Bottom005Centrality1030 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality1030 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterB005Cent1030 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Top005_Centrality1030_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Top005Centrality1030 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality1030 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterT005Cent1030 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Bottom005_Centrality3050_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Bottom005Centrality3050 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality3050 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterB005Cent3050 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Top005_Centrality3050_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Top005Centrality3050 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality3050 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterT005Cent3050 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Bottom005_Centrality5070_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Bottom005Centrality5070 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality5070 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterB005Cent5070 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIQ2Top005_Centrality5070_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIQ2Top005Centrality5070 + process.HLTRecoMETHfSequence + process.hltGlobalSumETHfFilterCentrality5070 + process.HLTDoCaloHcalMethod050nsMultiFitSequence + process.hltEvtPlaneProducer + process.hltEvtPlaneFilterT005Cent5070 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIFullTrack12L1MinimumBiasHF1AND + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter12 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack12_L1Centrality010_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIFullTrack12L1Centrality010 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter12 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack12_L1Centrality30100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack12Centrality30100 + process.hltPreHIFullTrack12L1Centrality30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter12 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIFullTrack18L1MinimumBiasHF1AND + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter18 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack18_L1Centrality010_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext010MinimumumBiasHF1AND + process.hltPreHIFullTrack18L1Centrality010 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter18 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack18_L1Centrality30100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack12Centrality30100 + process.hltPreHIFullTrack18L1Centrality30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter18 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack24_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack16 + process.hltPreHIFullTrack24 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter24 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack24_L1Centrality30100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack16Centrality30100 + process.hltPreHIFullTrack24L1Centrality30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter24 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack34_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack24 + process.hltPreHIFullTrack34 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter34 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack34_L1Centrality30100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack24Centrality30100 + process.hltPreHIFullTrack34L1Centrality30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter34 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack45_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack24 + process.hltPreHIFullTrack45 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter45 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIFullTrack45_L1Centrality30100_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleTrack24Centrality30100 + process.hltPreHIFullTrack45L1Centrality30100 + process.HLTHIPixelClusterSplitting + process.HLTDoHITrackingLocalStripSequenceZeroSuppression + process.HLTHIIterativeTrackingForGlobalPt8 + process.hltHIFullTrackSelectedTracks + process.hltHIFullTrackCandsForHITrackTrigger + process.hltHIFullTrackFilter45 + process.HLTDoHIStripZeroSuppressionRepacker + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0BptxAND + process.hltPreHIL1DoubleMu0 + process.hltHIDoubleMu0L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1AND + process.hltPreHIL1DoubleMu02HF + process.hltHIDoubleMu0MinBiasL1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0HFTower0 + process.hltPreHIL1DoubleMu02HF0 + process.hltHIDoubleMu0HFTower0Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu10_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu10BptxAND + process.hltPreHIL1DoubleMu10 + process.hltHIDoubleMu10L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_NHitQ_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0BptxAND + process.hltPreHIL2DoubleMu0NHitQ + process.hltHIDoubleMu0L1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0NHitQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_NHitQ_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1AND + process.hltPreHIL2DoubleMu0NHitQ2HF + process.hltHIDoubleMu0MinBiasL1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0NHitQ2HFFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_NHitQ_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0HFTower0 + process.hltPreHIL2DoubleMu0NHitQ2HF0 + process.hltHIDoubleMu0HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0NHitQ2HF0Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3_NHitQ10_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL2Mu3NHitQ102HF + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQ2HFL2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu3_NHitQ10_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3HFTower0 + process.hltPreHIL2Mu3NHitQ102HF0 + process.hltHIL1SingleMu3HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQ2HF0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu3_NHitQ15_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3MinBiasHF1AND + process.hltPreHIL3Mu3NHitQ152HF + process.hltHIL1SingleMu3MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQ2HFL2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu3NHit152HFL3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu3_NHitQ15_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu3HFTower0 + process.hltPreHIL3Mu3NHitQ152HF0 + process.hltHIL1SingleMu3HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu3N10HitQ2HF0L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu3NHit152HF0L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu5_NHitQ10_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu5MinBiasHF1AND + process.hltPreHIL2Mu5NHitQ102HF + process.hltHIL1SingleMu5MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu5N10HitQ2HFL2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu5_NHitQ10_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu5HFTower0 + process.hltPreHIL2Mu5NHitQ102HF0 + process.hltHIL1SingleMu5HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu5N10HitQ2HF0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu5_NHitQ15_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu5MinBiasHF1AND + process.hltPreHIL3Mu5NHitQ152HF + process.hltHIL1SingleMu5MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu5N10HitQ2HFL2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu5NHit152HFL3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu5_NHitQ15_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu5HFTower0 + process.hltPreHIL3Mu5NHitQ152HF0 + process.hltHIL1SingleMu5HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu5N10HitQ2HF0L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu5NHit152HF0L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu7_NHitQ10_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu7MinBiasHF1AND + process.hltPreHIL2Mu7NHitQ102HF + process.hltHIL1SingleMu7MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu7N10HitQ2HFL2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu7_NHitQ10_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu7HFTower0 + process.hltPreHIL2Mu7NHitQ102HF0 + process.hltHIL1SingleMu7HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu7N10HitQ2HF0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu7_NHitQ15_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu7MinBiasHF1AND + process.hltPreHIL3Mu7NHitQ152HF + process.hltHIL1SingleMu7MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu7N10HitQ2HFL2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu7NHit152HFL3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu7_NHitQ15_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu7HFTower0 + process.hltPreHIL3Mu7NHitQ152HF0 + process.hltHIL1SingleMu7HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu7N10HitQ2HF0L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu7NHit152HF0L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu15_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12BptxAND + process.hltPreHIL2Mu15 + process.hltHIL1SingleMu12Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu15L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu15_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12MinBiasHF1AND + process.hltPreHIL2Mu152HF + process.hltHIL1SingleMu12MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu152HFFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu15_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12HFTower0 + process.hltPreHIL2Mu152HF0 + process.hltHIL1SingleMu12HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu15N10HitQ2HF0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu15_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12BptxAND + process.hltPreHIL3Mu15 + process.hltHIL1SingleMu12Filtered + process.HLTL2muonrecoSequence + process.hltHIL3Mu15L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu15L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu15_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12MinBiasHF1AND + process.hltPreHIL3Mu152HF + process.hltHIL1SingleMu12MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL3Mu152HFL2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu152HFL3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu15_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu12HFTower0 + process.hltPreHIL3Mu152HF0 + process.hltHIL1SingleMu12HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL3Mu152HF0L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu152HF0L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu20_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16BptxAND + process.hltPreHIL2Mu20 + process.hltHIL1SingleMu16Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu20L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu20_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16MinBiasHF1AND + process.hltPreHIL2Mu202HF + process.hltHIL1SingleMu16MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL2Mu202HFL2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2Mu20_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16HFTower0 + process.hltPreHIL2Mu202HF0 + process.hltHIL1SingleMu16HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL2Mu202HF0L2Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu20_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16BptxAND + process.hltPreHIL3Mu20 + process.hltHIL1SingleMu16Filtered + process.HLTL2muonrecoSequence + process.hltHIL3Mu20L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHIL3SingleMu20L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu20_2HF_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16MinBiasHF1AND + process.hltPreHIL3Mu202HF + process.hltHIL1SingleMu16MinBiasFiltered + process.HLTL2muonrecoSequence + process.hltHIL3Mu202HFL2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu202HFL3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3Mu20_2HF0_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleMu16HFTower0 + process.hltPreHIL3Mu202HF0 + process.hltHIL1SingleMu16HFTower0Filtered + process.HLTL2muonrecoSequence + process.hltHIL3Mu202HF0L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHISingleMu202HF0L3Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_2HF_Cent30100_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30to100 + process.hltPreHIL1DoubleMu02HFCent30100 + process.hltHIDoubleMu0MinBiasCent30to100L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_2HF0_Cent30100_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0HFTower0Centrality30to100 + process.hltPreHIL1DoubleMu02HF0Cent30100 + process.hltHIDoubleMu0HFTower0Cent30to100L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30to100 + process.hltPreHIL2DoubleMu02HFCent30100NHitQ + process.hltHIDoubleMu0MinBiasCent30to100L1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu02HFcent30100NHitQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1DoubleMu0_Cent30_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL1DoubleMu0Cent30 + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0HFTower0Centrality30to100 + process.hltPreHIL2DoubleMu02HF0Cent30100NHitQ + process.hltHIDoubleMu0HFTower0Cent30to100L1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu02HF0cent30100NHitQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL2DoubleMu0Cent30OSNHitQ + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0cent30OSNHitQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL2DoubleMu0_Cent30_NHitQ_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL2DoubleMu0Cent30NHitQ + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTL2muonrecoSequence + process.hltHIL2DoubleMu0cent30NHitQFiltered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMu0_Cent30_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL3DoubleMu0Cent30 + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTL2muonrecoSequence + process.hltHIDimuonOpenCentrality30L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHIDimuonOpenCentrality30L3Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL3DoubleMu0Cent30OSm2p5to4p5 + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTL2muonrecoSequence + process.hltHIDimuonOpenCentrality30L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHIDimuonOpenCentrality30OSm2p5to4p5L3Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0MinBiasHF1ANDCentrality30 + process.hltPreHIL3DoubleMu0Cent30OSm7to14 + process.hltHIDoubleMu0MinBiasCent30L1Filtered + process.HLTL2muonrecoSequence + process.hltHIDimuonOpenCentrality30L2Filtered + process.HLTHIL3muonrecoSequence + process.hltHIDimuonOpenCentrality30OSm7to14L3Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0BptxAND + process.hltPreHIL3DoubleMu0OSm2p5to4p5 + process.hltHIDoubleMu0L1Filtered + process.HLTL2muonrecoSequence + process.hltHIDimuonOpenL2FilteredNoMBHFgated + process.HLTHIL3muonrecoSequence + process.hltHIDimuonOpenOSm2p5to4p5L3Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL3DoubleMu0_OS_m7to14_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMu0BptxAND + process.hltPreHIL3DoubleMu0OSm7to14 + process.hltHIDoubleMu0L1Filtered + process.HLTL2muonrecoSequence + process.hltHIDimuonOpenL2FilteredNoMBHFgated + process.HLTHIL3muonrecoSequence + process.hltHIDimuonOpenOSm7to14L3Filter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1SingleMuOpenNotHF2_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MuOpenNotMinimumBiasHF2AND + process.hltPreHIUPCL1SingleMuOpenNotHF2 + process.hltL1MuOpenNotHF2L1Filtered0 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MuOpenNotMinimumBiasHF2AND + process.hltPreHIUPCSingleMuNotHF2PixelSingleTrack + process.hltL1MuOpenNotHF2L1Filtered0 + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotHF2_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotMinimumBiasHF2AND + process.hltPreHIUPCL1DoubleMuOpenNotHF2 + process.hltL1MuOpenNotHF2L1Filtered2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotMinimumBiasHF2AND + process.hltPreHIUPCDoubleMuNotHF2PixelSingleTrack + process.hltL1MuOpenNotHF2L1Filtered2 + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1SingleEG2NotHF2_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG2NotMinimumBiasHF2AND + process.hltPreHIUPCL1SingleEG2NotHF2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG2NotMinimumBiasHF2AND + process.hltPreHIUPCSingleEG2NotHF2PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleEG2NotHF2_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG2NotMinimumBiasHF2AND + process.hltPreHIUPCL1DoubleEG2NotHF2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG2NotMinimumBiasHF2AND + process.hltPreHIUPCDoubleEG2NotHF2PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1SingleEG5NotHF2_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG5NotMinimumBiasHF2AND + process.hltPreHIUPCL1SingleEG5NotHF2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG5NotMinimumBiasHF2AND + process.hltPreHIUPCSingleEG5NotHF2PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotHF1_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotMinimumBiasHF1AND + process.hltPreHIUPCL1DoubleMuOpenNotHF1 + process.hltL1MuOpenL1Filtered2 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotMinimumBiasHF1AND + process.hltPreHIUPCDoubleMuNotHF1PixelSingleTrack + process.hltL1MuOpenL1Filtered2 + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleEG2NotZDCAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG2NotZDCAND + process.hltPreHIUPCL1DoubleEG2NotZDCAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleEG2NotZDCAND + process.hltPreHIUPCL1DoubleEG2NotZDCANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotZDCAND + process.hltPreHIUPCL1DoubleMuOpenNotZDCAND + process.hltL1MuOpenL1Filtered3 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1DoubleMuOpenNotZDCAND + process.hltPreHIUPCL1DoubleMuOpenNotZDCANDPixelSingleTrack + process.hltL1MuOpenL1Filtered3 + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1EG2NotZDCAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG2NotZDCAND + process.hltPreHIUPCL1EG2NotZDCAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1EG2NotZDCAND + process.hltPreHIUPCEG2NotZDCANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1MuOpenNotZDCAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MuOpenNotZDCAND + process.hltPreHIUPCL1MuOpenNotZDCAND + process.hltL1MuOpenL1Filtered4 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MuOpenNotZDCAND + process.hltPreHIUPCL1MuOpenNotZDCANDPixelSingleTrack + process.hltL1MuOpenL1Filtered4 + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotHFplusANDminusTH0BptxAND + process.hltPreHIUPCL1NotHFplusANDminusTH0BptxAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotHFplusANDminusTH0BptxAND + process.hltPreHIUPCL1NotHFplusANDminusTH0BptxANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 + process.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 + process.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1DoubleMuOpenNotHFplusANDminustTH0BptxAND + process.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0 + process.hltL1DoubleMuOpenTH0L1Filtered + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1DoubleMuOpenNotHFplusANDminustTH0BptxAND + process.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack + process.hltL1DoubleMuOpenTH0L1Filtered + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1CastorMediumJet_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorMediumJet + process.hltPreHIL1CastorMediumJet + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1CastorMediumJetAK4CaloJet20_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorMediumJet + process.hltPreHIL1CastorMediumJetAK4CaloJet20 + process.HLTPuAK4CaloJetsUPCSequence + process.hltSinglePuAK4CaloJet20Eta5p150nsMultiFit + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HICastorMediumJetPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1CastorMediumJet + process.hltPreHICastorMediumJetPixelSingleTrack + process.hltL1CastorMediumJetFiltered0UPC + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotMinimumBiasHF2_AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotMinimumBiasHF2AND + process.hltPreHIUPCL1NotMinimumBiasHF2AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotMinimumBiasHF2AND + process.hltPreHIUPCL1NotMinimumBiasHF2ANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1ZdcOR_BptxAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1ZdcORBptxAND + process.hltPreHIUPCL1ZdcORBptxAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1ZdcORBptxAND + process.hltPreHIUPCL1ZdcORBptxANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1ZdcXOR_BptxAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1ZdcXORBptxAND + process.hltPreHIUPCL1ZdcXORBptxAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1ZdcXORBptxAND + process.hltPreHIUPCL1ZdcXORBptxANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotZdcOR_BptxAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotZdcORBptxAND + process.hltPreHIUPCL1NotZdcORBptxAND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sHLTHIUPCL1NotZdcORBptxAND + process.hltPreHIUPCL1NotZdcORBptxANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTRecopixelvertexingSequenceForUPC + process.hltPixelCandsForUPC + process.hltSinglePixelTrackForUPC + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIZeroBias_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreHIZeroBias + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HICentralityVeto_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHICentralityVeto + process.HLTDoHILocalPixelSequence + process.hltPixelActivityFilterCentralityVeto + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Tech5_BPTX_PlusOnly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech5 + process.hltPreHIL1Tech5BPTXPlusOnly + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Tech6_BPTX_MinusOnly_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Tech6 + process.hltPreHIL1Tech6BPTXMinusOnly + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Tech7_NoBPTX_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sTech7 + process.hltPreHIL1Tech7NoBPTX + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1MinimumBiasHF1OR_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1OR + process.hltPreHIL1MinimumBiasHF1OR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1MinimumBiasHF2OR_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF2OR + process.hltPreHIL1MinimumBiasHF2OR + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1MinimumBiasHF1AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIL1MinimumBiasHF1AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1MinimumBiasHF2AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF2AND + process.hltPreHIL1MinimumBiasHF2AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sMinimumBiasHF1AND + process.hltPreHIL1MinimumBiasHF1ANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIZeroBiasPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreHIZeroBiasPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext70100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext70100MinimumumBiasHF1AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext70100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext70100MinimumumBiasHF1ANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext50100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext50100MinimumumBiasHF1AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext50100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext50100MinimumumBiasHF1ANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext30100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext30100MinimumumBiasHF1AND + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1Centralityext30100MinimumumBiasHF1AND + process.hltPreHIL1Centralityext30100MinimumumBiasHF1ANDPixelSingleTrack + process.HLTDoHILocalPixelSequence + process.HLTPixelTrackingForHITrackTrigger + process.hltHISinglePixelTrackFilter + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIPhysics_v1 = cms.Path( process.HLTBeginSequence + process.hltPreHIPhysics + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_HIRandom_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreHIRandom + process.HLTDoHIStripZeroSuppression + process.HLTEndSequence )
process.HLT_EcalCalibration_v2 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreEcalCalibration + process.hltEcalCalibrationRaw + process.HLTEndSequence )
process.HLT_HcalCalibration_v1 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHcalCalibration + process.hltHcalCalibTypeFilter + process.hltHcalCalibrationRaw + process.HLTEndSequence )
process.AlCa_EcalPhiSymForHI_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF1AND + process.hltPreAlCaEcalPhiSymForHI + process.HLTDoFullUnpackingEgammaEcal50nsMultiFitSequence + process.hltEcal50nsMultifitPhiSymFilter + process.HLTEndSequence )
process.AlCa_RPCMuonNoTriggersForHI_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPCForHI + process.hltPreAlCaRPCMuonNoTriggersForHI + process.hltRPCMuonNoTriggersL1Filtered0ForHI + process.HLTMuonLocalRecoSequence + process.HLTEndSequence )
process.AlCa_RPCMuonNoHitsForHI_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPCForHI + process.hltPreAlCaRPCMuonNoHitsForHI + process.HLTMuonLocalRecoSequence + process.hltRPCPointProducer + process.hltRPCFilter + process.HLTEndSequence )
process.AlCa_RPCMuonNormalisationForHI_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaRPCForHI + process.hltPreAlCaRPCMuonNormalisationForHI + process.hltRPCMuonNormaL1Filtered0ForHI + process.HLTMuonLocalRecoSequence + process.HLTEndSequence )
process.AlCa_LumiPixels_Random_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreAlCaLumiPixelsRandom + process.hltFEDSelectorLumiPixels + process.HLTEndSequence )
process.AlCa_LumiPixels_ZeroBias_v2 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBias + process.hltPreAlCaLumiPixelsZeroBias + process.hltFEDSelectorLumiPixels + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtDigis + process.hltScalersRawToDigi + process.hltFEDSelector + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW + process.hltBoolFalse )
process.HLTAnalyzerEndpath = cms.EndPath( process.hltGtDigis + process.hltPreAnalyzerEndpath + process.hltL1GtTrigReport + process.hltTrigReport )
process.HIPhysicsMuonsOutput = cms.EndPath( process.hltGtDigis + process.hltPreHIPhysicsMuonsOutput + process.hltOutputHIPhysicsMuons )
process.HIPhysicsHardProbesOutput = cms.EndPath( process.hltGtDigis + process.hltPreHIPhysicsHardProbesOutput + process.hltOutputHIPhysicsHardProbes )
process.HIPhysicsMinBiasUPCOutput = cms.EndPath( process.hltGtDigis + process.hltPreHIPhysicsMinBiasUPCOutput + process.hltOutputHIPhysicsMinBiasUPC )

# load the DQMStore and DQMRootOutputModule
process.load( "DQMServices.Core.DQMStore_cfi" )
process.DQMStore.enableMultiThread = True

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)
process.DQMOutput = cms.EndPath( process.dqmOutput + process.hltGtDigis + process.hltPreDQMOutput + process.hltPreDQMOutputSmart + process.hltOutputDQM )
process.DQMCalibrationOutput = cms.EndPath( process.hltGtDigis + process.hltPreDQMCalibrationOutput + process.hltOutputDQMCalibration )
process.DQMEventDisplayOutput = cms.EndPath( process.hltGtDigis + process.hltPreDQMEventDisplayOutput + process.hltPreDQMEventDisplayOutputSmart + process.hltOutputDQMEventDisplay )
process.RPCMONOutput = cms.EndPath( process.hltGtDigis + process.hltPreRPCMONOutput + process.hltOutputRPCMON )
process.CalibrationOutput = cms.EndPath( process.hltGtDigis + process.hltPreCalibrationOutput + process.hltOutputCalibration )
process.EcalCalibrationOutput = cms.EndPath( process.hltGtDigis + process.hltPreEcalCalibrationOutput + process.hltOutputEcalCalibration )
process.ALCAPHISYMOutput = cms.EndPath( process.hltGtDigis + process.hltPreALCAPHISYMOutput + process.hltOutputALCAPHISYM )
process.ALCALUMIPIXELSOutput = cms.EndPath( process.hltGtDigis + process.hltPreALCALUMIPIXELSOutput + process.hltOutputALCALUMIPIXELS )
process.HIExpressOutput = cms.EndPath( process.hltGtDigis + process.hltPreHIExpressOutput + process.hltPreHIExpressOutputSmart + process.hltOutputHIExpress )
process.NanoDSTOutput = cms.EndPath( process.hltGtDigis + process.hltPreNanoDSTOutput + process.hltOutputNanoDST )


process.HLTSchedule = cms.Schedule( *(process.HLTriggerFirstPath, process.DST_Physics_v1, process.HLT_HIPuAK4CaloJet40_Eta5p1_v2, process.HLT_HIPuAK4CaloJet60_Eta5p1_v2, process.HLT_HIPuAK4CaloJet80_Eta5p1_v2, process.HLT_HIPuAK4CaloJet100_Eta5p1_v2, process.HLT_HIPuAK4CaloJet110_Eta5p1_v2, process.HLT_HIPuAK4CaloJet120_Eta5p1_v2, process.HLT_HIPuAK4CaloJet150_Eta5p1_v2, process.HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v2, process.HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v2, process.HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v2, process.HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v2, process.HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v2, process.HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v2, process.HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v2, process.HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v2, process.HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v2, process.HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v2, process.HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v2, process.HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v2, process.HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v2, process.HLT_HIPuAK4CaloDJet60_Eta2p1_v2, process.HLT_HIPuAK4CaloDJet80_Eta2p1_v2, process.HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v2, process.HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v2, process.HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v2, process.HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt20_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt30_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt40_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt50_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt60_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt70_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v2, process.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v1, process.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v1, process.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v1, process.HLT_HISinglePhoton10_Eta1p5_v2, process.HLT_HISinglePhoton15_Eta1p5_v2, process.HLT_HISinglePhoton20_Eta1p5_v2, process.HLT_HISinglePhoton30_Eta1p5_v2, process.HLT_HISinglePhoton40_Eta1p5_v2, process.HLT_HISinglePhoton50_Eta1p5_v2, process.HLT_HISinglePhoton60_Eta1p5_v2, process.HLT_HISinglePhoton10_Eta1p5_Cent50_100_v2, process.HLT_HISinglePhoton15_Eta1p5_Cent50_100_v2, process.HLT_HISinglePhoton20_Eta1p5_Cent50_100_v2, process.HLT_HISinglePhoton30_Eta1p5_Cent50_100_v2, process.HLT_HISinglePhoton40_Eta1p5_Cent50_100_v2, process.HLT_HISinglePhoton10_Eta1p5_Cent30_100_v2, process.HLT_HISinglePhoton15_Eta1p5_Cent30_100_v2, process.HLT_HISinglePhoton20_Eta1p5_Cent30_100_v2, process.HLT_HISinglePhoton30_Eta1p5_Cent30_100_v2, process.HLT_HISinglePhoton40_Eta1p5_Cent30_100_v2, process.HLT_HISinglePhoton40_Eta2p1_v2, process.HLT_HISinglePhoton10_Eta3p1_v2, process.HLT_HISinglePhoton15_Eta3p1_v2, process.HLT_HISinglePhoton20_Eta3p1_v2, process.HLT_HISinglePhoton30_Eta3p1_v2, process.HLT_HISinglePhoton40_Eta3p1_v2, process.HLT_HISinglePhoton50_Eta3p1_v2, process.HLT_HISinglePhoton60_Eta3p1_v2, process.HLT_HISinglePhoton10_Eta3p1_Cent50_100_v2, process.HLT_HISinglePhoton15_Eta3p1_Cent50_100_v2, process.HLT_HISinglePhoton20_Eta3p1_Cent50_100_v2, process.HLT_HISinglePhoton30_Eta3p1_Cent50_100_v2, process.HLT_HISinglePhoton40_Eta3p1_Cent50_100_v2, process.HLT_HISinglePhoton10_Eta3p1_Cent30_100_v2, process.HLT_HISinglePhoton15_Eta3p1_Cent30_100_v2, process.HLT_HISinglePhoton20_Eta3p1_Cent30_100_v2, process.HLT_HISinglePhoton30_Eta3p1_Cent30_100_v2, process.HLT_HISinglePhoton40_Eta3p1_Cent30_100_v2, process.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v2, process.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v2, process.HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v2, process.HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v2, process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v2, process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v2, process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v2, process.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v2, process.HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v2, process.HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v2, process.HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v2, process.HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v2, process.HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v2, process.HLT_HIUCC100_v2, process.HLT_HIUCC020_v2, process.HLT_HIQ2Bottom005_Centrality1030_v2, process.HLT_HIQ2Top005_Centrality1030_v2, process.HLT_HIQ2Bottom005_Centrality3050_v2, process.HLT_HIQ2Top005_Centrality3050_v2, process.HLT_HIQ2Bottom005_Centrality5070_v2, process.HLT_HIQ2Top005_Centrality5070_v2, process.HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v2, process.HLT_HIFullTrack12_L1Centrality010_v2, process.HLT_HIFullTrack12_L1Centrality30100_v2, process.HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v2, process.HLT_HIFullTrack18_L1Centrality010_v2, process.HLT_HIFullTrack18_L1Centrality30100_v2, process.HLT_HIFullTrack24_v2, process.HLT_HIFullTrack24_L1Centrality30100_v2, process.HLT_HIFullTrack34_v2, process.HLT_HIFullTrack34_L1Centrality30100_v2, process.HLT_HIFullTrack45_v2, process.HLT_HIFullTrack45_L1Centrality30100_v2, process.HLT_HIL1DoubleMu0_v1, process.HLT_HIL1DoubleMu0_2HF_v1, process.HLT_HIL1DoubleMu0_2HF0_v1, process.HLT_HIL1DoubleMu10_v1, process.HLT_HIL2DoubleMu0_NHitQ_v2, process.HLT_HIL2DoubleMu0_NHitQ_2HF_v1, process.HLT_HIL2DoubleMu0_NHitQ_2HF0_v1, process.HLT_HIL2Mu3_NHitQ10_2HF_v1, process.HLT_HIL2Mu3_NHitQ10_2HF0_v1, process.HLT_HIL3Mu3_NHitQ15_2HF_v1, process.HLT_HIL3Mu3_NHitQ15_2HF0_v1, process.HLT_HIL2Mu5_NHitQ10_2HF_v1, process.HLT_HIL2Mu5_NHitQ10_2HF0_v1, process.HLT_HIL3Mu5_NHitQ15_2HF_v1, process.HLT_HIL3Mu5_NHitQ15_2HF0_v1, process.HLT_HIL2Mu7_NHitQ10_2HF_v1, process.HLT_HIL2Mu7_NHitQ10_2HF0_v1, process.HLT_HIL3Mu7_NHitQ15_2HF_v1, process.HLT_HIL3Mu7_NHitQ15_2HF0_v1, process.HLT_HIL2Mu15_v2, process.HLT_HIL2Mu15_2HF_v1, process.HLT_HIL2Mu15_2HF0_v1, process.HLT_HIL3Mu15_v1, process.HLT_HIL3Mu15_2HF_v1, process.HLT_HIL3Mu15_2HF0_v1, process.HLT_HIL2Mu20_v1, process.HLT_HIL2Mu20_2HF_v1, process.HLT_HIL2Mu20_2HF0_v1, process.HLT_HIL3Mu20_v1, process.HLT_HIL3Mu20_2HF_v1, process.HLT_HIL3Mu20_2HF0_v1, process.HLT_HIL1DoubleMu0_2HF_Cent30100_v1, process.HLT_HIL1DoubleMu0_2HF0_Cent30100_v1, process.HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v1, process.HLT_HIL1DoubleMu0_Cent30_v1, process.HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v1, process.HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v1, process.HLT_HIL2DoubleMu0_Cent30_NHitQ_v1, process.HLT_HIL3DoubleMu0_Cent30_v1, process.HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v1, process.HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v1, process.HLT_HIL3DoubleMu0_OS_m2p5to4p5_v1, process.HLT_HIL3DoubleMu0_OS_m7to14_v1, process.HLT_HIUPCL1SingleMuOpenNotHF2_v1, process.HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v1, process.HLT_HIUPCL1DoubleMuOpenNotHF2_v1, process.HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v1, process.HLT_HIUPCL1SingleEG2NotHF2_v1, process.HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v1, process.HLT_HIUPCL1DoubleEG2NotHF2_v1, process.HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v1, process.HLT_HIUPCL1SingleEG5NotHF2_v1, process.HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v1, process.HLT_HIUPCL1DoubleMuOpenNotHF1_v1, process.HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v1, process.HLT_HIUPCL1DoubleEG2NotZDCAND_v1, process.HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v1, process.HLT_HIUPCL1DoubleMuOpenNotZDCAND_v1, process.HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v1, process.HLT_HIUPCL1EG2NotZDCAND_v1, process.HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v1, process.HLT_HIUPCL1MuOpenNotZDCAND_v1, process.HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v1, process.HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v1, process.HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v1, process.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v2, process.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2, process.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v2, process.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v2, process.HLT_HIL1CastorMediumJet_v1, process.HLT_HIL1CastorMediumJetAK4CaloJet20_v2, process.HLT_HICastorMediumJetPixel_SingleTrack_v1, process.HLT_HIUPCL1NotMinimumBiasHF2_AND_v1, process.HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v1, process.HLT_HIUPCL1ZdcOR_BptxAND_v1, process.HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v1, process.HLT_HIUPCL1ZdcXOR_BptxAND_v1, process.HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v1, process.HLT_HIUPCL1NotZdcOR_BptxAND_v1, process.HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v1, process.HLT_HIZeroBias_v1, process.HLT_HICentralityVeto_v1, process.HLT_HIL1Tech5_BPTX_PlusOnly_v1, process.HLT_HIL1Tech6_BPTX_MinusOnly_v1, process.HLT_HIL1Tech7_NoBPTX_v1, process.HLT_HIL1MinimumBiasHF1OR_v1, process.HLT_HIL1MinimumBiasHF2OR_v1, process.HLT_HIL1MinimumBiasHF1AND_v1, process.HLT_HIL1MinimumBiasHF2AND_v1, process.HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v1, process.HLT_HIZeroBiasPixel_SingleTrack_v1, process.HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v1, process.HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v1, process.HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v1, process.HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v1, process.HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v1, process.HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v1, process.HLT_HIPhysics_v1, process.HLT_HIRandom_v1, process.HLT_EcalCalibration_v2, process.HLT_HcalCalibration_v1, process.AlCa_EcalPhiSymForHI_v2, process.AlCa_RPCMuonNoTriggersForHI_v1, process.AlCa_RPCMuonNoHitsForHI_v1, process.AlCa_RPCMuonNormalisationForHI_v1, process.AlCa_LumiPixels_Random_v1, process.AlCa_LumiPixels_ZeroBias_v2, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath, process.HIPhysicsMuonsOutput, process.HIPhysicsHardProbesOutput, process.HIPhysicsMinBiasUPCOutput, process.DQMOutput, process.DQMCalibrationOutput, process.DQMEventDisplayOutput, process.RPCMONOutput, process.CalibrationOutput, process.EcalCalibrationOutput, process.ALCAPHISYMOutput, process.ALCALUMIPIXELSOutput, process.HIExpressOutput, process.NanoDSTOutput ))


process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:RelVal_Raw_HIon_2015_v2_DATA.root',
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreExpressCosmicsOutputSmart' in process.__dict__:
    process.hltPreExpressCosmicsOutputSmart.hltResults = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreExpressOutputSmart' in process.__dict__:
    process.hltPreExpressOutputSmart.hltResults        = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreDQMForHIOutputSmart' in process.__dict__:
    process.hltPreDQMForHIOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreDQMForPPOutputSmart' in process.__dict__:
    process.hltPreDQMForPPOutputSmart.hltResults       = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreHLTDQMResultsOutputSmart' in process.__dict__:
    process.hltPreHLTDQMResultsOutputSmart.hltResults  = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreHLTDQMOutputSmart' in process.__dict__:
    process.hltPreHLTDQMOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltPreHLTMONOutputSmart' in process.__dict__:
    process.hltPreHLTMONOutputSmart.hltResults         = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', 'HLTHIon2015v2' )
    process.hltDQMHLTScalers.processname                      = 'HLTHIon2015v2'

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname              = 'HLTHIon2015v2'

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# enable the TrigReport and TimeReport
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:run2_hlt_HIon_2015_v2')
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    for pset in process.GlobalTag.toGet.value():
        pset.connect = pset.connect.value().replace('frontier://FrontierProd/', 'frontier://FrontierProd/')
    # fix for multi-run processing
    process.GlobalTag.RefreshEachRun = cms.untracked.bool( False )
    process.GlobalTag.ReconnectEachRun = cms.untracked.bool( False )

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')
    process.MessageLogger.categories.append('FastReport')

# add specific customizations
_customInfo = {}
_customInfo['menuType'  ]= "HIon_2015_v2"
_customInfo['globalTags']= {}
_customInfo['globalTags'][True ] = "auto:run2_hlt_HIon_2015_v2"
_customInfo['globalTags'][False] = "auto:run2_mc_HIon_2015_v2"
_customInfo['inputFiles']={}
_customInfo['inputFiles'][True]  = "file:RelVal_Raw_HIon_2015_v2_DATA.root"
_customInfo['inputFiles'][False] = "file:RelVal_Raw_HIon_2015_v2_MC.root"
_customInfo['maxEvents' ]=  100
_customInfo['globalTag' ]= "auto:run2_hlt_HIon_2015_v2"
_customInfo['inputFile' ]=  ['file:RelVal_Raw_HIon_2015_v2_DATA.root']
_customInfo['realData'  ]=  True
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
process = customizeHLTforAll(process,_customInfo)

