# /dev/CMSSW_8_0_0/HIon/V119 (CMSSW_8_0_10)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )

fragment.HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_8_0_0/HIon/V119')
)

fragment.HLTPSetInitialStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.2 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 2 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  seedPairPenalty = cms.int32( 0 ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetInitialStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilter" ) ),
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
  estimator = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.HLTPSetDetachedStepTrajectoryFilterBase = cms.PSet( 
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
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 2 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  seedPairPenalty = cms.int32( 0 ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetDetachedStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedStepTrajectoryFilter" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 3 ),
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
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.HLTPSetPixelPairStepTrajectoryFilterBase = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.1 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 2 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  seedPairPenalty = cms.int32( 0 ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetPixelPairStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilter" ) ),
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
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.HLTPSetMixedStepTrajectoryFilterBase = cms.PSet( 
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
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetMixedStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForMixedStep" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMixedStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMixedStepTrajectoryFilter" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 2 ),
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
  estimator = cms.string( "hltESPChi2ChargeTightMeasurementEstimator16" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.HLTPSetPixelLessStepTrajectoryFilterBase = cms.PSet( 
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
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPixelLessStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelLessStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelLessStepTrajectoryFilter" ) ),
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
  estimator = cms.string( "hltESPChi2ChargeTightMeasurementEstimator16" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 4 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.transferSystem = cms.PSet( 
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
fragment.HLTPSetTrajectoryBuilderForGsfElectrons = cms.PSet( 
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
fragment.HLTIter4PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTIter3PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTIter2PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 1 ),
  seedExtension = cms.int32( 1 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.2 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 1 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetbJetRegionalTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 8 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetTrajectoryFilterL3 = cms.PSet( 
  minPt = cms.double( 0.5 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 1000000000 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetTrajectoryFilterForElectrons = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minPt = cms.double( 2.0 ),
  minHitsMinPt = cms.int32( -1 ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minimumNumberOfHits = cms.int32( 5 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetMuonCkfTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minimumNumberOfHits = cms.int32( 5 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetMuTrackJpsiTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 10.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 8 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetMuTrackJpsiEffTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 9 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetCkfTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetCkf3HitTrajectoryFilter = cms.PSet( 
  minPt = cms.double( 0.9 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( -1 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTIter4PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter4PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter4ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeLooseMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  minNrOfHitsForRebuild = cms.untracked.int32( 4 )
)
fragment.HLTIter3PSetTrajectoryBuilderIT = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter3PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 1 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  MeasurementTrackerName = cms.string( "hltIter3ESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2ChargeLooseMeasurementEstimator16" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 )
)
fragment.HLTIter2PSetTrajectoryBuilderIT = cms.PSet( 
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
fragment.HLTIter1PSetTrajectoryBuilderIT = cms.PSet( 
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
fragment.HLTPSetTrajectoryBuilderForElectrons = cms.PSet( 
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
fragment.HLTPSetMuTrackJpsiTrajectoryBuilder = cms.PSet( 
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
fragment.HLTPSetMuTrackJpsiEffTrajectoryBuilder = cms.PSet( 
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
fragment.HLTPSetMuonCkfTrajectoryBuilderSeedHit = cms.PSet( 
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
fragment.HLTPSetMuonCkfTrajectoryBuilder = cms.PSet( 
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
fragment.HLTPSetPvClusterComparer = cms.PSet( 
  track_pt_min = cms.double( 2.5 ),
  track_pt_max = cms.double( 10.0 ),
  track_chi2_max = cms.double( 9999999.0 ),
  track_prob_min = cms.double( -1.0 )
)
fragment.HLTIter0PSetTrajectoryBuilderIT = cms.PSet( 
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
fragment.HLTIter0PSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  maxCCCLostHits = cms.int32( 1 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPvClusterComparerForBTag = cms.PSet( 
  track_pt_min = cms.double( 0.1 ),
  track_pt_max = cms.double( 20.0 ),
  track_chi2_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 )
)
fragment.HLTSeedFromConsecutiveHitsTripletOnlyCreator = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsTripletOnlyCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
fragment.HLTSeedFromConsecutiveHitsCreator = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterial" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "" )
)
fragment.HLTIter0HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
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
fragment.HLTIter2HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
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
fragment.HLTIter2HighPtTkMuPSetTrajectoryFilterIT = cms.PSet( 
  minPt = cms.double( 0.3 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 3 ),
  minimumNumberOfHits = cms.int32( 5 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPvClusterComparerForIT = cms.PSet( 
  track_pt_min = cms.double( 1.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_chi2_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 )
)
fragment.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
fragment.HLTSiStripClusterChargeCutLoose = cms.PSet(  value = cms.double( 1620.0 ) )
fragment.HLTSiStripClusterChargeCutTight = cms.PSet(  value = cms.double( 1945.0 ) )
fragment.HLTSeedFromConsecutiveHitsCreatorIT = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
fragment.HLTSeedFromProtoTracks = cms.PSet( 
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  MinOneOverPtError = cms.double( 1.0 ),
  magneticField = cms.string( "ParabolicMf" ),
  TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  forceKinematicWithRegionDirection = cms.bool( False )
)
fragment.HLTPSetMuonTrackingRegionBuilder8356 = cms.PSet( 
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
fragment.HLTPSetDetachedCkfTrajectoryBuilderForHI = cms.PSet( 
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
fragment.HLTPSetDetachedCkfTrajectoryFilterForHI = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 6 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.3 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 1 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  constantValueForLostHitsFractionFilter = cms.double( 0.701 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPixelPairCkfTrajectoryFilterForHI = cms.PSet( 
  minPt = cms.double( 1.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPixelPairCkfTrajectoryBuilderForHI = cms.PSet( 
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
fragment.HLTSiStripClusterChargeCutForHI = cms.PSet(  value = cms.double( 2069.0 ) )
fragment.HLTPSetDetachedCkfTrajectoryFilterForHIGlobalPt8 = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 6 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 8.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 1 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  constantValueForLostHitsFractionFilter = cms.double( 0.701 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetDetachedCkfTrajectoryBuilderForHIGlobalPt8 = cms.PSet( 
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
fragment.HLTPSetPixelPairCkfTrajectoryFilterForHIGlobalPt8 = cms.PSet( 
  minPt = cms.double( 8.0 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  minimumNumberOfHits = cms.int32( 6 ),
  nSigmaMinPt = cms.double( 5.0 ),
  chargeSignificance = cms.double( -1.0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTPSetPixelPairCkfTrajectoryBuilderForHIGlobalPt8 = cms.PSet( 
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
fragment.HLTPSetInitialCkfTrajectoryBuilderForHI = cms.PSet( 
  propagatorAlong = cms.string( "PropagatorWithMaterialForHI" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialCkfTrajectoryFilterForHI" ) ),
  maxCand = cms.int32( 5 ),
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  intermediateCleaning = cms.bool( False ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  updator = cms.string( "hltESPKFUpdator" ),
  alwaysUseInvalidHits = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 )
)
fragment.HLTPSetInitialCkfTrajectoryFilterForHI = cms.PSet( 
  minimumNumberOfHits = cms.int32( 6 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  maxLostHits = cms.int32( 999 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxConsecLostHits = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minPt = cms.double( 0.9 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  maxCCCLostHits = cms.int32( 9999 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 )
)
fragment.HLTSiStripClusterChargeCutTiny = cms.PSet(  value = cms.double( 800.0 ) )
fragment.HLTPSetTobTecStepTrajectoryBuilder = cms.PSet( 
  MeasurementTrackerName = cms.string( "" ),
  useSameTrajFilter = cms.bool( False ),
  minNrOfHitsForRebuild = cms.int32( 4 ),
  alwaysUseInvalidHits = cms.bool( False ),
  maxCand = cms.int32( 2 ),
  estimator = cms.string( "hltESPChi2ChargeTightMeasurementEstimator16" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTobTecStepTrajectoryFilterBase" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTobTecStepInOutTrajectoryFilterBase" ) ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  intermediateCleaning = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  requireSeedHitsInRebuild = cms.bool( True ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
)
fragment.HLTPSetTobTecStepTrajectoryFilterBase = cms.PSet( 
  maxLostHits = cms.int32( 0 ),
  minimumNumberOfHits = cms.int32( 5 ),
  seedPairPenalty = cms.int32( 1 ),
  minPt = cms.double( 0.1 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  strictSeedExtension = cms.bool( False ),
  seedExtension = cms.int32( 0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetTobTecStepInOutTrajectoryFilterBase = cms.PSet( 
  maxLostHits = cms.int32( 0 ),
  minimumNumberOfHits = cms.int32( 4 ),
  seedPairPenalty = cms.int32( 1 ),
  minPt = cms.double( 0.1 ),
  minHitsMinPt = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  chargeSignificance = cms.double( -1.0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  strictSeedExtension = cms.bool( False ),
  seedExtension = cms.int32( 0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetLowPtStepTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  bestHitOnly = cms.bool( True ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetLowPtStepTrajectoryFilter" ) ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetLowPtStepTrajectoryFilter" ) ),
  useSameTrajFilter = cms.bool( True ),
  maxCand = cms.int32( 4 ),
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
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.7 )
)
fragment.HLTPSetDetachedStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CompositeTrajectoryFilter" ),
  filters = cms.VPSet( 
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedStepTrajectoryFilterBase" )    )
  )
)
fragment.HLTPSetInitialStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CompositeTrajectoryFilter" ),
  filters = cms.VPSet( 
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterBase" )    )
  )
)
fragment.HLTPSetPixelPairStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CompositeTrajectoryFilter" ),
  filters = cms.VPSet( 
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilterBase" )    )
  )
)
fragment.HLTPSetLowPtStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  seedPairPenalty = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.075 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxCCCLostHits = cms.int32( 1 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetMixedStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 3 ),
  seedPairPenalty = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.1 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 999 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.4 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.HLTPSetPixelLessStepTrajectoryFilter = cms.PSet( 
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  minimumNumberOfHits = cms.int32( 4 ),
  seedPairPenalty = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  minPt = cms.double( 0.1 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHits = cms.int32( 0 ),
  maxConsecLostHits = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedExtension = cms.int32( 0 ),
  strictSeedExtension = cms.bool( False ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsForLoopers = cms.int32( 13 )
)
fragment.streams = cms.PSet( 
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
fragment.datasets = cms.PSet( 
  AlCaLumiPixels = cms.vstring( 'AlCa_LumiPixels_Random_v2',
    'AlCa_LumiPixels_ZeroBias_v3' ),
  AlCaPhiSym = cms.vstring( 'AlCa_EcalPhiSymForHI_v3' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration_v3' ),
  EventDisplay = cms.vstring( 'HLT_HIPuAK4CaloJet150_Eta5p1_v3',
    'HLT_HISinglePhoton60_Eta3p1_v3' ),
  HIEWQExo = cms.vstring( 'HLT_HIL1DoubleMu10_v2',
    'HLT_HIL2Mu15_2HF0_v2',
    'HLT_HIL2Mu15_2HF_v2',
    'HLT_HIL2Mu15_v3',
    'HLT_HIL2Mu20_2HF0_v2',
    'HLT_HIL2Mu20_2HF_v2',
    'HLT_HIL2Mu20_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v3',
    'HLT_HIL3Mu15_2HF0_v2',
    'HLT_HIL3Mu15_2HF_v2',
    'HLT_HIL3Mu15_v2',
    'HLT_HIL3Mu20_2HF0_v2',
    'HLT_HIL3Mu20_2HF_v2',
    'HLT_HIL3Mu20_v2' ),
  HIExpressPhysics = cms.vstring( 'HLT_HICentralityVeto_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v3',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v3',
    'HLT_HIFullTrack34_v3',
    'HLT_HIL1DoubleMu10_v2',
    'HLT_HIL1MinimumBiasHF1AND_v2',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v2',
    'HLT_HIL2Mu20_2HF_v2',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v2',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v3',
    'HLT_HIRandom_v2',
    'HLT_HISinglePhoton60_Eta1p5_v3',
    'HLT_HISinglePhoton60_Eta3p1_v3',
    'HLT_HIUCC020_v3',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v2',
    'HLT_HIZeroBias_v2' ),
  HIFlowCorr = cms.vstring( 'HLT_HIQ2Bottom005_Centrality1030_v3',
    'HLT_HIQ2Bottom005_Centrality3050_v3',
    'HLT_HIQ2Bottom005_Centrality5070_v3',
    'HLT_HIQ2Top005_Centrality1030_v3',
    'HLT_HIQ2Top005_Centrality3050_v3',
    'HLT_HIQ2Top005_Centrality5070_v3',
    'HLT_HIUCC020_v3',
    'HLT_HIUCC100_v3' ),
  HIForward = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v2',
    'HLT_HIL1CastorMediumJetAK4CaloJet20_v3',
    'HLT_HIL1CastorMediumJet_v2',
    'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v2',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleEG2NotHF2_v2',
    'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleEG2NotZDCAND_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHF1_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHF2_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v3',
    'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v2',
    'HLT_HIUPCL1EG2NotZDCAND_v2',
    'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1MuOpenNotZDCAND_v2',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v3',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v2',
    'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotMinimumBiasHF2_AND_v2',
    'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotZdcOR_BptxAND_v2',
    'HLT_HIUPCL1SingleEG2NotHF2_v2',
    'HLT_HIUPCL1SingleEG5NotHF2_v2',
    'HLT_HIUPCL1SingleMuOpenNotHF2_v2',
    'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1ZdcOR_BptxAND_v2',
    'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1ZdcXOR_BptxAND_v2',
    'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v2' ),
  HIHardProbes = cms.vstring( 'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt50_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt70_v3',
    'HLT_HIFullTrack12_L1Centrality010_v3',
    'HLT_HIFullTrack12_L1Centrality30100_v3',
    'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v3',
    'HLT_HIFullTrack18_L1Centrality010_v3',
    'HLT_HIFullTrack18_L1Centrality30100_v3',
    'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v3',
    'HLT_HIFullTrack24_L1Centrality30100_v3',
    'HLT_HIFullTrack24_v3',
    'HLT_HIFullTrack34_L1Centrality30100_v3',
    'HLT_HIFullTrack34_v3',
    'HLT_HIFullTrack45_L1Centrality30100_v3',
    'HLT_HIFullTrack45_v3',
    'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v3',
    'HLT_HIPuAK4CaloDJet60_Eta2p1_v3',
    'HLT_HIPuAK4CaloDJet80_Eta2p1_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v3',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v3',
    'HLT_HIPuAK4CaloJet110_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet150_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v3',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v3',
    'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton10_Eta1p5_v3',
    'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton10_Eta3p1_v3',
    'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton15_Eta1p5_v3',
    'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton15_Eta3p1_v3',
    'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton20_Eta1p5_v3',
    'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton20_Eta3p1_v3',
    'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton30_Eta1p5_v3',
    'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v3' ),
  HIHardProbesPeripheral = cms.vstring( 'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v3',
    'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v3' ),
  HIMinimumBias1 = cms.vstring( 'HLT_HICentralityVeto_v2',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v2',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v2',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v2',
    'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1MinimumBiasHF1OR_v2',
    'HLT_HIL1MinimumBiasHF2AND_v2',
    'HLT_HIL1MinimumBiasHF2OR_v2',
    'HLT_HIL1Tech5_BPTX_PlusOnly_v2',
    'HLT_HIL1Tech6_BPTX_MinusOnly_v2',
    'HLT_HIL1Tech7_NoBPTX_v2',
    'HLT_HIPhysics_v2',
    'HLT_HIRandom_v2',
    'HLT_HIZeroBiasPixel_SingleTrack_v2',
    'HLT_HIZeroBias_v2' ),
  HIMinimumBias2 = cms.vstring( 'HLT_HIL1MinimumBiasHF1AND_v2' ),
  HIOniaCentral30L2L3 = cms.vstring( 'HLT_HIL1DoubleMu0_Cent30_v2',
    'HLT_HIL2DoubleMu0_Cent30_NHitQ_v2',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v2',
    'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v2',
    'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v2',
    'HLT_HIL3DoubleMu0_Cent30_v2' ),
  HIOniaL1DoubleMu0 = cms.vstring( 'HLT_HIL1DoubleMu0_2HF0_v2',
    'HLT_HIL1DoubleMu0_2HF_v2',
    'HLT_HIL1DoubleMu0_v2',
    'HLT_HIL2DoubleMu0_NHitQ_2HF0_v2',
    'HLT_HIL2DoubleMu0_NHitQ_2HF_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v3',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v2',
    'HLT_HIL3DoubleMu0_OS_m7to14_v2' ),
  HIOniaPeripheral30100 = cms.vstring( 'HLT_HIL1DoubleMu0_2HF0_Cent30100_v2',
    'HLT_HIL1DoubleMu0_2HF_Cent30100_v2',
    'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v2',
    'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v2' ),
  HIOniaTnP = cms.vstring( 'HLT_HIL2Mu3_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu3_NHitQ10_2HF_v2',
    'HLT_HIL2Mu5_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu5_NHitQ10_2HF_v2',
    'HLT_HIL2Mu7_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu7_NHitQ10_2HF_v2',
    'HLT_HIL3Mu3_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu3_NHitQ15_2HF_v2',
    'HLT_HIL3Mu5_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu5_NHitQ15_2HF_v2',
    'HLT_HIL3Mu7_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu7_NHitQ15_2HF_v2' ),
  HIPhoton40AndZ = cms.vstring( 'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v3',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v3',
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v3',
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v3',
    'HLT_HISinglePhoton40_Eta1p5_v3',
    'HLT_HISinglePhoton40_Eta2p1_v3',
    'HLT_HISinglePhoton40_Eta3p1_v3',
    'HLT_HISinglePhoton50_Eta1p5_v3',
    'HLT_HISinglePhoton50_Eta3p1_v3',
    'HLT_HISinglePhoton60_Eta1p5_v3',
    'HLT_HISinglePhoton60_Eta3p1_v3' ),
  L1Accept = cms.vstring( 'DST_Physics_v2' ),
  OnlineMonitor = cms.vstring( 'HLT_HICastorMediumJetPixel_SingleTrack_v2',
    'HLT_HICentralityVeto_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt20_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt30_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v2',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt40_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt50_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt60_v3',
    'HLT_HIDmesonHITrackingGlobal_Dpt70_v3',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v3',
    'HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v3',
    'HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v3',
    'HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v3',
    'HLT_HIFullTrack12_L1Centrality010_v3',
    'HLT_HIFullTrack12_L1Centrality30100_v3',
    'HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v3',
    'HLT_HIFullTrack18_L1Centrality010_v3',
    'HLT_HIFullTrack18_L1Centrality30100_v3',
    'HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v3',
    'HLT_HIFullTrack24_L1Centrality30100_v3',
    'HLT_HIFullTrack24_v3',
    'HLT_HIFullTrack34_L1Centrality30100_v3',
    'HLT_HIFullTrack34_v3',
    'HLT_HIFullTrack45_L1Centrality30100_v3',
    'HLT_HIFullTrack45_v3',
    'HLT_HIL1CastorMediumJetAK4CaloJet20_v3',
    'HLT_HIL1CastorMediumJet_v2',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v2',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v2',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v2',
    'HLT_HIL1DoubleMu0_2HF0_Cent30100_v2',
    'HLT_HIL1DoubleMu0_2HF0_v2',
    'HLT_HIL1DoubleMu0_2HF_Cent30100_v2',
    'HLT_HIL1DoubleMu0_2HF_v2',
    'HLT_HIL1DoubleMu0_Cent30_v2',
    'HLT_HIL1DoubleMu0_v2',
    'HLT_HIL1DoubleMu10_v2',
    'HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v2',
    'HLT_HIL1MinimumBiasHF1AND_v2',
    'HLT_HIL1MinimumBiasHF1OR_v2',
    'HLT_HIL1MinimumBiasHF2AND_v2',
    'HLT_HIL1MinimumBiasHF2OR_v2',
    'HLT_HIL1Tech5_BPTX_PlusOnly_v2',
    'HLT_HIL1Tech6_BPTX_MinusOnly_v2',
    'HLT_HIL1Tech7_NoBPTX_v2',
    'HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v2',
    'HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v2',
    'HLT_HIL2DoubleMu0_Cent30_NHitQ_v2',
    'HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v2',
    'HLT_HIL2DoubleMu0_NHitQ_2HF0_v2',
    'HLT_HIL2DoubleMu0_NHitQ_2HF_v2',
    'HLT_HIL2DoubleMu0_NHitQ_v3',
    'HLT_HIL2Mu15_2HF0_v2',
    'HLT_HIL2Mu15_2HF_v2',
    'HLT_HIL2Mu15_v3',
    'HLT_HIL2Mu20_2HF0_v2',
    'HLT_HIL2Mu20_2HF_v2',
    'HLT_HIL2Mu20_v2',
    'HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v3',
    'HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v3',
    'HLT_HIL2Mu3_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu3_NHitQ10_2HF_v2',
    'HLT_HIL2Mu5_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu5_NHitQ10_2HF_v2',
    'HLT_HIL2Mu7_NHitQ10_2HF0_v2',
    'HLT_HIL2Mu7_NHitQ10_2HF_v2',
    'HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v2',
    'HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v2',
    'HLT_HIL3DoubleMu0_Cent30_v2',
    'HLT_HIL3DoubleMu0_OS_m2p5to4p5_v2',
    'HLT_HIL3DoubleMu0_OS_m7to14_v2',
    'HLT_HIL3Mu15_2HF0_v2',
    'HLT_HIL3Mu15_2HF_v2',
    'HLT_HIL3Mu15_v2',
    'HLT_HIL3Mu20_2HF0_v2',
    'HLT_HIL3Mu20_2HF_v2',
    'HLT_HIL3Mu20_v2',
    'HLT_HIL3Mu3_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu3_NHitQ15_2HF_v2',
    'HLT_HIL3Mu5_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu5_NHitQ15_2HF_v2',
    'HLT_HIL3Mu7_NHitQ15_2HF0_v2',
    'HLT_HIL3Mu7_NHitQ15_2HF_v2',
    'HLT_HIPhysics_v2',
    'HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v3',
    'HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v3',
    'HLT_HIPuAK4CaloDJet60_Eta2p1_v3',
    'HLT_HIPuAK4CaloDJet80_Eta2p1_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet100_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v3',
    'HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v3',
    'HLT_HIPuAK4CaloJet110_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet120_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet150_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet40_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet60_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v3',
    'HLT_HIPuAK4CaloJet80_Eta5p1_v3',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v3',
    'HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v3',
    'HLT_HIQ2Bottom005_Centrality1030_v3',
    'HLT_HIQ2Bottom005_Centrality3050_v3',
    'HLT_HIQ2Bottom005_Centrality5070_v3',
    'HLT_HIQ2Top005_Centrality1030_v3',
    'HLT_HIQ2Top005_Centrality3050_v3',
    'HLT_HIQ2Top005_Centrality5070_v3',
    'HLT_HIRandom_v2',
    'HLT_HISinglePhoton10_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton10_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton10_Eta1p5_v3',
    'HLT_HISinglePhoton10_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton10_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton10_Eta3p1_v3',
    'HLT_HISinglePhoton15_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton15_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton15_Eta1p5_v3',
    'HLT_HISinglePhoton15_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton15_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton15_Eta3p1_v3',
    'HLT_HISinglePhoton20_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton20_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton20_Eta1p5_v3',
    'HLT_HISinglePhoton20_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton20_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton20_Eta3p1_v3',
    'HLT_HISinglePhoton30_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton30_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton30_Eta1p5_v3',
    'HLT_HISinglePhoton30_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton30_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton30_Eta3p1_v3',
    'HLT_HISinglePhoton40_Eta1p5_Cent30_100_v3',
    'HLT_HISinglePhoton40_Eta1p5_Cent50_100_v3',
    'HLT_HISinglePhoton40_Eta1p5_v3',
    'HLT_HISinglePhoton40_Eta2p1_v3',
    'HLT_HISinglePhoton40_Eta3p1_Cent30_100_v3',
    'HLT_HISinglePhoton40_Eta3p1_Cent50_100_v3',
    'HLT_HISinglePhoton40_Eta3p1_v3',
    'HLT_HISinglePhoton50_Eta1p5_v3',
    'HLT_HISinglePhoton50_Eta3p1_v3',
    'HLT_HISinglePhoton60_Eta1p5_v3',
    'HLT_HISinglePhoton60_Eta3p1_v3',
    'HLT_HIUCC020_v3',
    'HLT_HIUCC100_v3',
    'HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v2',
    'HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleEG2NotHF2_v2',
    'HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleEG2NotZDCAND_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHF1_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHF2_v2',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3',
    'HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v3',
    'HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1DoubleMuOpenNotZDCAND_v2',
    'HLT_HIUPCL1EG2NotZDCAND_v2',
    'HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1MuOpenNotZDCAND_v2',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3',
    'HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v3',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v2',
    'HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotMinimumBiasHF2_AND_v2',
    'HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1NotZdcOR_BptxAND_v2',
    'HLT_HIUPCL1SingleEG2NotHF2_v2',
    'HLT_HIUPCL1SingleEG5NotHF2_v2',
    'HLT_HIUPCL1SingleMuOpenNotHF2_v2',
    'HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1ZdcOR_BptxAND_v2',
    'HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v2',
    'HLT_HIUPCL1ZdcXOR_BptxAND_v2',
    'HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v2',
    'HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v2',
    'HLT_HIZeroBiasPixel_SingleTrack_v2',
    'HLT_HIZeroBias_v2' ),
  RPCMonitor = cms.vstring( 'AlCa_RPCMuonNoHitsForHI_v2',
    'AlCa_RPCMuonNoTriggersForHI_v2',
    'AlCa_RPCMuonNormalisationForHI_v2' ),
  TestEnablesEcalHcal = cms.vstring( 'HLT_EcalCalibration_v3',
    'HLT_HcalCalibration_v2' ),
  TestEnablesEcalHcalDQM = cms.vstring( 'HLT_EcalCalibration_v3',
    'HLT_HcalCalibration_v2' )
)

fragment.CSCChannelMapperESSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "CSCChannelMapperRecord" ),
    firstValid = cms.vuint32( 1 )
)
fragment.CSCINdexerESSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "CSCIndexerRecord" ),
    firstValid = cms.vuint32( 1 )
)
fragment.GlobalParametersRcdSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "L1TGlobalParametersRcd" ),
    firstValid = cms.vuint32( 1 )
)
fragment.StableParametersRcdSource = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "L1TGlobalStableParametersRcd" ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "JetTagComputerRecord" ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    iovIsRunNotTime = cms.bool( True ),
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    firstValid = cms.vuint32( 1 )
)

fragment.hltESPPixelLessStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelLessStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPTobTecStepFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 30.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPTobTecStepRKFitter" ),
  MinNumberOfHits = cms.int32( 7 ),
  Smoother = cms.string( "hltESPTobTecStepRKSmoother" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  ComponentName = cms.string( "hltESPTobTecStepFitterSmoother" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPTobTecStepFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 30.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPTobTecStepRKFitterForLoopers" ),
  MinNumberOfHits = cms.int32( 7 ),
  Smoother = cms.string( "hltESPTobTecStepRKSmootherForLoopers" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  ComponentName = cms.string( "hltESPTobTecStepFitterSmootherForLoopers" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPLowPtStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPTobTecStepRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  ComponentName = cms.string( "hltESPTobTecStepRKSmoother" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPTobTecStepRKTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  ComponentName = cms.string( "hltESPTobTecStepRKSmootherForLoopers" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPTobTecStepRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 7 ),
  ComponentName = cms.string( "hltESPTobTecStepRKFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPTobTecStepRKTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 7 ),
  ComponentName = cms.string( "hltESPTobTecStepRKFitterForLoopers" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPTobTecStepFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepFlexibleKFFittingSmoother" ),
  appendToDataLabel = cms.string( "" ),
  standardFitter = cms.string( "hltESPTobTecStepFitterSmoother" ),
  looperFitter = cms.string( "hltESPTobTecStepFitterSmootherForLoopers" )
)
fragment.hltESPTobTecStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTobTecStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.09 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPChi2ChargeTightMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 0.5 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeTightMeasurementEstimator16" ),
  MaxChi2 = cms.double( 16.0 )
)
fragment.hltESPInitialStepChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 0.5 ),
  clusterChargeCut = cms.PSet(  value = cms.double( 800.0 ) ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  MaxChi2 = cms.double( 30.0 )
)
fragment.hltESPTobTecStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "hltESPTobTecStepClusterShapeHitFilter" ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par" )
)
fragment.hltESPPixelLessStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "hltESPPixelLessStepClusterShapeHitFilter" ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par" )
)
fragment.hltESPMixedStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "hltESPMixedStepClusterShapeHitFilter" ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par" )
)
fragment.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" )
)
fragment.CSCChannelMapperESProducer = cms.ESProducer( "CSCChannelMapperESProducer",
  AlgoName = cms.string( "CSCChannelMapperPostls1" )
)
fragment.CSCIndexerESProducer = cms.ESProducer( "CSCIndexerESProducer",
  AlgoName = cms.string( "CSCIndexerPostls1" )
)
fragment.CSCObjectMapESProducer = cms.ESProducer( "CSCObjectMapESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
fragment.CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  appendToDataLabel = cms.string( "" ),
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
fragment.CaloTowerTopologyEP = cms.ESProducer( "CaloTowerTopologyEP",
  appendToDataLabel = cms.string( "" )
)
fragment.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  ComponentName = cms.string( "ClusterShapeHitFilter" ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  PixelShapeFile = cms.string( "RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape.par" )
)
fragment.DTObjectMapESProducer = cms.ESProducer( "DTObjectMapESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.MaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.OppositeMaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.OppositePropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( 0.1 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.PropagatorWithMaterialForLoopers = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForLoopers" ),
  Mass = cms.double( 0.1396 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 4.0 ),
  useRungeKutta = cms.bool( False )
)
fragment.PropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStep" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( 0.1 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
fragment.SimpleSecondaryVertex3TrkComputer = cms.ESProducer( "SimpleSecondaryVertexESProducer",
  minTracks = cms.uint32( 3 ),
  minVertices = cms.uint32( 1 ),
  use3d = cms.bool( True ),
  unBoost = cms.bool( False ),
  useSignificance = cms.bool( True )
)
fragment.StableParameters = cms.ESProducer( "StableParametersTrivialProducer",
  NumberL1JetCounts = cms.uint32( 12 ),
  NumberL1NoIsoEG = cms.uint32( 4 ),
  NumberL1CenJet = cms.uint32( 4 ),
  NumberL1Tau = cms.uint32( 8 ),
  NumberConditionChips = cms.uint32( 1 ),
  NumberL1EGamma = cms.uint32( 12 ),
  TotalBxInEvent = cms.int32( 5 ),
  NumberL1Mu = cms.uint32( 4 ),
  PinsOnConditionChip = cms.uint32( 512 ),
  WordLength = cms.int32( 64 ),
  PinsOnChip = cms.uint32( 512 ),
  OrderOfChip = cms.vint32( 1 ),
  IfMuEtaNumberBits = cms.uint32( 6 ),
  OrderConditionChip = cms.vint32( 1 ),
  appendToDataLabel = cms.string( "" ),
  NumberL1TauJet = cms.uint32( 4 ),
  NumberL1Jet = cms.uint32( 12 ),
  NumberPhysTriggers = cms.uint32( 512 ),
  NumberL1Muon = cms.uint32( 12 ),
  UnitLength = cms.int32( 8 ),
  NumberL1IsoEG = cms.uint32( 4 ),
  NumberTechnicalTriggers = cms.uint32( 64 ),
  NumberL1ForJet = cms.uint32( 4 ),
  IfCaloEtaNumberBits = cms.uint32( 4 ),
  NumberPsbBoards = cms.int32( 7 ),
  NumberChips = cms.uint32( 5 ),
  NumberPhysTriggersExtended = cms.uint32( 64 )
)
fragment.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
fragment.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
fragment.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
fragment.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" ),
  SimpleMagneticField = cms.string( "" )
)
fragment.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  includeBadChambers = cms.bool( False )
)
fragment.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
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
fragment.hcalDDDRecConstants = cms.ESProducer( "HcalDDDRecConstantsESModule",
  appendToDataLabel = cms.string( "" )
)
fragment.hcalDDDSimConstants = cms.ESProducer( "HcalDDDSimConstantsESModule",
  appendToDataLabel = cms.string( "" )
)
fragment.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
fragment.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
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
fragment.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  recordLabel = cms.string( "HLT" ),
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
fragment.hltCombinedSecondaryVertexV2 = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  recordLabel = cms.string( "HLT" ),
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
fragment.hltDisplacedDijethltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
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
fragment.hltDisplacedDijethltESPTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
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
fragment.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" )
)
fragment.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  PropagationDirection = cms.string( "oppositeToMomentum" )
)
fragment.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.hltESPChi2ChargeLooseMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 0.5 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeLooseMeasurementEstimator16" ),
  MaxChi2 = cms.double( 16.0 )
)
fragment.hltESPChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 0.5 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTiny" ) ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  MaxChi2 = cms.double( 16.0 )
)
fragment.hltESPChi2ChargeMeasurementEstimator2000 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator2000" ),
  MaxChi2 = cms.double( 2000.0 )
)
fragment.hltESPChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  MaxChi2 = cms.double( 30.0 )
)
fragment.hltESPChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 0.5 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTiny" ) ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  MaxChi2 = cms.double( 9.0 )
)
fragment.hltESPChi2ChargeMeasurementEstimator9ForHI = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutForHI" ) ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  nSigma = cms.double( 3.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9ForHI" ),
  MaxChi2 = cms.double( 9.0 )
)
fragment.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 16.0 )
)
fragment.hltESPChi2MeasurementEstimator30 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator30" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 30.0 )
)
fragment.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 9.0 )
)
fragment.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" )
)
fragment.hltESPDetachedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPDisplacedDijethltPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
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
fragment.hltESPDisplacedDijethltPromptTrackCountingESProducerLong = cms.ESProducer( "PromptTrackCountingESProducer",
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
fragment.hltESPDisplacedDijethltTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
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
fragment.hltESPDisplacedDijethltTrackCounting2D2ndLong = cms.ESProducer( "TrackCountingESProducer",
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
fragment.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPElectronMaterialEffects = cms.ESProducer( "GsfMaterialEffectsESProducer",
  BetheHeitlerParametrization = cms.string( "BetheHeitler_cdfmom_nC6_O5.par" ),
  EnergyLossUpdator = cms.string( "GsfBetheHeitlerUpdator" ),
  ComponentName = cms.string( "hltESPElectronMaterialEffects" ),
  MultipleScatteringUpdator = cms.string( "MultipleScatteringUpdator" ),
  Mass = cms.double( 5.11E-4 ),
  BetheHeitlerCorrection = cms.int32( 2 )
)
fragment.hltESPFastSteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
fragment.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
fragment.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFlexibleKFFittingSmoother" ),
  appendToDataLabel = cms.string( "" ),
  standardFitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  looperFitter = cms.string( "hltESPKFFittingSmootherForLoopers" )
)
fragment.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
fragment.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" )
)
fragment.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPGsfTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPGsfTrajectorySmoother" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  ComponentName = cms.string( "hltESPGsfElectronFittingSmoother" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" )
)
fragment.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ErrorRescaling = cms.double( 100.0 ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" )
)
fragment.hltESPInitialStepChi2MeasurementEstimator36 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2MeasurementEstimator36" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 36.0 )
)
fragment.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( -1.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  MinNumberOfHits = cms.int32( 5 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPKFFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherForLoopers" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  EstimateCut = cms.double( 20.0 ),
  appendToDataLabel = cms.string( "" ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  Fitter = cms.string( "hltESPRKTrajectoryFitter" ),
  MinNumberOfHits = cms.int32( 3 ),
  Smoother = cms.string( "hltESPRKTrajectorySmoother" ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  MaxFractionOutliers = cms.double( 0.3 ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  RejectTracks = cms.bool( True )
)
fragment.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPKFTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
fragment.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPKFTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
fragment.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" )
)
fragment.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  DistanceMeasure = cms.string( "KullbackLeibler" )
)
fragment.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
fragment.hltESPMixedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedStepTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
)
fragment.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
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
fragment.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  DoLorentz = cms.bool( True ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False )
)
fragment.hltESPPixelPairStepChi2MeasurementEstimator25 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  appendToDataLabel = cms.string( "" ),
  MinimalTolerance = cms.double( 10.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  ComponentName = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" ),
  nSigma = cms.double( 3.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MaxChi2 = cms.double( 25.0 )
)
fragment.hltESPPixelPairTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelPairTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKTrajectoryFitter" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
fragment.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPRKTrajectorySmoother" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" )
)
fragment.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  ptMin = cms.double( -1.0 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
fragment.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagator" )
)
fragment.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAny" )
)
fragment.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" )
)
fragment.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
fragment.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
fragment.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
fragment.hltESPStripCPEfromTrackAngle = cms.ESProducer( "StripCPEESProducer",
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
fragment.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" )
)
fragment.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" )
)
fragment.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" )
)
fragment.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  StripCPE = cms.string( "Fake" ),
  Matcher = cms.string( "StandardMatcher" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" )
)
fragment.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
fragment.hltESPTrajectoryCleanerBySharedSeeds = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedSeeds" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedSeeds" ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  appendToDataLabel = cms.string( "" ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  appendToDataLabel = cms.string( "" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" )
)
fragment.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  includeBadChambers = cms.bool( False )
)
fragment.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  includeBadChambers = cms.bool( False )
)
fragment.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  SimpleMagneticField = cms.string( "ParabolicMf" )
)
fragment.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  includeBadChambers = cms.bool( False )
)
fragment.siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  )
)
fragment.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
fragment.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer( "SiStripBackPlaneCorrectionDepESProducer",
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
fragment.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
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

fragment.hltGetConditions = cms.EDAnalyzer( "EventSetupRecordDataGetter",
    toGet = cms.VPSet( 
    ),
    verbose = cms.untracked.bool( False )
)
fragment.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
fragment.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
fragment.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
fragment.hltGtStage2Digis = cms.EDProducer( "L1TRawToDigi",
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    CTP7 = cms.untracked.bool( False ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    Setup = cms.string( "stage2::GTSetup" ),
    MinFeds = cms.uint32( 0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    MTF7 = cms.untracked.bool( False ),
    FWId = cms.uint32( 0 ),
    debug = cms.untracked.bool( False ),
    FedIds = cms.vint32( 1404 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    FWOverride = cms.bool( False )
)
fragment.hltCaloStage2Digis = cms.EDProducer( "L1TRawToDigi",
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    CTP7 = cms.untracked.bool( False ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    Setup = cms.string( "stage2::CaloSetup" ),
    MinFeds = cms.uint32( 0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    MTF7 = cms.untracked.bool( False ),
    FWId = cms.uint32( 0 ),
    debug = cms.untracked.bool( False ),
    FedIds = cms.vint32( 1360, 1366 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    FWOverride = cms.bool( False )
)
fragment.hltGmtStage2Digis = cms.EDProducer( "L1TRawToDigi",
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    CTP7 = cms.untracked.bool( False ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    Setup = cms.string( "stage2::GMTSetup" ),
    MinFeds = cms.uint32( 0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    MTF7 = cms.untracked.bool( False ),
    FWId = cms.uint32( 0 ),
    debug = cms.untracked.bool( False ),
    FedIds = cms.vint32( 1402 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    FWOverride = cms.bool( False )
)
fragment.hltGtStage2ObjectMap = cms.EDProducer( "L1TGlobalProducer",
    L1DataBxInEvent = cms.int32( 5 ),
    JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    AlgorithmTriggersUnmasked = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    Verbosity = cms.untracked.int32( 0 ),
    EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    ProduceL1GtDaqRecord = cms.bool( True ),
    PrescaleSet = cms.uint32( 1 ),
    EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    TriggerMenuLuminosity = cms.string( "startup" ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    PrescaleCSVFile = cms.string( "prescale_L1TGlobal.csv" ),
    TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    BstLengthBytes = cms.int32( -1 ),
    MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltScalersRawToDigi = cms.EDProducer( "ScalersRawToDigi",
    scalersInputTag = cms.InputTag( "rawDataCollector" )
)
fragment.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    maxZ = cms.double( 40.0 ),
    src = cms.InputTag( "hltScalersRawToDigi" ),
    gtEvmLabel = cms.InputTag( "" ),
    changeToCMSCoordinates = cms.bool( False ),
    setSigmaZ = cms.double( 0.0 ),
    maxRadius = cms.double( 2.0 )
)
fragment.hltPreDSTPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
fragment.hltL1sV0MinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet40Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEcalDigis = cms.EDProducer( "EcalRawToDigi",
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
fragment.hltEcalUncalibRecHit50nsMultiFit = cms.EDProducer( "EcalUncalibRecHitProducer",
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
fragment.hltEcalDetIdToBeRecovered = cms.EDProducer( "EcalDetIdToBeRecoveredProducer",
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
fragment.hltEcalRecHit50nsMultiFit = cms.EDProducer( "EcalRecHitProducer",
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
fragment.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
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
fragment.hltHbherecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0 ),
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
    applyTimeSlew = cms.bool( True ),
    applyTimeConstraint = cms.bool( True ),
    applyPulseJitter = cms.bool( False ),
    pulseShapeParameters = cms.PSet(  ),
    timingshapedcutsParameters = cms.PSet( 
      ignorelowest = cms.bool( True ),
      win_offset = cms.double( 0.0 ),
      ignorehighest = cms.bool( False ),
      win_gain = cms.double( 1.0 ),
      tfilterEnvelope = cms.vdouble( 4.0, 12.04, 13.0, 10.56, 23.5, 8.82, 37.0, 7.38, 56.0, 6.3, 81.0, 5.64, 114.5, 5.44, 175.5, 5.38, 350.5, 5.14 )
    ),
    ts4Min = cms.double( 5.0 ),
    ts3chi2 = cms.double( 5.0 ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
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
fragment.hltHfrecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0 ),
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
    applyTimeSlew = cms.bool( True ),
    applyTimeConstraint = cms.bool( True ),
    applyPulseJitter = cms.bool( False ),
    pulseShapeParameters = cms.PSet(  ),
    timingshapedcutsParameters = cms.PSet(  ),
    ts4Min = cms.double( 5.0 ),
    ts3chi2 = cms.double( 5.0 ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
    meanTime = cms.double( -2.5 ),
    flagParameters = cms.PSet(  ),
    fitTimes = cms.int32( 1 ),
    timeMax = cms.double( 10.0 ),
    timeSigma = cms.double( 5.0 ),
    pedSigma = cms.double( 0.5 ),
    meanPed = cms.double( 0.0 ),
    hscpParameters = cms.PSet(  )
)
fragment.hltHorecoMethod0 = cms.EDProducer( "HcalHitReconstructor",
    pedestalUpperLimit = cms.double( 2.7 ),
    timeSlewPars = cms.vdouble( 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0, 12.2999, -2.19142, 0.0 ),
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
    applyTimeSlew = cms.bool( True ),
    applyTimeConstraint = cms.bool( True ),
    applyPulseJitter = cms.bool( False ),
    pulseShapeParameters = cms.PSet(  ),
    timingshapedcutsParameters = cms.PSet(  ),
    ts4Min = cms.double( 5.0 ),
    ts3chi2 = cms.double( 5.0 ),
    noise = cms.double( 1.0 ),
    applyPedConstraint = cms.bool( True ),
    applyUnconstrainedFit = cms.bool( False ),
    ts4Max = cms.double( 500.0 ),
    meanTime = cms.double( -2.5 ),
    flagParameters = cms.PSet(  ),
    fitTimes = cms.int32( 1 ),
    timeMax = cms.double( 10.0 ),
    timeSigma = cms.double( 5.0 ),
    pedSigma = cms.double( 0.5 ),
    meanPed = cms.double( 0.0 ),
    hscpParameters = cms.PSet(  )
)
fragment.hltTowerMakerHcalMethod050nsMultiFitForAll = cms.EDProducer( "CaloTowersCreator",
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
    HcalPhase = cms.int32( 0 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    UseRejectedHitsOnly = cms.bool( False ),
    EBThreshold = cms.double( 0.07 ),
    HEDGrid = cms.vdouble(  ),
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
fragment.hltPuAK4CaloJets50nsMultiFit = cms.EDProducer( "FastjetJetProducer",
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
fragment.hltPuAK4CaloJetsIDPassed50nsMultiFit = cms.EDProducer( "HLTCaloJetIDProducer",
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
fragment.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 = cms.EDProducer( "FixedGridRhoProducerFastjet",
    gridSpacing = cms.double( 0.55 ),
    maxRapidity = cms.double( 5.0 ),
    pfCandidatesTag = cms.InputTag( "hltTowerMakerHcalMethod050nsMultiFitForAll" )
)
fragment.hltAK4CaloRelativeCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    algorithm = cms.string( "AK4CaloHLT" ),
    level = cms.string( "L2Relative" )
)
fragment.hltAK4CaloAbsoluteCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    algorithm = cms.string( "AK4CaloHLT" ),
    level = cms.string( "L3Absolute" )
)
fragment.hltAK4CaloResidualCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    algorithm = cms.string( "AK4CaloHLT" ),
    level = cms.string( "L2L3Residual" )
)
fragment.hltPuAK4CaloCorrector = cms.EDProducer( "ChainedJetCorrectorProducer",
    correctors = cms.VInputTag( 'hltAK4CaloRelativeCorrector','hltAK4CaloAbsoluteCorrector','hltAK4CaloResidualCorrector' )
)
fragment.hltPuAK4CaloJetsCorrected50nsMultiFit = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJets50nsMultiFit" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
fragment.hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsIDPassed50nsMultiFit" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
fragment.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltSiStripRawToDigi = cms.EDProducer( "SiStripRawToDigiModule",
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
fragment.hltSiStripZeroSuppression = cms.EDProducer( "SiStripZeroSuppression",
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
fragment.hltSiStripDigiToZSRaw = cms.EDProducer( "SiStripDigiToRawModule",
    CopyBufferHeader = cms.bool( False ),
    InputDigiLabel = cms.string( "VirginRaw" ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" ),
    UseFedKey = cms.bool( False ),
    RawDataTag = cms.InputTag( "rawDataCollector" ),
    FedReadoutMode = cms.string( "ZERO_SUPPRESSED" ),
    UseWrongDigiType = cms.bool( False )
)
fragment.hltSiStripRawDigiToVirginRaw = cms.EDProducer( "SiStripDigiToRawModule",
    CopyBufferHeader = cms.bool( False ),
    InputDigiLabel = cms.string( "VirginRaw" ),
    InputModuleLabel = cms.string( "hltSiStripZeroSuppression" ),
    UseFedKey = cms.bool( False ),
    RawDataTag = cms.InputTag( "rawDataCollector" ),
    FedReadoutMode = cms.string( "VIRGIN_RAW" ),
    UseWrongDigiType = cms.bool( False )
)
fragment.virginRawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripRawDigiToVirginRaw' )
)
fragment.rawDataRepacker = cms.EDProducer( "RawDataCollectorByLabel",
    verbose = cms.untracked.int32( 0 ),
    RawCollectionList = cms.VInputTag( 'hltSiStripDigiToZSRaw','source','rawDataCollector' )
)
fragment.hltL1sSingleS1Jet28BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet60Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltL1sSingleJet44BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet44_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet80Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltL1sSingleS1Jet56BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet56_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet100Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet110Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet110Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 110.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet120Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet120Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 120.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltL1sSingleS1Jet64BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet64_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet150Eta5p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet150Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 150.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltL1sSingleS1Jet16Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet40Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet28Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet60Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet44Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet44_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet80Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIPuAK4CaloJet100Eta5p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet16Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet40Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet28Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet28_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet60Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet44Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet44_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIPuAK4CaloJet80Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIPuAK4CaloJet100Eta5p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIPuAK4CaloJet80Jet35Eta1p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet80Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 35.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet80Jet35Eta0p7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet80Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 35.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet100Jet35Eta1p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet100Eta1p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet100Jet35Eta0p7 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet100Eta0p750nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 0.7 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloJet804545Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTriplePuAK4CaloJet45Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 45.0 ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 80.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIPuAK4CaloDJet60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltEta2CaloJetsEta2p1ForJets = cms.EDFilter( "CaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    cut = cms.string( "abs(eta)<2.1" )
)
fragment.hltReduceJetMultEta2p1Forjets = cms.EDFilter( "LargestEtCaloJetSelector",
    maxNumber = cms.uint32( 3 ),
    filter = cms.bool( False ),
    src = cms.InputTag( "hltEta2CaloJetsEta2p1ForJets" )
)
fragment.hltJets4bTaggerCaloJet60Eta2p1Forjets = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
    etMin = cms.double( 60.0 )
)
fragment.hltHIJetsForCoreTracking = cms.EDFilter( "CandPtrSelector",
    src = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    cut = cms.string( "pt > 100 && abs(eta) < 2.4" )
)
fragment.hltHISiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    UseQualityInfo = cms.bool( False ),
    UsePilotBlade = cms.bool( False ),
    UsePhase1 = cms.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    IncludeErrors = cms.bool( False ),
    ErrorList = cms.vint32(  ),
    Regions = cms.PSet(  ),
    Timing = cms.untracked.bool( False ),
    CablingMapLabel = cms.string( "" ),
    UserErrorList = cms.vint32(  )
)
fragment.hltHISiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
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
fragment.hltHISiPixelClustersCache = cms.EDProducer( "SiPixelClusterShapeCacheProducer",
    src = cms.InputTag( "hltHISiPixelClusters" ),
    onDemand = cms.bool( False )
)
fragment.hltHISiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltHISiPixelClusters" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
fragment.hltHIPixelClusterVertices = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 30.0 ),
    zStep = cms.double( 0.1 ),
    minZ = cms.double( -30.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHits" )
)
fragment.hltHIPixelLayerTriplets = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIPixel3ProtoTracks = cms.EDProducer( "PixelTrackProducer",
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
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIPixelMedianVertex = cms.EDProducer( "HIPixelMedianVtxProducer",
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    FitThreshold = cms.int32( 5 ),
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    PtMin = cms.double( 0.075 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
fragment.hltHISelectedProtoTracks = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracks" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    maxD0Significance = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertex" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 )
)
fragment.hltHIPixelAdaptiveVertex = cms.EDProducer( "PrimaryVertexProducer",
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
fragment.hltHIBestAdaptiveVertex = cms.EDFilter( "HIBestVertexSelection",
    maxNumber = cms.uint32( 1 ),
    src = cms.InputTag( "hltHIPixelAdaptiveVertex" )
)
fragment.hltHISelectedVertex = cms.EDProducer( "HIBestVertexProducer",
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" )
)
fragment.hltHISiPixelClustersAfterSplitting = cms.EDProducer( "JetCoreClusterSplitter",
    verbose = cms.bool( False ),
    deltaRmax = cms.double( 0.1 ),
    forceXError = cms.double( 100.0 ),
    vertices = cms.InputTag( "hltHISelectedVertex" ),
    chargePerUnit = cms.double( 2000.0 ),
    forceYError = cms.double( 150.0 ),
    centralMIPCharge = cms.double( 26000.0 ),
    pixelClusters = cms.InputTag( "hltHISiPixelClusters" ),
    ptMin = cms.double( 100.0 ),
    chargeFractionMin = cms.double( 2.0 ),
    cores = cms.InputTag( "hltHIJetsForCoreTracking" ),
    fractionalWidth = cms.double( 0.4 ),
    pixelCPE = cms.string( "hltESPPixelCPEGeneric" )
)
fragment.hltHISiPixelClustersCacheAfterSplitting = cms.EDProducer( "SiPixelClusterShapeCacheProducer",
    src = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    onDemand = cms.bool( False )
)
fragment.hltHISiPixelRecHitsAfterSplitting = cms.EDProducer( "SiPixelRecHitConverter",
    VerboseLevel = cms.untracked.int32( 0 ),
    src = cms.InputTag( "hltHISiPixelClustersAfterSplitting" ),
    CPE = cms.string( "hltESPPixelCPEGeneric" )
)
fragment.hltHIPixelClusterVerticesAfterSplitting = cms.EDProducer( "HIPixelClusterVtxProducer",
    maxZ = cms.double( 30.0 ),
    zStep = cms.double( 0.1 ),
    minZ = cms.double( -30.0 ),
    pixelRecHits = cms.string( "hltHISiPixelRecHitsAfterSplitting" )
)
fragment.hltHIPixelLayerTripletsAfterSplitting = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIPixel3ProtoTracksAfterSplitting = cms.EDProducer( "PixelTrackProducer",
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
        VertexCollection = cms.InputTag( "hltHIPixelClusterVerticesAfterSplitting" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIPixelMedianVertexAfterSplitting = cms.EDProducer( "HIPixelMedianVtxProducer",
    PeakFindThreshold = cms.uint32( 100 ),
    PeakFindMaxZ = cms.double( 30.0 ),
    FitThreshold = cms.int32( 5 ),
    TrackCollection = cms.InputTag( "hltHIPixel3ProtoTracksAfterSplitting" ),
    PtMin = cms.double( 0.075 ),
    PeakFindBinsPerCm = cms.int32( 10 ),
    FitMaxZ = cms.double( 0.1 ),
    FitBinsPerCm = cms.int32( 500 )
)
fragment.hltHISelectedProtoTracksAfterSplitting = cms.EDFilter( "HIProtoTrackSelection",
    src = cms.InputTag( "hltHIPixel3ProtoTracksAfterSplitting" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    maxD0Significance = cms.double( 5.0 ),
    minZCut = cms.double( 0.2 ),
    VertexCollection = cms.InputTag( "hltHIPixelMedianVertexAfterSplitting" ),
    ptMin = cms.double( 0.0 ),
    nSigmaZ = cms.double( 5.0 )
)
fragment.hltHIPixelAdaptiveVertexAfterSplitting = cms.EDProducer( "PrimaryVertexProducer",
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
fragment.hltHIBestAdaptiveVertexAfterSplitting = cms.EDFilter( "HIBestVertexSelection",
    maxNumber = cms.uint32( 1 ),
    src = cms.InputTag( "hltHIPixelAdaptiveVertexAfterSplitting" )
)
fragment.hltHISelectedVertexAfterSplitting = cms.EDProducer( "HIBestVertexProducer",
    adaptiveVertexCollection = cms.InputTag( "hltHIBestAdaptiveVertex" ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    medianVertexCollection = cms.InputTag( "hltHIPixelMedianVertex" )
)
fragment.hltHITrackingSiStripRawToClustersFacilityZeroSuppression = cms.EDProducer( "SiStripClusterizer",
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
fragment.hltHISiStripClustersZeroSuppression = cms.EDProducer( "MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(  ),
    stripClusterProducer = cms.string( "hltHITrackingSiStripRawToClustersFacilityZeroSuppression" ),
    pixelClusterProducer = cms.string( "hltHISiPixelClustersAfterSplitting" ),
    switchOffPixelsIfEmpty = cms.bool( True ),
    inactiveStripDetectorLabels = cms.VInputTag(  ),
    skipClusters = cms.InputTag( "" ),
    measurementTracker = cms.string( "hltESPMeasurementTracker" )
)
fragment.hltHIPixel3PrimTracksForjets = cms.EDProducer( "PixelTrackProducer",
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
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        measurementTrackerName = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
        whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIPixelTrackSeedsForjets = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
fragment.hltHIPrimTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter0TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIter1ClustersRefRemovalForjets = cms.EDProducer( "HITrackClusterRemover",
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
fragment.hltHIIter1MaskedMeasurementTrackerEventForjets = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter1ClustersRefRemovalForjets" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
fragment.hltHIDetachedPixelLayerTripletsForjets = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIDetachedPixelTracksForjets = cms.EDProducer( "PixelTrackProducer",
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
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        measurementTrackerName = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
        whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIDetachedPixelTrackSeedsForjets = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
fragment.hltHIDetachedTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIDetachedGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter1TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIter2ClustersRefRemovalForjets = cms.EDProducer( "HITrackClusterRemover",
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
fragment.hltHIIter2MaskedMeasurementTrackerEventForjets = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter2ClustersRefRemovalForjets" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
fragment.hltHIPixelLayerPairsForjets = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIPixelPairSeedsForjets = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
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
        mode = cms.string( "VerticesFixed" ),
        nSigmaZBeamSpot = cms.double( 3.0 ),
        searchOpt = cms.bool( True ),
        zErrorBeamSpot = cms.double( 15.0 ),
        zErrorVetex = cms.double( 0.1 ),
        maxNRegions = cms.int32( 100 ),
        vertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        measurementTrackerName = cms.InputTag( "hltHISiStripClustersZeroSuppression" ),
        whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
        useMultipleScattering = cms.bool( False )
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
fragment.hltHIPixelPairTrackCandidatesForjets = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIPixelPairGlobalPrimTracksForjets = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter2TrackSelectionForjets = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIterTrackingMergedHighPurityForjets = cms.EDProducer( "TrackListMerger",
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
fragment.hltHIIterTrackingMergedTightForjets = cms.EDProducer( "TrackListMerger",
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
fragment.hltHIFullTrackCandsForDmesonjets = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    particleType = cms.string( "pi+" )
)
fragment.hltHIFullTrackFilterForDmesonjets = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltTktkVtxForDmesonjetsCaloJet60 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonjetsCaloJet60 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIPuAK4CaloDJet80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltJets4bTaggerCaloJet80Eta2p1Forjets = cms.EDFilter( "EtMinCaloJetSelector",
    filter = cms.bool( False ),
    src = cms.InputTag( "hltReduceJetMultEta2p1Forjets" ),
    etMin = cms.double( 80.0 )
)
fragment.hltTktkVtxForDmesonjetsCaloJet80 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonjetsCaloJet80 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIPuAK4CaloBJetCSV60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIVerticesL3 = cms.EDProducer( "PrimaryVertexProducer",
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
fragment.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    tracks = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    useAssigned = cms.bool( False ),
    coneSize = cms.double( 0.4 ),
    pvSrc = cms.InputTag( "" )
)
fragment.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 = cms.EDProducer( "TrackIPProducer",
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
fragment.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 = cms.EDProducer( "SecondaryVertexProducer",
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
fragment.hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltCombinedSecondaryVertex" ),
    tagInfos = cms.VInputTag( 'hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1','hltL3SecondaryVertexTagInfosCaloJet60Eta2p1' )
)
fragment.hltBLifetimeL3FilterCSVCaloJet60Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 )
)
fragment.hltPreHIPuAK4CaloBJetCSV80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    tracks = cms.InputTag( "hltHIIterTrackingMergedTightForjets" ),
    useAssigned = cms.bool( False ),
    coneSize = cms.double( 0.4 ),
    pvSrc = cms.InputTag( "" )
)
fragment.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 = cms.EDProducer( "TrackIPProducer",
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
fragment.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 = cms.EDProducer( "SecondaryVertexProducer",
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
fragment.hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "hltCombinedSecondaryVertex" ),
    tagInfos = cms.VInputTag( 'hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1','hltL3SecondaryVertexTagInfosCaloJet80Eta2p1' )
)
fragment.hltBLifetimeL3FilterCSVCaloJet80Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 )
)
fragment.hltPreHIPuAK4CaloBJetSSV60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "SimpleSecondaryVertex3TrkComputer" ),
    tagInfos = cms.VInputTag( 'hltL3SecondaryVertexTagInfosCaloJet60Eta2p1' )
)
fragment.hltBLifetimeL3FilterSSVCaloJet60Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet60Eta2p1Forjets" ),
    MinTag = cms.double( 0.01 ),
    MaxTag = cms.double( 99999.0 )
)
fragment.hltPreHIPuAK4CaloBJetSSV80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1 = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "SimpleSecondaryVertex3TrkComputer" ),
    tagInfos = cms.VInputTag( 'hltL3SecondaryVertexTagInfosCaloJet80Eta2p1' )
)
fragment.hltBLifetimeL3FilterSSVCaloJet80Eta2p1 = cms.EDFilter( "HLTCaloJetTag",
    saveTags = cms.bool( True ),
    MinJets = cms.int32( 1 ),
    JetTags = cms.InputTag( "hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1" ),
    TriggerType = cms.int32( 86 ),
    Jets = cms.InputTag( "hltJets4bTaggerCaloJet80Eta2p1Forjets" ),
    MinTag = cms.double( 0.01 ),
    MaxTag = cms.double( 99999.0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPixel3PrimTracksForGlobalPt8 = cms.EDProducer( "PixelTrackProducer",
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
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIPixelTrackSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
fragment.hltHIPrimTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter0TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIter1ClustersRefRemovalForGlobalPt8 = cms.EDProducer( "HITrackClusterRemover",
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
fragment.hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8 = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter1ClustersRefRemovalForGlobalPt8" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
fragment.hltHIDetachedPixelLayerTripletsForGlobalPt8 = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIDetachedPixelTracksForGlobalPt8 = cms.EDProducer( "PixelTrackProducer",
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
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltHIDetachedPixelTrackSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
fragment.hltHIDetachedTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIDetachedGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter1TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIter2ClustersRefRemovalForGlobalPt8 = cms.EDProducer( "HITrackClusterRemover",
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
fragment.hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8 = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    clustersToSkip = cms.InputTag( "hltHIIter2ClustersRefRemovalForGlobalPt8" ),
    OnDemand = cms.bool( False ),
    src = cms.InputTag( "hltHISiStripClustersZeroSuppression" )
)
fragment.hltHIPixelLayerPairsForGlobalPt8 = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltHIPixelPairSeedsForGlobalPt8 = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
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
        VertexCollection = cms.InputTag( "hltHISelectedVertexAfterSplitting" ),
        useMultipleScattering = cms.bool( False )
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
fragment.hltHIPixelPairTrackCandidatesForGlobalPt8 = cms.EDProducer( "CkfTrackCandidateMaker",
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
fragment.hltHIPixelPairGlobalPrimTracksForGlobalPt8 = cms.EDProducer( "TrackProducer",
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
fragment.hltHIIter2TrackSelectionForGlobalPt8 = cms.EDProducer( "HIMultiTrackSelector",
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
fragment.hltHIIterTrackingMergedHighPurityForGlobalPt8 = cms.EDProducer( "TrackListMerger",
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
fragment.hltHIIterTrackingMergedTightForGlobalPt8 = cms.EDProducer( "TrackListMerger",
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
fragment.hltHIFullTrackCandsForDmesonGlobalPt8 = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIIterTrackingMergedTightForGlobalPt8" ),
    particleType = cms.string( "pi+" )
)
fragment.hltHIFullTrackFilterForDmesonGlobalPt8 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltTktkVtxForDmesonGlobal8Dpt20 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp20 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltL1sCentralityext30100MinimumumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext30_100_MinimumumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sCentralityext50100MinimumumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext50_100_MinimumumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet16BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet16_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTktkVtxForDmesonGlobal8Dpt30 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp30 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt40 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTktkVtxForDmesonGlobal8Dpt40 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp40 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleS1Jet32BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet32_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt50 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTktkVtxForDmesonGlobal8Dpt50 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp50 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIDmesonHITrackingGlobalDpt60 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTktkVtxForDmesonGlobal8Dpt60 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp60 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltL1sSingleS1Jet52BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleS1Jet52_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt70 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTktkVtxForDmesonGlobal8Dpt70 = cms.EDProducer( "HLTDisplacedtktkVtxProducer",
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
fragment.hltTktkFilterForDmesonGlobal8Dp70 = cms.EDFilter( "HLTDisplacedtktkFilter",
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
fragment.hltPreHIDmesonHITrackingGlobalDpt60Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt60Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sCentralityext010MinimumumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext0_10_MinimumumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton10Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltIslandBasicClusters50nsMultiFitHI = cms.EDProducer( "IslandClusterProducer",
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
    barrelClusterCollection = cms.string( "islandBarrelBasicClustersHI" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClustersHI" ),
    IslandBarrelSeedThr = cms.double( 0.5 )
)
fragment.hltHiIslandSuperClusters50nsMultiFitHI = cms.EDProducer( "HiSuperClusterProducer",
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
fragment.hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
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
fragment.hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI = cms.EDProducer( "HiEgammaSCCorrectionMaker",
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
fragment.hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI = cms.EDProducer( "HiSpikeCleaner",
    originalSuperClusterProducer = cms.InputTag( "hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI" ),
    recHitProducerEndcap = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    TimingCut = cms.untracked.double( 9999999.0 ),
    swissCutThr = cms.untracked.double( 0.95 ),
    recHitProducerBarrel = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    etCut = cms.double( 8.0 ),
    outputColl = cms.string( "" )
)
fragment.hltRecoHIEcalWithCleaningCandidate50nsMultiFit = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scIslandEndcapProducer = cms.InputTag( "hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI" ),
    scHybridBarrelProducer = cms.InputTag( "hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI" ),
    recoEcalCandidateCollection = cms.string( "" )
)
fragment.hltHIPhoton10Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton15Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton20Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton20Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltL1sSingleEG7BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton30Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton30Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 30.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltL1sSingleEG21BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton40Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton40Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton50Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton50Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 50.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltL1sSingleEG30BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG30_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton60Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton60Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltL1sSingleEG3Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG3_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton10Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton15Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton20Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG7Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton30Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG21Centralityext50100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_Centrality_ext50_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton40Eta1p5Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG3Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG3_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton10Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton15Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton20Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG7Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG7_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton30Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG21Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG21_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHISinglePhoton40Eta1p5Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton40Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton40Eta2p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton10Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton10Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 10.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton15Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton15Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton20Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton20Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton30Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton30Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 30.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton40Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton40Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton50Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton50Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 50.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton60Eta3p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPhoton60Eta3p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 60.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 3.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltPreHISinglePhoton10Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton15Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton20Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton30Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton40Eta3p1Cent50100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton10Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton15Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton20Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton30Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHISinglePhoton40Eta3p1Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIDoublePhoton15Eta1p5Mass501000 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 1.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( True ),
    lowerMassCut = cms.double( 50.0 ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    l1EGCand = cms.InputTag( "" ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
fragment.hltPreHIDoublePhoton15Eta1p5Mass501000R9HECut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIEgammaR9ID50nsMultiFit = cms.EDProducer( "EgammaHLTR9IDProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    ecalRechitEE = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' )
)
fragment.hltHIEgammaR9IDDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.6 ),
    thrOverEEE = cms.double( -1.0 ),
    varTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.6 ),
    lessThan = cms.bool( False ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" )
)
fragment.hltHIEgammaHoverE50nsMultiFit = cms.EDProducer( "EgammaHLTBcHcalIsolationProducersRegional",
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
fragment.hltHIEgammaHOverEDoublePhoton15Eta1p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.25 ),
    varTag = cms.InputTag( "hltHIEgammaHoverE50nsMultiFit" ),
    thrOverEEB = cms.double( 0.25 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 1 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta1p550nsMultiFit" )
)
fragment.hltPreHIDoublePhoton15Eta2p1Mass501000R9Cut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoublePhoton15Eta2p150nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltHIDoublePhoton15Eta2p1GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( True ),
    lowerMassCut = cms.double( 50.0 ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    l1EGCand = cms.InputTag( "" ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p150nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
fragment.hltHIEgammaR9IDDoublePhoton15Eta2p150nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.4 ),
    thrOverEEE = cms.double( -1.0 ),
    varTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.4 ),
    lessThan = cms.bool( False ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p150nsMultiFit" )
)
fragment.hltPreHIDoublePhoton15Eta2p5Mass501000R9SigmaHECut = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLT1Photon",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 15.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 81 )
)
fragment.hltHIDoublePhoton15Eta2p5GlobalMass501000Filter = cms.EDFilter( "HLTPMMassFilter",
    saveTags = cms.bool( True ),
    lowerMassCut = cms.double( 50.0 ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    isElectron1 = cms.untracked.bool( False ),
    isElectron2 = cms.untracked.bool( False ),
    l1EGCand = cms.InputTag( "" ),
    upperMassCut = cms.double( 1000.0 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" ),
    reqOppCharge = cms.untracked.bool( False ),
    nZcandcut = cms.int32( 1 )
)
fragment.hltHIEgammaR9IDDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.5 ),
    thrOverEEE = cms.double( -1.0 ),
    varTag = cms.InputTag( "hltHIEgammaR9ID50nsMultiFit" ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.4 ),
    lessThan = cms.bool( False ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" )
)
fragment.hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer = cms.EDProducer( "EgammaHLTClusterShapeProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ecalRechitEB = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEB' ),
    ecalRechitEE = cms.InputTag( 'hltEcalRecHit50nsMultiFit','EcalRecHitsEE' ),
    isIeta = cms.bool( True )
)
fragment.hltHIEgammaSigmaIEtaIEtaDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( 0.045 ),
    thrOverEEE = cms.double( -1.0 ),
    varTag = cms.InputTag( 'hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer','sigmaIEtaIEta5x5' ),
    thrOverEEB = cms.double( -1.0 ),
    thrRegularEB = cms.double( 0.02 ),
    lessThan = cms.bool( True ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" )
)
fragment.hltHIEgammaHOverEDoublePhoton15Eta2p550nsMultiFit = cms.EDFilter( "HLTEgammaGenericFilter",
    thrOverE2EE = cms.double( -1.0 ),
    saveTags = cms.bool( True ),
    useEt = cms.bool( False ),
    thrOverE2EB = cms.double( -1.0 ),
    thrRegularEE = cms.double( -1.0 ),
    thrOverEEE = cms.double( 0.2 ),
    varTag = cms.InputTag( "hltHIEgammaHoverE50nsMultiFit" ),
    thrOverEEB = cms.double( 0.3 ),
    thrRegularEB = cms.double( -1.0 ),
    lessThan = cms.bool( True ),
    l1EGCand = cms.InputTag( "hltRecoHIEcalWithCleaningCandidate50nsMultiFit" ),
    ncandcut = cms.int32( 2 ),
    candTag = cms.InputTag( "hltHIDoublePhoton15Eta2p550nsMultiFit" )
)
fragment.hltL1sSingleMu3MinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet40Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3MinBiasFiltered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3MinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
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
fragment.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
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
fragment.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
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
fragment.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
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
fragment.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
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
fragment.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
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
fragment.hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    doSynchro = cms.bool( False )
)
fragment.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    deadSource = cms.string( "File" ),
    maskSource = cms.string( "File" )
)
fragment.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
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
fragment.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGeneratorFromL1T",
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2OfflineMuonSeeds" ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    CentralBxOnly = cms.bool( True ),
    InputObjects = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1MaxEta = cms.double( 2.5 ),
    EtaMatchingBins = cms.vdouble( 0.0, 2.5 ),
    L1MinPt = cms.double( 0.0 ),
    L1MinQuality = cms.uint32( 1 ),
    GMTReadoutCollection = cms.InputTag( "" ),
    UseUnassociatedL1 = cms.bool( False ),
    UseOfflineSeed = cms.untracked.bool( True ),
    MatchDR = cms.vdouble( 0.3 ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" )
)
fragment.hltL2Muons = cms.EDProducer( "L2MuonProducer",
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
fragment.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
fragment.hltHIL2Mu3N10HitQL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltSinglePuAK4CaloJet40Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 40.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltL1sSingleMu3SingleCenJet28 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleCenJet28" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet60Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3CenJet28Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3SingleCenJet28" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu3N10HitQL2FilteredWithJet28 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu3SingleCenJet40 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleCenJet40" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet80Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3CenJet40Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3SingleCenJet40" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu3N10HitQL2FilteredWithJet40 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet100Eta2p1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSinglePuAK4CaloJet100Eta2p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 100.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHIL2Mu3Eta2p5HIPhoton10Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL2Mu3Eta2p5HIPhoton15Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL2Mu3Eta2p5HIPhoton20Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleMu3SingleEG12 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleEG12" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3Eta2p5HIPhoton30Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3EG12Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3SingleEG12" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu3N10HitQL2FilteredWithEG12 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu3SingleEG20 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_SingleEG20" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3Eta2p5HIPhoton40Eta1p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3EG20Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3SingleEG20" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu3N10HitQL2FilteredWithEG20 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIUCC100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltTowerMakerForHf = cms.EDProducer( "CaloTowersCreator",
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
    HcalPhase = cms.int32( 0 ),
    HESGrid = cms.vdouble(  ),
    EcutTower = cms.double( -1000.0 ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    HESWeights = cms.vdouble(  ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kProblematic',
      'kRecovered',
      'kTime',
      'kWeird',
      'kBad' ),
    HEDWeight = cms.double( 1.0 ),
    UseSymEETreshold = cms.bool( False ),
    HEDThreshold = cms.double( 0.8 ),
    UseRejectedHitsOnly = cms.bool( False ),
    EBThreshold = cms.double( 0.07 ),
    HEDGrid = cms.vdouble(  ),
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
fragment.hltMetForHf = cms.EDProducer( "CaloMETProducer",
    alias = cms.string( "RawCaloMET" ),
    calculateSignificance = cms.bool( False ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False ),
    src = cms.InputTag( "hltTowerMakerForHf" )
)
fragment.hltGlobalSumETHfFilter4470 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 4470.0 ),
    Max = cms.double( 6400.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
fragment.hltPixelActivityFilter40000 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 40000 )
)
fragment.hltPreHIUCC020 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltGlobalSumETHfFilter4680 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 4680.0 ),
    Max = cms.double( 6400.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
fragment.hltPixelActivityFilter60000 = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000000 ),
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 60000 )
)
fragment.hltL1sMinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIQ2Bottom005Centrality1030 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltGlobalSumETHfFilterCentrality1030 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 1200.0 ),
    Max = cms.double( 3380.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
fragment.hltEvtPlaneProducer = cms.EDProducer( "EvtPlaneProducer",
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
fragment.hltEvtPlaneFilterB005Cent1030 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
fragment.hltPreHIQ2Top005Centrality1030 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEvtPlaneFilterT005Cent1030 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.145 )
)
fragment.hltPreHIQ2Bottom005Centrality3050 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltGlobalSumETHfFilterCentrality3050 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 400.0 ),
    Max = cms.double( 1500.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
fragment.hltEvtPlaneFilterB005Cent3050 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
fragment.hltPreHIQ2Top005Centrality3050 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEvtPlaneFilterT005Cent3050 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.183 )
)
fragment.hltPreHIQ2Bottom005Centrality5070 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltGlobalSumETHfFilterCentrality5070 = cms.EDFilter( "HLTGlobalSumsCaloMET",
    saveTags = cms.bool( True ),
    observable = cms.string( "sumEt" ),
    MinN = cms.int32( 1 ),
    Min = cms.double( 60.0 ),
    Max = cms.double( 600.0 ),
    inputTag = cms.InputTag( "hltMetForHf" ),
    triggerType = cms.int32( 88 )
)
fragment.hltEvtPlaneFilterB005Cent5070 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 0.01 ),
    Vnlow = cms.double( 0.0 )
)
fragment.hltPreHIQ2Top005Centrality5070 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEvtPlaneFilterT005Cent5070 = cms.EDFilter( "EvtPlaneFilter",
    EPlabel = cms.InputTag( "hltEvtPlaneProducer" ),
    EPlvl = cms.int32( 0 ),
    EPidx = cms.int32( 8 ),
    Vnhigh = cms.double( 1.0 ),
    Vnlow = cms.double( 0.223 )
)
fragment.hltPreHIFullTrack12L1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIFullTrackSelectedTracks = cms.EDProducer( "AnalyticalTrackSelector",
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
fragment.hltHIFullTrackCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIFullTrackSelectedTracks" ),
    particleType = cms.string( "pi+" )
)
fragment.hltHIFullTrackFilter12 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltPreHIFullTrack12L1Centrality010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleTrack12Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack12_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIFullTrack12L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIFullTrack18L1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIFullTrackFilter18 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltPreHIFullTrack18L1Centrality010 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIFullTrack18L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleTrack16BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack16_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIFullTrack24 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIFullTrackFilter24 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltL1sSingleTrack16Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack16_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIFullTrack24L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleTrack24BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack24_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIFullTrack34 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIFullTrackFilter34 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltL1sSingleTrack24Centralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleTrack24_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIFullTrack34L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIFullTrack45 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIFullTrackFilter45 = cms.EDFilter( "HLTSingleVertexPixelTrackFilter",
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
fragment.hltPreHIFullTrack45L1Centrality30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleMu0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltL1sDoubleMu0MinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu02HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0MinBiasL1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0MinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltL1sDoubleMu0HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu02HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltL1sDoubleMu10BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu10_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu10 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu10L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu10BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIL2DoubleMu0NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu0NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL2DoubleMu0NHitQ2HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu0NHitQ2HFFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL2DoubleMu0NHitQ2HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu0NHitQ2HF0Filtered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL2Mu3NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2Mu3N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu3HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu3NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu3HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu3N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL3Mu3NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
fragment.hltHISiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
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
fragment.hltHISiStripClusters = cms.EDProducer( "MeasurementTrackerEventProducer",
    inactivePixelDetectorLabels = cms.VInputTag(  ),
    stripClusterProducer = cms.string( "hltHISiStripRawToClustersFacility" ),
    pixelClusterProducer = cms.string( "hltHISiPixelClusters" ),
    switchOffPixelsIfEmpty = cms.bool( True ),
    inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
    skipClusters = cms.InputTag( "" ),
    measurementTracker = cms.string( "hltESPMeasurementTracker" )
)
fragment.hltHIL3TrajSeedOIState = cms.EDProducer( "TSGFromL2Muon",
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
fragment.hltHIL3TrackCandidateFromL2OIState = cms.EDProducer( "CkfTrajectoryMaker",
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
fragment.hltHIL3TkTracksFromL2OIState = cms.EDProducer( "TrackProducer",
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
fragment.hltHIL3MuonsOIState = cms.EDProducer( "L3MuonProducer",
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
fragment.hltHIL3TrajSeedOIHit = cms.EDProducer( "TSGFromL2Muon",
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
        MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
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
fragment.hltHIL3TrackCandidateFromL2OIHit = cms.EDProducer( "CkfTrajectoryMaker",
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
fragment.hltHIL3TkTracksFromL2OIHit = cms.EDProducer( "TrackProducer",
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
fragment.hltHIL3MuonsOIHit = cms.EDProducer( "L3MuonProducer",
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
fragment.hltHIL3TkFromL2OICombination = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
fragment.hltHIL3TrajectorySeed = cms.EDProducer( "L3MuonTrajectorySeedCombiner",
    labels = cms.VInputTag( 'hltHIL3TrajSeedOIState','hltHIL3TrajSeedOIHit' )
)
fragment.hltHIL3TrackCandidateFromL2 = cms.EDProducer( "L3TrackCandCombiner",
    labels = cms.VInputTag( 'hltHIL3TrackCandidateFromL2OIHit','hltHIL3TrackCandidateFromL2OIState' )
)
fragment.hltHIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
fragment.hltHIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltHIL3MuonsOIState','hltHIL3MuonsOIHit' )
)
fragment.hltHIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputLinksObjects = cms.InputTag( "hltHIL3MuonsLinksCombination" ),
    InputObjects = cms.InputTag( "hltHIL3Muons" ),
    MuonPtOption = cms.string( "Tracker" )
)
fragment.hltHISingleMu3NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu3NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHISingleMu3NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltL1sSingleMu5MinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu5NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu5MinBiasFiltered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu5MinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu5N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu5HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu5NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu5HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu5HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu5N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL3Mu5NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHISingleMu5NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu5NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHISingleMu5NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltL1sSingleMu7MinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7_MinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu7NHitQ102HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu7MinBiasFiltered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu7MinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu7N10HitQ2HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu7HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu7NHitQ102HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu7HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu7HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu7N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL3Mu7NHitQ152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHISingleMu7NHit152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu7NHitQ152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHISingleMu7NHit152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltL1sSingleMu12BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu12Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu12BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu15L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu12MinimumBiasHF1ANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_MinimumBiasHF1_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu12MinBiasFiltered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu12MinimumBiasHF1ANDBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu152HFFiltered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu12HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu12_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu12HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu12HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu15N10HitQ2HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL3Mu15 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu15L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHISingleMu15L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu152HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu152HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHISingleMu152HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu152HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu152HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHISingleMu152HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltL1sSingleMu16BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu16Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu16BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu20L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu16MinimumBiasHF1ANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_MinimumBiasHF1_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu202HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu16MinBiasFiltered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu16MinimumBiasHF1ANDBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu202HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltL1sSingleMu16HFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu16_HFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL2Mu202HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL1SingleMu16HFTower0Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu16HFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltHIL2Mu202HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltPreHIL3Mu20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu20L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHIL3SingleMu20L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu202HF = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu202HFL2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHISingleMu202HFL3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltPreHIL3Mu202HF0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL3Mu202HF0L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHISingleMu202HF0L3Filtered = cms.EDFilter( "HLTMuonL3PreFilter",
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
fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND_Centrality_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu02HFCent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0MinBiasCent30to100L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext30100BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltL1sDoubleMu0HFplusANDminusTH0Centrliatiyext30100BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_HFplusANDminusTH0_Centrliatiy_ext30_100_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu02HF0Cent30100 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0HFTower0Cent30to100L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0HFplusANDminusTH0Centrliatiyext30100BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIL2DoubleMu02HFCent30100NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu02HFcent30100NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0_MinimumBiasHF1_AND_Centrality_ext0_30_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1DoubleMu0Cent30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDoubleMu0MinBiasCent30L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIL2DoubleMu02HF0Cent30100NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu02HF0cent30100NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL2DoubleMu0Cent30OSNHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu0cent30OSNHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL2DoubleMu0Cent30NHitQ = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIL2DoubleMu0cent30NHitQFiltered = cms.EDFilter( "HLTMuonDimuonL2FromL1TFilter",
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
fragment.hltPreHIL3DoubleMu0Cent30 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDimuonOpenCentrality30L2Filtered = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHIDimuonOpenCentrality30L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
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
fragment.hltPreHIL3DoubleMu0Cent30OSm2p5to4p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDimuonOpenCentrality30OSm2p5to4p5L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
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
fragment.hltPreHIL3DoubleMu0Cent30OSm7to14 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDimuonOpenCentrality30OSm7to14L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
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
fragment.hltPreHIL3DoubleMu0OSm2p5to4p5 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDimuonOpenL2FilteredNoMBHFgated = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
fragment.hltHIDimuonOpenOSm2p5to4p5L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
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
fragment.hltPreHIL3DoubleMu0OSm7to14 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIDimuonOpenOSm7to14L3Filter = cms.EDFilter( "HLTMuonDimuonL3Filter",
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
fragment.hltL1sMuOpenNotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1SingleMuOpenNotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1MuOpenNotHF2L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sMuOpenNotMinimumBiasHF2AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCSingleMuNotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPixelLayerTripletsForUPC = cms.EDProducer( "SeedingLayersEDProducer",
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
fragment.hltPixelTracksForUPC = cms.EDProducer( "PixelTrackProducer",
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
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltPixelCandsForUPC = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForUPC" ),
    particleType = cms.string( "pi+" )
)
fragment.hltSinglePixelTrackForUPC = cms.EDFilter( "CandViewCountFilter",
    src = cms.InputTag( "hltPixelCandsForUPC" ),
    minNumber = cms.uint32( 1 )
)
fragment.hltL1sDoubleMuOpenNotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1MuOpenNotHF2L1Filtered2 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMuOpenNotMinimumBiasHF2AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCDoubleMuNotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sEG2NotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_EG2_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1SingleEG2NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCSingleEG2NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleEG2NotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG2_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleEG2NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCDoubleEG2NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sEG5NotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_EG5_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1SingleEG5NotHF2 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCSingleEG5NotHF2PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleMuOpenNotMinimumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotMinimumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotHF1 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1MuOpenL1Filtered2 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMuOpenNotMinimumBiasHF1AND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCDoubleMuNotHF1PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleEG2NotZdcANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG2_NotZdc_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleEG2NotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1DoubleEG2NotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleMuOpenNotZdcANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen_NotZdc_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1MuOpenL1Filtered3 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMuOpenNotZdcANDBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sSingleEG2NotZDCANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2_NotZDC_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1EG2NotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCEG2NotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sMuOpenNotZdcANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MuOpen_NotZdc_AND_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1MuOpenNotZDCAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1MuOpenL1Filtered4 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sMuOpenNotZdcANDBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCL1MuOpenNotZDCANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sV0NotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotHFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1NotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1NotHFplusANDminusTH0BptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sNotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotHFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sDoubleMuOpenNotHFplusANDminusTH0BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpenNotHFplusANDminusTH0_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1DoubleMuOpenTH0L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMuOpenNotHFplusANDminusTH0BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sCastorMediumJetBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_CastorMediumJet_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1CastorMediumJet = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1CastorMediumJetAK4CaloJet20 = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPuAK4CaloJetsUPC = cms.EDProducer( "FastjetJetProducer",
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
fragment.hltPuAK4CaloJetsUPCIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
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
fragment.hltPuAK4CaloJetsUPCCorrected = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsUPC" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
fragment.hltPuAK4CaloJetsUPCCorrectedIDPassed = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltPuAK4CaloJetsUPCIDPassed" ),
    correctors = cms.VInputTag( 'hltPuAK4CaloCorrector' )
)
fragment.hltSinglePuAK4CaloJet20Eta5p150nsMultiFit = cms.EDFilter( "HLT1CaloJet",
    saveTags = cms.bool( True ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 5.1 ),
    MinMass = cms.double( -1.0 ),
    inputTag = cms.InputTag( "hltPuAK4CaloJetsUPCCorrectedIDPassed" ),
    MinE = cms.double( -1.0 ),
    triggerType = cms.int32( 85 )
)
fragment.hltPreHICastorMediumJetPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1CastorMediumJetFiltered0UPC = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sCastorMediumJetBptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltL1sNotMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotMinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1NotMinimumBiasHF2AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1NotMinimumBiasHF2ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sZdcORBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZdcOR_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1ZdcORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1ZdcORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sZdcXORBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZdcXOR_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1ZdcXORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1ZdcXORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sNotZdcORBptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_NotZdcOR_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIUPCL1NotZdcORBptxAND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIUPCL1NotZdcORBptxANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sZeroBias = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHICentralityVeto = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPixelActivityFilterCentralityVeto = cms.EDFilter( "HLTPixelActivityFilter",
    maxClusters = cms.uint32( 1000 ),
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltHISiPixelClusters" ),
    minClusters = cms.uint32( 3 )
)
fragment.hltL1sL1Tech5 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1Tech5BPTXPlusOnly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sL1Tech6 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1Tech6BPTXMinusOnly = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sL1Tech7 = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1Tech7NoBPTX = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sMinimumBiasHF1OR = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF1_OR" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1MinimumBiasHF1OR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sMinimumBiasHF2OR = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF2_OR" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1MinimumBiasHF2OR = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1MinimumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sMinimumBiasHF2AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF2_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1MinimumBiasHF2AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1MinimumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHIPixel3ProtoTracksForHITrackTrigger = cms.EDProducer( "PixelTrackProducer",
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
        VertexCollection = cms.InputTag( "hltHIPixelClusterVertices" ),
        useMultipleScattering = cms.bool( False ),
        useFakeVertices = cms.bool( False )
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
fragment.hltPixelCandsForHITrackTrigger = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltHIPixel3ProtoTracksForHITrackTrigger" ),
    particleType = cms.string( "pi+" )
)
fragment.hltHISinglePixelTrackFilter = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCandsForHITrackTrigger" ),
    saveTags = cms.bool( True ),
    MinTrks = cms.uint32( 1 ),
    MinPt = cms.double( 0.0 ),
    MinSep = cms.double( 1.0 )
)
fragment.hltPreHIZeroBiasPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1sCentralityext70100MinimumumBiasHF1AND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_Centrality_ext70_100_MinimumumBiasHF1_AND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreHIL1Centralityext70100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1Centralityext70100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1Centralityext50100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1Centralityext50100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1Centralityext30100MinimumumBiasHF1AND = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIL1Centralityext30100MinimumumBiasHF1ANDPixelSingleTrack = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltPreHIPhysics = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
fragment.hltPreHIRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
fragment.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 1024 )
)
fragment.hltPreHcalCalibration = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter",
    InputTag = cms.InputTag( "rawDataCollector" ),
    CalibTypes = cms.vint32( 1, 2, 3, 4, 5, 6 ),
    FilterSummary = cms.untracked.bool( False )
)
fragment.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 1024, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199 )
)
fragment.hltPreAlCaEcalPhiSymForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltEcalPreshowerDigis = cms.EDProducer( "ESRawToDigi",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    debugMode = cms.untracked.bool( False ),
    InstanceES = cms.string( "" ),
    ESdigiCollection = cms.string( "" ),
    LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" )
)
fragment.hltEcalPreshowerRecHit = cms.EDProducer( "ESRecHitProducer",
    ESRecoAlgo = cms.int32( 0 ),
    ESrechitCollection = cms.string( "EcalRecHitsES" ),
    algo = cms.string( "ESRecHitWorker" ),
    ESdigiCollection = cms.InputTag( "hltEcalPreshowerDigis" )
)
fragment.hltEcal50nsMultifitPhiSymFilter = cms.EDFilter( "HLTEcalPhiSymFilter",
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
fragment.hltL1sSingleMuOpenIorSingleMu12BptxAND = cms.EDFilter( "HLTL1TSeed",
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen OR L1_SingleMu12_BptxAND" ),
    L1EGammaInputTag = cms.InputTag( 'hltCaloStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltCaloStage2Digis','Jet' ),
    saveTags = cms.bool( True ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1EtSumInputTag = cms.InputTag( 'hltCaloStage2Digis','EtSum' ),
    L1TauInputTag = cms.InputTag( 'hltCaloStage2Digis','Tau' ),
    L1MuonInputTag = cms.InputTag( 'hltGmtStage2Digis','Muon' ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreAlCaRPCMuonNoTriggersForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltRPCMuonNoTriggersL1Filtered0ForHI = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpenIorSingleMu12BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32( 6 ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreAlCaRPCMuonNoHitsForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltRPCPointProducer = cms.EDProducer( "RPCPointProducer",
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
fragment.hltRPCFilter = cms.EDFilter( "HLTRPCFilter",
    rangestrips = cms.untracked.double( 1.0 ),
    rpcDTPoints = cms.InputTag( 'hltRPCPointProducer','RPCDTExtrapolatedPoints' ),
    rpcRecHits = cms.InputTag( "hltRpcRecHits" ),
    rpcCSCPoints = cms.InputTag( 'hltRPCPointProducer','RPCCSCExtrapolatedPoints' )
)
fragment.hltPreAlCaRPCMuonNormalisationForHI = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltRPCMuonNormaL1Filtered0ForHI = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpenIorSingleMu12BptxAND" ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.6 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  ),
    CandTag = cms.InputTag( 'hltGmtStage2Digis','Muon' )
)
fragment.hltPreAlCaLumiPixelsRandom = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltFEDSelectorLumiPixels = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 )
)
fragment.hltPreAlCaLumiPixelsZeroBias = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltFEDSelector = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1023, 1024 )
)
fragment.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
fragment.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
fragment.hltPreHLTAnalyzerEndpath = cms.EDFilter( "HLTPrescaler",
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" ),
    offset = cms.uint32( 0 )
)
fragment.hltL1TGlobalSummary = cms.EDAnalyzer( "L1TGlobalSummary",
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    MaxBx = cms.int32( 0 ),
    DumpRecord = cms.bool( False ),
    AlgInputTag = cms.InputTag( "hltGtStage2Digis" ),
    MinBx = cms.int32( 0 ),
    DumpTrigResults = cms.bool( False ),
    DumpTrigSummary = cms.bool( True )
)
fragment.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 ),
    serviceBy = cms.untracked.string( "never" ),
    resetBy = cms.untracked.string( "never" ),
    reportBy = cms.untracked.string( "job" ),
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)

fragment.HLTL1UnpackerSequence = cms.Sequence( fragment.hltGtStage2Digis + fragment.hltCaloStage2Digis + fragment.hltGmtStage2Digis + fragment.hltGtStage2ObjectMap )
fragment.HLTBeamSpot = cms.Sequence( fragment.hltScalersRawToDigi + fragment.hltOnlineBeamSpot )
fragment.HLTBeginSequence = cms.Sequence( fragment.hltTriggerType + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTEndSequence = cms.Sequence( fragment.hltBoolEnd )
fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshower50nsMultiFitSequence = cms.Sequence( fragment.hltEcalDigis + fragment.hltEcalUncalibRecHit50nsMultiFit + fragment.hltEcalDetIdToBeRecovered + fragment.hltEcalRecHit50nsMultiFit )
fragment.HLTDoLocalHcalMethod0Sequence = cms.Sequence( fragment.hltHcalDigis + fragment.hltHbherecoMethod0 + fragment.hltHfrecoMethod0 + fragment.hltHorecoMethod0 )
fragment.HLTDoCaloHcalMethod050nsMultiFitSequence = cms.Sequence( fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshower50nsMultiFitSequence + fragment.HLTDoLocalHcalMethod0Sequence + fragment.hltTowerMakerHcalMethod050nsMultiFitForAll )
fragment.HLTPuAK4CaloJetsReconstruction50nsMultiFitSequence = cms.Sequence( fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltPuAK4CaloJets50nsMultiFit + fragment.hltPuAK4CaloJetsIDPassed50nsMultiFit )
fragment.HLTPuAK4CaloCorrectorProducersSequence = cms.Sequence( fragment.hltAK4CaloRelativeCorrector + fragment.hltAK4CaloAbsoluteCorrector + fragment.hltAK4CaloResidualCorrector + fragment.hltPuAK4CaloCorrector )
fragment.HLTPuAK4CaloJetsCorrection50nsMultiFitSequence = cms.Sequence( fragment.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 + fragment.HLTPuAK4CaloCorrectorProducersSequence + fragment.hltPuAK4CaloJetsCorrected50nsMultiFit + fragment.hltPuAK4CaloJetsCorrectedIDPassed50nsMultiFit )
fragment.HLTPuAK4CaloJets50nsMultiFitSequence = cms.Sequence( fragment.HLTPuAK4CaloJetsReconstruction50nsMultiFitSequence + fragment.HLTPuAK4CaloJetsCorrection50nsMultiFitSequence )
fragment.HLTDoHIStripZeroSuppression = cms.Sequence( fragment.hltSiStripRawToDigi + fragment.hltSiStripZeroSuppression + fragment.hltSiStripDigiToZSRaw + fragment.hltSiStripRawDigiToVirginRaw + fragment.virginRawDataRepacker + fragment.rawDataRepacker )
fragment.HLTDoHILocalPixelSequence = cms.Sequence( fragment.hltHISiPixelDigis + fragment.hltHISiPixelClusters + fragment.hltHISiPixelClustersCache + fragment.hltHISiPixelRecHits )
fragment.HLTHIRecopixelvetexingSequence = cms.Sequence( fragment.hltHIPixelClusterVertices + fragment.hltHIPixelLayerTriplets + fragment.hltHIPixel3ProtoTracks + fragment.hltHIPixelMedianVertex + fragment.hltHISelectedProtoTracks + fragment.hltHIPixelAdaptiveVertex + fragment.hltHIBestAdaptiveVertex + fragment.hltHISelectedVertex )
fragment.HLTDoHILocalPixelSequenceAfterSplitting = cms.Sequence( fragment.hltHISiPixelClustersAfterSplitting + fragment.hltHISiPixelClustersCacheAfterSplitting + fragment.hltHISiPixelRecHitsAfterSplitting )
fragment.HLTHIRecopixelvetexingSequenceAfterSplitting = cms.Sequence( fragment.hltHIPixelClusterVerticesAfterSplitting + fragment.hltHIPixelLayerTripletsAfterSplitting + fragment.hltHIPixel3ProtoTracksAfterSplitting + fragment.hltHIPixelMedianVertexAfterSplitting + fragment.hltHISelectedProtoTracksAfterSplitting + fragment.hltHIPixelAdaptiveVertexAfterSplitting + fragment.hltHIBestAdaptiveVertexAfterSplitting + fragment.hltHISelectedVertexAfterSplitting )
fragment.HLTHIPixelClusterSplitting = cms.Sequence( fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltHIJetsForCoreTracking + fragment.HLTDoHILocalPixelSequence + fragment.HLTHIRecopixelvetexingSequence + fragment.HLTDoHILocalPixelSequenceAfterSplitting + fragment.HLTHIRecopixelvetexingSequenceAfterSplitting )
fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression = cms.Sequence( fragment.hltSiStripRawToDigi + fragment.hltSiStripZeroSuppression + fragment.hltHITrackingSiStripRawToClustersFacilityZeroSuppression + fragment.hltHISiStripClustersZeroSuppression )
fragment.HLTHIIterativeTrackingIteration0Forjets = cms.Sequence( fragment.hltHIPixel3PrimTracksForjets + fragment.hltHIPixelTrackSeedsForjets + fragment.hltHIPrimTrackCandidatesForjets + fragment.hltHIGlobalPrimTracksForjets + fragment.hltHIIter0TrackSelectionForjets )
fragment.HLTHIIterativeTrackingIteration1Forjets = cms.Sequence( fragment.hltHIIter1ClustersRefRemovalForjets + fragment.hltHIIter1MaskedMeasurementTrackerEventForjets + fragment.hltHIDetachedPixelLayerTripletsForjets + fragment.hltHIDetachedPixelTracksForjets + fragment.hltHIDetachedPixelTrackSeedsForjets + fragment.hltHIDetachedTrackCandidatesForjets + fragment.hltHIDetachedGlobalPrimTracksForjets + fragment.hltHIIter1TrackSelectionForjets )
fragment.HLTHIIterativeTrackingIteration2Forjets = cms.Sequence( fragment.hltHIIter2ClustersRefRemovalForjets + fragment.hltHIIter2MaskedMeasurementTrackerEventForjets + fragment.hltHIPixelLayerPairsForjets + fragment.hltHIPixelPairSeedsForjets + fragment.hltHIPixelPairTrackCandidatesForjets + fragment.hltHIPixelPairGlobalPrimTracksForjets + fragment.hltHIIter2TrackSelectionForjets )
fragment.HLTHIIterativeTrackingForJets = cms.Sequence( fragment.HLTHIIterativeTrackingIteration0Forjets + fragment.HLTHIIterativeTrackingIteration1Forjets + fragment.HLTHIIterativeTrackingIteration2Forjets + fragment.hltHIIterTrackingMergedHighPurityForjets + fragment.hltHIIterTrackingMergedTightForjets )
fragment.HLTDoHIStripZeroSuppressionRepacker = cms.Sequence( fragment.hltSiStripDigiToZSRaw + fragment.hltSiStripRawDigiToVirginRaw + fragment.virginRawDataRepacker + fragment.rawDataRepacker )
fragment.HLTBtagCSVSequenceL3CaloJet60Eta2p1 = cms.Sequence( fragment.hltHIVerticesL3 + fragment.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 + fragment.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 + fragment.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 + fragment.hltL3CombinedSecondaryVertexBJetTagsCaloJet60Eta2p1 )
fragment.HLTBtagCSVSequenceL3CaloJet80Eta2p1 = cms.Sequence( fragment.hltHIVerticesL3 + fragment.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 + fragment.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 + fragment.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 + fragment.hltL3CombinedSecondaryVertexBJetTagsCaloJet80Eta2p1 )
fragment.HLTBtagSSVSequenceL3CaloJet60Eta2p1 = cms.Sequence( fragment.hltHIVerticesL3 + fragment.hltFastPixelBLifetimeL3AssociatorCaloJet60Eta2p1 + fragment.hltFastPixelBLifetimeL3TagInfosCaloJet60Eta2p1 + fragment.hltL3SecondaryVertexTagInfosCaloJet60Eta2p1 + fragment.hltL3SimpleSecondaryVertexBJetTagsCaloJet60Eta2p1 )
fragment.HLTBtagSSVSequenceL3CaloJet80Eta2p1 = cms.Sequence( fragment.hltHIVerticesL3 + fragment.hltFastPixelBLifetimeL3AssociatorCaloJet80Eta2p1 + fragment.hltFastPixelBLifetimeL3TagInfosCaloJet80Eta2p1 + fragment.hltL3SecondaryVertexTagInfosCaloJet80Eta2p1 + fragment.hltL3SimpleSecondaryVertexBJetTagsCaloJet80Eta2p1 )
fragment.HLTHIIterativeTrackingIteration0ForGlobalPt8 = cms.Sequence( fragment.hltHIPixel3PrimTracksForGlobalPt8 + fragment.hltHIPixelTrackSeedsForGlobalPt8 + fragment.hltHIPrimTrackCandidatesForGlobalPt8 + fragment.hltHIGlobalPrimTracksForGlobalPt8 + fragment.hltHIIter0TrackSelectionForGlobalPt8 )
fragment.HLTHIIterativeTrackingIteration1ForGlobalPt8 = cms.Sequence( fragment.hltHIIter1ClustersRefRemovalForGlobalPt8 + fragment.hltHIIter1MaskedMeasurementTrackerEventForGlobalPt8 + fragment.hltHIDetachedPixelLayerTripletsForGlobalPt8 + fragment.hltHIDetachedPixelTracksForGlobalPt8 + fragment.hltHIDetachedPixelTrackSeedsForGlobalPt8 + fragment.hltHIDetachedTrackCandidatesForGlobalPt8 + fragment.hltHIDetachedGlobalPrimTracksForGlobalPt8 + fragment.hltHIIter1TrackSelectionForGlobalPt8 )
fragment.HLTHIIterativeTrackingIteration2ForGlobalPt8 = cms.Sequence( fragment.hltHIIter2ClustersRefRemovalForGlobalPt8 + fragment.hltHIIter2MaskedMeasurementTrackerEventForGlobalPt8 + fragment.hltHIPixelLayerPairsForGlobalPt8 + fragment.hltHIPixelPairSeedsForGlobalPt8 + fragment.hltHIPixelPairTrackCandidatesForGlobalPt8 + fragment.hltHIPixelPairGlobalPrimTracksForGlobalPt8 + fragment.hltHIIter2TrackSelectionForGlobalPt8 )
fragment.HLTHIIterativeTrackingForGlobalPt8 = cms.Sequence( fragment.HLTHIIterativeTrackingIteration0ForGlobalPt8 + fragment.HLTHIIterativeTrackingIteration1ForGlobalPt8 + fragment.HLTHIIterativeTrackingIteration2ForGlobalPt8 + fragment.hltHIIterTrackingMergedHighPurityForGlobalPt8 + fragment.hltHIIterTrackingMergedTightForGlobalPt8 )
fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence = cms.Sequence( fragment.hltIslandBasicClusters50nsMultiFitHI + fragment.hltHiIslandSuperClusters50nsMultiFitHI + fragment.hltHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI + fragment.hltHiCorrectedIslandEndcapSuperClusters50nsMultiFitHI + fragment.hltCleanedHiCorrectedIslandBarrelSuperClusters50nsMultiFitHI + fragment.hltRecoHIEcalWithCleaningCandidate50nsMultiFit )
fragment.HLTMuonLocalRecoSequence = cms.Sequence( fragment.hltMuonDTDigis + fragment.hltDt1DRecHits + fragment.hltDt4DSegments + fragment.hltMuonCSCDigis + fragment.hltCsc2DRecHits + fragment.hltCscSegments + fragment.hltMuonRPCDigis + fragment.hltRpcRecHits )
fragment.HLTL2muonrecoNocandSequence = cms.Sequence( fragment.HLTMuonLocalRecoSequence + fragment.hltL2OfflineMuonSeeds + fragment.hltL2MuonSeeds + fragment.hltL2Muons )
fragment.HLTL2muonrecoSequence = cms.Sequence( fragment.HLTL2muonrecoNocandSequence + fragment.hltL2MuonCandidates )
fragment.HLTDoLocalHfSequence = cms.Sequence( fragment.hltHcalDigis + fragment.hltHfrecoMethod0 + fragment.hltTowerMakerForHf )
fragment.HLTRecoMETHfSequence = cms.Sequence( fragment.HLTDoLocalHfSequence + fragment.hltMetForHf )
fragment.HLTDoHILocalPixelClustersSequence = cms.Sequence( fragment.hltHISiPixelDigis + fragment.hltHISiPixelClusters )
fragment.HLTDoHILocalStripSequence = cms.Sequence( fragment.hltSiStripExcludedFEDListProducer + fragment.hltHISiStripRawToClustersFacility + fragment.hltHISiStripClusters )
fragment.HLTHIL3muonTkCandidateSequence = cms.Sequence( fragment.HLTDoHILocalPixelSequence + fragment.HLTDoHILocalStripSequence + fragment.hltHIL3TrajSeedOIState + fragment.hltHIL3TrackCandidateFromL2OIState + fragment.hltHIL3TkTracksFromL2OIState + fragment.hltHIL3MuonsOIState + fragment.hltHIL3TrajSeedOIHit + fragment.hltHIL3TrackCandidateFromL2OIHit + fragment.hltHIL3TkTracksFromL2OIHit + fragment.hltHIL3MuonsOIHit + fragment.hltHIL3TkFromL2OICombination + fragment.hltHIL3TrajectorySeed + fragment.hltHIL3TrackCandidateFromL2 )
fragment.HLTHIL3muonrecoNocandSequence = cms.Sequence( fragment.HLTHIL3muonTkCandidateSequence + fragment.hltHIL3MuonsLinksCombination + fragment.hltHIL3Muons )
fragment.HLTHIL3muonrecoSequence = cms.Sequence( fragment.HLTHIL3muonrecoNocandSequence + fragment.hltHIL3MuonCandidates )
fragment.HLTRecopixelvertexingSequenceForUPC = cms.Sequence( fragment.hltPixelLayerTripletsForUPC + fragment.hltPixelTracksForUPC )
fragment.HLTPuAK4CaloJetsUPCReconstructionSequence = cms.Sequence( fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltPuAK4CaloJetsUPC + fragment.hltPuAK4CaloJetsUPCIDPassed )
fragment.HLTPuAK4CaloJetsUPCCorrectionSequence = cms.Sequence( fragment.hltFixedGridRhoFastjetAllCalo50nsMultiFitHcalMethod0 + fragment.HLTPuAK4CaloCorrectorProducersSequence + fragment.hltPuAK4CaloJetsUPCCorrected + fragment.hltPuAK4CaloJetsUPCCorrectedIDPassed )
fragment.HLTPuAK4CaloJetsUPCSequence = cms.Sequence( fragment.HLTPuAK4CaloJetsUPCReconstructionSequence + fragment.HLTPuAK4CaloJetsUPCCorrectionSequence )
fragment.HLTPixelTrackingForHITrackTrigger = cms.Sequence( fragment.hltHIPixelClusterVertices + fragment.hltHIPixelLayerTriplets + fragment.hltHIPixel3ProtoTracksForHITrackTrigger + fragment.hltPixelCandsForHITrackTrigger )
fragment.HLTBeginSequenceRandom = cms.Sequence( fragment.hltRandomEventsFilter + fragment.hltGtStage2Digis )
fragment.HLTBeginSequenceCalibration = cms.Sequence( fragment.hltCalibrationEventsFilter + fragment.hltGtStage2Digis )
fragment.HLTDoFullUnpackingEgammaEcal50nsMultiFitSequence = cms.Sequence( fragment.hltEcalDigis + fragment.hltEcalPreshowerDigis + fragment.hltEcalUncalibRecHit50nsMultiFit + fragment.hltEcalDetIdToBeRecovered + fragment.hltEcalRecHit50nsMultiFit + fragment.hltEcalPreshowerRecHit )

fragment.HLTriggerFirstPath = cms.Path( fragment.hltGetConditions + fragment.hltGetRaw + fragment.hltBoolFalse )
fragment.DST_Physics_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltPreDSTPhysics + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet40_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHIPuAK4CaloJet40Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet60_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28BptxAND + fragment.hltPreHIPuAK4CaloJet60Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloJet80Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet100_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet56BptxAND + fragment.hltPreHIPuAK4CaloJet100Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet110_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet56BptxAND + fragment.hltPreHIPuAK4CaloJet110Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet110Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet120_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet56BptxAND + fragment.hltPreHIPuAK4CaloJet120Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet120Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet150_Eta5p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet64BptxAND + fragment.hltPreHIPuAK4CaloJet150Eta5p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet150Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet16Centralityext30100BptxAND + fragment.hltPreHIPuAK4CaloJet40Eta5p1Cent30100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28Centralityext30100BptxAND + fragment.hltPreHIPuAK4CaloJet60Eta5p1Cent30100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext30100BptxAND + fragment.hltPreHIPuAK4CaloJet80Eta5p1Cent30100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext30100BptxAND + fragment.hltPreHIPuAK4CaloJet100Eta5p1Cent30100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet16Centralityext50100BptxAND + fragment.hltPreHIPuAK4CaloJet40Eta5p1Cent50100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet40Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28Centralityext50100BptxAND + fragment.hltPreHIPuAK4CaloJet60Eta5p1Cent50100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext50100BptxAND + fragment.hltPreHIPuAK4CaloJet80Eta5p1Cent50100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext50100BptxAND + fragment.hltPreHIPuAK4CaloJet100Eta5p1Cent50100 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloJet80Jet35Eta1p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta1p150nsMultiFit + fragment.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloJet80Jet35Eta0p7 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta0p750nsMultiFit + fragment.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet56BptxAND + fragment.hltPreHIPuAK4CaloJet100Jet35Eta1p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta1p150nsMultiFit + fragment.hltDoublePuAK4CaloJet35Eta1p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet56BptxAND + fragment.hltPreHIPuAK4CaloJet100Jet35Eta0p7 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta0p750nsMultiFit + fragment.hltDoublePuAK4CaloJet35Eta0p750nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloJet804545Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltTriplePuAK4CaloJet45Eta2p150nsMultiFit + fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloDJet60_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28BptxAND + fragment.hltPreHIPuAK4CaloDJet60Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet60Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.hltHIFullTrackCandsForDmesonjets + fragment.hltHIFullTrackFilterForDmesonjets + fragment.hltTktkVtxForDmesonjetsCaloJet60 + fragment.hltTktkFilterForDmesonjetsCaloJet60 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloDJet80_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloDJet80Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet80Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.hltHIFullTrackCandsForDmesonjets + fragment.hltHIFullTrackFilterForDmesonjets + fragment.hltTktkVtxForDmesonjetsCaloJet80 + fragment.hltTktkFilterForDmesonjetsCaloJet80 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28BptxAND + fragment.hltPreHIPuAK4CaloBJetCSV60Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet60Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.HLTBtagCSVSequenceL3CaloJet60Eta2p1 + fragment.hltBLifetimeL3FilterCSVCaloJet60Eta2p1 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloBJetCSV80Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet80Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.HLTBtagCSVSequenceL3CaloJet80Eta2p1 + fragment.hltBLifetimeL3FilterCSVCaloJet80Eta2p1 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28BptxAND + fragment.hltPreHIPuAK4CaloBJetSSV60Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet60Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.HLTBtagSSVSequenceL3CaloJet60Eta2p1 + fragment.hltBLifetimeL3FilterSSVCaloJet60Eta2p1 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIPuAK4CaloBJetSSV80Eta2p1 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + fragment.hltEta2CaloJetsEta2p1ForJets + fragment.hltReduceJetMultEta2p1Forjets + fragment.hltJets4bTaggerCaloJet80Eta2p1Forjets + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForJets + fragment.HLTBtagSSVSequenceL3CaloJet80Eta2p1 + fragment.hltBLifetimeL3FilterSSVCaloJet80Eta2p1 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt20 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt20 + fragment.hltTktkFilterForDmesonGlobal8Dp20 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext30100MinimumumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt20 + fragment.hltTktkFilterForDmesonGlobal8Dp20 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext50100MinimumumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent50100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt20 + fragment.hltTktkFilterForDmesonGlobal8Dp20 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet16BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt30 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt30 + fragment.hltTktkFilterForDmesonGlobal8Dp30 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet16Centralityext30100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt30 + fragment.hltTktkFilterForDmesonGlobal8Dp30 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet16Centralityext50100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent50100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt30 + fragment.hltTktkFilterForDmesonGlobal8Dp30 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt40 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt40 + fragment.hltTktkFilterForDmesonGlobal8Dp40 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28Centralityext30100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt40 + fragment.hltTktkFilterForDmesonGlobal8Dp40 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet28Centralityext50100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent50100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt40 + fragment.hltTktkFilterForDmesonGlobal8Dp40 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt50_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet32BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt50 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt50 + fragment.hltTktkFilterForDmesonGlobal8Dp50 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet44BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt60 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt60 + fragment.hltTktkFilterForDmesonGlobal8Dp60 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt70_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet52BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt70 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt70 + fragment.hltTktkFilterForDmesonGlobal8Dp70 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext30100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt60Cent30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt60 + fragment.hltTktkFilterForDmesonGlobal8Dp60 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleS1Jet44Centralityext50100BptxAND + fragment.hltPreHIDmesonHITrackingGlobalDpt60Cent50100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt60 + fragment.hltTktkFilterForDmesonGlobal8Dp60 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt20Cent010 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt20 + fragment.hltTktkFilterForDmesonGlobal8Dp20 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt30Cent010 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt30 + fragment.hltTktkFilterForDmesonGlobal8Dp30 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIDmesonHITrackingGlobalDpt40Cent010 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackCandsForDmesonGlobalPt8 + fragment.hltHIFullTrackFilterForDmesonGlobalPt8 + fragment.hltTktkVtxForDmesonGlobal8Dpt40 + fragment.hltTktkFilterForDmesonGlobal8Dp40 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton10Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton15Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton20Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7BptxAND + fragment.hltPreHISinglePhoton30Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHISinglePhoton40Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton50_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHISinglePhoton50Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton50Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton60_Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG30BptxAND + fragment.hltPreHISinglePhoton60Eta1p5 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton60Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta1p5_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton10Eta1p5Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta1p5_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton15Eta1p5Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta1p5_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton20Eta1p5Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta1p5_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7Centralityext50100BptxAND + fragment.hltPreHISinglePhoton30Eta1p5Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta1p5_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21Centralityext50100BptxAND + fragment.hltPreHISinglePhoton40Eta1p5Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta1p5_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton10Eta1p5Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta1p5_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton15Eta1p5Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta1p5_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton20Eta1p5Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta1p5_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7Centralityext30100BptxAND + fragment.hltPreHISinglePhoton30Eta1p5Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta1p5_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21Centralityext30100BptxAND + fragment.hltPreHISinglePhoton40Eta1p5Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHISinglePhoton40Eta2p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton10Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton15Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreHISinglePhoton20Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7BptxAND + fragment.hltPreHISinglePhoton30Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHISinglePhoton40Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton50_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHISinglePhoton50Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton50Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton60_Eta3p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG30BptxAND + fragment.hltPreHISinglePhoton60Eta3p1 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton60Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta3p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton10Eta3p1Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta3p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton15Eta3p1Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta3p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext50100BptxAND + fragment.hltPreHISinglePhoton20Eta3p1Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta3p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7Centralityext50100BptxAND + fragment.hltPreHISinglePhoton30Eta3p1Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta3p1_Cent50_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21Centralityext50100BptxAND + fragment.hltPreHISinglePhoton40Eta3p1Cent50100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton10_Eta3p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton10Eta3p1Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton15_Eta3p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton15Eta3p1Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton20_Eta3p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG3Centralityext30100BptxAND + fragment.hltPreHISinglePhoton20Eta3p1Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton30_Eta3p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG7Centralityext30100BptxAND + fragment.hltPreHISinglePhoton30Eta3p1Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HISinglePhoton40_Eta3p1_Cent30_100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21Centralityext30100BptxAND + fragment.hltPreHISinglePhoton40Eta3p1Cent30100 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta3p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHIDoublePhoton15Eta1p5Mass501000 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIDoublePhoton15Eta1p550nsMultiFit + fragment.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHIDoublePhoton15Eta1p5Mass501000R9HECut + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIDoublePhoton15Eta1p550nsMultiFit + fragment.hltHIDoublePhoton15Eta1p5GlobalMass501000Filter + fragment.hltHIEgammaR9ID50nsMultiFit + fragment.hltHIEgammaR9IDDoublePhoton15Eta1p550nsMultiFit + fragment.hltHIEgammaHoverE50nsMultiFit + fragment.hltHIEgammaHOverEDoublePhoton15Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHIDoublePhoton15Eta2p1Mass501000R9Cut + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIDoublePhoton15Eta2p150nsMultiFit + fragment.hltHIDoublePhoton15Eta2p1GlobalMass501000Filter + fragment.hltHIEgammaR9ID50nsMultiFit + fragment.hltHIEgammaR9IDDoublePhoton15Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG21BptxAND + fragment.hltPreHIDoublePhoton15Eta2p5Mass501000R9SigmaHECut + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIDoublePhoton15Eta2p550nsMultiFit + fragment.hltHIDoublePhoton15Eta2p5GlobalMass501000Filter + fragment.hltHIEgammaR9ID50nsMultiFit + fragment.hltHIEgammaR9IDDoublePhoton15Eta2p550nsMultiFit + fragment.hltHIEgammaSigmaIEtaIEta50nsMultiFitProducer + fragment.hltHIEgammaSigmaIEtaIEtaDoublePhoton15Eta2p550nsMultiFit + fragment.hltHIEgammaHoverE50nsMultiFit + fragment.hltHIEgammaHOverEDoublePhoton15Eta2p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet40Eta2p1 + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2Filtered + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet40Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3SingleCenJet28 + fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet60Eta2p1 + fragment.hltHIL1SingleMu3CenJet28Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2FilteredWithJet28 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet60Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3SingleCenJet40 + fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet80Eta2p1 + fragment.hltHIL1SingleMu3CenJet40Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2FilteredWithJet40 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet80Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3SingleCenJet40 + fragment.hltPreHIL2Mu3Eta2p5PuAK4CaloJet100Eta2p1 + fragment.hltHIL1SingleMu3CenJet40Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2FilteredWithJet40 + fragment.HLTPuAK4CaloJets50nsMultiFitSequence + fragment.hltSinglePuAK4CaloJet100Eta2p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL2Mu3Eta2p5HIPhoton10Eta1p5 + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2Filtered + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton10Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL2Mu3Eta2p5HIPhoton15Eta1p5 + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2Filtered + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton15Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL2Mu3Eta2p5HIPhoton20Eta1p5 + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2Filtered + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton20Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3SingleEG12 + fragment.hltPreHIL2Mu3Eta2p5HIPhoton30Eta1p5 + fragment.hltHIL1SingleMu3EG12Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2FilteredWithEG12 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton30Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3SingleEG20 + fragment.hltPreHIL2Mu3Eta2p5HIPhoton40Eta1p5 + fragment.hltHIL1SingleMu3EG20Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQL2FilteredWithEG20 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.HLTDoHIEcalClusWithCleaning50nsMultiFitSequence + fragment.hltHIPhoton40Eta1p550nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUCC100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIUCC100 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilter4470 + fragment.HLTDoHILocalPixelClustersSequence + fragment.hltPixelActivityFilter40000 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUCC020_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIUCC020 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilter4680 + fragment.HLTDoHILocalPixelClustersSequence + fragment.hltPixelActivityFilter60000 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Bottom005_Centrality1030_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Bottom005Centrality1030 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality1030 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterB005Cent1030 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Top005_Centrality1030_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Top005Centrality1030 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality1030 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterT005Cent1030 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Bottom005_Centrality3050_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Bottom005Centrality3050 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality3050 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterB005Cent3050 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Top005_Centrality3050_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Top005Centrality3050 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality3050 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterT005Cent3050 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Bottom005_Centrality5070_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Bottom005Centrality5070 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality5070 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterB005Cent5070 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIQ2Top005_Centrality5070_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIQ2Top005Centrality5070 + fragment.HLTRecoMETHfSequence + fragment.hltGlobalSumETHfFilterCentrality5070 + fragment.HLTDoCaloHcalMethod050nsMultiFitSequence + fragment.hltEvtPlaneProducer + fragment.hltEvtPlaneFilterT005Cent5070 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIFullTrack12L1MinimumBiasHF1AND + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter12 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack12_L1Centrality010_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIFullTrack12L1Centrality010 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter12 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack12_L1Centrality30100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack12Centralityext30100BptxAND + fragment.hltPreHIFullTrack12L1Centrality30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter12 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIFullTrack18L1MinimumBiasHF1AND + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter18 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack18_L1Centrality010_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext010MinimumumBiasHF1AND + fragment.hltPreHIFullTrack18L1Centrality010 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter18 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack18_L1Centrality30100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack12Centralityext30100BptxAND + fragment.hltPreHIFullTrack18L1Centrality30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter18 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack24_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack16BptxAND + fragment.hltPreHIFullTrack24 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter24 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack24_L1Centrality30100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack16Centralityext30100BptxAND + fragment.hltPreHIFullTrack24L1Centrality30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter24 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack34_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack24BptxAND + fragment.hltPreHIFullTrack34 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter34 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack34_L1Centrality30100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack24Centralityext30100BptxAND + fragment.hltPreHIFullTrack34L1Centrality30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter34 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack45_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack24BptxAND + fragment.hltPreHIFullTrack45 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter45 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIFullTrack45_L1Centrality30100_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleTrack24Centralityext30100BptxAND + fragment.hltPreHIFullTrack45L1Centrality30100 + fragment.HLTHIPixelClusterSplitting + fragment.HLTDoHITrackingLocalStripSequenceZeroSuppression + fragment.HLTHIIterativeTrackingForGlobalPt8 + fragment.hltHIFullTrackSelectedTracks + fragment.hltHIFullTrackCandsForHITrackTrigger + fragment.hltHIFullTrackFilter45 + fragment.HLTDoHIStripZeroSuppressionRepacker + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0BptxAND + fragment.hltPreHIL1DoubleMu0 + fragment.hltHIDoubleMu0L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1AND + fragment.hltPreHIL1DoubleMu02HF + fragment.hltHIDoubleMu0MinBiasL1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0HFplusANDminusTH0BptxAND + fragment.hltPreHIL1DoubleMu02HF0 + fragment.hltHIDoubleMu0HFTower0Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu10_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu10BptxAND + fragment.hltPreHIL1DoubleMu10 + fragment.hltHIDoubleMu10L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_NHitQ_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0BptxAND + fragment.hltPreHIL2DoubleMu0NHitQ + fragment.hltHIDoubleMu0L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu0NHitQFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_NHitQ_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1AND + fragment.hltPreHIL2DoubleMu0NHitQ2HF + fragment.hltHIDoubleMu0MinBiasL1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu0NHitQ2HFFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_NHitQ_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0HFplusANDminusTH0BptxAND + fragment.hltPreHIL2DoubleMu0NHitQ2HF0 + fragment.hltHIDoubleMu0HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu0NHitQ2HF0Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3_NHitQ10_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL2Mu3NHitQ102HF + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQ2HFL2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu3_NHitQ10_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3HFplusANDminusTH0BptxAND + fragment.hltPreHIL2Mu3NHitQ102HF0 + fragment.hltHIL1SingleMu3HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQ2HF0L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu3_NHitQ15_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3MinimumBiasHF1AND + fragment.hltPreHIL3Mu3NHitQ152HF + fragment.hltHIL1SingleMu3MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQ2HFL2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu3NHit152HFL3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu3_NHitQ15_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3HFplusANDminusTH0BptxAND + fragment.hltPreHIL3Mu3NHitQ152HF0 + fragment.hltHIL1SingleMu3HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu3N10HitQ2HF0L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu3NHit152HF0L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu5_NHitQ10_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5MinimumBiasHF1AND + fragment.hltPreHIL2Mu5NHitQ102HF + fragment.hltHIL1SingleMu5MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu5N10HitQ2HFL2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu5_NHitQ10_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5HFplusANDminusTH0BptxAND + fragment.hltPreHIL2Mu5NHitQ102HF0 + fragment.hltHIL1SingleMu5HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu5N10HitQ2HF0L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu5_NHitQ15_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5MinimumBiasHF1AND + fragment.hltPreHIL3Mu5NHitQ152HF + fragment.hltHIL1SingleMu5MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu5N10HitQ2HFL2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu5NHit152HFL3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu5_NHitQ15_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5HFplusANDminusTH0BptxAND + fragment.hltPreHIL3Mu5NHitQ152HF0 + fragment.hltHIL1SingleMu5HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu5N10HitQ2HF0L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu5NHit152HF0L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu7_NHitQ10_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu7MinimumBiasHF1AND + fragment.hltPreHIL2Mu7NHitQ102HF + fragment.hltHIL1SingleMu7MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu7N10HitQ2HFL2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu7_NHitQ10_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu7HFplusANDminusTH0BptxAND + fragment.hltPreHIL2Mu7NHitQ102HF0 + fragment.hltHIL1SingleMu7HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu7N10HitQ2HF0L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu7_NHitQ15_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu7MinimumBiasHF1AND + fragment.hltPreHIL3Mu7NHitQ152HF + fragment.hltHIL1SingleMu7MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu7N10HitQ2HFL2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu7NHit152HFL3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu7_NHitQ15_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu7HFplusANDminusTH0BptxAND + fragment.hltPreHIL3Mu7NHitQ152HF0 + fragment.hltHIL1SingleMu7HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu7N10HitQ2HF0L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu7NHit152HF0L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu15_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12BptxAND + fragment.hltPreHIL2Mu15 + fragment.hltHIL1SingleMu12Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu15L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu15_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12MinimumBiasHF1ANDBptxAND + fragment.hltPreHIL2Mu152HF + fragment.hltHIL1SingleMu12MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu152HFFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu15_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12HFplusANDminusTH0BptxAND + fragment.hltPreHIL2Mu152HF0 + fragment.hltHIL1SingleMu12HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu15N10HitQ2HF0L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu15_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12BptxAND + fragment.hltPreHIL3Mu15 + fragment.hltHIL1SingleMu12Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu15L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu15L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu15_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12MinimumBiasHF1ANDBptxAND + fragment.hltPreHIL3Mu152HF + fragment.hltHIL1SingleMu12MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu152HFL2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu152HFL3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu15_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu12HFplusANDminusTH0BptxAND + fragment.hltPreHIL3Mu152HF0 + fragment.hltHIL1SingleMu12HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu152HF0L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu152HF0L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu20_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16BptxAND + fragment.hltPreHIL2Mu20 + fragment.hltHIL1SingleMu16Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu20L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu20_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16MinimumBiasHF1ANDBptxAND + fragment.hltPreHIL2Mu202HF + fragment.hltHIL1SingleMu16MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu202HFL2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2Mu20_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16HFplusANDminusTH0BptxAND + fragment.hltPreHIL2Mu202HF0 + fragment.hltHIL1SingleMu16HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2Mu202HF0L2Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu20_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16BptxAND + fragment.hltPreHIL3Mu20 + fragment.hltHIL1SingleMu16Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu20L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHIL3SingleMu20L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu20_2HF_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16MinimumBiasHF1ANDBptxAND + fragment.hltPreHIL3Mu202HF + fragment.hltHIL1SingleMu16MinBiasFiltered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu202HFL2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu202HFL3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3Mu20_2HF0_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu16HFplusANDminusTH0BptxAND + fragment.hltPreHIL3Mu202HF0 + fragment.hltHIL1SingleMu16HFTower0Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL3Mu202HF0L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHISingleMu202HF0L3Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_2HF_Cent30100_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext30100BptxAND + fragment.hltPreHIL1DoubleMu02HFCent30100 + fragment.hltHIDoubleMu0MinBiasCent30to100L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_2HF0_Cent30100_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0HFplusANDminusTH0Centrliatiyext30100BptxAND + fragment.hltPreHIL1DoubleMu02HF0Cent30100 + fragment.hltHIDoubleMu0HFTower0Cent30to100L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext30100BptxAND + fragment.hltPreHIL2DoubleMu02HFCent30100NHitQ + fragment.hltHIDoubleMu0MinBiasCent30to100L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu02HFcent30100NHitQFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1DoubleMu0_Cent30_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL1DoubleMu0Cent30 + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0HFplusANDminusTH0Centrliatiyext30100BptxAND + fragment.hltPreHIL2DoubleMu02HF0Cent30100NHitQ + fragment.hltHIDoubleMu0HFTower0Cent30to100L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu02HF0cent30100NHitQFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL2DoubleMu0Cent30OSNHitQ + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu0cent30OSNHitQFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL2DoubleMu0_Cent30_NHitQ_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL2DoubleMu0Cent30NHitQ + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIL2DoubleMu0cent30NHitQFiltered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3DoubleMu0_Cent30_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL3DoubleMu0Cent30 + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIDimuonOpenCentrality30L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHIDimuonOpenCentrality30L3Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL3DoubleMu0Cent30OSm2p5to4p5 + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIDimuonOpenCentrality30L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHIDimuonOpenCentrality30OSm2p5to4p5L3Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0MinimumBiasHF1ANDCentralityext030BptxAND + fragment.hltPreHIL3DoubleMu0Cent30OSm7to14 + fragment.hltHIDoubleMu0MinBiasCent30L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIDimuonOpenCentrality30L2Filtered + fragment.HLTHIL3muonrecoSequence + fragment.hltHIDimuonOpenCentrality30OSm7to14L3Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3DoubleMu0_OS_m2p5to4p5_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0BptxAND + fragment.hltPreHIL3DoubleMu0OSm2p5to4p5 + fragment.hltHIDoubleMu0L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIDimuonOpenL2FilteredNoMBHFgated + fragment.HLTHIL3muonrecoSequence + fragment.hltHIDimuonOpenOSm2p5to4p5L3Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL3DoubleMu0_OS_m7to14_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0BptxAND + fragment.hltPreHIL3DoubleMu0OSm7to14 + fragment.hltHIDoubleMu0L1Filtered + fragment.HLTL2muonrecoSequence + fragment.hltHIDimuonOpenL2FilteredNoMBHFgated + fragment.HLTHIL3muonrecoSequence + fragment.hltHIDimuonOpenOSm7to14L3Filter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1SingleMuOpenNotHF2_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMuOpenNotMinimumBiasHF2AND + fragment.hltPreHIUPCL1SingleMuOpenNotHF2 + fragment.hltL1MuOpenNotHF2L1Filtered0 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMuOpenNotMinimumBiasHF2AND + fragment.hltPreHIUPCSingleMuNotHF2PixelSingleTrack + fragment.hltL1MuOpenNotHF2L1Filtered0 + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotHF2_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotMinimumBiasHF2AND + fragment.hltPreHIUPCL1DoubleMuOpenNotHF2 + fragment.hltL1MuOpenNotHF2L1Filtered2 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotMinimumBiasHF2AND + fragment.hltPreHIUPCDoubleMuNotHF2PixelSingleTrack + fragment.hltL1MuOpenNotHF2L1Filtered2 + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1SingleEG2NotHF2_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEG2NotMinimumBiasHF2AND + fragment.hltPreHIUPCL1SingleEG2NotHF2 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEG2NotMinimumBiasHF2AND + fragment.hltPreHIUPCSingleEG2NotHF2PixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleEG2NotHF2_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleEG2NotMinimumBiasHF2AND + fragment.hltPreHIUPCL1DoubleEG2NotHF2 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleEG2NotMinimumBiasHF2AND + fragment.hltPreHIUPCDoubleEG2NotHF2PixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1SingleEG5NotHF2_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEG5NotMinimumBiasHF2AND + fragment.hltPreHIUPCL1SingleEG5NotHF2 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEG5NotMinimumBiasHF2AND + fragment.hltPreHIUPCSingleEG5NotHF2PixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotHF1_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotMinimumBiasHF1AND + fragment.hltPreHIUPCL1DoubleMuOpenNotHF1 + fragment.hltL1MuOpenL1Filtered2 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotMinimumBiasHF1AND + fragment.hltPreHIUPCDoubleMuNotHF1PixelSingleTrack + fragment.hltL1MuOpenL1Filtered2 + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleEG2NotZDCAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleEG2NotZdcANDBptxAND + fragment.hltPreHIUPCL1DoubleEG2NotZDCAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleEG2NotZdcANDBptxAND + fragment.hltPreHIUPCL1DoubleEG2NotZDCANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotZDCAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotZdcANDBptxAND + fragment.hltPreHIUPCL1DoubleMuOpenNotZDCAND + fragment.hltL1MuOpenL1Filtered3 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotZdcANDBptxAND + fragment.hltPreHIUPCL1DoubleMuOpenNotZDCANDPixelSingleTrack + fragment.hltL1MuOpenL1Filtered3 + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1EG2NotZDCAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG2NotZDCANDBptxAND + fragment.hltPreHIUPCL1EG2NotZDCAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG2NotZDCANDBptxAND + fragment.hltPreHIUPCEG2NotZDCANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1MuOpenNotZDCAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMuOpenNotZdcANDBptxAND + fragment.hltPreHIUPCL1MuOpenNotZDCAND + fragment.hltL1MuOpenL1Filtered4 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMuOpenNotZdcANDBptxAND + fragment.hltPreHIUPCL1MuOpenNotZDCANDPixelSingleTrack + fragment.hltL1MuOpenL1Filtered4 + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0NotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1NotHFplusANDminusTH0BptxAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0NotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1NotHFplusANDminusTH0BptxANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0 + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1NotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0 + fragment.hltL1DoubleMuOpenTH0L1Filtered + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMuOpenNotHFplusANDminusTH0BptxAND + fragment.hltPreHIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0PixelSingleTrack + fragment.hltL1DoubleMuOpenTH0L1Filtered + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1CastorMediumJet_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCastorMediumJetBptxAND + fragment.hltPreHIL1CastorMediumJet + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1CastorMediumJetAK4CaloJet20_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCastorMediumJetBptxAND + fragment.hltPreHIL1CastorMediumJetAK4CaloJet20 + fragment.HLTPuAK4CaloJetsUPCSequence + fragment.hltSinglePuAK4CaloJet20Eta5p150nsMultiFit + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HICastorMediumJetPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCastorMediumJetBptxAND + fragment.hltPreHICastorMediumJetPixelSingleTrack + fragment.hltL1CastorMediumJetFiltered0UPC + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotMinimumBiasHF2_AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotMinimumBiasHF2AND + fragment.hltPreHIUPCL1NotMinimumBiasHF2AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotMinimumBiasHF2AND + fragment.hltPreHIUPCL1NotMinimumBiasHF2ANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1ZdcOR_BptxAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZdcORBptxAND + fragment.hltPreHIUPCL1ZdcORBptxAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZdcORBptxAND + fragment.hltPreHIUPCL1ZdcORBptxANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1ZdcXOR_BptxAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZdcXORBptxAND + fragment.hltPreHIUPCL1ZdcXORBptxAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZdcXORBptxAND + fragment.hltPreHIUPCL1ZdcXORBptxANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotZdcOR_BptxAND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotZdcORBptxAND + fragment.hltPreHIUPCL1NotZdcORBptxAND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sNotZdcORBptxAND + fragment.hltPreHIUPCL1NotZdcORBptxANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTRecopixelvertexingSequenceForUPC + fragment.hltPixelCandsForUPC + fragment.hltSinglePixelTrackForUPC + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIZeroBias_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreHIZeroBias + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HICentralityVeto_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHICentralityVeto + fragment.HLTDoHILocalPixelSequence + fragment.hltPixelActivityFilterCentralityVeto + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Tech5_BPTX_PlusOnly_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1Tech5 + fragment.hltPreHIL1Tech5BPTXPlusOnly + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Tech6_BPTX_MinusOnly_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1Tech6 + fragment.hltPreHIL1Tech6BPTXMinusOnly + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Tech7_NoBPTX_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1Tech7 + fragment.hltPreHIL1Tech7NoBPTX + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1MinimumBiasHF1OR_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1OR + fragment.hltPreHIL1MinimumBiasHF1OR + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1MinimumBiasHF2OR_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF2OR + fragment.hltPreHIL1MinimumBiasHF2OR + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1MinimumBiasHF1AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIL1MinimumBiasHF1AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1MinimumBiasHF2AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF2AND + fragment.hltPreHIL1MinimumBiasHF2AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMinimumBiasHF1AND + fragment.hltPreHIL1MinimumBiasHF1ANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTPixelTrackingForHITrackTrigger + fragment.hltHISinglePixelTrackFilter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIZeroBiasPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreHIZeroBiasPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTPixelTrackingForHITrackTrigger + fragment.hltHISinglePixelTrackFilter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext70100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext70100MinimumumBiasHF1AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext70100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext70100MinimumumBiasHF1ANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTPixelTrackingForHITrackTrigger + fragment.hltHISinglePixelTrackFilter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext50100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext50100MinimumumBiasHF1AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext50100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext50100MinimumumBiasHF1ANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTPixelTrackingForHITrackTrigger + fragment.hltHISinglePixelTrackFilter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext30100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext30100MinimumumBiasHF1AND + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCentralityext30100MinimumumBiasHF1AND + fragment.hltPreHIL1Centralityext30100MinimumumBiasHF1ANDPixelSingleTrack + fragment.HLTDoHILocalPixelSequence + fragment.HLTPixelTrackingForHITrackTrigger + fragment.hltHISinglePixelTrackFilter + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIPhysics_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltPreHIPhysics + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_HIRandom_v2 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreHIRandom + fragment.HLTDoHIStripZeroSuppression + fragment.HLTEndSequence )
fragment.HLT_EcalCalibration_v3 = cms.Path( fragment.HLTBeginSequenceCalibration + fragment.hltPreEcalCalibration + fragment.hltEcalCalibrationRaw + fragment.HLTEndSequence )
fragment.HLT_HcalCalibration_v2 = cms.Path( fragment.HLTBeginSequenceCalibration + fragment.hltPreHcalCalibration + fragment.hltHcalCalibTypeFilter + fragment.hltHcalCalibrationRaw + fragment.HLTEndSequence )
fragment.AlCa_EcalPhiSymForHI_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0MinimumBiasHF1AND + fragment.hltPreAlCaEcalPhiSymForHI + fragment.HLTDoFullUnpackingEgammaEcal50nsMultiFitSequence + fragment.hltEcal50nsMultifitPhiSymFilter + fragment.HLTEndSequence )
fragment.AlCa_RPCMuonNoTriggersForHI_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenIorSingleMu12BptxAND + fragment.hltPreAlCaRPCMuonNoTriggersForHI + fragment.hltRPCMuonNoTriggersL1Filtered0ForHI + fragment.HLTMuonLocalRecoSequence + fragment.HLTEndSequence )
fragment.AlCa_RPCMuonNoHitsForHI_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenIorSingleMu12BptxAND + fragment.hltPreAlCaRPCMuonNoHitsForHI + fragment.HLTMuonLocalRecoSequence + fragment.hltRPCPointProducer + fragment.hltRPCFilter + fragment.HLTEndSequence )
fragment.AlCa_RPCMuonNormalisationForHI_v2 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenIorSingleMu12BptxAND + fragment.hltPreAlCaRPCMuonNormalisationForHI + fragment.hltRPCMuonNormaL1Filtered0ForHI + fragment.HLTMuonLocalRecoSequence + fragment.HLTEndSequence )
fragment.AlCa_LumiPixels_Random_v2 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreAlCaLumiPixelsRandom + fragment.hltFEDSelectorLumiPixels + fragment.HLTEndSequence )
fragment.AlCa_LumiPixels_ZeroBias_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreAlCaLumiPixelsZeroBias + fragment.hltFEDSelectorLumiPixels + fragment.HLTEndSequence )
fragment.HLTriggerFinalPath = cms.Path( fragment.hltGtStage2Digis + fragment.hltScalersRawToDigi + fragment.hltFEDSelector + fragment.hltTriggerSummaryAOD + fragment.hltTriggerSummaryRAW + fragment.hltBoolFalse )
fragment.HLTAnalyzerEndpath = cms.EndPath( fragment.hltGtStage2Digis + fragment.hltPreHLTAnalyzerEndpath + fragment.hltL1TGlobalSummary + fragment.hltTrigReport )


fragment.HLTSchedule = cms.Schedule( *(fragment.HLTriggerFirstPath, fragment.DST_Physics_v2, fragment.HLT_HIPuAK4CaloJet40_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet60_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet80_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet100_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet110_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet120_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet150_Eta5p1_v3, fragment.HLT_HIPuAK4CaloJet40_Eta5p1_Cent30_100_v3, fragment.HLT_HIPuAK4CaloJet60_Eta5p1_Cent30_100_v3, fragment.HLT_HIPuAK4CaloJet80_Eta5p1_Cent30_100_v3, fragment.HLT_HIPuAK4CaloJet100_Eta5p1_Cent30_100_v3, fragment.HLT_HIPuAK4CaloJet40_Eta5p1_Cent50_100_v3, fragment.HLT_HIPuAK4CaloJet60_Eta5p1_Cent50_100_v3, fragment.HLT_HIPuAK4CaloJet80_Eta5p1_Cent50_100_v3, fragment.HLT_HIPuAK4CaloJet100_Eta5p1_Cent50_100_v3, fragment.HLT_HIPuAK4CaloJet80_Jet35_Eta1p1_v3, fragment.HLT_HIPuAK4CaloJet80_Jet35_Eta0p7_v3, fragment.HLT_HIPuAK4CaloJet100_Jet35_Eta1p1_v3, fragment.HLT_HIPuAK4CaloJet100_Jet35_Eta0p7_v3, fragment.HLT_HIPuAK4CaloJet80_45_45_Eta2p1_v3, fragment.HLT_HIPuAK4CaloDJet60_Eta2p1_v3, fragment.HLT_HIPuAK4CaloDJet80_Eta2p1_v3, fragment.HLT_HIPuAK4CaloBJetCSV60_Eta2p1_v3, fragment.HLT_HIPuAK4CaloBJetCSV80_Eta2p1_v3, fragment.HLT_HIPuAK4CaloBJetSSV60_Eta2p1_v3, fragment.HLT_HIPuAK4CaloBJetSSV80_Eta2p1_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent30_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent50_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent30_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent50_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent30_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent50_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt50_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt70_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent30_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt60_Cent50_100_v3, fragment.HLT_HIDmesonHITrackingGlobal_Dpt20_Cent0_10_v2, fragment.HLT_HIDmesonHITrackingGlobal_Dpt30_Cent0_10_v2, fragment.HLT_HIDmesonHITrackingGlobal_Dpt40_Cent0_10_v2, fragment.HLT_HISinglePhoton10_Eta1p5_v3, fragment.HLT_HISinglePhoton15_Eta1p5_v3, fragment.HLT_HISinglePhoton20_Eta1p5_v3, fragment.HLT_HISinglePhoton30_Eta1p5_v3, fragment.HLT_HISinglePhoton40_Eta1p5_v3, fragment.HLT_HISinglePhoton50_Eta1p5_v3, fragment.HLT_HISinglePhoton60_Eta1p5_v3, fragment.HLT_HISinglePhoton10_Eta1p5_Cent50_100_v3, fragment.HLT_HISinglePhoton15_Eta1p5_Cent50_100_v3, fragment.HLT_HISinglePhoton20_Eta1p5_Cent50_100_v3, fragment.HLT_HISinglePhoton30_Eta1p5_Cent50_100_v3, fragment.HLT_HISinglePhoton40_Eta1p5_Cent50_100_v3, fragment.HLT_HISinglePhoton10_Eta1p5_Cent30_100_v3, fragment.HLT_HISinglePhoton15_Eta1p5_Cent30_100_v3, fragment.HLT_HISinglePhoton20_Eta1p5_Cent30_100_v3, fragment.HLT_HISinglePhoton30_Eta1p5_Cent30_100_v3, fragment.HLT_HISinglePhoton40_Eta1p5_Cent30_100_v3, fragment.HLT_HISinglePhoton40_Eta2p1_v3, fragment.HLT_HISinglePhoton10_Eta3p1_v3, fragment.HLT_HISinglePhoton15_Eta3p1_v3, fragment.HLT_HISinglePhoton20_Eta3p1_v3, fragment.HLT_HISinglePhoton30_Eta3p1_v3, fragment.HLT_HISinglePhoton40_Eta3p1_v3, fragment.HLT_HISinglePhoton50_Eta3p1_v3, fragment.HLT_HISinglePhoton60_Eta3p1_v3, fragment.HLT_HISinglePhoton10_Eta3p1_Cent50_100_v3, fragment.HLT_HISinglePhoton15_Eta3p1_Cent50_100_v3, fragment.HLT_HISinglePhoton20_Eta3p1_Cent50_100_v3, fragment.HLT_HISinglePhoton30_Eta3p1_Cent50_100_v3, fragment.HLT_HISinglePhoton40_Eta3p1_Cent50_100_v3, fragment.HLT_HISinglePhoton10_Eta3p1_Cent30_100_v3, fragment.HLT_HISinglePhoton15_Eta3p1_Cent30_100_v3, fragment.HLT_HISinglePhoton20_Eta3p1_Cent30_100_v3, fragment.HLT_HISinglePhoton30_Eta3p1_Cent30_100_v3, fragment.HLT_HISinglePhoton40_Eta3p1_Cent30_100_v3, fragment.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_v3, fragment.HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v3, fragment.HLT_HIDoublePhoton15_Eta2p1_Mass50_1000_R9Cut_v3, fragment.HLT_HIDoublePhoton15_Eta2p5_Mass50_1000_R9SigmaHECut_v3, fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet40Eta2p1_v3, fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet60Eta2p1_v3, fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet80Eta2p1_v3, fragment.HLT_HIL2Mu3Eta2p5_PuAK4CaloJet100Eta2p1_v3, fragment.HLT_HIL2Mu3Eta2p5_HIPhoton10Eta1p5_v3, fragment.HLT_HIL2Mu3Eta2p5_HIPhoton15Eta1p5_v3, fragment.HLT_HIL2Mu3Eta2p5_HIPhoton20Eta1p5_v3, fragment.HLT_HIL2Mu3Eta2p5_HIPhoton30Eta1p5_v3, fragment.HLT_HIL2Mu3Eta2p5_HIPhoton40Eta1p5_v3, fragment.HLT_HIUCC100_v3, fragment.HLT_HIUCC020_v3, fragment.HLT_HIQ2Bottom005_Centrality1030_v3, fragment.HLT_HIQ2Top005_Centrality1030_v3, fragment.HLT_HIQ2Bottom005_Centrality3050_v3, fragment.HLT_HIQ2Top005_Centrality3050_v3, fragment.HLT_HIQ2Bottom005_Centrality5070_v3, fragment.HLT_HIQ2Top005_Centrality5070_v3, fragment.HLT_HIFullTrack12_L1MinimumBiasHF1_AND_v3, fragment.HLT_HIFullTrack12_L1Centrality010_v3, fragment.HLT_HIFullTrack12_L1Centrality30100_v3, fragment.HLT_HIFullTrack18_L1MinimumBiasHF1_AND_v3, fragment.HLT_HIFullTrack18_L1Centrality010_v3, fragment.HLT_HIFullTrack18_L1Centrality30100_v3, fragment.HLT_HIFullTrack24_v3, fragment.HLT_HIFullTrack24_L1Centrality30100_v3, fragment.HLT_HIFullTrack34_v3, fragment.HLT_HIFullTrack34_L1Centrality30100_v3, fragment.HLT_HIFullTrack45_v3, fragment.HLT_HIFullTrack45_L1Centrality30100_v3, fragment.HLT_HIL1DoubleMu0_v2, fragment.HLT_HIL1DoubleMu0_2HF_v2, fragment.HLT_HIL1DoubleMu0_2HF0_v2, fragment.HLT_HIL1DoubleMu10_v2, fragment.HLT_HIL2DoubleMu0_NHitQ_v3, fragment.HLT_HIL2DoubleMu0_NHitQ_2HF_v2, fragment.HLT_HIL2DoubleMu0_NHitQ_2HF0_v2, fragment.HLT_HIL2Mu3_NHitQ10_2HF_v2, fragment.HLT_HIL2Mu3_NHitQ10_2HF0_v2, fragment.HLT_HIL3Mu3_NHitQ15_2HF_v2, fragment.HLT_HIL3Mu3_NHitQ15_2HF0_v2, fragment.HLT_HIL2Mu5_NHitQ10_2HF_v2, fragment.HLT_HIL2Mu5_NHitQ10_2HF0_v2, fragment.HLT_HIL3Mu5_NHitQ15_2HF_v2, fragment.HLT_HIL3Mu5_NHitQ15_2HF0_v2, fragment.HLT_HIL2Mu7_NHitQ10_2HF_v2, fragment.HLT_HIL2Mu7_NHitQ10_2HF0_v2, fragment.HLT_HIL3Mu7_NHitQ15_2HF_v2, fragment.HLT_HIL3Mu7_NHitQ15_2HF0_v2, fragment.HLT_HIL2Mu15_v3, fragment.HLT_HIL2Mu15_2HF_v2, fragment.HLT_HIL2Mu15_2HF0_v2, fragment.HLT_HIL3Mu15_v2, fragment.HLT_HIL3Mu15_2HF_v2, fragment.HLT_HIL3Mu15_2HF0_v2, fragment.HLT_HIL2Mu20_v2, fragment.HLT_HIL2Mu20_2HF_v2, fragment.HLT_HIL2Mu20_2HF0_v2, fragment.HLT_HIL3Mu20_v2, fragment.HLT_HIL3Mu20_2HF_v2, fragment.HLT_HIL3Mu20_2HF0_v2, fragment.HLT_HIL1DoubleMu0_2HF_Cent30100_v2, fragment.HLT_HIL1DoubleMu0_2HF0_Cent30100_v2, fragment.HLT_HIL2DoubleMu0_2HF_Cent30100_NHitQ_v2, fragment.HLT_HIL1DoubleMu0_Cent30_v2, fragment.HLT_HIL2DoubleMu0_2HF0_Cent30100_NHitQ_v2, fragment.HLT_HIL2DoubleMu0_Cent30_OS_NHitQ_v2, fragment.HLT_HIL2DoubleMu0_Cent30_NHitQ_v2, fragment.HLT_HIL3DoubleMu0_Cent30_v2, fragment.HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v2, fragment.HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v2, fragment.HLT_HIL3DoubleMu0_OS_m2p5to4p5_v2, fragment.HLT_HIL3DoubleMu0_OS_m7to14_v2, fragment.HLT_HIUPCL1SingleMuOpenNotHF2_v2, fragment.HLT_HIUPCSingleMuNotHF2Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1DoubleMuOpenNotHF2_v2, fragment.HLT_HIUPCDoubleMuNotHF2Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1SingleEG2NotHF2_v2, fragment.HLT_HIUPCSingleEG2NotHF2Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1DoubleEG2NotHF2_v2, fragment.HLT_HIUPCDoubleEG2NotHF2Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1SingleEG5NotHF2_v2, fragment.HLT_HIUPCSingleEG5NotHF2Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1DoubleMuOpenNotHF1_v2, fragment.HLT_HIUPCDoubleMuNotHF1Pixel_SingleTrack_v2, fragment.HLT_HIUPCL1DoubleEG2NotZDCAND_v2, fragment.HLT_HIUPCL1DoubleEG2NotZDCANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1DoubleMuOpenNotZDCAND_v2, fragment.HLT_HIUPCL1DoubleMuOpenNotZDCANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1EG2NotZDCAND_v2, fragment.HLT_HIUPCEG2NotZDCANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1MuOpenNotZDCAND_v2, fragment.HLT_HIUPCL1MuOpenNotZDCANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1NotHFplusANDminusTH0BptxAND_v2, fragment.HLT_HIUPCL1NotHFplusANDminusTH0BptxANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0_v3, fragment.HLT_HIUPCL1NotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3, fragment.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0_v3, fragment.HLT_HIUPCL1DoubleMuOpenNotHFMinimumbiasHFplusANDminustTH0Pixel_SingleTrack_v3, fragment.HLT_HIL1CastorMediumJet_v2, fragment.HLT_HIL1CastorMediumJetAK4CaloJet20_v3, fragment.HLT_HICastorMediumJetPixel_SingleTrack_v2, fragment.HLT_HIUPCL1NotMinimumBiasHF2_AND_v2, fragment.HLT_HIUPCL1NotMinimumBiasHF2_ANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1ZdcOR_BptxAND_v2, fragment.HLT_HIUPCL1ZdcOR_BptxANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1ZdcXOR_BptxAND_v2, fragment.HLT_HIUPCL1ZdcXOR_BptxANDPixel_SingleTrack_v2, fragment.HLT_HIUPCL1NotZdcOR_BptxAND_v2, fragment.HLT_HIUPCL1NotZdcOR_BptxANDPixel_SingleTrack_v2, fragment.HLT_HIZeroBias_v2, fragment.HLT_HICentralityVeto_v2, fragment.HLT_HIL1Tech5_BPTX_PlusOnly_v2, fragment.HLT_HIL1Tech6_BPTX_MinusOnly_v2, fragment.HLT_HIL1Tech7_NoBPTX_v2, fragment.HLT_HIL1MinimumBiasHF1OR_v2, fragment.HLT_HIL1MinimumBiasHF2OR_v2, fragment.HLT_HIL1MinimumBiasHF1AND_v2, fragment.HLT_HIL1MinimumBiasHF2AND_v2, fragment.HLT_HIL1MinimumBiasHF1ANDPixel_SingleTrack_v2, fragment.HLT_HIZeroBiasPixel_SingleTrack_v2, fragment.HLT_HIL1Centralityext70100MinimumumBiasHF1AND_v2, fragment.HLT_HIL1Centralityext70100MinimumumBiasHF1ANDPixel_SingleTrack_v2, fragment.HLT_HIL1Centralityext50100MinimumumBiasHF1AND_v2, fragment.HLT_HIL1Centralityext50100MinimumumBiasHF1ANDPixel_SingleTrack_v2, fragment.HLT_HIL1Centralityext30100MinimumumBiasHF1AND_v2, fragment.HLT_HIL1Centralityext30100MinimumumBiasHF1ANDPixel_SingleTrack_v2, fragment.HLT_HIPhysics_v2, fragment.HLT_HIRandom_v2, fragment.HLT_EcalCalibration_v3, fragment.HLT_HcalCalibration_v2, fragment.AlCa_EcalPhiSymForHI_v3, fragment.AlCa_RPCMuonNoTriggersForHI_v2, fragment.AlCa_RPCMuonNoHitsForHI_v2, fragment.AlCa_RPCMuonNormalisationForHI_v2, fragment.AlCa_LumiPixels_Random_v2, fragment.AlCa_LumiPixels_ZeroBias_v3, fragment.HLTriggerFinalPath, fragment.HLTAnalyzerEndpath ))


# dummyfy hltGetConditions in cff's
if 'hltGetConditions' in fragment.__dict__ and 'HLTriggerFirstPath' in fragment.__dict__ :
    fragment.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    fragment.HLTriggerFirstPath.replace(fragment.hltGetConditions,fragment.hltDummyConditions)

# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment,"HIon")

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
fragment = customizeHLTforCMSSW(fragment,"HIon")

