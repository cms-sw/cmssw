# hltGetConfiguration /dev/CMSSW_15_0_0/Special --cff --data --type Special

# /dev/CMSSW_15_0_0/Special/V6 (CMSSW_15_0_0)

import FWCore.ParameterSet.Config as cms

fragment = cms.ProcessFragment( "HLT" )

fragment.load("Configuration.StandardSequences.Accelerators_cff")

fragment.HLTConfigVersion = cms.PSet(
  tableName = cms.string("/dev/CMSSW_15_0_0/Special/V6")
)

fragment.HLTGroupedCkfTrajectoryBuilderP5 = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTCkfBaseTrajectoryFilterP5" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  seedAs5DHit = cms.bool( False ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 1 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESChi2MeasurementEstimatorForP5" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True )
)
fragment.HLTCkfBaseTrajectoryFilterP5 = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minHitsAtHighEta = cms.int32( 5 ),
  minPt = cms.double( 0.5 ),
  maxConsecLostHits = cms.int32( 3 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  highEtaSwitch = cms.double( 5.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 4 )
)
fragment.HLTCkfBaseTrajectoryFilter_block = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minHitsAtHighEta = cms.int32( 5 ),
  minPt = cms.double( 0.9 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  highEtaSwitch = cms.double( 5.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 )
)
fragment.HLTIter4PSetTrajectoryBuilderIT = cms.PSet( 
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter4PSetTrajectoryFilterIT" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  maxCand = cms.int32( 1 ),
  alwaysUseInvalidHits = cms.bool( False ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  intermediateCleaning = cms.bool( True ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter0GroupedCkfTrajectoryBuilderIT = cms.PSet( 
  keepOriginalIfRebuildFails = cms.bool( False ),
  lockHits = cms.bool( True ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  intermediateCleaning = cms.bool( True ),
  bestHitOnly = cms.bool( True ),
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( False ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0PSetTrajectoryFilterIT" ) ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter4PSetTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 6 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.3 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 0 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPvClusterComparerForIT = cms.PSet( 
  track_chi2_max = cms.double( 20.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 ),
  track_pt_min = cms.double( 1.0 )
)
fragment.HLTPSetMuonCkfTrajectoryBuilder = cms.PSet( 
  rescaleErrorIfFail = cms.double( 1.0 ),
  ComponentType = cms.string( "MuonCkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryFilter" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  maxCand = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  intermediateCleaning = cms.bool( False ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  updator = cms.string( "hltESPKFUpdator" ),
  deltaEta = cms.double( -1.0 ),
  useSeedLayer = cms.bool( False ),
  deltaPhi = cms.double( -1.0 ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter0HighPtTkMuPSetTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.3 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPvClusterComparerForBTag = cms.PSet( 
  track_chi2_max = cms.double( 20.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 ),
  track_pt_min = cms.double( 0.1 )
)
fragment.HLTIter2GroupedCkfTrajectoryBuilderIT = cms.PSet( 
  keepOriginalIfRebuildFails = cms.bool( False ),
  lockHits = cms.bool( True ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter2PSetTrajectoryFilterIT" ) ),
  maxCand = cms.int32( 2 ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  intermediateCleaning = cms.bool( True ),
  bestHitOnly = cms.bool( True ),
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( False ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter2PSetTrajectoryFilterIT" ) ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTSiStripClusterChargeCutTight = cms.PSet(  value = cms.double( 1945.0 ) )
fragment.HLTPSetMuonTrackingRegionBuilder8356 = cms.PSet( 
  Rescale_Dz = cms.double( 3.0 ),
  Pt_fixed = cms.bool( False ),
  Eta_fixed = cms.bool( False ),
  Eta_min = cms.double( 0.1 ),
  DeltaZ = cms.double( 15.9 ),
  maxRegions = cms.int32( 2 ),
  EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
  UseVertex = cms.bool( False ),
  Z_fixed = cms.bool( True ),
  PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
  PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
  Rescale_phi = cms.double( 3.0 ),
  DeltaEta = cms.double( 0.2 ),
  precise = cms.bool( True ),
  OnDemand = cms.int32( -1 ),
  EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
  MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
  vertexCollection = cms.InputTag( "pixelVertices" ),
  Pt_min = cms.double( 1.5 ),
  beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
  Phi_fixed = cms.bool( False ),
  DeltaR = cms.double( 0.2 ),
  input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
  DeltaPhi = cms.double( 0.2 ),
  Phi_min = cms.double( 0.1 ),
  Rescale_eta = cms.double( 3.0 )
)
fragment.HLTIter2PSetTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 0.3 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetMuTrackJpsiTrajectoryBuilder = cms.PSet( 
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuTrackJpsiTrajectoryFilter" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  maxCand = cms.int32( 1 ),
  alwaysUseInvalidHits = cms.bool( False ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  intermediateCleaning = cms.bool( True ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetTrajectoryBuilderForGsfElectrons = cms.PSet( 
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 90.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "hltESPBwdElectronPropagator" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTrajectoryFilterForElectrons" ) ),
  propagatorAlong = cms.string( "hltESPFwdElectronPropagator" ),
  maxCand = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator2000" ),
  intermediateCleaning = cms.bool( False ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
fragment.HLTPSetMuonCkfTrajectoryFilter = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.9 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( -1 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 0.2 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTSeedFromProtoTracks = cms.PSet( 
  TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "ParabolicMf" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  MinOneOverPtError = cms.double( 1.0 )
)
fragment.HLTPSetMuTrackJpsiTrajectoryFilter = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 10.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 8 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTIter0PSetTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 0.3 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTSeedFromConsecutiveHitsCreator = cms.PSet( 
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  propagator = cms.string( "PropagatorWithMaterial" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  MinOneOverPtError = cms.double( 1.0 )
)
fragment.HLTSiStripClusterChargeCutForHI = cms.PSet(  value = cms.double( 2069.0 ) )
fragment.HLTPSetTrajectoryFilterForElectrons = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 2.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( -1 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( -1 ),
  maxLostHitsFraction = cms.double( 999.0 ),
  maxLostHits = cms.int32( 1 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTIter0HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
  lostHitPenalty = cms.double( 30.0 ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0HighPtTkMuPSetTrajectoryFilterIT" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  maxCand = cms.int32( 4 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  intermediateCleaning = cms.bool( True ),
  updator = cms.string( "hltESPKFUpdator" ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter1GroupedCkfTrajectoryBuilderIT = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter1PSetTrajectoryFilterIT" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 2 ),
  alwaysUseInvalidHits = cms.bool( False ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 5.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter1PSetTrajectoryFilterIT" ) )
)
fragment.HLTIter0IterL3MuonPSetGroupedCkfTrajectoryBuilderIT = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( True ),
  lostHitPenalty = cms.double( 1.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3MuonGroupedCkfTrajectoryFilterIT" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  minNrOfHitsForRebuild = cms.int32( 2 ),
  maxCand = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3MuonGroupedCkfTrajectoryFilterIT" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 1000.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter0IterL3FromL1MuonGroupedCkfTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.9 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 10.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTIter0IterL3FromL1MuonPSetGroupedCkfTrajectoryBuilderIT = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( True ),
  lostHitPenalty = cms.double( 1.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3FromL1MuonGroupedCkfTrajectoryFilterIT" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  minNrOfHitsForRebuild = cms.int32( 2 ),
  maxCand = cms.int32( 5 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3FromL1MuonGroupedCkfTrajectoryFilterIT" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 1000.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTIter0IterL3MuonGroupedCkfTrajectoryFilterIT = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.9 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 10.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetCkfBaseTrajectoryFilter_block = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 0.9 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTSiStripClusterChargeCutLoose = cms.PSet(  value = cms.double( 1620.0 ) )
fragment.HLTPSetInitialStepTrajectoryFilterShapePreSplittingPPOnAA = cms.PSet( 
  ComponentType = cms.string( "StripSubClusterShapeTrajectoryFilter" ),
  subclusterCutSN = cms.double( 12.0 ),
  trimMaxADC = cms.double( 30.0 ),
  seedCutMIPs = cms.double( 0.35 ),
  subclusterCutMIPs = cms.double( 0.45 ),
  subclusterWindow = cms.double( 0.7 ),
  maxNSat = cms.uint32( 3 ),
  trimMaxFracNeigh = cms.double( 0.25 ),
  maxTrimmedSizeDiffNeg = cms.double( 1.0 ),
  seedCutSN = cms.double( 7.0 ),
  layerMask = cms.PSet( 
    TOB = cms.bool( False ),
    TIB = cms.vuint32( 1, 2 ),
    TID = cms.vuint32( 1, 2 ),
    TEC = cms.bool( False )
  ),
  maxTrimmedSizeDiffPos = cms.double( 0.7 ),
  trimMaxFracTotal = cms.double( 0.15 )
)
fragment.HLTPSetInitialStepTrajectoryFilterBasePreSplittingForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  maxCCCLostHits = cms.int32( 0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minPt = cms.double( 1.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetInitialStepTrajectoryBuilderPreSplittingForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterPreSplittingForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetInitialStepTrajectoryFilterPreSplittingForFullTrackingPPOnAA = cms.PSet( 
  ComponentType = cms.string( "CompositeTrajectoryFilter" ),
  filters = cms.VPSet( 
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterBasePreSplittingForFullTrackingPPOnAA" )    ),
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterShapePreSplittingPPOnAA" )    )
  )
)
fragment.HLTPSetInitialStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  maxCCCLostHits = cms.int32( 0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minPt = cms.double( 1.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetInitialStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 1 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetLowPtQuadStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 1.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetLowPtQuadStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetLowPtQuadStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 4 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPLowPtQuadStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetHighPtTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 1.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetHighPtTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetHighPtTripletStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPHighPtTripletStepChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetLowPtTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 2.8 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetLowPtTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetLowPtTripletStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 4 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPLowPtTripletStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetDetachedQuadStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetDetachedTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPixelPairStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPixelPairStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( False ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPPixelPairStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelPairStepTrajectoryFilterInOutForFullTrackingPPOnAA" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetMixedTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 1.4 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPixelLessStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 0 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetPixelLessStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPixelLessStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 4 ),
  maxCand = cms.int32( 2 ),
  alwaysUseInvalidHits = cms.bool( False ),
  estimator = cms.string( "hltESPPixelLessStepChi2ChargeMeasurementEstimator16" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetTobTecStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 5 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 0 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetTobTecStepInOutTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 1 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 0 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetTobTecStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( False ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTobTecStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 4 ),
  maxCand = cms.int32( 2 ),
  alwaysUseInvalidHits = cms.bool( False ),
  estimator = cms.string( "hltESPTobTecStepChi2ChargeMeasurementEstimator16" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetTobTecStepInOutTrajectoryFilterForFullTrackingPPOnAA" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetJetCoreStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 9999 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetJetCoreStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetJetCoreStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 50 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetPixelPairStepTrajectoryFilterInOutForFullTrackingPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 1 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 5.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetMixedTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMixedTripletStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialForMixedStep" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 2 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPMixedTripletStepChi2ChargeMeasurementEstimator16" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetDetachedQuadStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedQuadStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPDetachedQuadStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetDetachedTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetDetachedTripletStepTrajectoryFilterForFullTrackingPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPDetachedTripletStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetInitialStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 4 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  maxCCCLostHits = cms.int32( 0 ),
  nSigmaMinPt = cms.double( 5.0 ),
  minPt = cms.double( 3.0 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetInitialStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( True ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterForDmesonPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 1 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetLowPtQuadStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 2.8 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetLowPtQuadStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetLowPtQuadStepTrajectoryFilterForDmesonPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 4 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPLowPtQuadStepChi2ChargeMeasurementEstimator9" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.HLTPSetHighPtTripletStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
  minimumNumberOfHits = cms.int32( 3 ),
  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
  seedExtension = cms.int32( 0 ),
  chargeSignificance = cms.double( -1.0 ),
  pixelSeedExtension = cms.bool( False ),
  strictSeedExtension = cms.bool( False ),
  nSigmaMinPt = cms.double( 5.0 ),
  maxCCCLostHits = cms.int32( 0 ),
  minPt = cms.double( 3.5 ),
  maxConsecLostHits = cms.int32( 1 ),
  extraNumberOfHitsBeforeTheFirstLoop = cms.int32( 4 ),
  constantValueForLostHitsFractionFilter = cms.double( 2.0 ),
  seedPairPenalty = cms.int32( 0 ),
  maxNumberOfHits = cms.int32( 100 ),
  minNumberOfHitsForLoopers = cms.int32( 13 ),
  minGoodStripCharge = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  minNumberOfHitsPerLoop = cms.int32( 4 ),
  minHitsMinPt = cms.int32( 3 ),
  maxLostHitsFraction = cms.double( 0.1 ),
  maxLostHits = cms.int32( 999 ),
  highEtaSwitch = cms.double( 5.0 ),
  minHitsAtHighEta = cms.int32( 5 )
)
fragment.HLTPSetHighPtTripletStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
  useSameTrajFilter = cms.bool( True ),
  ComponentType = cms.string( "GroupedCkfTrajectoryBuilder" ),
  keepOriginalIfRebuildFails = cms.bool( False ),
  lostHitPenalty = cms.double( 30.0 ),
  lockHits = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  maxDPhiForLooperReconstruction = cms.double( 2.0 ),
  maxPtForLooperReconstruction = cms.double( 0.0 ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetHighPtTripletStepTrajectoryFilterForDmesonPPOnAA" ) ),
  propagatorAlong = cms.string( "PropagatorWithMaterialParabolicMf" ),
  minNrOfHitsForRebuild = cms.int32( 5 ),
  maxCand = cms.int32( 3 ),
  alwaysUseInvalidHits = cms.bool( True ),
  estimator = cms.string( "hltESPHighPtTripletStepChi2ChargeMeasurementEstimator30" ),
  inOutTrajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfBaseTrajectoryFilter_block" ) ),
  intermediateCleaning = cms.bool( True ),
  foundHitBonus = cms.double( 10.0 ),
  updator = cms.string( "hltESPKFUpdator" ),
  bestHitOnly = cms.bool( True ),
  seedAs5DHit = cms.bool( False )
)
fragment.streams = cms.PSet( 
  ALCALumiPixelsCountsExpress = cms.vstring( 'AlCaLumiPixelsCountsExpress' ),
  ALCALumiPixelsCountsGated = cms.vstring( 'AlCaLumiPixelsCountsGated' ),
  ALCALumiPixelsCountsPrompt = cms.vstring( 'AlCaLumiPixelsCountsPrompt' ),
  ALCALumiPixelsCountsPromptHighRate0 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate0' ),
  ALCALumiPixelsCountsPromptHighRate1 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate1' ),
  ALCALumiPixelsCountsPromptHighRate2 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate2' ),
  ALCALumiPixelsCountsPromptHighRate3 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate3' ),
  ALCALumiPixelsCountsPromptHighRate4 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate4' ),
  ALCALumiPixelsCountsPromptHighRate5 = cms.vstring( 'AlCaLumiPixelsCountsPromptHighRate5' ),
  ALCAP0 = cms.vstring( 'AlCaP0' ),
  ALCAPHISYM = cms.vstring( 'AlCaPhiSym' ),
  ALCAPPSExpress = cms.vstring( 'AlCaPPSExpress' ),
  ALCAPPSPrompt = cms.vstring( 'AlCaPPSPrompt' ),
  Calibration = cms.vstring( 'TestEnablesEcalHcal' ),
  DQM = cms.vstring( 'OnlineMonitor' ),
  DQMCalibration = cms.vstring( 'TestEnablesEcalHcalDQM' ),
  DQMEventDisplay = cms.vstring( 'EventDisplay' ),
  DQMGPUvsCPU = cms.vstring( 'DQMGPUvsCPU' ),
  DQMOnlineBeamspot = cms.vstring( 'DQMOnlineBeamspot' ),
  DQMPPSRandom = cms.vstring( 'DQMPPSRandom' ),
  EcalCalibration = cms.vstring( 'EcalLaser' ),
  Express = cms.vstring( 'ExpressPhysics' ),
  ExpressAlignment = cms.vstring( 'ExpressAlignment' ),
  ExpressCosmics = cms.vstring( 'ExpressCosmics' ),
  HLTMonitor = cms.vstring( 'HLTMonitor' ),
  NanoDST = cms.vstring( 'L1Accept' ),
  PhysicsCommissioning = cms.vstring( 'Commissioning',
    'Cosmics',
    'HLTPhysics',
    'HcalNZS',
    'MinimumBias',
    'MuonShower',
    'NoBPTX',
    'ZeroBias' ),
  PhysicsSpecialHLTPhysics0 = cms.vstring( 'SpecialHLTPhysics0' ),
  PhysicsSpecialHLTPhysics1 = cms.vstring( 'SpecialHLTPhysics1' ),
  PhysicsSpecialHLTPhysics10 = cms.vstring( 'SpecialHLTPhysics10' ),
  PhysicsSpecialHLTPhysics11 = cms.vstring( 'SpecialHLTPhysics11' ),
  PhysicsSpecialHLTPhysics12 = cms.vstring( 'SpecialHLTPhysics12' ),
  PhysicsSpecialHLTPhysics13 = cms.vstring( 'SpecialHLTPhysics13' ),
  PhysicsSpecialHLTPhysics14 = cms.vstring( 'SpecialHLTPhysics14' ),
  PhysicsSpecialHLTPhysics15 = cms.vstring( 'SpecialHLTPhysics15' ),
  PhysicsSpecialHLTPhysics16 = cms.vstring( 'SpecialHLTPhysics16' ),
  PhysicsSpecialHLTPhysics17 = cms.vstring( 'SpecialHLTPhysics17' ),
  PhysicsSpecialHLTPhysics18 = cms.vstring( 'SpecialHLTPhysics18' ),
  PhysicsSpecialHLTPhysics19 = cms.vstring( 'SpecialHLTPhysics19' ),
  PhysicsSpecialHLTPhysics2 = cms.vstring( 'SpecialHLTPhysics2' ),
  PhysicsSpecialHLTPhysics3 = cms.vstring( 'SpecialHLTPhysics3' ),
  PhysicsSpecialHLTPhysics4 = cms.vstring( 'SpecialHLTPhysics4' ),
  PhysicsSpecialHLTPhysics5 = cms.vstring( 'SpecialHLTPhysics5' ),
  PhysicsSpecialHLTPhysics6 = cms.vstring( 'SpecialHLTPhysics6' ),
  PhysicsSpecialHLTPhysics7 = cms.vstring( 'SpecialHLTPhysics7' ),
  PhysicsSpecialHLTPhysics8 = cms.vstring( 'SpecialHLTPhysics8' ),
  PhysicsSpecialHLTPhysics9 = cms.vstring( 'SpecialHLTPhysics9' ),
  PhysicsSpecialRandom0 = cms.vstring( 'SpecialRandom0',
    'SpecialRandom1' ),
  PhysicsSpecialRandom1 = cms.vstring( 'SpecialRandom2',
    'SpecialRandom3' ),
  PhysicsSpecialRandom2 = cms.vstring( 'SpecialRandom4',
    'SpecialRandom5' ),
  PhysicsSpecialRandom3 = cms.vstring( 'SpecialRandom6',
    'SpecialRandom7' ),
  PhysicsSpecialRandom4 = cms.vstring( 'SpecialRandom8',
    'SpecialRandom9' ),
  PhysicsSpecialRandom5 = cms.vstring( 'SpecialRandom10',
    'SpecialRandom11' ),
  PhysicsSpecialRandom6 = cms.vstring( 'SpecialRandom12',
    'SpecialRandom13' ),
  PhysicsSpecialRandom7 = cms.vstring( 'SpecialRandom14',
    'SpecialRandom15' ),
  PhysicsSpecialRandom8 = cms.vstring( 'SpecialRandom16',
    'SpecialRandom17' ),
  PhysicsSpecialRandom9 = cms.vstring( 'SpecialRandom18',
    'SpecialRandom19' ),
  PhysicsSpecialZeroBias0 = cms.vstring( 'SpecialZeroBias0',
    'SpecialZeroBias1' ),
  PhysicsSpecialZeroBias1 = cms.vstring( 'SpecialZeroBias2',
    'SpecialZeroBias3' ),
  PhysicsSpecialZeroBias10 = cms.vstring( 'SpecialZeroBias20',
    'SpecialZeroBias21' ),
  PhysicsSpecialZeroBias11 = cms.vstring( 'SpecialZeroBias22',
    'SpecialZeroBias23' ),
  PhysicsSpecialZeroBias12 = cms.vstring( 'SpecialZeroBias24',
    'SpecialZeroBias25' ),
  PhysicsSpecialZeroBias13 = cms.vstring( 'SpecialZeroBias26',
    'SpecialZeroBias27' ),
  PhysicsSpecialZeroBias14 = cms.vstring( 'SpecialZeroBias28',
    'SpecialZeroBias29' ),
  PhysicsSpecialZeroBias15 = cms.vstring( 'SpecialZeroBias30',
    'SpecialZeroBias31' ),
  PhysicsSpecialZeroBias2 = cms.vstring( 'SpecialZeroBias4',
    'SpecialZeroBias5' ),
  PhysicsSpecialZeroBias3 = cms.vstring( 'SpecialZeroBias6',
    'SpecialZeroBias7' ),
  PhysicsSpecialZeroBias4 = cms.vstring( 'SpecialZeroBias8',
    'SpecialZeroBias9' ),
  PhysicsSpecialZeroBias5 = cms.vstring( 'SpecialZeroBias10',
    'SpecialZeroBias11' ),
  PhysicsSpecialZeroBias6 = cms.vstring( 'SpecialZeroBias12',
    'SpecialZeroBias13' ),
  PhysicsSpecialZeroBias7 = cms.vstring( 'SpecialZeroBias14',
    'SpecialZeroBias15' ),
  PhysicsSpecialZeroBias8 = cms.vstring( 'SpecialZeroBias16',
    'SpecialZeroBias17' ),
  PhysicsSpecialZeroBias9 = cms.vstring( 'SpecialZeroBias18',
    'SpecialZeroBias19' ),
  PhysicsVRRandom0 = cms.vstring( 'VRRandom0',
    'VRRandom1' ),
  PhysicsVRRandom1 = cms.vstring( 'VRRandom2',
    'VRRandom3' ),
  PhysicsVRRandom2 = cms.vstring( 'VRRandom4',
    'VRRandom5' ),
  PhysicsVRRandom3 = cms.vstring( 'VRRandom6',
    'VRRandom7' ),
  PhysicsVRRandom4 = cms.vstring( 'VRRandom8',
    'VRRandom9' ),
  PhysicsVRRandom5 = cms.vstring( 'VRRandom10',
    'VRRandom11' ),
  PhysicsVRRandom6 = cms.vstring( 'VRRandom12',
    'VRRandom13' ),
  PhysicsVRRandom7 = cms.vstring( 'VRRandom14',
    'VRRandom15' ),
  RPCMON = cms.vstring( 'RPCMonitor' )
)
fragment.datasets = cms.PSet( 
  AlCaLumiPixelsCountsExpress = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_Random_v10' ),
  AlCaLumiPixelsCountsGated = cms.vstring( 'AlCa_LumiPixelsCounts_ZeroBiasGated_v5' ),
  AlCaLumiPixelsCountsPrompt = cms.vstring( 'AlCa_LumiPixelsCounts_Random_v10',
    'AlCa_LumiPixelsCounts_ZeroBias_v12' ),
  AlCaLumiPixelsCountsPromptHighRate0 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaLumiPixelsCountsPromptHighRate1 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaLumiPixelsCountsPromptHighRate2 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaLumiPixelsCountsPromptHighRate3 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaLumiPixelsCountsPromptHighRate4 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaLumiPixelsCountsPromptHighRate5 = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
    'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' ),
  AlCaP0 = cms.vstring( 'AlCa_EcalEtaEBonly_v25',
    'AlCa_EcalEtaEEonly_v25',
    'AlCa_EcalPi0EBonly_v25',
    'AlCa_EcalPi0EEonly_v25' ),
  AlCaPPSExpress = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
    'HLT_PPSMaxTracksPerRP4_v9' ),
  AlCaPPSPrompt = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
    'HLT_PPSMaxTracksPerRP4_v9' ),
  AlCaPhiSym = cms.vstring( 'AlCa_EcalPhiSym_v20' ),
  Commissioning = cms.vstring( 'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v14',
    'HLT_L1SingleMuCosmics_EMTF_v4' ),
  Cosmics = cms.vstring( 'HLT_L1SingleMu3_v5',
    'HLT_L1SingleMu5_v5',
    'HLT_L1SingleMu7_v5',
    'HLT_L1SingleMuCosmics_v8',
    'HLT_L1SingleMuOpen_DT_v6',
    'HLT_L1SingleMuOpen_v6' ),
  DQMGPUvsCPU = cms.vstring( 'DQM_EcalReconstruction_v12',
    'DQM_HcalReconstruction_v10',
    'DQM_PixelReconstruction_v12' ),
  DQMOnlineBeamspot = cms.vstring( 'HLT_HT300_Beamspot_v23',
    'HLT_HT60_Beamspot_v22',
    'HLT_ZeroBias_Beamspot_v16' ),
  DQMPPSRandom = cms.vstring( 'HLT_PPSRandom_v1' ),
  EcalLaser = cms.vstring( 'HLT_EcalCalibration_v4' ),
  EventDisplay = cms.vstring( 'HLT_BptxOR_v6',
    'HLT_L1ETM120_v4',
    'HLT_L1ETM150_v4',
    'HLT_L1HTT120er_v4',
    'HLT_L1HTT160er_v4',
    'HLT_L1HTT200er_v4',
    'HLT_L1HTT255er_v4',
    'HLT_L1HTT280er_v4',
    'HLT_L1HTT320er_v4',
    'HLT_L1HTT360er_v4',
    'HLT_L1HTT400er_v4',
    'HLT_L1HTT450er_v4',
    'HLT_L1SingleEG10er2p5_v4',
    'HLT_L1SingleEG15er2p5_v4',
    'HLT_L1SingleEG26er2p5_v4',
    'HLT_L1SingleEG28er1p5_v4',
    'HLT_L1SingleEG28er2p1_v4',
    'HLT_L1SingleEG28er2p5_v4',
    'HLT_L1SingleEG34er2p5_v4',
    'HLT_L1SingleEG36er2p5_v4',
    'HLT_L1SingleEG38er2p5_v4',
    'HLT_L1SingleEG40er2p5_v4',
    'HLT_L1SingleEG42er2p5_v4',
    'HLT_L1SingleEG45er2p5_v4',
    'HLT_L1SingleEG50_v4',
    'HLT_L1SingleEG8er2p5_v4',
    'HLT_L1SingleJet120_v4',
    'HLT_L1SingleJet180_v4',
    'HLT_L1SingleJet60_v4',
    'HLT_L1SingleJet90_v4',
    'HLT_L1SingleMu7_v5',
    'HLT_Physics_v14' ),
  ExpressAlignment = cms.vstring( 'HLT_HT300_Beamspot_PixelClusters_WP2_v7',
    'HLT_HT300_Beamspot_v23',
    'HLT_HT60_Beamspot_v22',
    'HLT_PixelClusters_WP2_v4',
    'HLT_ZeroBias_Beamspot_v16' ),
  ExpressCosmics = cms.vstring( 'HLT_L1SingleMuCosmics_v8',
    'HLT_L1SingleMuOpen_DT_v6',
    'HLT_L1SingleMuOpen_v6',
    'HLT_Random_v3' ),
  ExpressPhysics = cms.vstring( 'HLT_BptxOR_v6',
    'HLT_L1SingleEG10er2p5_v4',
    'HLT_L1SingleEG15er2p5_v4',
    'HLT_L1SingleEG26er2p5_v4',
    'HLT_L1SingleEG28er1p5_v4',
    'HLT_L1SingleEG28er2p1_v4',
    'HLT_L1SingleEG28er2p5_v4',
    'HLT_L1SingleEG34er2p5_v4',
    'HLT_L1SingleEG36er2p5_v4',
    'HLT_L1SingleEG38er2p5_v4',
    'HLT_L1SingleEG40er2p5_v4',
    'HLT_L1SingleEG42er2p5_v4',
    'HLT_L1SingleEG45er2p5_v4',
    'HLT_L1SingleEG50_v4',
    'HLT_L1SingleEG8er2p5_v4',
    'HLT_L1SingleJet60_v4',
    'HLT_Physics_v14',
    'HLT_PixelClusters_WP1_v4',
    'HLT_PixelClusters_WP2_v4',
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v8',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
    'HLT_ZeroBias_IsolatedBunches_v12',
    'HLT_ZeroBias_v13' ),
  HLTMonitor = cms.vstring( 'HLT_L1SingleMuCosmics_CosmicTracking_v1',
    'HLT_L1SingleMuCosmics_PointingCosmicTracking_v1' ),
  HLTPhysics = cms.vstring( 'HLT_Physics_v14' ),
  HcalNZS = cms.vstring( 'HLT_HcalNZS_v21',
    'HLT_HcalPhiSym_v23' ),
  L1Accept = cms.vstring( 'DST_Physics_v16',
    'DST_ZeroBias_v11' ),
  MinimumBias = cms.vstring( 'HLT_BptxOR_v6',
    'HLT_L1ETM120_v4',
    'HLT_L1ETM150_v4',
    'HLT_L1EXT_HCAL_LaserMon1_v5',
    'HLT_L1EXT_HCAL_LaserMon4_v5',
    'HLT_L1HTT120er_v4',
    'HLT_L1HTT160er_v4',
    'HLT_L1HTT200er_v4',
    'HLT_L1HTT255er_v4',
    'HLT_L1HTT280er_v4',
    'HLT_L1HTT320er_v4',
    'HLT_L1HTT360er_v4',
    'HLT_L1HTT400er_v4',
    'HLT_L1HTT450er_v4',
    'HLT_L1SingleEG10er2p5_v4',
    'HLT_L1SingleEG15er2p5_v4',
    'HLT_L1SingleEG26er2p5_v4',
    'HLT_L1SingleEG28er1p5_v4',
    'HLT_L1SingleEG28er2p1_v4',
    'HLT_L1SingleEG28er2p5_v4',
    'HLT_L1SingleEG34er2p5_v4',
    'HLT_L1SingleEG36er2p5_v4',
    'HLT_L1SingleEG38er2p5_v4',
    'HLT_L1SingleEG40er2p5_v4',
    'HLT_L1SingleEG42er2p5_v4',
    'HLT_L1SingleEG45er2p5_v4',
    'HLT_L1SingleEG50_v4',
    'HLT_L1SingleEG8er2p5_v4',
    'HLT_L1SingleJet10erHE_v5',
    'HLT_L1SingleJet120_v4',
    'HLT_L1SingleJet12erHE_v5',
    'HLT_L1SingleJet180_v4',
    'HLT_L1SingleJet200_v5',
    'HLT_L1SingleJet35_v5',
    'HLT_L1SingleJet60_v4',
    'HLT_L1SingleJet8erHE_v5',
    'HLT_L1SingleJet90_v4' ),
  MuonShower = cms.vstring( 'HLT_CscCluster_Cosmic_v4' ),
  NoBPTX = cms.vstring( 'HLT_CDC_L2cosmic_10_er1p0_v10',
    'HLT_CDC_L2cosmic_5p5_er1p0_v10',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v14',
    'HLT_L2Mu10_NoVertex_NoBPTX_v15',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13' ),
  OnlineMonitor = cms.vstring( 'DQM_Random_v1',
    'DQM_ZeroBias_v3',
    'HLT_BptxOR_v6',
    'HLT_CDC_L2cosmic_10_er1p0_v10',
    'HLT_CDC_L2cosmic_5p5_er1p0_v10',
    'HLT_HcalNZS_v21',
    'HLT_HcalPhiSym_v23',
    'HLT_IsoTrackHB_v14',
    'HLT_IsoTrackHE_v14',
    'HLT_L1DoubleMu0_v5',
    'HLT_L1ETM120_v4',
    'HLT_L1ETM150_v4',
    'HLT_L1FatEvents_v5',
    'HLT_L1HTT120er_v4',
    'HLT_L1HTT160er_v4',
    'HLT_L1HTT200er_v4',
    'HLT_L1HTT255er_v4',
    'HLT_L1HTT280er_v4',
    'HLT_L1HTT320er_v4',
    'HLT_L1HTT360er_v4',
    'HLT_L1HTT400er_v4',
    'HLT_L1HTT450er_v4',
    'HLT_L1SingleEG10er2p5_v4',
    'HLT_L1SingleEG15er2p5_v4',
    'HLT_L1SingleEG26er2p5_v4',
    'HLT_L1SingleEG28er1p5_v4',
    'HLT_L1SingleEG28er2p1_v4',
    'HLT_L1SingleEG28er2p5_v4',
    'HLT_L1SingleEG34er2p5_v4',
    'HLT_L1SingleEG36er2p5_v4',
    'HLT_L1SingleEG38er2p5_v4',
    'HLT_L1SingleEG40er2p5_v4',
    'HLT_L1SingleEG42er2p5_v4',
    'HLT_L1SingleEG45er2p5_v4',
    'HLT_L1SingleEG50_v4',
    'HLT_L1SingleEG8er2p5_v4',
    'HLT_L1SingleJet120_v4',
    'HLT_L1SingleJet180_v4',
    'HLT_L1SingleJet200_v5',
    'HLT_L1SingleJet35_v5',
    'HLT_L1SingleJet60_v4',
    'HLT_L1SingleJet90_v4',
    'HLT_L1SingleMuCosmics_v8',
    'HLT_L1SingleMuOpen_v6',
    'HLT_L2Mu10_NoVertex_NoBPTX3BX_v14',
    'HLT_L2Mu10_NoVertex_NoBPTX_v15',
    'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14',
    'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13',
    'HLT_Physics_v14',
    'HLT_PixelClusters_WP1_v4',
    'HLT_PixelClusters_WP2_v4',
    'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v8',
    'HLT_ZeroBias_FirstBXAfterTrain_v10',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
    'HLT_ZeroBias_FirstCollisionInTrain_v11',
    'HLT_ZeroBias_IsolatedBunches_v12',
    'HLT_ZeroBias_LastCollisionInTrain_v10',
    'HLT_ZeroBias_v13' ),
  RPCMonitor = cms.vstring( 'AlCa_RPCMuonNormalisation_v23' ),
  SpecialHLTPhysics0 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics1 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics10 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics11 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics12 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics13 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics14 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics15 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics16 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics17 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics18 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics19 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics2 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics3 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics4 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics5 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics6 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics7 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics8 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialHLTPhysics9 = cms.vstring( 'HLT_SpecialHLTPhysics_v7' ),
  SpecialRandom0 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom1 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom10 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom11 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom12 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom13 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom14 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom15 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom16 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom17 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom18 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom19 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom2 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom3 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom4 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom5 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom6 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom7 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom8 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialRandom9 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  SpecialZeroBias0 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias1 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias10 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias11 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias12 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias13 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias14 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias15 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias16 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias17 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias18 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias19 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias2 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias20 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias21 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias22 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias23 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias24 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias25 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias26 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias27 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias28 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias29 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias3 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias30 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias31 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias4 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias5 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias6 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias7 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias8 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  SpecialZeroBias9 = cms.vstring( 'HLT_SpecialZeroBias_v6',
    'HLT_ZeroBias_Gated_v4',
    'HLT_ZeroBias_HighRate_v4' ),
  TestEnablesEcalHcal = cms.vstring( 'HLT_EcalCalibration_v4',
    'HLT_HcalCalibration_v6' ),
  TestEnablesEcalHcalDQM = cms.vstring( 'HLT_EcalCalibration_v4',
    'HLT_HcalCalibration_v6' ),
  VRRandom0 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom1 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom10 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom11 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom12 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom13 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom14 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom15 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom2 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom3 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom4 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom5 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom6 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom7 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom8 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  VRRandom9 = cms.vstring( 'HLT_Random_HighRate_v1' ),
  ZeroBias = cms.vstring( 'HLT_Random_v3',
    'HLT_ZeroBias_Alignment_v8',
    'HLT_ZeroBias_FirstBXAfterTrain_v10',
    'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
    'HLT_ZeroBias_FirstCollisionInTrain_v11',
    'HLT_ZeroBias_IsolatedBunches_v12',
    'HLT_ZeroBias_LastCollisionInTrain_v10',
    'HLT_ZeroBias_v13' )
)

fragment.CSCChannelMapperESSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "CSCChannelMapperRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.CSCINdexerESSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "CSCIndexerRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.GlobalParametersRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1TGlobalParametersRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.HcalTimeSlewEP = cms.ESSource( "HcalTimeSlewEP",
    appendToDataLabel = cms.string( "HBHE" ),
    timeSlewParametersM2 = cms.VPSet( 
      cms.PSet(  tmax = cms.double( 16.0 ),
        tzero = cms.double( 23.960177 ),
        slope = cms.double( -3.178648 )
      ),
      cms.PSet(  tmax = cms.double( 10.0 ),
        tzero = cms.double( 11.977461 ),
        slope = cms.double( -1.5610227 )
      ),
      cms.PSet(  tmax = cms.double( 6.25 ),
        tzero = cms.double( 9.109694 ),
        slope = cms.double( -1.075824 )
      )
    ),
    timeSlewParametersM3 = cms.VPSet( 
      cms.PSet(  tspar0_siPM = cms.double( 0.0 ),
        tspar2_siPM = cms.double( 0.0 ),
        tspar2 = cms.double( 0.0 ),
        cap = cms.double( 6.0 ),
        tspar1 = cms.double( -2.19142 ),
        tspar0 = cms.double( 12.2999 ),
        tspar1_siPM = cms.double( 0.0 )
      ),
      cms.PSet(  tspar0_siPM = cms.double( 0.0 ),
        tspar2_siPM = cms.double( 0.0 ),
        tspar2 = cms.double( 32.0 ),
        cap = cms.double( 6.0 ),
        tspar1 = cms.double( -3.2 ),
        tspar0 = cms.double( 15.5 ),
        tspar1_siPM = cms.double( 0.0 )
      ),
      cms.PSet(  tspar0_siPM = cms.double( 0.0 ),
        tspar2_siPM = cms.double( 0.0 ),
        tspar2 = cms.double( 0.0 ),
        cap = cms.double( 6.0 ),
        tspar1 = cms.double( -2.19142 ),
        tspar0 = cms.double( 12.2999 ),
        tspar1_siPM = cms.double( 0.0 )
      ),
      cms.PSet(  tspar0_siPM = cms.double( 0.0 ),
        tspar2_siPM = cms.double( 0.0 ),
        tspar2 = cms.double( 0.0 ),
        cap = cms.double( 6.0 ),
        tspar1 = cms.double( -2.19142 ),
        tspar0 = cms.double( 12.2999 ),
        tspar1_siPM = cms.double( 0.0 )
      )
    )
)
fragment.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "JetTagComputerRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSPFRecHitHCALParamsRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "PFRecHitHCALParamsRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSPFRecHitHCALTopologyRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "PFRecHitHCALTopologyRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.hltESSTfGraphRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "TfGraphRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
fragment.ppsPixelTopologyESSource = cms.ESSource( "PPSPixelTopologyESSource",
    RunType = cms.string( "Run3" ),
    PitchSimY = cms.double( 0.15 ),
    PitchSimX = cms.double( 0.1 ),
    thickness = cms.double( 0.23 ),
    noOfPixelSimX = cms.int32( 160 ),
    noOfPixelSimY = cms.int32( 104 ),
    noOfPixels = cms.int32( 16640 ),
    simXWidth = cms.double( 16.6 ),
    simYWidth = cms.double( 16.2 ),
    deadEdgeWidth = cms.double( 0.2 ),
    activeEdgeSigma = cms.double( 0.02 ),
    physActiveEdgeDist = cms.double( 0.15 ),
    appendToDataLabel = cms.string( "" )
)

fragment.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "anyDirection" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
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
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" ),
  MapAuto = cms.untracked.bool( False ),
  SkipHE = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.CaloTowerTopologyEP = cms.ESProducer( "CaloTowerTopologyEP",
  appendToDataLabel = cms.string( "" )
)
fragment.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  dump = cms.untracked.vstring(  ),
  appendToDataLabel = cms.string( "" )
)
fragment.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "ClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.DTObjectMapESProducer = cms.ESProducer( "DTObjectMapESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.GlobalParameters = cms.ESProducer( "StableParametersTrivialProducer",
  TotalBxInEvent = cms.int32( 5 ),
  NumberPhysTriggers = cms.uint32( 512 ),
  NumberL1Muon = cms.uint32( 8 ),
  NumberL1EGamma = cms.uint32( 12 ),
  NumberL1Jet = cms.uint32( 12 ),
  NumberL1Tau = cms.uint32( 12 ),
  NumberChips = cms.uint32( 1 ),
  PinsOnChip = cms.uint32( 512 ),
  OrderOfChip = cms.vint32( 1 ),
  NumberL1IsoEG = cms.uint32( 4 ),
  NumberL1JetCounts = cms.uint32( 12 ),
  UnitLength = cms.int32( 8 ),
  NumberL1ForJet = cms.uint32( 4 ),
  IfCaloEtaNumberBits = cms.uint32( 4 ),
  IfMuEtaNumberBits = cms.uint32( 6 ),
  NumberL1TauJet = cms.uint32( 4 ),
  NumberL1Mu = cms.uint32( 4 ),
  NumberConditionChips = cms.uint32( 1 ),
  NumberPsbBoards = cms.int32( 7 ),
  NumberL1CenJet = cms.uint32( 4 ),
  PinsOnConditionChip = cms.uint32( 512 ),
  NumberL1NoIsoEG = cms.uint32( 4 ),
  NumberTechnicalTriggers = cms.uint32( 64 ),
  NumberPhysTriggersExtended = cms.uint32( 64 ),
  WordLength = cms.int32( 64 ),
  OrderConditionChip = cms.vint32( 1 ),
  appendToDataLabel = cms.string( "" )
)
fragment.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.MaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.OppositeMaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.OppositePropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( 0.1 )
)
fragment.PropagatorWithMaterialForLoopers = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 4.0 ),
  ComponentName = cms.string( "PropagatorWithMaterialForLoopers" ),
  Mass = cms.double( 0.1396 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.PropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStep" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( 0.1 )
)
fragment.SiPixelTemplateStoreESProducer = cms.ESProducer( "SiPixelTemplateStoreESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.SiStripClusterizerConditionsESProducer = cms.ESProducer( "SiStripClusterizerConditionsESProducer",
  QualityLabel = cms.string( "" ),
  Label = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 ),
  appendToDataLabel = cms.string( "" )
)
fragment.SimpleSecondaryVertex3TrkComputer = cms.ESProducer( "SimpleSecondaryVertexESProducer",
  use3d = cms.bool( True ),
  useSignificance = cms.bool( True ),
  unBoost = cms.bool( False ),
  minTracks = cms.uint32( 3 ),
  minVertices = cms.uint32( 1 ),
  appendToDataLabel = cms.string( "" )
)
fragment.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorAny" ),
  NoErrorPropagation = cms.bool( False ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" ),
  appendToDataLabel = cms.string( "" )
)
fragment.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
fragment.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" ),
  PluginName = cms.string( "" ),
  SimpleMagneticField = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.ctppsGeometryESModule = cms.ESProducer( "CTPPSGeometryESModule",
  verbosity = cms.untracked.uint32( 1 ),
  buildMisalignedGeometry = cms.bool( False ),
  isRun2 = cms.bool( False ),
  dbTag = cms.string( "" ),
  compactViewTag = cms.string( "" ),
  fromPreprocessedDB = cms.untracked.bool( True ),
  fromDD4hep = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.ctppsInterpolatedOpticalFunctionsESSource = cms.ESProducer( "CTPPSInterpolatedOpticalFunctionsESSource",
  lhcInfoLabel = cms.string( "" ),
  lhcInfoPerFillLabel = cms.string( "" ),
  lhcInfoPerLSLabel = cms.string( "" ),
  opticsLabel = cms.string( "" ),
  useNewLHCInfo = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
fragment.ecalElectronicsMappingHostESProducer = cms.ESProducer( "EcalElectronicsMappingHostESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.ecalMultifitConditionsHostESProducer = cms.ESProducer( "EcalMultifitConditionsHostESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
  flagMask = cms.PSet( 
    kBad = cms.vstring( 'kFaultyHardware',
      'kDead',
      'kKilled' ),
    kGood = cms.vstring( 'kGood' ),
    kRecovered = cms.vstring( 'kLeadingEdgeRecovered',
      'kTowerRecovered' ),
    kProblematic = cms.vstring( 'kPoorReco',
      'kPoorCalib',
      'kNoisy',
      'kSaturated' ),
    kWeird = cms.vstring( 'kWeird',
      'kDiWeird' ),
    kTime = cms.vstring( 'kOutOfTime' )
  ),
  dbstatusMask = cms.PSet( 
    kBad = cms.vstring( 'kNonRespondingIsolated',
      'kDeadVFE',
      'kDeadFE',
      'kNoDataNoTP' ),
    kGood = cms.vstring( 'kOk' ),
    kRecovered = cms.vstring(  ),
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
    kWeird = cms.vstring(  ),
    kTime = cms.vstring(  )
  ),
  timeThresh = cms.double( 2.0 )
)
fragment.hcalChannelPropertiesESProd = cms.ESProducer( "HcalChannelPropertiesEP" )
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
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
fragment.hcalMahiConditionsESProducer = cms.ESProducer( "HcalMahiConditionsESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  phase = cms.uint32( 1 ),
  RecoveredRecHitBits = cms.vstring( '' ),
  SeverityLevels = cms.VPSet( 
    cms.PSet(  ChannelStatus = cms.vstring( '' ),
      RecHitFlags = cms.vstring( '' ),
      Level = cms.int32( 0 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellCaloTowerProb' ),
      RecHitFlags = cms.vstring( '' ),
      Level = cms.int32( 1 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellExcludeFromHBHENoiseSummary' ),
      RecHitFlags = cms.vstring( 'HBHEIsolatedNoise',
        'HFAnomalousHit' ),
      Level = cms.int32( 5 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( '' ),
      RecHitFlags = cms.vstring( 'HBHEHpdHitMultiplicity',
        'HBHESpikeNoise',
        'HBHETS4TS5Noise',
        'HBHEOOTPU',
        'HBHEFlatNoise',
        'HBHENegativeNoise' ),
      Level = cms.int32( 8 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( '' ),
      RecHitFlags = cms.vstring( 'HFLongShort',
        'HFS8S1Ratio',
        'HFPET',
        'HFSignalAsymmetry' ),
      Level = cms.int32( 11 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellCaloTowerMask' ),
      RecHitFlags = cms.vstring(  ),
      Level = cms.int32( 12 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellHot' ),
      RecHitFlags = cms.vstring( '' ),
      Level = cms.int32( 15 )
    ),
    cms.PSet(  ChannelStatus = cms.vstring( 'HcalCellOff',
  'HcalCellDead' ),
      RecHitFlags = cms.vstring( '' ),
      Level = cms.int32( 20 )
    )
  ),
  DropChannelStatusBits = cms.vstring( 'HcalCellMask',
    'HcalCellOff',
    'HcalCellDead' ),
  appendToDataLabel = cms.string( "" )
)
fragment.hcalRecoParamWithPulseShapeESProducer = cms.ESProducer( "HcalRecoParamWithPulseShapeESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hcalSiPMCharacteristicsESProducer = cms.ESProducer( "HcalSiPMCharacteristicsESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer( "CandidateBoostedDoubleSecondaryVertexESProducer",
  useCondDB = cms.bool( False ),
  gbrForestLabel = cms.string( "" ),
  weightFile = cms.FileInPath( "RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v4.weights.xml.gz" ),
  useGBRForest = cms.bool( True ),
  useAdaBoost = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  trackPseudoSelection = cms.PSet( 
    maxDistToAxis = cms.double( 0.07 ),
    totalHitsMin = cms.uint32( 0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dValMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    qualityClass = cms.string( "any" ),
    jetDeltaRMax = cms.double( 0.3 ),
    normChi2Max = cms.double( 99999.9 ),
    pixelHitsMin = cms.uint32( 0 ),
    sip2dSigMin = cms.double( 2.0 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  trackSelection = cms.PSet( 
    maxDistToAxis = cms.double( 0.07 ),
    totalHitsMin = cms.uint32( 0 ),
    ptMin = cms.double( 0.0 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dValMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    maxDecayLen = cms.double( 5.0 ),
    qualityClass = cms.string( "any" ),
    jetDeltaRMax = cms.double( 0.3 ),
    normChi2Max = cms.double( 99999.9 ),
    pixelHitsMin = cms.uint32( 0 ),
    sip2dSigMin = cms.double( -99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 )
  ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackFlip = cms.bool( False ),
  useTrackWeights = cms.bool( True ),
  SoftLeptonFlip = cms.bool( False ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  minimumTrackWeight = cms.double( 0.5 ),
  charmCut = cms.double( 1.5 ),
  trackSort = cms.string( "sip2dSig" ),
  trackMultiplicityMin = cms.uint32( 3 ),
  vertexFlip = cms.bool( False ),
  useCategories = cms.bool( True ),
  categoryVariableName = cms.string( "vertexCategory" ),
  calibrationRecords = cms.vstring( 'CombinedSVRecoVertex',
    'CombinedSVPseudoVertex',
    'CombinedSVNoVertex' ),
  calibrationRecord = cms.string( "" ),
  recordLabel = cms.string( "HLT" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltCombinedSecondaryVertexV2 = cms.ESProducer( "CombinedSecondaryVertexESProducer",
  trackPseudoSelection = cms.PSet( 
    max_pT_dRcut = cms.double( 0.1 ),
    b_dR = cms.double( 0.6263 ),
    min_pT = cms.double( 120.0 ),
    b_pT = cms.double( 0.3684 ),
    ptMin = cms.double( 0.0 ),
    max_pT_trackPTcut = cms.double( 3.0 ),
    max_pT = cms.double( 500.0 ),
    useVariableJTA = cms.bool( False ),
    maxDecayLen = cms.double( 5.0 ),
    qualityClass = cms.string( "any" ),
    normChi2Max = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 ),
    a_dR = cms.double( -0.001053 ),
    maxDistToAxis = cms.double( 0.07 ),
    totalHitsMin = cms.uint32( 0 ),
    a_pT = cms.double( 0.005263 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dValMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    min_pT_dRcut = cms.double( 0.5 ),
    jetDeltaRMax = cms.double( 0.3 ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip2dSigMin = cms.double( 2.0 )
  ),
  trackSelection = cms.PSet( 
    max_pT_dRcut = cms.double( 0.1 ),
    b_dR = cms.double( 0.6263 ),
    min_pT = cms.double( 120.0 ),
    b_pT = cms.double( 0.3684 ),
    ptMin = cms.double( 0.0 ),
    max_pT_trackPTcut = cms.double( 3.0 ),
    max_pT = cms.double( 500.0 ),
    useVariableJTA = cms.bool( False ),
    maxDecayLen = cms.double( 5.0 ),
    qualityClass = cms.string( "any" ),
    normChi2Max = cms.double( 99999.9 ),
    sip2dValMin = cms.double( -99999.9 ),
    sip3dValMin = cms.double( -99999.9 ),
    a_dR = cms.double( -0.001053 ),
    maxDistToAxis = cms.double( 0.07 ),
    totalHitsMin = cms.uint32( 0 ),
    a_pT = cms.double( 0.005263 ),
    sip2dSigMax = cms.double( 99999.9 ),
    sip2dValMax = cms.double( 99999.9 ),
    sip3dSigMax = cms.double( 99999.9 ),
    sip3dValMax = cms.double( 99999.9 ),
    min_pT_dRcut = cms.double( 0.5 ),
    jetDeltaRMax = cms.double( 0.3 ),
    pixelHitsMin = cms.uint32( 0 ),
    sip3dSigMin = cms.double( -99999.9 ),
    sip2dSigMin = cms.double( -99999.9 )
  ),
  trackPairV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.03 ) ),
  pseudoVertexV0Filter = cms.PSet(  k0sMassWindow = cms.double( 0.05 ) ),
  trackFlip = cms.bool( False ),
  useTrackWeights = cms.bool( True ),
  SoftLeptonFlip = cms.bool( False ),
  pseudoMultiplicityMin = cms.uint32( 2 ),
  correctVertexMass = cms.bool( True ),
  minimumTrackWeight = cms.double( 0.5 ),
  charmCut = cms.double( 1.5 ),
  trackSort = cms.string( "sip2dSig" ),
  trackMultiplicityMin = cms.uint32( 3 ),
  vertexFlip = cms.bool( False ),
  useCategories = cms.bool( True ),
  categoryVariableName = cms.string( "vertexCategory" ),
  calibrationRecords = cms.vstring( 'CombinedSVIVFV2RecoVertex',
    'CombinedSVIVFV2PseudoVertex',
    'CombinedSVIVFV2NoVertex' ),
  calibrationRecord = cms.string( "" ),
  recordLabel = cms.string( "HLT" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltDisplacedDijethltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  nthTrack = cms.int32( -1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.1 ),
  maxImpactParameterSig = cms.double( 999999.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  trackQualityClass = cms.string( "any" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltDisplacedDijethltESPTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
  minimumImpactParameter = cms.double( 0.05 ),
  useSignedImpactParameterSig = cms.bool( False ),
  nthTrack = cms.int32( 1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  a_dR = cms.double( -0.001053 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  b_pT = cms.double( 0.3684 ),
  min_pT = cms.double( 120.0 ),
  max_pT = cms.double( 500.0 ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_dRcut = cms.double( 0.1 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESChi2MeasurementEstimatorForP5 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 100.0 ),
  nSigma = cms.double( 4.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 100000.0 ),
  ComponentName = cms.string( "hltESChi2MeasurementEstimatorForP5" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESFittingSmootherRKP5 = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESFittingSmootherRKP5" ),
  Fitter = cms.string( "hltESPRKTrajectoryFitter" ),
  Smoother = cms.string( "hltESPRKTrajectorySmoother" ),
  EstimateCut = cms.double( 20.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 4 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  LogPixelProbabilityCut = cms.double( 0.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.hltESPChi2ChargeLooseMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeLooseMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeMeasurementEstimator2000 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 2000.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator2000" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator30" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeMeasurementEstimator9ForHI = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeMeasurementEstimator9ForHI" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutForHI" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2ChargeTightMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2ChargeTightMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2MeasurementEstimator100 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 40.0 ),
  nSigma = cms.double( 4.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1.0E12 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator100" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2MeasurementEstimator30 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator30" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDetachedQuadStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPDetachedQuadStepChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDetachedQuadStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedQuadStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPDetachedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPDetachedTripletStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPDetachedTripletStepChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDetachedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPDisplacedDijethltPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
  nthTrack = cms.int32( -1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.1 ),
  maxImpactParameterSig = cms.double( 999999.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  trackQualityClass = cms.string( "any" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDisplacedDijethltPromptTrackCountingESProducerLong = cms.ESProducer( "PromptTrackCountingESProducer",
  nthTrack = cms.int32( -1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.2 ),
  maxImpactParameterSig = cms.double( 999999.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  trackQualityClass = cms.string( "any" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDisplacedDijethltPromptTrackCountingESProducerShortSig5 = cms.ESProducer( "PromptTrackCountingESProducer",
  nthTrack = cms.int32( -1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  deltaRmin = cms.double( 0.0 ),
  maxImpactParameter = cms.double( 0.05 ),
  maxImpactParameterSig = cms.double( 5.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 999999.0 ),
  trackQualityClass = cms.string( "any" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDisplacedDijethltTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
  minimumImpactParameter = cms.double( 0.05 ),
  useSignedImpactParameterSig = cms.bool( False ),
  nthTrack = cms.int32( 1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  a_dR = cms.double( -0.001053 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  b_pT = cms.double( 0.3684 ),
  min_pT = cms.double( 120.0 ),
  max_pT = cms.double( 500.0 ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_dRcut = cms.double( 0.1 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDisplacedDijethltTrackCounting2D1stLoose = cms.ESProducer( "TrackCountingESProducer",
  minimumImpactParameter = cms.double( 0.03 ),
  useSignedImpactParameterSig = cms.bool( False ),
  nthTrack = cms.int32( 1 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  a_dR = cms.double( -0.001053 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  b_pT = cms.double( 0.3684 ),
  min_pT = cms.double( 120.0 ),
  max_pT = cms.double( 500.0 ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_dRcut = cms.double( 0.1 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDisplacedDijethltTrackCounting2D2ndLong = cms.ESProducer( "TrackCountingESProducer",
  minimumImpactParameter = cms.double( 0.2 ),
  useSignedImpactParameterSig = cms.bool( True ),
  nthTrack = cms.int32( 2 ),
  impactParameterType = cms.int32( 1 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 999999.0 ),
  maximumDistanceToJetAxis = cms.double( 9999999.0 ),
  trackQualityClass = cms.string( "any" ),
  useVariableJTA = cms.bool( False ),
  a_dR = cms.double( -0.001053 ),
  b_dR = cms.double( 0.6263 ),
  a_pT = cms.double( 0.005263 ),
  b_pT = cms.double( 0.3684 ),
  min_pT = cms.double( 120.0 ),
  max_pT = cms.double( 500.0 ),
  min_pT_dRcut = cms.double( 0.5 ),
  max_pT_dRcut = cms.double( 0.1 ),
  max_pT_trackPTcut = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
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
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  NoErrorPropagation = cms.bool( False ),
  PropagationDirection = cms.string( "anyDirection" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  NoErrorPropagation = cms.bool( False ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFittingSmootherIT" ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  EstimateCut = cms.double( -1.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 3 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFittingSmootherRK" ),
  Fitter = cms.string( "hltESPTrajectoryFitterRK" ),
  Smoother = cms.string( "hltESPTrajectorySmootherRK" ),
  EstimateCut = cms.double( -1.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 5 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFlexibleKFFittingSmoother" ),
  standardFitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  looperFitter = cms.string( "hltESPKFFittingSmootherForLoopers" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
fragment.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPGsfElectronFittingSmoother" ),
  Fitter = cms.string( "hltESPGsfTrajectoryFitter" ),
  Smoother = cms.string( "hltESPGsfTrajectorySmoother" ),
  EstimateCut = cms.double( -1.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 5 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  ErrorRescaling = cms.double( 100.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPHighPtTripletStepChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPHighPtTripletStepChi2ChargeMeasurementEstimator30" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPInitialStepChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2ChargeMeasurementEstimator30" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPInitialStepChi2MeasurementEstimator36 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 36.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2MeasurementEstimator36" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmoother" ),
  Fitter = cms.string( "hltESPKFTrajectoryFitter" ),
  Smoother = cms.string( "hltESPKFTrajectorySmoother" ),
  EstimateCut = cms.double( -1.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 5 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 5 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmootherForLoopers" ),
  Fitter = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  Smoother = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  EstimateCut = cms.double( 20.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 3 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  Fitter = cms.string( "hltESPRKTrajectoryFitter" ),
  Smoother = cms.string( "hltESPRKTrajectorySmoother" ),
  EstimateCut = cms.double( 20.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 3 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( True ),
  NoInvalidHitsBeginEnd = cms.bool( True ),
  LogPixelProbabilityCut = cms.double( -14.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  DistanceMeasure = cms.string( "KullbackLeibler" ),
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPLowPtQuadStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPLowPtQuadStepChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPLowPtQuadStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtQuadStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPLowPtStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPLowPtTripletStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPLowPtTripletStepChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPLowPtTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "hltESPMeasurementTracker" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  SiStripQualityLabel = cms.string( "" ),
  UseStripModuleQualityDB = cms.bool( True ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( True ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  MaskBadAPVFibers = cms.bool( True ),
  UseStripStripQualityDB = cms.bool( True ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  badStripCuts = cms.PSet( 
    TOB = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TIB = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TID = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    ),
    TEC = cms.PSet( 
      maxBad = cms.uint32( 4 ),
      maxConsecutiveBad = cms.uint32( 2 )
    )
  ),
  UsePixelModuleQualityDB = cms.bool( True ),
  DebugPixelModuleQualityDB = cms.untracked.bool( False ),
  UsePixelROCQualityDB = cms.bool( True ),
  DebugPixelROCQualityDB = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPMixedStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPMixedStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPMixedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPMixedTripletStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPMixedTripletStepChi2ChargeMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPMixedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPFRecHitHCALParams = cms.ESProducer( "PFRecHitHCALParamsESProducer@alpaka",
  energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
  energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltESPPFRecHitHCALTopology = cms.ESProducer( "PFRecHitHCALTopologyESProducer@alpaka",
  usePFThresholdsFromDB = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltESPPixelCPEFastParamsHIonPhase1 = cms.ESProducer( "PixelCPEFastParamsESProducerAlpakaHIonPhase1@alpaka",
  LoadTemplatesFromDB = cms.bool( True ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  useLAWidthFromDB = cms.bool( True ),
  lAOffset = cms.double( 0.0 ),
  lAWidthBPix = cms.double( 0.0 ),
  lAWidthFPix = cms.double( 0.0 ),
  doLorentzFromAlignment = cms.bool( False ),
  useLAFromDB = cms.bool( True ),
  xerr_barrel_l1 = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_l1 = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_barrel_ln = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_ln = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_endcap = cms.vdouble( 0.002, 0.002 ),
  yerr_endcap = cms.vdouble( 0.0021 ),
  xerr_barrel_l1_def = cms.double( 0.0103 ),
  yerr_barrel_l1_def = cms.double( 0.0021 ),
  xerr_barrel_ln_def = cms.double( 0.0103 ),
  yerr_barrel_ln_def = cms.double( 0.0021 ),
  xerr_endcap_def = cms.double( 0.002 ),
  yerr_endcap_def = cms.double( 7.5E-4 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  ComponentName = cms.string( "PixelCPEFastParamsHIonPhase1" ),
  MagneticFieldRecord = cms.ESInputTag( "","" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelCPEFastParamsPhase1 = cms.ESProducer( "PixelCPEFastParamsESProducerAlpakaPhase1@alpaka",
  LoadTemplatesFromDB = cms.bool( True ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  useLAWidthFromDB = cms.bool( True ),
  lAOffset = cms.double( 0.0 ),
  lAWidthBPix = cms.double( 0.0 ),
  lAWidthFPix = cms.double( 0.0 ),
  doLorentzFromAlignment = cms.bool( False ),
  useLAFromDB = cms.bool( True ),
  xerr_barrel_l1 = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_l1 = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_barrel_ln = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_ln = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_endcap = cms.vdouble( 0.002, 0.002 ),
  yerr_endcap = cms.vdouble( 0.0021 ),
  xerr_barrel_l1_def = cms.double( 0.0103 ),
  yerr_barrel_l1_def = cms.double( 0.0021 ),
  xerr_barrel_ln_def = cms.double( 0.0103 ),
  yerr_barrel_ln_def = cms.double( 0.0021 ),
  xerr_endcap_def = cms.double( 0.002 ),
  yerr_endcap_def = cms.double( 7.5E-4 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  ComponentName = cms.string( "PixelCPEFastParams" ),
  MagneticFieldRecord = cms.ESInputTag( "","" ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
  LoadTemplatesFromDB = cms.bool( True ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  useLAWidthFromDB = cms.bool( False ),
  lAOffset = cms.double( 0.0 ),
  lAWidthBPix = cms.double( 0.0 ),
  lAWidthFPix = cms.double( 0.0 ),
  doLorentzFromAlignment = cms.bool( False ),
  useLAFromDB = cms.bool( True ),
  xerr_barrel_l1 = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_l1 = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_barrel_ln = cms.vdouble( 0.00115, 0.0012, 8.8E-4 ),
  yerr_barrel_ln = cms.vdouble( 0.00375, 0.0023, 0.0025, 0.0025, 0.0023, 0.0023, 0.0021, 0.0021, 0.0024 ),
  xerr_endcap = cms.vdouble( 0.002, 0.002 ),
  yerr_endcap = cms.vdouble( 0.0021 ),
  xerr_barrel_l1_def = cms.double( 0.0103 ),
  yerr_barrel_l1_def = cms.double( 0.0021 ),
  xerr_barrel_ln_def = cms.double( 0.0103 ),
  yerr_barrel_ln_def = cms.double( 0.0021 ),
  xerr_endcap_def = cms.double( 0.002 ),
  yerr_endcap_def = cms.double( 7.5E-4 ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  size_cutX = cms.double( 3.0 ),
  size_cutY = cms.double( 3.0 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  inflate_errors = cms.bool( False ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  NoTemplateErrorsWhenNoTrkAngles = cms.bool( False ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  IrradiationBiasCorrection = cms.bool( True ),
  DoCosmics = cms.bool( False ),
  isPhase2 = cms.bool( False ),
  SmallPitch = cms.bool( False ),
  ComponentName = cms.string( "hltESPPixelCPEGeneric" ),
  MagneticFieldRecord = cms.ESInputTag( "","" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
  LoadTemplatesFromDB = cms.bool( True ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 ),
  useLAWidthFromDB = cms.bool( True ),
  lAOffset = cms.double( 0.0 ),
  lAWidthBPix = cms.double( 0.0 ),
  lAWidthFPix = cms.double( 0.0 ),
  doLorentzFromAlignment = cms.bool( False ),
  useLAFromDB = cms.bool( True ),
  barrelTemplateID = cms.int32( 0 ),
  forwardTemplateID = cms.int32( 0 ),
  directoryWithTemplates = cms.int32( 0 ),
  speed = cms.int32( -2 ),
  UseClusterSplitter = cms.bool( False ),
  ComponentName = cms.string( "hltESPPixelCPETemplateReco" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelLessStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPPixelLessStepChi2ChargeMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelLessStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPPixelLessStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelLessStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelLessStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPPixelPairStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1.0E12 ),
  ComponentName = cms.string( "hltESPPixelPairStepChi2ChargeMeasurementEstimator9" ),
  pTChargeCutThreshold = cms.double( 15.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutLoose" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelPairStepChi2MeasurementEstimator25 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 25.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPPixelPairTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelPairTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPRKTrajectoryFitter" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPRKTrajectorySmoother" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( True ),
  ptMin = cms.double( -1.0 )
)
fragment.hltESPSiPixelCablingSoA = cms.ESProducer( "SiPixelCablingSoAESProducer@alpaka",
  CablingMapLabel = cms.string( "" ),
  UseQualityInfo = cms.bool( False ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltESPSiPixelGainCalibrationForHLTSoA = cms.ESProducer( "SiPixelGainCalibrationForHLTSoAESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  NoErrorPropagation = cms.bool( False ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
  NoErrorPropagation = cms.bool( False ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useTuningForL2Speed = cms.bool( False ),
  useIsYokeFlag = cms.bool( True ),
  endcapShiftInZNeg = cms.double( 0.0 ),
  SetVBFPointer = cms.bool( False ),
  AssumeNoMaterial = cms.bool( False ),
  endcapShiftInZPos = cms.double( 0.0 ),
  useInTeslaFromMagField = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useEndcapShiftsInZ = cms.bool( False ),
  sendLogWarning = cms.bool( False ),
  useMatVolumes = cms.bool( True ),
  debug = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  useMagVolumes = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPStripCPEfromTrackAngle = cms.ESProducer( "StripCPEESProducer",
  ComponentName = cms.string( "hltESPStripCPEfromTrackAngle" ),
  ComponentType = cms.string( "StripCPEfromTrackAngle" ),
  parameters = cms.PSet( 
    mTIB_P1 = cms.double( 0.202 ),
    maxChgOneMIP = cms.double( 6000.0 ),
    mTEC_P0 = cms.double( -1.885 ),
    mTOB_P1 = cms.double( 0.253 ),
    mTEC_P1 = cms.double( 0.471 ),
    mLC_P2 = cms.double( 0.3 ),
    mLC_P1 = cms.double( 0.618 ),
    mTOB_P0 = cms.double( -1.026 ),
    mLC_P0 = cms.double( -0.326 ),
    useLegacyError = cms.bool( False ),
    mTIB_P0 = cms.double( -0.742 ),
    mTID_P1 = cms.double( 0.433 ),
    mTID_P0 = cms.double( -1.427 )
  )
)
fragment.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPTobTecStepChi2ChargeMeasurementEstimator16" ),
  pTChargeCutThreshold = cms.double( -1.0 ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPTobTecStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepFitterSmoother" ),
  Fitter = cms.string( "hltESPTobTecStepRKFitter" ),
  Smoother = cms.string( "hltESPTobTecStepRKSmoother" ),
  EstimateCut = cms.double( 30.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 7 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepFitterSmootherForLoopers" ),
  Fitter = cms.string( "hltESPTobTecStepRKFitterForLoopers" ),
  Smoother = cms.string( "hltESPTobTecStepRKSmootherForLoopers" ),
  EstimateCut = cms.double( 30.0 ),
  MaxFractionOutliers = cms.double( 0.3 ),
  MaxNumberOfOutliers = cms.int32( 3 ),
  MinDof = cms.int32( 2 ),
  NoOutliersBeginEnd = cms.bool( False ),
  MinNumberOfHits = cms.int32( 7 ),
  MinNumberOfHitsHighEta = cms.int32( 5 ),
  HighEtaSwitch = cms.double( 5.0 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  LogPixelProbabilityCut = cms.double( -16.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepFlexibleKFFittingSmoother" ),
  standardFitter = cms.string( "hltESPTobTecStepFitterSmoother" ),
  looperFitter = cms.string( "hltESPTobTecStepFitterSmootherForLoopers" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKFitter" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepRKTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKFitterForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKSmoother" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepRKTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKSmootherForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTobTecStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTobTecStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.09 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPTrackAlgoPriorityOrder = cms.ESProducer( "TrackAlgoPriorityOrderESProducer",
  ComponentName = cms.string( "hltESPTrackAlgoPriorityOrder" ),
  algoOrder = cms.vstring(  ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTrackSelectionTfCKF = cms.ESProducer( "TfGraphDefProducer",
  ComponentName = cms.string( "hltESPTrackSelectionTfCKF" ),
  FileName = cms.FileInPath( "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/CKF_Run3_12_5_0_pre5.pb" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
fragment.hltESPTrajectoryCleanerBySharedHitsP5 = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHitsP5" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltOnlineBeamSpotESProducer = cms.ESProducer( "OnlineBeamSpotESProducer",
  timeThreshold = cms.int32( 48 ),
  sigmaZThreshold = cms.double( 2.0 ),
  sigmaXYThreshold = cms.double( 4.0 ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltPixelTracksCleanerBySharedHits = cms.ESProducer( "PixelTrackCleanerBySharedHitsESProducer",
  ComponentName = cms.string( "hltPixelTracksCleanerBySharedHits" ),
  useQuadrupletAlgo = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.hltTrackCleaner = cms.ESProducer( "TrackCleanerESProducer",
  ComponentName = cms.string( "hltTrackCleaner" ),
  appendToDataLabel = cms.string( "" )
)
fragment.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
fragment.multipleScatteringParametrisationMakerESProducer = cms.ESProducer( "MultipleScatteringParametrisationMakerESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( True ),
  includeGEM = cms.bool( True ),
  includeME0 = cms.bool( False )
)
fragment.muonSeededTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "muonSeededTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.1 ),
  ValidHitBonus = cms.double( 1000.0 ),
  MissingHitPenalty = cms.double( 1.0 ),
  allowSharedFirstHit = cms.bool( True )
)
fragment.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  PluginName = cms.string( "" ),
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  appendToDataLabel = cms.string( "" )
)
fragment.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
fragment.siPixelGainCalibrationForHLTGPU = cms.ESProducer( "SiPixelGainCalibrationForHLTGPUESProducer",
  appendToDataLabel = cms.string( "" )
)
fragment.siPixelROCsStatusAndMappingWrapperESProducer = cms.ESProducer( "SiPixelROCsStatusAndMappingWrapperESProducer",
  ComponentName = cms.string( "" ),
  CablingMapLabel = cms.string( "" ),
  UseQualityInfo = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
fragment.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
fragment.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer( "SiStripBackPlaneCorrectionDepESProducer",
  LatencyRecord = cms.PSet( 
    label = cms.untracked.string( "" ),
    record = cms.string( "SiStripLatencyRcd" )
  ),
  BackPlaneCorrectionPeakMode = cms.PSet( 
    label = cms.untracked.string( "peak" ),
    record = cms.string( "SiStripBackPlaneCorrectionRcd" )
  ),
  BackPlaneCorrectionDeconvMode = cms.PSet( 
    label = cms.untracked.string( "deconvolution" ),
    record = cms.string( "SiStripBackPlaneCorrectionRcd" )
  )
)
fragment.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
  LatencyRecord = cms.PSet( 
    label = cms.untracked.string( "" ),
    record = cms.string( "SiStripLatencyRcd" )
  ),
  LorentzAnglePeakMode = cms.PSet( 
    label = cms.untracked.string( "peak" ),
    record = cms.string( "SiStripLorentzAngleRcd" )
  ),
  LorentzAngleDeconvMode = cms.PSet( 
    label = cms.untracked.string( "deconvolution" ),
    record = cms.string( "SiStripLorentzAngleRcd" )
  )
)
fragment.zdcTopologyEP = cms.ESProducer( "ZdcTopologyEP",
  appendToDataLabel = cms.string( "" )
)

fragment.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
fragment.hltPSetMap = cms.EDProducer( "ParameterSetBlobProducer" )
fragment.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
fragment.hltBackend = cms.EDProducer( "AlpakaBackendProducer@alpaka"
)
fragment.hltStatusOnGPUFilter = cms.EDFilter( "AlpakaBackendFilter",
    producer = cms.InputTag( 'hltBackend','backend' ),
    backends = cms.vstring( 'CudaAsync',
      'ROCmAsync' )
)
fragment.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
fragment.hltGtStage2Digis = cms.EDProducer( "L1TRawToDigi",
    FedIds = cms.vint32( 1404 ),
    Setup = cms.string( "stage2::GTSetup" ),
    FWId = cms.uint32( 0 ),
    DmxFWId = cms.uint32( 0 ),
    FWOverride = cms.bool( False ),
    TMTCheck = cms.bool( True ),
    CTP7 = cms.untracked.bool( False ),
    MTF7 = cms.untracked.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    debug = cms.untracked.bool( False ),
    MinFeds = cms.uint32( 0 )
)
fragment.hltGtStage2ObjectMap = cms.EDProducer( "L1TGlobalProducer",
    MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' ),
    CICADAInputTag = cms.InputTag( 'hltGtStage2Digis','CICADAScore' ),
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    AlgoBlkInputTag = cms.InputTag( "hltGtStage2Digis" ),
    GetPrescaleColumnFromData = cms.bool( False ),
    AlgorithmTriggersUnprescaled = cms.bool( True ),
    RequireMenuToMatchAlgoBlkInput = cms.bool( True ),
    AlgorithmTriggersUnmasked = cms.bool( True ),
    useMuonShowers = cms.bool( True ),
    produceAXOL1TLScore = cms.bool( False ),
    resetPSCountersEachLumiSec = cms.bool( True ),
    semiRandomInitialPSCounters = cms.bool( False ),
    ProduceL1GtDaqRecord = cms.bool( True ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 ),
    L1DataBxInEvent = cms.int32( 5 ),
    AlternativeNrBxBoardDaq = cms.uint32( 0 ),
    BstLengthBytes = cms.int32( -1 ),
    PrescaleSet = cms.uint32( 1 ),
    Verbosity = cms.untracked.int32( 0 ),
    PrintL1Menu = cms.untracked.bool( False ),
    TriggerMenuLuminosity = cms.string( "startup" )
)
fragment.hltOnlineMetaDataDigis = cms.EDProducer( "OnlineMetaDataRawToDigi",
    onlineMetaDataInputLabel = cms.InputTag( "rawDataCollector" )
)
fragment.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    changeToCMSCoordinates = cms.bool( False ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 0.0 ),
    beamMode = cms.untracked.uint32( 11 ),
    src = cms.InputTag( "" ),
    gtEvmLabel = cms.InputTag( "" ),
    maxRadius = cms.double( 2.0 ),
    useTransientRecord = cms.bool( True )
)
fragment.hltL1sZeroBiasIorAlwaysTrueIorIsolatedBunch = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_AlwaysTrue OR L1_IsolatedBunch" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaEcalPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltEcalDigisLegacy = cms.EDProducer( "EcalRawToDigi",
    tccUnpacking = cms.bool( True ),
    FedLabel = cms.InputTag( "listfeds" ),
    srpUnpacking = cms.bool( True ),
    syncCheck = cms.bool( True ),
    feIdCheck = cms.bool( True ),
    silentMode = cms.untracked.bool( True ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    orderedFedList = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    eventPut = cms.bool( True ),
    numbTriggerTSamples = cms.int32( 1 ),
    numbXtalTSamples = cms.int32( 10 ),
    orderedDCCIdList = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    DoRegional = cms.bool( False ),
    feUnpacking = cms.bool( True ),
    forceToKeepFRData = cms.bool( False ),
    headerUnpacking = cms.bool( True ),
    memUnpacking = cms.bool( True )
)
fragment.hltEcalDigisSoA = cms.EDProducer( "EcalRawToDigiPortable@alpaka",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    maxChannelsEB = cms.uint32( 61200 ),
    maxChannelsEE = cms.uint32( 14648 ),
    digisLabelEB = cms.string( "ebDigis" ),
    digisLabelEE = cms.string( "eeDigis" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltEcalDigis = cms.EDProducer( "EcalDigisFromPortableProducer",
    digisInLabelEB = cms.InputTag( 'hltEcalDigisSoA','ebDigis' ),
    digisInLabelEE = cms.InputTag( 'hltEcalDigisSoA','eeDigis' ),
    digisOutLabelEB = cms.string( "ebDigis" ),
    digisOutLabelEE = cms.string( "eeDigis" ),
    produceDummyIntegrityCollections = cms.bool( False )
)
fragment.hltEcalUncalibRecHitSoA = cms.EDProducer( "EcalUncalibRecHitProducerPortable@alpaka",
    digisLabelEB = cms.InputTag( 'hltEcalDigisSoA','ebDigis' ),
    digisLabelEE = cms.InputTag( 'hltEcalDigisSoA','eeDigis' ),
    recHitsLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    recHitsLabelEE = cms.string( "EcalUncalibRecHitsEE" ),
    EBtimeFitLimits_Lower = cms.double( 0.2 ),
    EBtimeFitLimits_Upper = cms.double( 1.4 ),
    EEtimeFitLimits_Lower = cms.double( 0.2 ),
    EEtimeFitLimits_Upper = cms.double( 1.4 ),
    EBtimeConstantTerm = cms.double( 0.6 ),
    EEtimeConstantTerm = cms.double( 1.0 ),
    EBtimeNconst = cms.double( 28.5 ),
    EEtimeNconst = cms.double( 31.8 ),
    outOfTimeThresholdGain12pEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain12mEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain61pEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain61mEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain12pEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain12mEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain61pEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain61mEE = cms.double( 1000.0 ),
    amplitudeThresholdEB = cms.double( 10.0 ),
    amplitudeThresholdEE = cms.double( 10.0 ),
    EBtimeFitParameters = cms.vdouble( -2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621 ),
    EEtimeFitParameters = cms.vdouble( -2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277 ),
    EBamplitudeFitParameters = cms.vdouble( 1.138, 1.652 ),
    EEamplitudeFitParameters = cms.vdouble( 1.89, 1.4 ),
    kernelMinimizeThreads = cms.untracked.vuint32( 32, 1, 1 ),
    shouldRunTimingComputation = cms.bool( True ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltEcalUncalibRecHit = cms.EDProducer( "EcalUncalibRecHitSoAToLegacy",
    inputCollectionEB = cms.InputTag( 'hltEcalUncalibRecHitSoA','EcalUncalibRecHitsEB' ),
    outputLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    isPhase2 = cms.bool( False ),
    inputCollectionEE = cms.InputTag( 'hltEcalUncalibRecHitSoA','EcalUncalibRecHitsEE' ),
    outputLabelEE = cms.string( "EcalUncalibRecHitsEE" )
)
fragment.hltEcalDetIdToBeRecovered = cms.EDProducer( "EcalDetIdToBeRecoveredProducer",
    ebSrFlagCollection = cms.InputTag( "hltEcalDigisLegacy" ),
    eeSrFlagCollection = cms.InputTag( "hltEcalDigisLegacy" ),
    ebIntegrityGainErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityGainErrors' ),
    ebIntegrityGainSwitchErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityGainSwitchErrors' ),
    ebIntegrityChIdErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityChIdErrors' ),
    eeIntegrityGainErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityGainErrors' ),
    eeIntegrityGainSwitchErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityGainSwitchErrors' ),
    eeIntegrityChIdErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityChIdErrors' ),
    integrityTTIdErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityTTIdErrors' ),
    integrityBlockSizeErrors = cms.InputTag( 'hltEcalDigisLegacy','EcalIntegrityBlockSizeErrors' ),
    ebDetIdToBeRecovered = cms.string( "ebDetId" ),
    eeDetIdToBeRecovered = cms.string( "eeDetId" ),
    ebFEToBeRecovered = cms.string( "ebFE" ),
    eeFEToBeRecovered = cms.string( "eeFE" )
)
fragment.hltEcalRecHit = cms.EDProducer( "EcalRecHitProducer",
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHit','EcalUncalibRecHitsEB' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    ChannelStatusToBeExcluded = cms.vstring(  ),
    killDeadChannels = cms.bool( True ),
    algo = cms.string( "EcalRecHitWorkerSimple" ),
    EBLaserMIN = cms.double( 0.5 ),
    EELaserMIN = cms.double( 0.5 ),
    EBLaserMAX = cms.double( 3.0 ),
    EELaserMAX = cms.double( 8.0 ),
    timeCalibTag = cms.ESInputTag( "","" ),
    timeOffsetTag = cms.ESInputTag( "","" ),
    skipTimeCalib = cms.bool( False ),
    laserCorrection = cms.bool( True ),
    flagsMapDBReco = cms.PSet( 
      kDead = cms.vstring( 'kNoDataNoTP' ),
      kGood = cms.vstring( 'kOk',
        'kDAC',
        'kNoLaser',
        'kNoisy' ),
      kTowerRecovered = cms.vstring( 'kDeadFE' ),
      kNoisy = cms.vstring( 'kNNoisy',
        'kFixedG6',
        'kFixedG1' ),
      kNeighboursRecovered = cms.vstring( 'kFixedG0',
        'kNonRespondingIsolated',
        'kDeadVFE' )
    ),
    algoRecover = cms.string( "EcalRecHitWorkerRecover" ),
    recoverEBIsolatedChannels = cms.bool( False ),
    recoverEEIsolatedChannels = cms.bool( False ),
    recoverEBVFE = cms.bool( False ),
    recoverEEVFE = cms.bool( False ),
    recoverEBFE = cms.bool( False ),
    recoverEEFE = cms.bool( False ),
    dbStatusToBeExcludedEE = cms.vint32( 14, 78, 142 ),
    dbStatusToBeExcludedEB = cms.vint32( 14, 78, 142 ),
    logWarningEtThreshold_EB_FE = cms.double( -1.0 ),
    logWarningEtThreshold_EE_FE = cms.double( -1.0 ),
    ebDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebDetId' ),
    eeDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeDetId' ),
    ebFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebFE' ),
    eeFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeFE' ),
    singleChannelRecoveryMethod = cms.string( "NeuralNetworks" ),
    singleChannelRecoveryThreshold = cms.double( 8.0 ),
    sum8ChannelRecoveryThreshold = cms.double( 0.0 ),
    bdtWeightFileNoCracks = cms.FileInPath( "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/bdtgAllRH_8GT700MeV_noCracks_ZskimData2017_v1.xml" ),
    bdtWeightFileCracks = cms.FileInPath( "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/bdtgAllRH_8GT700MeV_onlyCracks_ZskimData2017_v1.xml" ),
    triggerPrimitiveDigiCollection = cms.InputTag( 'hltEcalDigisLegacy','EcalTriggerPrimitives' ),
    cleaningConfig = cms.PSet( 
      cThreshold_endcap = cms.double( 15.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e6e2thresh = cms.double( 0.04 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      cThreshold_double = cms.double( 10.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 )
    )
)
fragment.hltEcalPreshowerDigis = cms.EDProducer( "ESRawToDigi",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    debugMode = cms.untracked.bool( False ),
    InstanceES = cms.string( "" ),
    LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ),
    ESdigiCollection = cms.string( "" )
)
fragment.hltEcalPreshowerRecHit = cms.EDProducer( "ESRecHitProducer",
    ESrechitCollection = cms.string( "EcalRecHitsES" ),
    ESdigiCollection = cms.InputTag( "hltEcalPreshowerDigis" ),
    algo = cms.string( "ESRecHitWorker" ),
    ESRecoAlgo = cms.int32( 0 )
)
fragment.hltEcalPhiSymFilter = cms.EDFilter( "HLTEcalPhiSymFilter",
    barrelDigiCollection = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    endcapDigiCollection = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    barrelUncalibHitCollection = cms.InputTag( 'hltEcalUncalibRecHit','EcalUncalibRecHitsEB' ),
    endcapUncalibHitCollection = cms.InputTag( 'hltEcalUncalibRecHit','EcalUncalibRecHitsEE' ),
    barrelHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    statusThreshold = cms.uint32( 3 ),
    useRecoFlag = cms.bool( False ),
    ampCut_barrelP = cms.vdouble( 14.31759, 14.33355, 14.34853, 14.36281, 14.37667, 14.39011, 14.40334, 14.41657, 14.42994, 14.44359, 14.45759, 14.47222, 14.48748, 14.50358, 14.52052, 14.53844, 14.55755, 14.57778, 14.59934, 14.62216, 14.64645, 14.67221, 14.69951, 14.72849, 14.75894, 14.79121, 14.82502, 14.86058, 14.89796, 14.93695, 14.97783, 15.02025, 15.06442, 15.11041, 15.15787, 15.20708, 15.25783, 15.31026, 15.36409, 15.41932, 15.47602, 15.53384, 15.5932, 15.65347, 15.715, 15.77744, 15.84086, 15.90505, 15.97001, 16.03539, 16.10147, 16.16783, 16.23454, 16.30146, 16.36824, 16.43502, 16.50159, 16.56781, 16.63354, 16.69857, 16.76297, 16.82625, 16.88862, 16.94973, 17.00951, 17.06761, 17.12403, 17.1787, 17.23127, 17.28167, 17.32955, 17.37491, 17.41754, 17.45723, 17.49363, 17.52688, 17.55642, 17.58218, 17.60416, 17.62166, 17.63468, 17.64315, 17.64665, 17.6449, 17.6379 ),
    ampCut_barrelM = cms.vdouble( 17.6379, 17.6449, 17.64665, 17.64315, 17.63468, 17.62166, 17.60416, 17.58218, 17.55642, 17.52688, 17.49363, 17.45723, 17.41754, 17.37491, 17.32955, 17.28167, 17.23127, 17.1787, 17.12403, 17.06761, 17.00951, 16.94973, 16.88862, 16.82625, 16.76297, 16.69857, 16.63354, 16.56781, 16.50159, 16.43502, 16.36824, 16.30146, 16.23454, 16.16783, 16.10147, 16.03539, 15.97001, 15.90505, 15.84086, 15.77744, 15.715, 15.65347, 15.5932, 15.53384, 15.47602, 15.41932, 15.36409, 15.31026, 15.25783, 15.20708, 15.15787, 15.11041, 15.06442, 15.02025, 14.97783, 14.93695, 14.89796, 14.86058, 14.82502, 14.79121, 14.75894, 14.72849, 14.69951, 14.67221, 14.64645, 14.62216, 14.59934, 14.57778, 14.55755, 14.53844, 14.52052, 14.50358, 14.48748, 14.47222, 14.45759, 14.44359, 14.42994, 14.41657, 14.40334, 14.39011, 14.37667, 14.36281, 14.34853, 14.33355, 14.31759 ),
    ampCut_endcapP = cms.vdouble( 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5 ),
    ampCut_endcapM = cms.vdouble( 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0 ),
    phiSymBarrelDigiCollection = cms.string( "phiSymEcalDigisEB" ),
    phiSymEndcapDigiCollection = cms.string( "phiSymEcalDigisEE" )
)
fragment.hltFEDSelectorL1 = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1404 )
)
fragment.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
fragment.hltL1sAlCaEcalPi0Eta = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_AlwaysTrue OR L1_IsolatedBunch OR L1_SingleEG8er2p5 OR L1_SingleEG10er2p5 OR L1_SingleEG15er2p5 OR L1_SingleEG26er2p5 OR L1_SingleEG34er2p5 OR L1_SingleEG36er2p5 OR L1_SingleEG38er2p5 OR L1_SingleEG40er2p5 OR L1_SingleEG42er2p5 OR L1_SingleEG45er2p5 OR L1_SingleEG60 OR L1_SingleIsoEG26er2p5 OR L1_SingleIsoEG28er2p5 OR L1_SingleIsoEG30er2p5 OR L1_SingleIsoEG32er2p5 OR L1_SingleIsoEG34er2p5 OR L1_SingleIsoEG24er2p1 OR L1_SingleIsoEG26er2p1 OR L1_SingleIsoEG28er2p1 OR L1_SingleIsoEG30er2p1 OR L1_SingleIsoEG32er2p1 OR L1_DoubleEG_22_10_er2p5 OR L1_DoubleEG_25_14_er2p5 OR L1_DoubleEG_25_12_er2p5 OR L1_SingleJet90 OR L1_SingleJet120 OR L1_SingleJet140er2p5 OR L1_SingleJet160er2p5 OR L1_SingleJet180 OR L1_SingleJet200 OR L1_DoubleJet40er2p5 OR L1_DoubleJet100er2p5 OR L1_DoubleJet120er2p5 OR L1_QuadJet60er2p5 OR L1_HTT120er OR L1_HTT160er OR L1_HTT200er OR L1_HTT255er OR L1_HTT280er OR L1_HTT320er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaEcalEtaEBonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltSimple3x3Clusters = cms.EDProducer( "EgammaHLTNxNClusterProducer",
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelHitProducer = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHitProducer = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    barrelClusterCollection = cms.string( "Simple3x3ClustersBarrel" ),
    endcapClusterCollection = cms.string( "Simple3x3ClustersEndcap" ),
    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    posCalcParameters = cms.PSet( 
      T0_barl = cms.double( 7.4 ),
      T0_endcPresh = cms.double( 1.2 ),
      LogWeighted = cms.bool( True ),
      T0_endc = cms.double( 3.1 ),
      X0 = cms.double( 0.89 ),
      W0 = cms.double( 4.2 )
    ),
    maxNumberofSeeds = cms.int32( 700 ),
    maxNumberofClusters = cms.int32( 300 ),
    debugLevel = cms.int32( 0 )
)
fragment.hltAlCaEtaRecHitsFilterEBonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
    barrelHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    doSelBarrel = cms.bool( True ),
    barrelSelection = cms.PSet( 
      massHighPi0Cand = cms.double( 0.156 ),
      ptMinForIsolation = cms.double( 1.0 ),
      seleMinvMaxBarrel = cms.double( 0.8 ),
      massLowPi0Cand = cms.double( 0.084 ),
      seleS9S25Gamma = cms.double( 0.8 ),
      seleBeltDeta = cms.double( 0.1 ),
      seleS4S9GammaBarrel_region2 = cms.double( 0.9 ),
      barrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
      removePi0CandidatesForEta = cms.bool( True ),
      seleMinvMinBarrel = cms.double( 0.2 ),
      seleS4S9GammaBarrel_region1 = cms.double( 0.9 ),
      selePtPairBarrel_region1 = cms.double( 3.0 ),
      selePtPairBarrel_region2 = cms.double( 3.0 ),
      seleBeltDR = cms.double( 0.3 ),
      region1_Barrel = cms.double( 1.0 ),
      seleIsoBarrel_region1 = cms.double( 0.5 ),
      selePtGammaBarrel_region1 = cms.double( 0.65 ),
      seleIsoBarrel_region2 = cms.double( 0.5 ),
      selePtGammaBarrel_region2 = cms.double( 1.4 ),
      store5x5RecHitEB = cms.bool( True ),
      seleNxtalBarrel_region2 = cms.uint32( 6 ),
      seleNxtalBarrel_region1 = cms.uint32( 6 )
    ),
    doSelEndcap = cms.bool( False ),
    endcapSelection = cms.PSet( 
      seleBeltDetaEndCap = cms.double( 0.05 ),
      selePtPairMaxEndCap_region3 = cms.double( 2.5 ),
      seleS4S9GammaEndCap_region2 = cms.double( 0.65 ),
      seleS4S9GammaEndCap_region1 = cms.double( 0.65 ),
      seleNxtalEndCap_region2 = cms.uint32( 6 ),
      seleNxtalEndCap_region3 = cms.uint32( 6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      selePtPairEndCap_region1 = cms.double( 1.5 ),
      endcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
      selePtPairEndCap_region2 = cms.double( 1.5 ),
      seleS4S9GammaEndCap_region3 = cms.double( 0.65 ),
      selePtGammaEndCap_region3 = cms.double( 0.5 ),
      selePtGammaEndCap_region2 = cms.double( 0.5 ),
      selePtGammaEndCap_region1 = cms.double( 0.5 ),
      region1_EndCap = cms.double( 1.8 ),
      region2_EndCap = cms.double( 2.0 ),
      store5x5RecHitEE = cms.bool( False ),
      seleIsoEndCap_region3 = cms.double( 0.5 ),
      seleIsoEndCap_region2 = cms.double( 0.5 ),
      seleMinvMinEndCap = cms.double( 0.05 ),
      selePtPairEndCap_region3 = cms.double( 99.0 ),
      seleIsoEndCap_region1 = cms.double( 0.5 ),
      seleBeltDREndCap = cms.double( 0.2 ),
      seleMinvMaxEndCap = cms.double( 0.3 ),
      seleNxtalEndCap_region1 = cms.uint32( 6 ),
      seleS9S25GammaEndCap = cms.double( 0.0 )
    ),
    storeRecHitES = cms.bool( False ),
    preshowerSelection = cms.PSet( 
      preshClusterEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      ESCollection = cms.string( "etaEcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshSeededNstrip = cms.int32( 15 ),
      preshCalibGamma = cms.double( 0.024 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 )
    ),
    debugLevel = cms.int32( 0 )
)
fragment.hltAlCaEtaEBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    EBRecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEBonlyRegional','etaEcalRecHitsEB' ),
    EERecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEBonlyRegional','etaEcalRecHitsEB' ),
    EBRecalibRecHitCollection = cms.string( "etaEcalRecHitsEB" ),
    EERecalibRecHitCollection = cms.string( "etaEcalRecHitsEE" ),
    doEnergyScale = cms.bool( False ),
    doIntercalib = cms.bool( False ),
    doLaserCorrections = cms.bool( False ),
    doEnergyScaleInverse = cms.bool( False ),
    doIntercalibInverse = cms.bool( False ),
    doLaserCorrectionsInverse = cms.bool( False )
)
fragment.hltAlCaEtaEBRechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "barrel" ),
    digisIn = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    digisOut = cms.string( "etaEBDigis" ),
    recHits = cms.InputTag( 'hltAlCaEtaEBUncalibrator','etaEcalRecHitsEB' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "etaEBSrFlags" )
)
fragment.hltPreAlCaEcalEtaEEonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltAlCaEtaRecHitsFilterEEonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
    barrelHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    doSelBarrel = cms.bool( False ),
    barrelSelection = cms.PSet( 
      massHighPi0Cand = cms.double( 0.163 ),
      ptMinForIsolation = cms.double( 1.0 ),
      seleMinvMaxBarrel = cms.double( 0.8 ),
      massLowPi0Cand = cms.double( 0.104 ),
      seleS9S25Gamma = cms.double( 0.0 ),
      seleBeltDeta = cms.double( 0.05 ),
      seleS4S9GammaBarrel_region2 = cms.double( 0.65 ),
      barrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
      removePi0CandidatesForEta = cms.bool( False ),
      seleMinvMinBarrel = cms.double( 0.3 ),
      seleS4S9GammaBarrel_region1 = cms.double( 0.65 ),
      selePtPairBarrel_region1 = cms.double( 1.5 ),
      selePtPairBarrel_region2 = cms.double( 1.5 ),
      seleBeltDR = cms.double( 0.2 ),
      region1_Barrel = cms.double( 1.0 ),
      seleIsoBarrel_region1 = cms.double( 0.5 ),
      selePtGammaBarrel_region1 = cms.double( 1.0 ),
      seleIsoBarrel_region2 = cms.double( 0.5 ),
      selePtGammaBarrel_region2 = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( False ),
      seleNxtalBarrel_region2 = cms.uint32( 6 ),
      seleNxtalBarrel_region1 = cms.uint32( 6 )
    ),
    doSelEndcap = cms.bool( True ),
    endcapSelection = cms.PSet( 
      seleBeltDetaEndCap = cms.double( 0.1 ),
      selePtPairMaxEndCap_region3 = cms.double( 999.0 ),
      seleS4S9GammaEndCap_region2 = cms.double( 0.9 ),
      seleS4S9GammaEndCap_region1 = cms.double( 0.9 ),
      seleNxtalEndCap_region2 = cms.uint32( 6 ),
      seleNxtalEndCap_region3 = cms.uint32( 6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      selePtPairEndCap_region1 = cms.double( 3.0 ),
      endcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
      selePtPairEndCap_region2 = cms.double( 3.0 ),
      seleS4S9GammaEndCap_region3 = cms.double( 0.9 ),
      selePtGammaEndCap_region3 = cms.double( 1.0 ),
      selePtGammaEndCap_region2 = cms.double( 1.0 ),
      selePtGammaEndCap_region1 = cms.double( 1.0 ),
      region1_EndCap = cms.double( 1.8 ),
      region2_EndCap = cms.double( 2.0 ),
      store5x5RecHitEE = cms.bool( True ),
      seleIsoEndCap_region3 = cms.double( 0.5 ),
      seleIsoEndCap_region2 = cms.double( 0.5 ),
      seleMinvMinEndCap = cms.double( 0.2 ),
      selePtPairEndCap_region3 = cms.double( 3.0 ),
      seleIsoEndCap_region1 = cms.double( 0.5 ),
      seleBeltDREndCap = cms.double( 0.3 ),
      seleMinvMaxEndCap = cms.double( 0.8 ),
      seleNxtalEndCap_region1 = cms.uint32( 6 ),
      seleS9S25GammaEndCap = cms.double( 0.85 )
    ),
    storeRecHitES = cms.bool( True ),
    preshowerSelection = cms.PSet( 
      preshClusterEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      ESCollection = cms.string( "etaEcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshSeededNstrip = cms.int32( 15 ),
      preshCalibGamma = cms.double( 0.024 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 )
    ),
    debugLevel = cms.int32( 0 )
)
fragment.hltAlCaEtaEEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    EBRecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEEonlyRegional','etaEcalRecHitsEE' ),
    EERecHitCollection = cms.InputTag( 'hltAlCaEtaRecHitsFilterEEonlyRegional','etaEcalRecHitsEE' ),
    EBRecalibRecHitCollection = cms.string( "etaEcalRecHitsEB" ),
    EERecalibRecHitCollection = cms.string( "etaEcalRecHitsEE" ),
    doEnergyScale = cms.bool( False ),
    doIntercalib = cms.bool( False ),
    doLaserCorrections = cms.bool( False ),
    doEnergyScaleInverse = cms.bool( False ),
    doIntercalibInverse = cms.bool( False ),
    doLaserCorrectionsInverse = cms.bool( False )
)
fragment.hltAlCaEtaEERechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "endcap" ),
    digisIn = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    digisOut = cms.string( "etaEEDigis" ),
    recHits = cms.InputTag( 'hltAlCaEtaEEUncalibrator','etaEcalRecHitsEE' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "etaEESrFlags" )
)
fragment.hltPreAlCaEcalPi0EBonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltAlCaPi0RecHitsFilterEBonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
    barrelHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    doSelBarrel = cms.bool( True ),
    barrelSelection = cms.PSet( 
      massHighPi0Cand = cms.double( 0.163 ),
      ptMinForIsolation = cms.double( 1.0 ),
      seleMinvMaxBarrel = cms.double( 0.22 ),
      massLowPi0Cand = cms.double( 0.104 ),
      seleS9S25Gamma = cms.double( 0.0 ),
      seleBeltDeta = cms.double( 0.05 ),
      seleS4S9GammaBarrel_region2 = cms.double( 0.9 ),
      barrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
      removePi0CandidatesForEta = cms.bool( False ),
      seleMinvMinBarrel = cms.double( 0.06 ),
      seleS4S9GammaBarrel_region1 = cms.double( 0.88 ),
      selePtPairBarrel_region1 = cms.double( 2.0 ),
      selePtPairBarrel_region2 = cms.double( 1.75 ),
      seleBeltDR = cms.double( 0.2 ),
      region1_Barrel = cms.double( 1.0 ),
      seleIsoBarrel_region1 = cms.double( 0.5 ),
      selePtGammaBarrel_region1 = cms.double( 0.65 ),
      seleIsoBarrel_region2 = cms.double( 0.5 ),
      selePtGammaBarrel_region2 = cms.double( 0.65 ),
      store5x5RecHitEB = cms.bool( False ),
      seleNxtalBarrel_region2 = cms.uint32( 6 ),
      seleNxtalBarrel_region1 = cms.uint32( 6 )
    ),
    doSelEndcap = cms.bool( False ),
    endcapSelection = cms.PSet( 
      seleBeltDetaEndCap = cms.double( 0.05 ),
      selePtPairMaxEndCap_region3 = cms.double( 2.5 ),
      seleS4S9GammaEndCap_region2 = cms.double( 0.65 ),
      seleS4S9GammaEndCap_region1 = cms.double( 0.65 ),
      seleNxtalEndCap_region2 = cms.uint32( 6 ),
      seleNxtalEndCap_region3 = cms.uint32( 6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      selePtPairEndCap_region1 = cms.double( 1.5 ),
      endcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
      selePtPairEndCap_region2 = cms.double( 1.5 ),
      seleS4S9GammaEndCap_region3 = cms.double( 0.65 ),
      selePtGammaEndCap_region3 = cms.double( 0.5 ),
      selePtGammaEndCap_region2 = cms.double( 0.5 ),
      selePtGammaEndCap_region1 = cms.double( 0.5 ),
      region1_EndCap = cms.double( 1.8 ),
      region2_EndCap = cms.double( 2.0 ),
      store5x5RecHitEE = cms.bool( False ),
      seleIsoEndCap_region3 = cms.double( 0.5 ),
      seleIsoEndCap_region2 = cms.double( 0.5 ),
      seleMinvMinEndCap = cms.double( 0.05 ),
      selePtPairEndCap_region3 = cms.double( 99.0 ),
      seleIsoEndCap_region1 = cms.double( 0.5 ),
      seleBeltDREndCap = cms.double( 0.2 ),
      seleMinvMaxEndCap = cms.double( 0.3 ),
      seleNxtalEndCap_region1 = cms.uint32( 6 ),
      seleS9S25GammaEndCap = cms.double( 0.0 )
    ),
    storeRecHitES = cms.bool( False ),
    preshowerSelection = cms.PSet( 
      preshClusterEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      ESCollection = cms.string( "pi0EcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshSeededNstrip = cms.int32( 15 ),
      preshCalibGamma = cms.double( 0.024 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 )
    ),
    debugLevel = cms.int32( 0 )
)
fragment.hltAlCaPi0EBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    EBRecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEBonlyRegional','pi0EcalRecHitsEB' ),
    EERecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEBonlyRegional','pi0EcalRecHitsEB' ),
    EBRecalibRecHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    EERecalibRecHitCollection = cms.string( "pi0EcalRecHitsEE" ),
    doEnergyScale = cms.bool( False ),
    doIntercalib = cms.bool( False ),
    doLaserCorrections = cms.bool( False ),
    doEnergyScaleInverse = cms.bool( False ),
    doIntercalibInverse = cms.bool( False ),
    doLaserCorrectionsInverse = cms.bool( False )
)
fragment.hltAlCaPi0EBRechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "barrel" ),
    digisIn = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    digisOut = cms.string( "pi0EBDigis" ),
    recHits = cms.InputTag( 'hltAlCaPi0EBUncalibrator','pi0EcalRecHitsEB' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "pi0EBSrFlags" )
)
fragment.hltPreAlCaEcalPi0EEonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltAlCaPi0RecHitsFilterEEonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
    barrelHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    barrelClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersBarrel' ),
    endcapClusters = cms.InputTag( 'hltSimple3x3Clusters','Simple3x3ClustersEndcap' ),
    useRecoFlag = cms.bool( False ),
    flagLevelRecHitsToUse = cms.int32( 1 ),
    useDBStatus = cms.bool( True ),
    statusLevelRecHitsToUse = cms.int32( 1 ),
    doSelBarrel = cms.bool( False ),
    barrelSelection = cms.PSet( 
      massHighPi0Cand = cms.double( 0.163 ),
      ptMinForIsolation = cms.double( 1.0 ),
      seleMinvMaxBarrel = cms.double( 0.22 ),
      massLowPi0Cand = cms.double( 0.104 ),
      seleS9S25Gamma = cms.double( 0.0 ),
      seleBeltDeta = cms.double( 0.05 ),
      seleS4S9GammaBarrel_region2 = cms.double( 0.65 ),
      barrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
      removePi0CandidatesForEta = cms.bool( False ),
      seleMinvMinBarrel = cms.double( 0.06 ),
      seleS4S9GammaBarrel_region1 = cms.double( 0.65 ),
      selePtPairBarrel_region1 = cms.double( 1.5 ),
      selePtPairBarrel_region2 = cms.double( 1.5 ),
      seleBeltDR = cms.double( 0.2 ),
      region1_Barrel = cms.double( 1.0 ),
      seleIsoBarrel_region1 = cms.double( 0.5 ),
      selePtGammaBarrel_region1 = cms.double( 0.5 ),
      seleIsoBarrel_region2 = cms.double( 0.5 ),
      selePtGammaBarrel_region2 = cms.double( 0.5 ),
      store5x5RecHitEB = cms.bool( False ),
      seleNxtalBarrel_region2 = cms.uint32( 6 ),
      seleNxtalBarrel_region1 = cms.uint32( 6 )
    ),
    doSelEndcap = cms.bool( True ),
    endcapSelection = cms.PSet( 
      seleBeltDetaEndCap = cms.double( 0.05 ),
      selePtPairMaxEndCap_region3 = cms.double( 999.0 ),
      seleS4S9GammaEndCap_region2 = cms.double( 0.92 ),
      seleS4S9GammaEndCap_region1 = cms.double( 0.85 ),
      seleNxtalEndCap_region2 = cms.uint32( 6 ),
      seleNxtalEndCap_region3 = cms.uint32( 6 ),
      ptMinForIsolationEndCap = cms.double( 0.5 ),
      selePtPairEndCap_region1 = cms.double( 3.75 ),
      endcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
      selePtPairEndCap_region2 = cms.double( 2.0 ),
      seleS4S9GammaEndCap_region3 = cms.double( 0.92 ),
      selePtGammaEndCap_region3 = cms.double( 0.95 ),
      selePtGammaEndCap_region2 = cms.double( 0.95 ),
      selePtGammaEndCap_region1 = cms.double( 1.1 ),
      region1_EndCap = cms.double( 1.8 ),
      region2_EndCap = cms.double( 2.0 ),
      store5x5RecHitEE = cms.bool( False ),
      seleIsoEndCap_region3 = cms.double( 0.5 ),
      seleIsoEndCap_region2 = cms.double( 0.5 ),
      seleMinvMinEndCap = cms.double( 0.05 ),
      selePtPairEndCap_region3 = cms.double( 2.0 ),
      seleIsoEndCap_region1 = cms.double( 0.5 ),
      seleBeltDREndCap = cms.double( 0.2 ),
      seleMinvMaxEndCap = cms.double( 0.3 ),
      seleNxtalEndCap_region1 = cms.uint32( 6 ),
      seleS9S25GammaEndCap = cms.double( 0.0 )
    ),
    storeRecHitES = cms.bool( True ),
    preshowerSelection = cms.PSet( 
      preshClusterEnergyCut = cms.double( 0.0 ),
      debugLevelES = cms.string( "" ),
      ESCollection = cms.string( "pi0EcalRecHitsES" ),
      preshNclust = cms.int32( 4 ),
      preshStripEnergyCut = cms.double( 0.0 ),
      preshCalibPlaneY = cms.double( 0.7 ),
      preshSeededNstrip = cms.int32( 15 ),
      preshCalibGamma = cms.double( 0.024 ),
      preshCalibPlaneX = cms.double( 1.0 ),
      preshCalibMIP = cms.double( 9.0E-5 )
    ),
    debugLevel = cms.int32( 0 )
)
fragment.hltAlCaPi0EEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
    EBRecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEEonlyRegional','pi0EcalRecHitsEE' ),
    EERecHitCollection = cms.InputTag( 'hltAlCaPi0RecHitsFilterEEonlyRegional','pi0EcalRecHitsEE' ),
    EBRecalibRecHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    EERecalibRecHitCollection = cms.string( "pi0EcalRecHitsEE" ),
    doEnergyScale = cms.bool( False ),
    doIntercalib = cms.bool( False ),
    doLaserCorrections = cms.bool( False ),
    doEnergyScaleInverse = cms.bool( False ),
    doIntercalibInverse = cms.bool( False ),
    doLaserCorrectionsInverse = cms.bool( False )
)
fragment.hltAlCaPi0EERechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "endcap" ),
    digisIn = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    digisOut = cms.string( "pi0EEDigis" ),
    recHits = cms.InputTag( 'hltAlCaPi0EEUncalibrator','pi0EcalRecHitsEE' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "pi0EESrFlags" )
)
fragment.hltL1sSingleMu5IorSingleMu14erIorSingleMu16er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5 OR L1_SingleMu7 OR L1_SingleMu18 OR L1_SingleMu20 OR L1_SingleMu22 OR L1_SingleMu25" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaRPCMuonNormalisation = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltRPCMuonNormaL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu5IorSingleMu14erIorSingleMu16er" ),
    MaxEta = cms.double( 2.4 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltFEDSelectorTCDS = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1024, 1025 )
)
fragment.hltFEDSelectorDT = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1369, 1370, 1371 )
)
fragment.hltFEDSelectorRPC = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 790, 791, 792, 793, 794, 795, 821 )
)
fragment.hltFEDSelectorCSC = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 831, 832, 833, 834, 835, 836, 837, 838, 839, 841, 842, 843, 844, 845, 846, 847, 848, 849, 851, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 869 )
)
fragment.hltFEDSelectorGEM = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478 )
)
fragment.hltFEDSelectorTwinMux = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1390, 1391, 1393, 1394, 1395 )
)
fragment.hltFEDSelectorOMTF = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1380, 1381 )
)
fragment.hltFEDSelectorCPPF = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1386 )
)
fragment.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
fragment.hltPreAlCaLumiPixelsCountsRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPixelTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "pixel" ),
    acceptedCombinations = cms.untracked.vstring(  ),
    DcsStatusLabel = cms.untracked.InputTag( "" ),
    DCSRecordLabel = cms.untracked.InputTag( "hltOnlineMetaDataDigis" )
)
fragment.hltOnlineBeamSpotDevice = cms.EDProducer( "BeamSpotDeviceProducer@alpaka",
    src = cms.InputTag( "hltOnlineBeamSpot" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltSiPixelClustersSoA = cms.EDProducer( "SiPixelRawToClusterPhase1@alpaka",
    IncludeErrors = cms.bool( True ),
    UseQualityInfo = cms.bool( False ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    VCaltoElectronGain = cms.double( 1.0 ),
    VCaltoElectronGain_L1 = cms.double( 1.0 ),
    VCaltoElectronOffset = cms.double( 0.0 ),
    VCaltoElectronOffset_L1 = cms.double( 0.0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    Regions = cms.PSet(  ),
    CablingMapLabel = cms.string( "" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltSiPixelClusters = cms.EDProducer( "SiPixelDigisClustersFromSoAAlpakaPhase1",
    src = cms.InputTag( "hltSiPixelClustersSoA" ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    produceDigis = cms.bool( False ),
    storeDigis = cms.bool( False )
)
fragment.hltSiPixelDigiErrors = cms.EDProducer( "SiPixelDigiErrorsFromSoAAlpaka",
    digiErrorSoASrc = cms.InputTag( "hltSiPixelClustersSoA" ),
    fmtErrorsSoASrc = cms.InputTag( "hltSiPixelClustersSoA" ),
    CablingMapLabel = cms.string( "" ),
    UsePhase1 = cms.bool( True ),
    ErrorList = cms.vint32( 29 ),
    UserErrorList = cms.vint32( 40 )
)
fragment.hltSiPixelRecHitsSoA = cms.EDProducer( "SiPixelRecHitAlpakaPhase1@alpaka",
    beamSpot = cms.InputTag( "hltOnlineBeamSpotDevice" ),
    src = cms.InputTag( "hltSiPixelClustersSoA" ),
    CPE = cms.string( "PixelCPEFastParams" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltSiPixelRecHits = cms.EDProducer( "SiPixelRecHitFromSoAAlpakaPhase1",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    src = cms.InputTag( "hltSiPixelClusters" )
)
fragment.hltAlcaPixelClusterCounts = cms.EDProducer( "AlcaPCCEventProducer",
    pixelClusterLabel = cms.InputTag( "hltSiPixelClusters" ),
    trigstring = cms.untracked.string( "alcaPCCEvent" ),
    savePerROCInfo = cms.bool( True )
)
fragment.hltL1sZeroBias = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaLumiPixelsCountsZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sDQMPixelReconstruction = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreDQMPixelReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltOnlineBeamSpotDeviceSerialSync = cms.EDProducer( "alpaka_serial_sync::BeamSpotDeviceProducer",
    src = cms.InputTag( "hltOnlineBeamSpot" )
)
fragment.hltSiPixelClustersSoASerialSync = cms.EDProducer( "alpaka_serial_sync::SiPixelRawToClusterPhase1",
    IncludeErrors = cms.bool( True ),
    UseQualityInfo = cms.bool( False ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    VCaltoElectronGain = cms.double( 1.0 ),
    VCaltoElectronGain_L1 = cms.double( 1.0 ),
    VCaltoElectronOffset = cms.double( 0.0 ),
    VCaltoElectronOffset_L1 = cms.double( 0.0 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    Regions = cms.PSet(  ),
    CablingMapLabel = cms.string( "" )
)
fragment.hltSiPixelClustersSerialSync = cms.EDProducer( "SiPixelDigisClustersFromSoAAlpakaPhase1",
    src = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    produceDigis = cms.bool( False ),
    storeDigis = cms.bool( False )
)
fragment.hltSiPixelDigiErrorsSerialSync = cms.EDProducer( "SiPixelDigiErrorsFromSoAAlpaka",
    digiErrorSoASrc = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    fmtErrorsSoASrc = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    CablingMapLabel = cms.string( "" ),
    UsePhase1 = cms.bool( True ),
    ErrorList = cms.vint32( 29 ),
    UserErrorList = cms.vint32( 40 )
)
fragment.hltSiPixelRecHitsSoASerialSync = cms.EDProducer( "alpaka_serial_sync::SiPixelRecHitAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpotDeviceSerialSync" ),
    src = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    CPE = cms.string( "PixelCPEFastParams" )
)
fragment.hltSiPixelRecHitsSerialSync = cms.EDProducer( "SiPixelRecHitFromSoAAlpakaPhase1",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    src = cms.InputTag( "hltSiPixelClustersSerialSync" )
)
fragment.hltPixelTracksSoA = cms.EDProducer( "CAHitNtupletAlpakaPhase1@alpaka",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    CPE = cms.string( "PixelCPEFastParams" ),
    ptmin = cms.double( 0.9 ),
    CAThetaCutBarrel = cms.double( 0.002 ),
    CAThetaCutForward = cms.double( 0.003 ),
    hardCurvCut = cms.double( 0.0328407225 ),
    dcaCutInnerTriplet = cms.double( 0.15 ),
    dcaCutOuterTriplet = cms.double( 0.25 ),
    earlyFishbone = cms.bool( True ),
    lateFishbone = cms.bool( False ),
    fillStatistics = cms.bool( False ),
    minHitsPerNtuplet = cms.uint32( 3 ),
    minHitsForSharingCut = cms.uint32( 10 ),
    fitNas4 = cms.bool( False ),
    doClusterCut = cms.bool( True ),
    doZ0Cut = cms.bool( True ),
    doPtCut = cms.bool( True ),
    useRiemannFit = cms.bool( False ),
    doSharedHitCut = cms.bool( True ),
    dupPassThrough = cms.bool( False ),
    useSimpleTripletCleaner = cms.bool( True ),
    maxNumberOfDoublets = cms.uint32( 524288 ),
    idealConditions = cms.bool( False ),
    includeJumpingForwardDoublets = cms.bool( True ),
    cellZ0Cut = cms.double( 12.0 ),
    cellPtCut = cms.double( 0.5 ),
    trackQualityCuts = cms.PSet( 
      chi2MaxPt = cms.double( 10.0 ),
      tripletMaxTip = cms.double( 0.3 ),
      chi2Scale = cms.double( 8.0 ),
      quadrupletMaxTip = cms.double( 0.5 ),
      quadrupletMinPt = cms.double( 0.3 ),
      quadrupletMaxZip = cms.double( 12.0 ),
      tripletMaxZip = cms.double( 12.0 ),
      tripletMinPt = cms.double( 0.5 ),
      chi2Coeff = cms.vdouble( 0.9, 1.8 )
    ),
    minYsizeB1 = cms.int32( 1 ),
    minYsizeB2 = cms.int32( 1 ),
    phiCuts = cms.vint32( 522, 730, 730, 522, 626, 626, 522, 522, 626, 626, 626, 522, 522, 522, 522, 522, 522, 522, 522 ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltPixelTracks = cms.EDProducer( "PixelTrackProducerFromSoAAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    trackSrc = cms.InputTag( "hltPixelTracksSoA" ),
    pixelRecHitLegacySrc = cms.InputTag( "hltSiPixelRecHits" ),
    minNumberOfHits = cms.int32( 0 ),
    minQuality = cms.string( "loose" )
)
fragment.hltPixelVerticesSoA = cms.EDProducer( "PixelVertexProducerAlpakaPhase1@alpaka",
    oneKernel = cms.bool( True ),
    useDensity = cms.bool( True ),
    useDBSCAN = cms.bool( False ),
    useIterative = cms.bool( False ),
    doSplitting = cms.bool( True ),
    minT = cms.int32( 2 ),
    eps = cms.double( 0.07 ),
    errmax = cms.double( 0.01 ),
    chi2max = cms.double( 9.0 ),
    maxVertices = cms.int32( 256 ),
    PtMin = cms.double( 0.5 ),
    PtMax = cms.double( 75.0 ),
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoA" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltPixelVertices = cms.EDProducer( "PixelVertexProducerFromSoAAlpaka",
    TrackCollection = cms.InputTag( "hltPixelTracks" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    src = cms.InputTag( "hltPixelVerticesSoA" )
)
fragment.hltTrimmedPixelVertices = cms.EDProducer( "PixelVertexCollectionTrimmer",
    src = cms.InputTag( "hltPixelVertices" ),
    maxVtx = cms.uint32( 100 ),
    fractionSumPt2 = cms.double( 0.3 ),
    minSumPt2 = cms.double( 0.0 ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) )
)
fragment.hltPixelTracksSoASerialSync = cms.EDProducer( "alpaka_serial_sync::CAHitNtupletAlpakaPhase1",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    CPE = cms.string( "PixelCPEFastParams" ),
    ptmin = cms.double( 0.9 ),
    CAThetaCutBarrel = cms.double( 0.002 ),
    CAThetaCutForward = cms.double( 0.003 ),
    hardCurvCut = cms.double( 0.0328407225 ),
    dcaCutInnerTriplet = cms.double( 0.15 ),
    dcaCutOuterTriplet = cms.double( 0.25 ),
    earlyFishbone = cms.bool( True ),
    lateFishbone = cms.bool( False ),
    fillStatistics = cms.bool( False ),
    minHitsPerNtuplet = cms.uint32( 3 ),
    minHitsForSharingCut = cms.uint32( 10 ),
    fitNas4 = cms.bool( False ),
    doClusterCut = cms.bool( True ),
    doZ0Cut = cms.bool( True ),
    doPtCut = cms.bool( True ),
    useRiemannFit = cms.bool( False ),
    doSharedHitCut = cms.bool( True ),
    dupPassThrough = cms.bool( False ),
    useSimpleTripletCleaner = cms.bool( True ),
    maxNumberOfDoublets = cms.uint32( 524288 ),
    idealConditions = cms.bool( False ),
    includeJumpingForwardDoublets = cms.bool( True ),
    cellZ0Cut = cms.double( 12.0 ),
    cellPtCut = cms.double( 0.5 ),
    trackQualityCuts = cms.PSet( 
      chi2MaxPt = cms.double( 10.0 ),
      tripletMaxTip = cms.double( 0.3 ),
      chi2Scale = cms.double( 8.0 ),
      quadrupletMaxTip = cms.double( 0.5 ),
      quadrupletMinPt = cms.double( 0.3 ),
      quadrupletMaxZip = cms.double( 12.0 ),
      tripletMaxZip = cms.double( 12.0 ),
      tripletMinPt = cms.double( 0.5 ),
      chi2Coeff = cms.vdouble( 0.9, 1.8 )
    ),
    minYsizeB1 = cms.int32( 1 ),
    minYsizeB2 = cms.int32( 1 ),
    phiCuts = cms.vint32( 522, 730, 730, 522, 626, 626, 522, 522, 626, 626, 626, 522, 522, 522, 522, 522, 522, 522, 522 )
)
fragment.hltPixelTracksSerialSync = cms.EDProducer( "PixelTrackProducerFromSoAAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    trackSrc = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    pixelRecHitLegacySrc = cms.InputTag( "hltSiPixelRecHitsSerialSync" ),
    minNumberOfHits = cms.int32( 0 ),
    minQuality = cms.string( "loose" )
)
fragment.hltPixelVerticesSoASerialSync = cms.EDProducer( "alpaka_serial_sync::PixelVertexProducerAlpakaPhase1",
    oneKernel = cms.bool( True ),
    useDensity = cms.bool( True ),
    useDBSCAN = cms.bool( False ),
    useIterative = cms.bool( False ),
    doSplitting = cms.bool( True ),
    minT = cms.int32( 2 ),
    eps = cms.double( 0.07 ),
    errmax = cms.double( 0.01 ),
    chi2max = cms.double( 9.0 ),
    maxVertices = cms.int32( 256 ),
    PtMin = cms.double( 0.5 ),
    PtMax = cms.double( 75.0 ),
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoASerialSync" )
)
fragment.hltPixelVerticesSerialSync = cms.EDProducer( "PixelVertexProducerFromSoAAlpaka",
    TrackCollection = cms.InputTag( "hltPixelTracksSerialSync" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    src = cms.InputTag( "hltPixelVerticesSoASerialSync" )
)
fragment.hltTrimmedPixelVerticesSerialSync = cms.EDProducer( "PixelVertexCollectionTrimmer",
    src = cms.InputTag( "hltPixelVerticesSerialSync" ),
    maxVtx = cms.uint32( 100 ),
    fractionSumPt2 = cms.double( 0.3 ),
    minSumPt2 = cms.double( 0.0 ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) )
)
fragment.hltSiPixelRecHitsSoAMonitorCPU = cms.EDProducer( "SiPixelPhase1MonitorRecHitsSoAAlpaka",
    pixelHitsSrc = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    TopFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsCPU" )
)
fragment.hltSiPixelRecHitsSoAMonitorGPU = cms.EDProducer( "SiPixelPhase1MonitorRecHitsSoAAlpaka",
    pixelHitsSrc = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    TopFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsGPU" )
)
fragment.hltSiPixelRecHitsSoACompareGPUvsCPU = cms.EDProducer( "SiPixelPhase1CompareRecHits",
    pixelHitsReferenceSoA = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    pixelHitsTargetSoA = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU" ),
    minD2cut = cms.double( 1.0E-4 )
)
fragment.hltPixelTracksSoAMonitorCPU = cms.EDProducer( "SiPixelPhase1MonitorTrackSoAAlpaka",
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackCPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" )
)
fragment.hltPixelTracksSoAMonitorGPU = cms.EDProducer( "SiPixelPhase1MonitorTrackSoAAlpaka",
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackGPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" )
)
fragment.hltPixelTracksSoACompareGPUvsCPU = cms.EDProducer( "SiPixelPhase1CompareTracks",
    pixelTrackReferenceSoA = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    pixelTrackTargetSoA = cms.InputTag( "hltPixelTracksSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" ),
    deltaR2cut = cms.double( 4.0E-4 )
)
fragment.hltPixelVerticesSoAMonitorCPU = cms.EDProducer( "SiPixelMonitorVertexSoAAlpaka",
    pixelVertexSrc = cms.InputTag( "hltPixelVerticesSoASerialSync" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexCPU" )
)
fragment.hltPixelVerticesSoAMonitorGPU = cms.EDProducer( "SiPixelMonitorVertexSoAAlpaka",
    pixelVertexSrc = cms.InputTag( "hltPixelVerticesSoA" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexGPU" )
)
fragment.hltPixelVerticesSoACompareGPUvsCPU = cms.EDProducer( "SiPixelCompareVertices",
    pixelVertexReferenceSoA = cms.InputTag( "hltPixelVerticesSoASerialSync" ),
    pixelVertexTargetSoA = cms.InputTag( "hltPixelVerticesSoA" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU" ),
    dzCut = cms.double( 1.0 )
)
fragment.hltL1sDQMEcalReconstruction = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreDQMEcalReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltEcalDigisSoASerialSync = cms.EDProducer( "alpaka_serial_sync::EcalRawToDigiPortable",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    maxChannelsEB = cms.uint32( 61200 ),
    maxChannelsEE = cms.uint32( 14648 ),
    digisLabelEB = cms.string( "ebDigis" ),
    digisLabelEE = cms.string( "eeDigis" )
)
fragment.hltEcalDigisSerialSync = cms.EDProducer( "EcalDigisFromPortableProducer",
    digisInLabelEB = cms.InputTag( 'hltEcalDigisSoASerialSync','ebDigis' ),
    digisInLabelEE = cms.InputTag( 'hltEcalDigisSoASerialSync','eeDigis' ),
    digisOutLabelEB = cms.string( "ebDigis" ),
    digisOutLabelEE = cms.string( "eeDigis" ),
    produceDummyIntegrityCollections = cms.bool( False )
)
fragment.hltEcalUncalibRecHitSoASerialSync = cms.EDProducer( "alpaka_serial_sync::EcalUncalibRecHitProducerPortable",
    digisLabelEB = cms.InputTag( 'hltEcalDigisSoASerialSync','ebDigis' ),
    digisLabelEE = cms.InputTag( 'hltEcalDigisSoASerialSync','eeDigis' ),
    recHitsLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    recHitsLabelEE = cms.string( "EcalUncalibRecHitsEE" ),
    EBtimeFitLimits_Lower = cms.double( 0.2 ),
    EBtimeFitLimits_Upper = cms.double( 1.4 ),
    EEtimeFitLimits_Lower = cms.double( 0.2 ),
    EEtimeFitLimits_Upper = cms.double( 1.4 ),
    EBtimeConstantTerm = cms.double( 0.6 ),
    EEtimeConstantTerm = cms.double( 1.0 ),
    EBtimeNconst = cms.double( 28.5 ),
    EEtimeNconst = cms.double( 31.8 ),
    outOfTimeThresholdGain12pEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain12mEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain61pEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain61mEB = cms.double( 1000.0 ),
    outOfTimeThresholdGain12pEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain12mEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain61pEE = cms.double( 1000.0 ),
    outOfTimeThresholdGain61mEE = cms.double( 1000.0 ),
    amplitudeThresholdEB = cms.double( 10.0 ),
    amplitudeThresholdEE = cms.double( 10.0 ),
    EBtimeFitParameters = cms.vdouble( -2.015452, 3.130702, -12.3473, 41.88921, -82.83944, 91.01147, -50.35761, 11.05621 ),
    EEtimeFitParameters = cms.vdouble( -2.390548, 3.553628, -17.62341, 67.67538, -133.213, 140.7432, -75.41106, 16.20277 ),
    EBamplitudeFitParameters = cms.vdouble( 1.138, 1.652 ),
    EEamplitudeFitParameters = cms.vdouble( 1.89, 1.4 ),
    kernelMinimizeThreads = cms.untracked.vuint32( 32, 1, 1 ),
    shouldRunTimingComputation = cms.bool( True )
)
fragment.hltEcalUncalibRecHitSerialSync = cms.EDProducer( "EcalUncalibRecHitSoAToLegacy",
    inputCollectionEB = cms.InputTag( 'hltEcalUncalibRecHitSoASerialSync','EcalUncalibRecHitsEB' ),
    outputLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    isPhase2 = cms.bool( False ),
    inputCollectionEE = cms.InputTag( 'hltEcalUncalibRecHitSoASerialSync','EcalUncalibRecHitsEE' ),
    outputLabelEE = cms.string( "EcalUncalibRecHitsEE" )
)
fragment.hltEcalRecHitSerialSync = cms.EDProducer( "EcalRecHitProducer",
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHitSerialSync','EcalUncalibRecHitsEE' ),
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalUncalibRecHitSerialSync','EcalUncalibRecHitsEB' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    ChannelStatusToBeExcluded = cms.vstring(  ),
    killDeadChannels = cms.bool( True ),
    algo = cms.string( "EcalRecHitWorkerSimple" ),
    EBLaserMIN = cms.double( 0.5 ),
    EELaserMIN = cms.double( 0.5 ),
    EBLaserMAX = cms.double( 3.0 ),
    EELaserMAX = cms.double( 8.0 ),
    timeCalibTag = cms.ESInputTag( "","" ),
    timeOffsetTag = cms.ESInputTag( "","" ),
    skipTimeCalib = cms.bool( False ),
    laserCorrection = cms.bool( True ),
    flagsMapDBReco = cms.PSet( 
      kDead = cms.vstring( 'kNoDataNoTP' ),
      kGood = cms.vstring( 'kOk',
        'kDAC',
        'kNoLaser',
        'kNoisy' ),
      kTowerRecovered = cms.vstring( 'kDeadFE' ),
      kNoisy = cms.vstring( 'kNNoisy',
        'kFixedG6',
        'kFixedG1' ),
      kNeighboursRecovered = cms.vstring( 'kFixedG0',
        'kNonRespondingIsolated',
        'kDeadVFE' )
    ),
    algoRecover = cms.string( "EcalRecHitWorkerRecover" ),
    recoverEBIsolatedChannels = cms.bool( False ),
    recoverEEIsolatedChannels = cms.bool( False ),
    recoverEBVFE = cms.bool( False ),
    recoverEEVFE = cms.bool( False ),
    recoverEBFE = cms.bool( False ),
    recoverEEFE = cms.bool( False ),
    dbStatusToBeExcludedEE = cms.vint32( 14, 78, 142 ),
    dbStatusToBeExcludedEB = cms.vint32( 14, 78, 142 ),
    logWarningEtThreshold_EB_FE = cms.double( -1.0 ),
    logWarningEtThreshold_EE_FE = cms.double( -1.0 ),
    ebDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebDetId' ),
    eeDetIdToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeDetId' ),
    ebFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','ebFE' ),
    eeFEToBeRecovered = cms.InputTag( 'hltEcalDetIdToBeRecovered','eeFE' ),
    singleChannelRecoveryMethod = cms.string( "NeuralNetworks" ),
    singleChannelRecoveryThreshold = cms.double( 8.0 ),
    sum8ChannelRecoveryThreshold = cms.double( 0.0 ),
    bdtWeightFileNoCracks = cms.FileInPath( "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/bdtgAllRH_8GT700MeV_noCracks_ZskimData2017_v1.xml" ),
    bdtWeightFileCracks = cms.FileInPath( "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/bdtgAllRH_8GT700MeV_onlyCracks_ZskimData2017_v1.xml" ),
    triggerPrimitiveDigiCollection = cms.InputTag( 'hltEcalDigisLegacy','EcalTriggerPrimitives' ),
    cleaningConfig = cms.PSet( 
      cThreshold_endcap = cms.double( 15.0 ),
      tightenCrack_e1_double = cms.double( 2.0 ),
      cThreshold_barrel = cms.double( 4.0 ),
      e6e2thresh = cms.double( 0.04 ),
      e4e1Threshold_barrel = cms.double( 0.08 ),
      e4e1Threshold_endcap = cms.double( 0.3 ),
      tightenCrack_e4e1_single = cms.double( 3.0 ),
      cThreshold_double = cms.double( 10.0 ),
      e4e1_b_barrel = cms.double( -0.024 ),
      tightenCrack_e6e2_double = cms.double( 3.0 ),
      e4e1_a_barrel = cms.double( 0.04 ),
      tightenCrack_e1_single = cms.double( 2.0 ),
      e4e1_a_endcap = cms.double( 0.02 ),
      e4e1_b_endcap = cms.double( -0.0125 ),
      ignoreOutOfTimeThresh = cms.double( 1.0E9 )
    )
)
fragment.hltL1sDQMHcalReconstruction = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreDQMHcalReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    HcalFirstFED = cms.untracked.int32( 700 ),
    firstSample = cms.int32( 0 ),
    lastSample = cms.int32( 9 ),
    FilterDataQuality = cms.bool( True ),
    FEDs = cms.untracked.vint32(  ),
    UnpackZDC = cms.untracked.bool( True ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackUMNio = cms.untracked.bool( True ),
    UnpackTTP = cms.untracked.bool( False ),
    silent = cms.untracked.bool( True ),
    saveQIE10DataNSamples = cms.untracked.vint32(  ),
    saveQIE10DataTags = cms.untracked.vstring(  ),
    saveQIE11DataNSamples = cms.untracked.vint32(  ),
    saveQIE11DataTags = cms.untracked.vstring(  ),
    ComplainEmptyData = cms.untracked.bool( False ),
    UnpackerMode = cms.untracked.int32( 0 ),
    ExpectedOrbitMessageTime = cms.untracked.int32( -1 ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    ElectronicsMap = cms.string( "" )
)
fragment.hltHcalDigisSoA = cms.EDProducer( "HcalDigisSoAProducer@alpaka",
    hbheDigisLabel = cms.InputTag( "hltHcalDigis" ),
    qie11DigiLabel = cms.InputTag( "hltHcalDigis" ),
    digisLabelF01HE = cms.string( "f01HEDigis" ),
    digisLabelF5HB = cms.string( "f5HBDigis" ),
    digisLabelF3HB = cms.string( "f3HBDigis" ),
    maxChannelsF01HE = cms.uint32( 10000 ),
    maxChannelsF5HB = cms.uint32( 10000 ),
    maxChannelsF3HB = cms.uint32( 10000 ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltHbheRecoSoA = cms.EDProducer( "HBHERecHitProducerPortable@alpaka",
    maxTimeSamples = cms.uint32( 10 ),
    kprep1dChannelsPerBlock = cms.uint32( 32 ),
    digisLabelF01HE = cms.InputTag( 'hltHcalDigisSoA','f01HEDigis' ),
    digisLabelF5HB = cms.InputTag( 'hltHcalDigisSoA','f5HBDigis' ),
    digisLabelF3HB = cms.InputTag( 'hltHcalDigisSoA','f3HBDigis' ),
    recHitsLabelM0HBHE = cms.string( "" ),
    sipmQTSShift = cms.int32( 0 ),
    sipmQNTStoSum = cms.int32( 3 ),
    firstSampleShift = cms.int32( 0 ),
    useEffectivePedestals = cms.bool( True ),
    meanTime = cms.double( 0.0 ),
    timeSigmaSiPM = cms.double( 2.5 ),
    timeSigmaHPD = cms.double( 5.0 ),
    ts4Thresh = cms.double( 0.0 ),
    applyTimeSlew = cms.bool( True ),
    tzeroTimeSlewParameters = cms.vdouble( 23.960177, 11.977461, 9.109694 ),
    slopeTimeSlewParameters = cms.vdouble( -3.178648, -1.5610227, -1.075824 ),
    tmaxTimeSlewParameters = cms.vdouble( 16.0, 10.0, 6.25 ),
    kernelMinimizeThreads = cms.vuint32( 16, 1, 1 ),
    pulseOffsets = cms.vint32( -3, -2, -1, 0, 1, 2, 3, 4 ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltHbhereco = cms.EDProducer( "HcalRecHitSoAToLegacy",
    src = cms.InputTag( "hltHbheRecoSoA" )
)
fragment.hltHfprereco = cms.EDProducer( "HFPreReconstructor",
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    forceSOI = cms.int32( -1 ),
    soiShift = cms.int32( 0 ),
    dropZSmarkedPassed = cms.bool( True ),
    tsFromDB = cms.bool( False ),
    sumAllTimeSlices = cms.bool( False )
)
fragment.hltHfreco = cms.EDProducer( "HFPhase1Reconstructor",
    inputLabel = cms.InputTag( "hltHfprereco" ),
    algoConfigClass = cms.string( "HFPhase1PMTParams" ),
    useChannelQualityFromDB = cms.bool( False ),
    checkChannelQualityForDepth3and4 = cms.bool( False ),
    algorithm = cms.PSet( 
      tfallIfNoTDC = cms.double( -101.0 ),
      triseIfNoTDC = cms.double( -100.0 ),
      rejectAllFailures = cms.bool( True ),
      energyWeights = cms.vdouble( 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0 ),
      soiPhase = cms.uint32( 1 ),
      timeShift = cms.double( 0.0 ),
      tlimits = cms.vdouble( -1000.0, 1000.0, -1000.0, 1000.0 ),
      Class = cms.string( "HFFlexibleTimeCheck" )
    ),
    runHFStripFilter = cms.bool( False ),
    HFStripFilter = cms.PSet( 
      seedHitIetaMax = cms.int32( 35 ),
      verboseLevel = cms.untracked.int32( 10 ),
      maxThreshold = cms.double( 100.0 ),
      stripThreshold = cms.double( 40.0 ),
      wedgeCut = cms.double( 0.05 ),
      lstrips = cms.int32( 2 ),
      maxStripTime = cms.double( 10.0 ),
      gap = cms.int32( 2 ),
      timeMax = cms.double( 6.0 )
    ),
    setNoiseFlags = cms.bool( True ),
    S9S1stat = cms.PSet( 
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      isS8S1 = cms.bool( False ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      short_optimumSlope = cms.vdouble( -99999.0, 0.0164905, 0.0238698, 0.0321383, 0.041296, 0.0513428, 0.0622789, 0.0741041, 0.0868186, 0.100422, 0.135313, 0.136289, 0.0589927 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    S8S1stat = cms.PSet( 
      shortEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      isS8S1 = cms.bool( True ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      longEnergyParams = cms.vdouble( 40.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0 ),
      short_optimumSlope = cms.vdouble( 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    ),
    PETstat = cms.PSet( 
      shortEnergyParams = cms.vdouble( 35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093 ),
      shortETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      long_R_29 = cms.vdouble( 0.8 ),
      longETParams = cms.vdouble( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ),
      longEnergyParams = cms.vdouble( 43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62 ),
      short_R_29 = cms.vdouble( 0.8 ),
      long_R = cms.vdouble( 0.98 ),
      short_R = cms.vdouble( 0.8 ),
      HcalAcceptSeverityLevel = cms.int32( 9 )
    )
)
fragment.hltHoreco = cms.EDProducer( "HcalHitReconstructor",
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 ),
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    correctTiming = cms.bool( False ),
    dropZSmarkedPassed = cms.bool( True ),
    firstAuxTS = cms.int32( 4 ),
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    tsFromDB = cms.bool( True ),
    useLeakCorrection = cms.bool( False ),
    recoParamsFromDB = cms.bool( True ),
    setNegativeFlags = cms.bool( False ),
    saturationParameters = cms.PSet(  maxADCvalue = cms.int32( 127 ) ),
    setSaturationFlags = cms.bool( False ),
    Subdetector = cms.string( "HO" ),
    digiTimeFromDB = cms.bool( True ),
    hfTimingTrustParameters = cms.PSet(  ),
    setTimingTrustFlags = cms.bool( False ),
    setNoiseFlags = cms.bool( False ),
    digistat = cms.PSet(  ),
    HFInWindowStat = cms.PSet(  ),
    S9S1stat = cms.PSet(  ),
    S8S1stat = cms.PSet(  ),
    PETstat = cms.PSet(  ),
    dataOOTCorrectionName = cms.string( "" ),
    dataOOTCorrectionCategory = cms.string( "Data" ),
    mcOOTCorrectionName = cms.string( "" ),
    mcOOTCorrectionCategory = cms.string( "MC" )
)
fragment.hltHcalDigisSoASerialSync = cms.EDProducer( "alpaka_serial_sync::HcalDigisSoAProducer",
    hbheDigisLabel = cms.InputTag( "hltHcalDigis" ),
    qie11DigiLabel = cms.InputTag( "hltHcalDigis" ),
    digisLabelF01HE = cms.string( "f01HEDigis" ),
    digisLabelF5HB = cms.string( "f5HBDigis" ),
    digisLabelF3HB = cms.string( "f3HBDigis" ),
    maxChannelsF01HE = cms.uint32( 10000 ),
    maxChannelsF5HB = cms.uint32( 10000 ),
    maxChannelsF3HB = cms.uint32( 10000 )
)
fragment.hltHbheRecoSoASerialSync = cms.EDProducer( "alpaka_serial_sync::HBHERecHitProducerPortable",
    maxTimeSamples = cms.uint32( 10 ),
    kprep1dChannelsPerBlock = cms.uint32( 32 ),
    digisLabelF01HE = cms.InputTag( 'hltHcalDigisSoASerialSync','f01HEDigis' ),
    digisLabelF5HB = cms.InputTag( 'hltHcalDigisSoASerialSync','f5HBDigis' ),
    digisLabelF3HB = cms.InputTag( 'hltHcalDigisSoASerialSync','f3HBDigis' ),
    recHitsLabelM0HBHE = cms.string( "" ),
    sipmQTSShift = cms.int32( 0 ),
    sipmQNTStoSum = cms.int32( 3 ),
    firstSampleShift = cms.int32( 0 ),
    useEffectivePedestals = cms.bool( True ),
    meanTime = cms.double( 0.0 ),
    timeSigmaSiPM = cms.double( 2.5 ),
    timeSigmaHPD = cms.double( 5.0 ),
    ts4Thresh = cms.double( 0.0 ),
    applyTimeSlew = cms.bool( True ),
    tzeroTimeSlewParameters = cms.vdouble( 23.960177, 11.977461, 9.109694 ),
    slopeTimeSlewParameters = cms.vdouble( -3.178648, -1.5610227, -1.075824 ),
    tmaxTimeSlewParameters = cms.vdouble( 16.0, 10.0, 6.25 ),
    kernelMinimizeThreads = cms.vuint32( 16, 1, 1 ),
    pulseOffsets = cms.vint32( -3, -2, -1, 0, 1, 2, 3, 4 )
)
fragment.hltHbherecoSerialSync = cms.EDProducer( "HcalRecHitSoAToLegacy",
    src = cms.InputTag( "hltHbheRecoSoASerialSync" )
)
fragment.hltParticleFlowRecHitHBHESoA = cms.EDProducer( "PFRecHitSoAProducerHCAL@alpaka",
    producers = cms.VPSet( 
      cms.PSet(  src = cms.InputTag( "hltHbheRecoSoA" ),
        params = cms.ESInputTag( "hltESPPFRecHitHCALParams","" )
      )
    ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    synchronise = cms.untracked.bool( False ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltParticleFlowRecHitHBHE = cms.EDProducer( "LegacyPFRecHitProducer",
    src = cms.InputTag( "hltParticleFlowRecHitHBHESoA" )
)
fragment.hltParticleFlowClusterHBHESoA = cms.EDProducer( "PFClusterSoAProducer@alpaka",
    pfRecHits = cms.InputTag( "hltParticleFlowRecHitHBHESoA" ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    seedFinder = cms.PSet( 
      thresholdsByDetector = cms.VPSet( 
        cms.PSet(  seedingThresholdPt = cms.double( 0.0 ),
          seedingThreshold = cms.vdouble( 0.125, 0.25, 0.35, 0.35 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  seedingThresholdPt = cms.double( 0.0 ),
          seedingThreshold = cms.vdouble( 0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      nNeighbours = cms.int32( 4 )
    ),
    initialClusteringStep = cms.PSet(  thresholdsByDetector = cms.VPSet( 
  cms.PSet(  gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
    detector = cms.string( "HCAL_BARREL1" )
  ),
  cms.PSet(  gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
    detector = cms.string( "HCAL_ENDCAP" )
  )
) ),
    pfClusterBuilder = cms.PSet( 
      minFracTot = cms.double( 1.0E-20 ),
      stoppingTolerance = cms.double( 1.0E-8 ),
      positionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      maxIterations = cms.uint32( 5 ),
      recHitEnergyNorms = cms.VPSet( 
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      showerSigma = cms.double( 10.0 ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      excludeOtherSeeds = cms.bool( True ),
      timeResolutionCalcBarrel = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeResolutionCalcEndcap = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      )
    ),
    synchronise = cms.bool( False ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
fragment.hltParticleFlowClusterHBHE = cms.EDProducer( "LegacyPFClusterProducer",
    src = cms.InputTag( "hltParticleFlowClusterHBHESoA" ),
    PFRecHitsLabelIn = cms.InputTag( "hltParticleFlowRecHitHBHESoA" ),
    recHitsSource = cms.InputTag( "hltParticleFlowRecHitHBHE" ),
    usePFThresholdsFromDB = cms.bool( True ),
    pfClusterBuilder = cms.PSet( 
      minFracTot = cms.double( 1.0E-20 ),
      stoppingTolerance = cms.double( 1.0E-8 ),
      positionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( 5 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      maxIterations = cms.uint32( 5 ),
      minChi2Prob = cms.double( 0.0 ),
      allCellsPositionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( -1 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      algoName = cms.string( "Basic2DGenericPFlowClusterizer" ),
      recHitEnergyNorms = cms.VPSet( 
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
          depths = cms.vint32( 1, 2, 3, 4 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
          depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      maxNSigmaTime = cms.double( 10.0 ),
      showerSigma = cms.double( 10.0 ),
      timeSigmaEE = cms.double( 10.0 ),
      clusterTimeResFromSeed = cms.bool( False ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      excludeOtherSeeds = cms.bool( True ),
      timeResolutionCalcBarrel = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeResolutionCalcEndcap = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeSigmaEB = cms.double( 10.0 )
    )
)
fragment.hltParticleFlowClusterHCAL = cms.EDProducer( "PFMultiDepthClusterProducer",
    clustersSource = cms.InputTag( "hltParticleFlowClusterHBHE" ),
    energyCorrector = cms.PSet(  ),
    pfClusterBuilder = cms.PSet( 
      allCellsPositionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( -1 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      algoName = cms.string( "PFMultiDepthClusterizer" ),
      nSigmaPhi = cms.double( 2.0 ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      nSigmaEta = cms.double( 2.0 )
    ),
    positionReCalc = cms.PSet(  ),
    usePFThresholdsFromDB = cms.bool( True )
)
fragment.hltParticleFlowRecHitHBHESoASerialSync = cms.EDProducer( "alpaka_serial_sync::PFRecHitSoAProducerHCAL",
    producers = cms.VPSet( 
      cms.PSet(  src = cms.InputTag( "hltHbheRecoSoASerialSync" ),
        params = cms.ESInputTag( "hltESPPFRecHitHCALParams","" )
      )
    ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    synchronise = cms.untracked.bool( False )
)
fragment.hltParticleFlowRecHitHBHESerialSync = cms.EDProducer( "LegacyPFRecHitProducer",
    src = cms.InputTag( "hltParticleFlowRecHitHBHESoASerialSync" )
)
fragment.hltParticleFlowClusterHBHESoASerialSync = cms.EDProducer( "alpaka_serial_sync::PFClusterSoAProducer",
    pfRecHits = cms.InputTag( "hltParticleFlowRecHitHBHESoASerialSync" ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    seedFinder = cms.PSet( 
      thresholdsByDetector = cms.VPSet( 
        cms.PSet(  seedingThresholdPt = cms.double( 0.0 ),
          seedingThreshold = cms.vdouble( 0.125, 0.25, 0.35, 0.35 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  seedingThresholdPt = cms.double( 0.0 ),
          seedingThreshold = cms.vdouble( 0.1375, 0.275, 0.275, 0.275, 0.275, 0.275, 0.275 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      nNeighbours = cms.int32( 4 )
    ),
    initialClusteringStep = cms.PSet(  thresholdsByDetector = cms.VPSet( 
  cms.PSet(  gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
    detector = cms.string( "HCAL_BARREL1" )
  ),
  cms.PSet(  gatheringThreshold = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
    detector = cms.string( "HCAL_ENDCAP" )
  )
) ),
    pfClusterBuilder = cms.PSet( 
      minFracTot = cms.double( 1.0E-20 ),
      stoppingTolerance = cms.double( 1.0E-8 ),
      positionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      maxIterations = cms.uint32( 5 ),
      recHitEnergyNorms = cms.VPSet( 
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      showerSigma = cms.double( 10.0 ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      excludeOtherSeeds = cms.bool( True ),
      timeResolutionCalcBarrel = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeResolutionCalcEndcap = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      )
    ),
    synchronise = cms.bool( False )
)
fragment.hltParticleFlowClusterHBHESerialSync = cms.EDProducer( "LegacyPFClusterProducer",
    src = cms.InputTag( "hltParticleFlowClusterHBHESoASerialSync" ),
    PFRecHitsLabelIn = cms.InputTag( "hltParticleFlowRecHitHBHESoASerialSync" ),
    recHitsSource = cms.InputTag( "hltParticleFlowRecHitHBHESerialSync" ),
    usePFThresholdsFromDB = cms.bool( True ),
    pfClusterBuilder = cms.PSet( 
      minFracTot = cms.double( 1.0E-20 ),
      stoppingTolerance = cms.double( 1.0E-8 ),
      positionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( 5 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      maxIterations = cms.uint32( 5 ),
      minChi2Prob = cms.double( 0.0 ),
      allCellsPositionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( -1 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      algoName = cms.string( "Basic2DGenericPFlowClusterizer" ),
      recHitEnergyNorms = cms.VPSet( 
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
          depths = cms.vint32( 1, 2, 3, 4 ),
          detector = cms.string( "HCAL_BARREL1" )
        ),
        cms.PSet(  recHitEnergyNorm = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
          depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
          detector = cms.string( "HCAL_ENDCAP" )
        )
      ),
      maxNSigmaTime = cms.double( 10.0 ),
      showerSigma = cms.double( 10.0 ),
      timeSigmaEE = cms.double( 10.0 ),
      clusterTimeResFromSeed = cms.bool( False ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      excludeOtherSeeds = cms.bool( True ),
      timeResolutionCalcBarrel = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeResolutionCalcEndcap = cms.PSet( 
        corrTermLowE = cms.double( 0.0 ),
        threshLowE = cms.double( 6.0 ),
        noiseTerm = cms.double( 21.86 ),
        constantTermLowE = cms.double( 4.24 ),
        noiseTermLowE = cms.double( 8.0 ),
        threshHighE = cms.double( 15.0 ),
        constantTerm = cms.double( 2.82 )
      ),
      timeSigmaEB = cms.double( 10.0 )
    )
)
fragment.hltParticleFlowClusterHCALSerialSync = cms.EDProducer( "PFMultiDepthClusterProducer",
    clustersSource = cms.InputTag( "hltParticleFlowClusterHBHESerialSync" ),
    energyCorrector = cms.PSet(  ),
    pfClusterBuilder = cms.PSet( 
      allCellsPositionCalc = cms.PSet( 
        minAllowedNormalization = cms.double( 1.0E-9 ),
        posCalcNCrystals = cms.int32( -1 ),
        algoName = cms.string( "Basic2DGenericPFlowPositionCalc" ),
        logWeightDenominatorByDetector = cms.VPSet( 
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.4, 0.3, 0.3, 0.3 ),
            depths = cms.vint32( 1, 2, 3, 4 ),
            detector = cms.string( "HCAL_BARREL1" )
          ),
          cms.PSet(  logWeightDenominator = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
            depths = cms.vint32( 1, 2, 3, 4, 5, 6, 7 ),
            detector = cms.string( "HCAL_ENDCAP" )
          )
        ),
        minFractionInCalc = cms.double( 1.0E-9 )
      ),
      algoName = cms.string( "PFMultiDepthClusterizer" ),
      nSigmaPhi = cms.double( 2.0 ),
      minFractionToKeep = cms.double( 1.0E-7 ),
      nSigmaEta = cms.double( 2.0 )
    ),
    positionReCalc = cms.PSet(  ),
    usePFThresholdsFromDB = cms.bool( True )
)
fragment.hltPreDQMRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDQMZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDSTZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltFEDSelectorL1uGTTest = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1405 )
)
fragment.hltPreDSTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
fragment.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 1024 )
)
fragment.hltPreHcalCalibration = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 1024, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199 )
)
fragment.hltL1EventNumberNZS = cms.EDFilter( "HLTL1NumberFilter",
    rawInput = cms.InputTag( "rawDataCollector" ),
    period = cms.uint32( 4096 ),
    invert = cms.bool( False ),
    fedId = cms.int32( 1024 ),
    useTCDSEventNumber = cms.bool( False )
)
fragment.hltL1sHcalNZS = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_SingleJet90 OR L1_SingleJet120 OR L1_SingleJet140er2p5 OR L1_SingleJet160er2p5 OR L1_SingleJet180 OR L1_SingleJet200 OR L1_DoubleJet40er2p5 OR L1_DoubleJet100er2p5 OR L1_DoubleJet120er2p5 OR L1_QuadJet60er2p5 OR L1_HTT120er OR L1_HTT160er OR L1_HTT200er OR L1_HTT255er OR L1_HTT280er OR L1_HTT320er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreHcalNZS = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEGorSingleorDoubleMu = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG30er2p1 OR L1_SingleIsoEG28er2p1 OR L1_SingleEG10er2p5 OR L1_SingleEG15er2p5 OR L1_SingleEG26er2p5 OR L1_SingleEG34er2p5 OR L1_SingleEG36er2p5 OR L1_SingleEG38er2p5 OR L1_SingleEG40er2p5 OR L1_SingleEG42er2p5 OR L1_SingleEG45er2p5 OR L1_SingleEG60 OR L1_SingleMu3 OR L1_SingleMu5 OR L1_SingleMu7 OR L1_SingleMu18 OR L1_SingleMu20 OR L1_SingleMu22 OR L1_SingleMu25 OR L1_DoubleMu_12_5 OR L1_DoubleMu_15_7 OR L1_SingleIsoEG24er2p1 OR L1_SingleIsoEG26er2p5 OR L1_SingleIsoEG28er2p5 OR L1_SingleIsoEG30er2p5 OR L1_SingleIsoEG32er2p5 OR L1_SingleIsoEG34er2p5 OR L1_DoubleEG_22_10_er2p5 OR L1_DoubleEG_25_12_er2p5 OR L1_DoubleEG_25_14_er2p5 OR L1_TripleEG_18_17_8_er2p5 OR L1_TripleMu_5_3_3 OR L1_TripleMu_5_5_3" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreHcalPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1EventNumberL1Fat = cms.EDFilter( "HLTL1NumberFilter",
    rawInput = cms.InputTag( "rawDataCollector" ),
    period = cms.uint32( 107 ),
    invert = cms.bool( False ),
    fedId = cms.int32( 1024 ),
    useTCDSEventNumber = cms.bool( True )
)
fragment.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBiasAlignment = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBiasBeamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBSumThreshold = cms.double( 0.2 ),
    HF2Weight = cms.double( 1.0 ),
    EBWeight = cms.double( 1.0 ),
    hfInput = cms.InputTag( "hltHfreco" ),
    EESumThreshold = cms.double( 0.45 ),
    HOThreshold0 = cms.double( 3.5 ),
    HOThresholdPlus1 = cms.double( 3.5 ),
    HOThresholdMinus1 = cms.double( 3.5 ),
    HOThresholdPlus2 = cms.double( 3.5 ),
    HOThresholdMinus2 = cms.double( 3.5 ),
    HBGrid = cms.vdouble(  ),
    HBThreshold1 = cms.double( 0.4 ),
    HBThreshold2 = cms.double( 0.3 ),
    HBThreshold = cms.double( 0.3 ),
    EEWeights = cms.vdouble(  ),
    HF1Threshold = cms.double( 0.5 ),
    HF2Weights = cms.vdouble(  ),
    HOWeights = cms.vdouble(  ),
    EEGrid = cms.vdouble(  ),
    HEDWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    UseHO = cms.bool( False ),
    HBWeights = cms.vdouble(  ),
    HESWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Grid = cms.vdouble(  ),
    HEDWeights = cms.vdouble(  ),
    HF1Grid = cms.vdouble(  ),
    EBWeights = cms.vdouble(  ),
    HOWeight = cms.double( 1.0E-99 ),
    EBThreshold = cms.double( 0.07 ),
    EEThreshold = cms.double( 0.3 ),
    UseEtEBTreshold = cms.bool( False ),
    UseSymEBTreshold = cms.bool( False ),
    UseEtEETreshold = cms.bool( False ),
    UseSymEETreshold = cms.bool( False ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    HcalThreshold = cms.double( -1000.0 ),
    HF2Threshold = cms.double( 0.85 ),
    HESThreshold1 = cms.double( 0.1 ),
    HESThreshold = cms.double( 0.2 ),
    HF1Weights = cms.vdouble(  ),
    hoInput = cms.InputTag( "hltHoreco" ),
    HESGrid = cms.vdouble(  ),
    HESWeights = cms.vdouble(  ),
    HEDThreshold1 = cms.double( 0.1 ),
    HEDThreshold = cms.double( 0.2 ),
    EcutTower = cms.double( -1000.0 ),
    HEDGrid = cms.vdouble(  ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHit:EcalRecHitsEB','hltEcalRecHit:EcalRecHitsEE' ),
    HBWeight = cms.double( 1.0 ),
    HOGrid = cms.vdouble(  ),
    EBGrid = cms.vdouble(  ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    HcalAcceptSeverityLevel = cms.uint32( 9 ),
    EcalRecHitSeveritiesToBeExcluded = cms.vstring( 'kTime',
      'kWeird',
      'kBad' ),
    UseHcalRecoveredHits = cms.bool( False ),
    UseEcalRecoveredHits = cms.bool( False ),
    UseRejectedHitsOnly = cms.bool( False ),
    HcalAcceptSeverityLevelForRejectedHit = cms.uint32( 9999 ),
    EcalSeveritiesToBeUsedInBadTowers = cms.vstring(  ),
    UseRejectedRecoveredHcalHits = cms.bool( False ),
    UseRejectedRecoveredEcalHits = cms.bool( False ),
    missingHcalRescaleFactorForEcal = cms.double( 0.0 ),
    AllowMissingInputs = cms.bool( False ),
    HcalPhase = cms.int32( 1 ),
    usePFThresholdsFromDB = cms.bool( True ),
    EcalRecHitThresh = cms.bool( False )
)
fragment.hltAK4CaloJetsPF = cms.EDProducer( "FastjetJetProducer",
    useMassDropTagger = cms.bool( False ),
    useFiltering = cms.bool( False ),
    useDynamicFiltering = cms.bool( False ),
    useTrimming = cms.bool( False ),
    usePruning = cms.bool( False ),
    useCMSBoostedTauSeedingAlgorithm = cms.bool( False ),
    useKtPruning = cms.bool( False ),
    useConstituentSubtraction = cms.bool( False ),
    useSoftDrop = cms.bool( False ),
    correctShape = cms.bool( False ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    muCut = cms.double( -1.0 ),
    yCut = cms.double( -1.0 ),
    rFilt = cms.double( -1.0 ),
    rFiltFactor = cms.double( -1.0 ),
    trimPtFracMin = cms.double( -1.0 ),
    zcut = cms.double( -1.0 ),
    rcut_factor = cms.double( -1.0 ),
    csRho_EtaMax = cms.double( -1.0 ),
    csRParam = cms.double( -1.0 ),
    beta = cms.double( -1.0 ),
    R0 = cms.double( -1.0 ),
    gridMaxRapidity = cms.double( -1.0 ),
    gridSpacing = cms.double( -1.0 ),
    DzTrVtxMax = cms.double( 0.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    MaxVtxZ = cms.double( 15.0 ),
    subjetPtMin = cms.double( -1.0 ),
    muMin = cms.double( -1.0 ),
    muMax = cms.double( -1.0 ),
    yMin = cms.double( -1.0 ),
    yMax = cms.double( -1.0 ),
    dRMin = cms.double( -1.0 ),
    dRMax = cms.double( -1.0 ),
    maxDepth = cms.int32( -1 ),
    nFilt = cms.int32( -1 ),
    MinVtxNdof = cms.int32( 5 ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetType = cms.string( "CaloJet" ),
    jetAlgorithm = cms.string( "AntiKt" ),
    rParam = cms.double( 0.4 ),
    inputEtMin = cms.double( 0.3 ),
    inputEMin = cms.double( 0.0 ),
    jetPtMin = cms.double( 1.0 ),
    doPVCorrection = cms.bool( False ),
    doAreaFastjet = cms.bool( False ),
    doRhoFastjet = cms.bool( False ),
    doPUOffsetCorr = cms.bool( False ),
    puPtMin = cms.double( 10.0 ),
    nSigmaPU = cms.double( 1.0 ),
    radiusPU = cms.double( 0.4 ),
    subtractorName = cms.string( "" ),
    useExplicitGhosts = cms.bool( False ),
    doAreaDiskApprox = cms.bool( False ),
    voronoiRfact = cms.double( -9.0 ),
    Rho_EtaMax = cms.double( 4.4 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    Active_Area_Repeats = cms.int32( 5 ),
    GhostArea = cms.double( 0.01 ),
    restrictInputs = cms.bool( False ),
    maxInputs = cms.uint32( 1 ),
    writeCompound = cms.bool( False ),
    writeJetsWithConst = cms.bool( False ),
    doFastJetNonUniform = cms.bool( False ),
    useDeterministicSeed = cms.bool( True ),
    minSeed = cms.uint32( 0 ),
    verbosity = cms.int32( 0 ),
    puWidth = cms.double( 0.0 ),
    nExclude = cms.uint32( 0 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    puCenters = cms.vdouble(  ),
    applyWeight = cms.bool( False ),
    srcWeights = cms.InputTag( "" ),
    minimumTowersFraction = cms.double( 0.0 ),
    jetCollInstanceName = cms.string( "" ),
    sumRecHits = cms.bool( False )
)
fragment.hltAK4CaloJetsPFEt5 = cms.EDFilter( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltAK4CaloJetsPF" ),
    filter = cms.bool( False ),
    etMin = cms.double( 5.0 )
)
fragment.hltMuonDTDigis = cms.EDProducer( "DTuROSRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    debug = cms.untracked.bool( False )
)
fragment.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    recAlgoConfig = cms.PSet( 
      maxTime = cms.double( 420.0 ),
      debug = cms.untracked.bool( False ),
      stepTwoFromDigi = cms.bool( False ),
      tTrigModeConfig = cms.PSet( 
        debug = cms.untracked.bool( False ),
        tofCorrType = cms.int32( 0 ),
        tTrigLabel = cms.string( "" ),
        wirePropCorrType = cms.int32( 0 ),
        doTOFCorrection = cms.bool( True ),
        vPropWire = cms.double( 24.4 ),
        doT0Correction = cms.bool( True ),
        doWirePropCorrection = cms.bool( True ),
        t0Label = cms.string( "" )
      ),
      useUncertDB = cms.bool( True ),
      doVdriftCorr = cms.bool( True ),
      minTime = cms.double( -3.0 ),
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      readLegacyTTrigDB = cms.bool( True ),
      readLegacyVDriftDB = cms.bool( True )
    ),
    recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
    debug = cms.untracked.bool( False ),
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" )
)
fragment.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    Reco4DAlgoConfig = cms.PSet( 
      Reco2DAlgoConfig = cms.PSet( 
        AlphaMaxPhi = cms.double( 1.0 ),
        debug = cms.untracked.bool( False ),
        segmCleanerMode = cms.int32( 2 ),
        AlphaMaxTheta = cms.double( 0.9 ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
        recAlgoConfig = cms.PSet( 
          maxTime = cms.double( 420.0 ),
          debug = cms.untracked.bool( False ),
          stepTwoFromDigi = cms.bool( False ),
          tTrigModeConfig = cms.PSet( 
            debug = cms.untracked.bool( False ),
            tofCorrType = cms.int32( 0 ),
            tTrigLabel = cms.string( "" ),
            wirePropCorrType = cms.int32( 0 ),
            doTOFCorrection = cms.bool( True ),
            vPropWire = cms.double( 24.4 ),
            doT0Correction = cms.bool( True ),
            doWirePropCorrection = cms.bool( True ),
            t0Label = cms.string( "" )
          ),
          useUncertDB = cms.bool( True ),
          doVdriftCorr = cms.bool( True ),
          minTime = cms.double( -3.0 ),
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          readLegacyTTrigDB = cms.bool( True ),
          readLegacyVDriftDB = cms.bool( True )
        ),
        MaxAllowedHits = cms.uint32( 50 ),
        nUnSharedHitsMin = cms.int32( 2 ),
        nSharedHitsMax = cms.int32( 2 ),
        performT0SegCorrection = cms.bool( False ),
        perform_delta_rejecting = cms.bool( False )
      ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      debug = cms.untracked.bool( False ),
      segmCleanerMode = cms.int32( 2 ),
      AllDTRecHits = cms.bool( True ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
      recAlgoConfig = cms.PSet( 
        maxTime = cms.double( 420.0 ),
        debug = cms.untracked.bool( False ),
        stepTwoFromDigi = cms.bool( False ),
        tTrigModeConfig = cms.PSet( 
          debug = cms.untracked.bool( False ),
          tofCorrType = cms.int32( 0 ),
          tTrigLabel = cms.string( "" ),
          wirePropCorrType = cms.int32( 0 ),
          doTOFCorrection = cms.bool( True ),
          vPropWire = cms.double( 24.4 ),
          doT0Correction = cms.bool( True ),
          doWirePropCorrection = cms.bool( True ),
          t0Label = cms.string( "" )
        ),
        useUncertDB = cms.bool( True ),
        doVdriftCorr = cms.bool( True ),
        minTime = cms.double( -3.0 ),
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        readLegacyTTrigDB = cms.bool( True ),
        readLegacyVDriftDB = cms.bool( True )
      ),
      nUnSharedHitsMin = cms.int32( 2 ),
      nSharedHitsMax = cms.int32( 2 ),
      performT0SegCorrection = cms.bool( False ),
      perform_delta_rejecting = cms.bool( False )
    ),
    debug = cms.untracked.bool( False ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" )
)
fragment.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    InputObjects = cms.InputTag( "rawDataCollector" ),
    UseExaminer = cms.bool( True ),
    ExaminerMask = cms.uint32( 535558134 ),
    UseSelectiveUnpacking = cms.bool( True ),
    ErrorMask = cms.uint32( 0 ),
    UnpackStatusDigis = cms.bool( False ),
    UseFormatStatus = cms.bool( True ),
    useRPCs = cms.bool( False ),
    useGEMs = cms.bool( False ),
    useCSCShowers = cms.bool( False ),
    Debug = cms.untracked.bool( False ),
    PrintEventNumber = cms.untracked.bool( False ),
    runDQM = cms.untracked.bool( False ),
    VisualFEDInspect = cms.untracked.bool( False ),
    VisualFEDShort = cms.untracked.bool( False ),
    FormatedEventDump = cms.untracked.bool( False ),
    SuppressZeroLCT = cms.untracked.bool( True ),
    DisableMappingCheck = cms.untracked.bool( False ),
    B904Setup = cms.untracked.bool( False ),
    B904vmecrate = cms.untracked.int32( 1 ),
    B904dmb = cms.untracked.int32( 3 )
)
fragment.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    CSCStripPeakThreshold = cms.double( 10.0 ),
    CSCStripClusterChargeCut = cms.double( 25.0 ),
    CSCStripxtalksOffset = cms.double( 0.03 ),
    UseAverageTime = cms.bool( False ),
    UseParabolaFit = cms.bool( False ),
    UseFivePoleFit = cms.bool( True ),
    CSCWireClusterDeltaT = cms.int32( 1 ),
    CSCUseCalibrations = cms.bool( True ),
    CSCUseStaticPedestals = cms.bool( False ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    readBadChannels = cms.bool( False ),
    readBadChambers = cms.bool( True ),
    CSCUseTimingCorrections = cms.bool( True ),
    CSCUseGasGainCorrections = cms.bool( False ),
    CSCDebug = cms.untracked.bool( False ),
    CSCstripWireDeltaTime = cms.int32( 8 ),
    XTasymmetry_ME1a = cms.double( 0.023 ),
    XTasymmetry_ME1b = cms.double( 0.01 ),
    XTasymmetry_ME12 = cms.double( 0.015 ),
    XTasymmetry_ME13 = cms.double( 0.02 ),
    XTasymmetry_ME21 = cms.double( 0.023 ),
    XTasymmetry_ME22 = cms.double( 0.023 ),
    XTasymmetry_ME31 = cms.double( 0.023 ),
    XTasymmetry_ME32 = cms.double( 0.023 ),
    XTasymmetry_ME41 = cms.double( 0.023 ),
    ConstSyst_ME1a = cms.double( 0.01 ),
    ConstSyst_ME1b = cms.double( 0.02 ),
    ConstSyst_ME12 = cms.double( 0.02 ),
    ConstSyst_ME13 = cms.double( 0.03 ),
    ConstSyst_ME21 = cms.double( 0.03 ),
    ConstSyst_ME22 = cms.double( 0.03 ),
    ConstSyst_ME31 = cms.double( 0.03 ),
    ConstSyst_ME32 = cms.double( 0.03 ),
    ConstSyst_ME41 = cms.double( 0.03 ),
    NoiseLevel_ME1a = cms.double( 9.0 ),
    NoiseLevel_ME1b = cms.double( 6.0 ),
    NoiseLevel_ME12 = cms.double( 7.0 ),
    NoiseLevel_ME13 = cms.double( 4.0 ),
    NoiseLevel_ME21 = cms.double( 5.0 ),
    NoiseLevel_ME22 = cms.double( 7.0 ),
    NoiseLevel_ME31 = cms.double( 5.0 ),
    NoiseLevel_ME32 = cms.double( 7.0 ),
    NoiseLevel_ME41 = cms.double( 5.0 ),
    CSCUseReducedWireTimeWindow = cms.bool( True ),
    CSCWireTimeWindowLow = cms.int32( 5 ),
    CSCWireTimeWindowHigh = cms.int32( 11 )
)
fragment.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_type = cms.int32( 1 ),
    algo_psets = cms.VPSet( 
      cms.PSet(  parameters_per_chamber_type = cms.vint32( 1, 2, 3, 4, 5, 6, 5, 6, 5, 6 ),
        algo_psets = cms.VPSet( 
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 35.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.006 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.005 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 100.0 ),
            chi2_str = cms.double( 50.0 )
          ),
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 35.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.005 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.004 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 100.0 ),
            chi2_str = cms.double( 50.0 )
          ),
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 35.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.004 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.003 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 100.0 ),
            chi2_str = cms.double( 50.0 )
          ),
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 20.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.003 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.002 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 60.0 ),
            chi2_str = cms.double( 30.0 )
          ),
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 60.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.007 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.005 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 180.0 ),
            chi2_str = cms.double( 80.0 )
          ),
          cms.PSet(  wideSeg = cms.double( 3.0 ),
            chi2Norm_2D_ = cms.double( 35.0 ),
            dRIntMax = cms.double( 2.0 ),
            doCollisions = cms.bool( True ),
            dPhiMax = cms.double( 0.006 ),
            dRMax = cms.double( 1.5 ),
            dPhiIntMax = cms.double( 0.004 ),
            minLayersApart = cms.int32( 1 ),
            chi2Max = cms.double( 100.0 ),
            chi2_str = cms.double( 50.0 )
          )
        ),
        algo_name = cms.string( "CSCSegAlgoRU" ),
        chamber_types = cms.vstring( 'ME1/a',
          'ME1/b',
          'ME1/2',
          'ME1/3',
          'ME2/1',
          'ME2/2',
          'ME3/1',
          'ME3/2',
          'ME4/1',
          'ME4/2' )
      )
    )
)
fragment.hltMuonRPCDigisCPPF = cms.EDProducer( "RPCAMCRawToDigi",
    inputTag = cms.InputTag( "rawDataCollector" ),
    calculateCRC = cms.bool( True ),
    fillCounters = cms.bool( True ),
    RPCAMCUnpacker = cms.string( "RPCCPPFUnpacker" ),
    RPCAMCUnpackerSettings = cms.PSet( 
      bxMin = cms.int32( -2 ),
      cppfDaqDelay = cms.int32( 2 ),
      fillAMCCounters = cms.bool( True ),
      bxMax = cms.int32( 2 )
    )
)
fragment.hltOmtfDigis = cms.EDProducer( "OmtfUnpacker",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    skipRpc = cms.bool( False ),
    skipCsc = cms.bool( False ),
    skipDt = cms.bool( False ),
    skipMuon = cms.bool( False ),
    useRpcConnectionFile = cms.bool( False ),
    rpcConnectionFile = cms.string( "" ),
    outputTag = cms.string( "" )
)
fragment.hltMuonRPCDigisTwinMux = cms.EDProducer( "RPCTwinMuxRawToDigi",
    inputTag = cms.InputTag( "rawDataCollector" ),
    calculateCRC = cms.bool( True ),
    fillCounters = cms.bool( True ),
    bxMin = cms.int32( -2 ),
    bxMax = cms.int32( 2 )
)
fragment.hltMuonRPCDigis = cms.EDProducer( "RPCDigiMerger",
    inputTagSimRPCDigis = cms.InputTag( "" ),
    inputTagTwinMuxDigis = cms.InputTag( "hltMuonRPCDigisTwinMux" ),
    inputTagOMTFDigis = cms.InputTag( "hltOmtfDigis" ),
    inputTagCPPFDigis = cms.InputTag( "hltMuonRPCDigisCPPF" ),
    InputLabel = cms.InputTag( "rawDataCollector" ),
    bxMinTwinMux = cms.int32( -2 ),
    bxMaxTwinMux = cms.int32( 2 ),
    bxMinOMTF = cms.int32( -3 ),
    bxMaxOMTF = cms.int32( 4 ),
    bxMinCPPF = cms.int32( -2 ),
    bxMaxCPPF = cms.int32( 2 )
)
fragment.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskSource = cms.string( "File" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    deadSource = cms.string( "File" ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" )
)
fragment.hltMuonGEMDigis = cms.EDProducer( "GEMRawToDigiModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    useDBEMap = cms.bool( True ),
    keepDAQStatus = cms.bool( False ),
    readMultiBX = cms.bool( False ),
    ge21Off = cms.bool( True ),
    fedIdStart = cms.uint32( 1467 ),
    fedIdEnd = cms.uint32( 1478 )
)
fragment.hltGemRecHits = cms.EDProducer( "GEMRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    recAlgo = cms.string( "GEMRecHitStandardAlgo" ),
    gemDigiLabel = cms.InputTag( "hltMuonGEMDigis" ),
    applyMasking = cms.bool( True ),
    ge21Off = cms.bool( False )
)
fragment.hltGemSegments = cms.EDProducer( "GEMSegmentProducer",
    gemRecHitLabel = cms.InputTag( "hltGemRecHits" ),
    enableGE0 = cms.bool( True ),
    enableGE12 = cms.bool( False ),
    ge0_name = cms.string( "GE0SegAlgoRU" ),
    algo_name = cms.string( "GEMSegmentAlgorithm" ),
    ge0_pset = cms.PSet( 
      maxChi2GoodSeg = cms.double( 50.0 ),
      maxChi2Prune = cms.double( 50.0 ),
      maxNumberOfHitsPerLayer = cms.uint32( 100 ),
      maxETASeeds = cms.double( 0.1 ),
      maxPhiAdditional = cms.double( 0.001096605744 ),
      minNumberOfHits = cms.uint32( 4 ),
      doCollisions = cms.bool( True ),
      maxPhiSeeds = cms.double( 0.001096605744 ),
      requireCentralBX = cms.bool( True ),
      maxChi2Additional = cms.double( 100.0 ),
      allowWideSegments = cms.bool( True ),
      maxNumberOfHits = cms.uint32( 300 ),
      maxTOFDiff = cms.double( 25.0 )
    ),
    algo_pset = cms.PSet( 
      dYclusBoxMax = cms.double( 5.0 ),
      dXclusBoxMax = cms.double( 1.0 ),
      maxRecHitsInCluster = cms.int32( 4 ),
      preClustering = cms.bool( True ),
      preClusteringUseChaining = cms.bool( True ),
      dEtaChainBoxMax = cms.double( 0.05 ),
      clusterOnlySameBXRecHits = cms.bool( True ),
      minHitsPerSegment = cms.uint32( 2 ),
      dPhiChainBoxMax = cms.double( 0.02 )
    )
)
fragment.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
    beamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    scaleDT = cms.bool( True ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
    ME0RecSegmentLabel = cms.InputTag( "me0Segments" ),
    EnableDTMeasurement = cms.bool( True ),
    EnableCSCMeasurement = cms.bool( True ),
    EnableME0Measurement = cms.bool( False ),
    crackEtas = cms.vdouble( 0.2, 1.6, 1.7 ),
    crackWindow = cms.double( 0.04 ),
    deltaPhiSearchWindow = cms.double( 0.25 ),
    deltaEtaSearchWindow = cms.double( 0.2 ),
    deltaEtaCrackSearchWindow = cms.double( 0.25 ),
    CSC_01 = cms.vdouble( 0.166, 0.0, 0.0, 0.031, 0.0, 0.0 ),
    CSC_12 = cms.vdouble( -0.161, 0.254, -0.047, 0.042, -0.007, 0.0 ),
    CSC_02 = cms.vdouble( 0.612, -0.207, 0.0, 0.067, -0.001, 0.0 ),
    CSC_13 = cms.vdouble( 0.901, -1.302, 0.533, 0.045, 0.005, 0.0 ),
    CSC_03 = cms.vdouble( 0.787, -0.338, 0.029, 0.101, -0.008, 0.0 ),
    CSC_14 = cms.vdouble( 0.606, -0.181, -0.002, 0.111, -0.003, 0.0 ),
    CSC_23 = cms.vdouble( -0.081, 0.113, -0.029, 0.015, 0.008, 0.0 ),
    CSC_24 = cms.vdouble( 0.004, 0.021, -0.002, 0.053, 0.0, 0.0 ),
    CSC_34 = cms.vdouble( 0.062, -0.067, 0.019, 0.021, 0.003, 0.0 ),
    DT_12 = cms.vdouble( 0.183, 0.054, -0.087, 0.028, 0.002, 0.0 ),
    DT_13 = cms.vdouble( 0.315, 0.068, -0.127, 0.051, -0.002, 0.0 ),
    DT_14 = cms.vdouble( 0.359, 0.052, -0.107, 0.072, -0.004, 0.0 ),
    DT_23 = cms.vdouble( 0.13, 0.023, -0.057, 0.028, 0.004, 0.0 ),
    DT_24 = cms.vdouble( 0.176, 0.014, -0.051, 0.051, 0.003, 0.0 ),
    DT_34 = cms.vdouble( 0.044, 0.004, -0.013, 0.029, 0.003, 0.0 ),
    OL_1213 = cms.vdouble( 0.96, -0.737, 0.0, 0.052, 0.0, 0.0 ),
    OL_1222 = cms.vdouble( 0.848, -0.591, 0.0, 0.062, 0.0, 0.0 ),
    OL_1232 = cms.vdouble( 0.184, 0.0, 0.0, 0.066, 0.0, 0.0 ),
    OL_2213 = cms.vdouble( 0.117, 0.0, 0.0, 0.044, 0.0, 0.0 ),
    OL_2222 = cms.vdouble( 0.107, 0.0, 0.0, 0.04, 0.0, 0.0 ),
    SME_11 = cms.vdouble( 3.295, -1.527, 0.112, 0.378, 0.02, 0.0 ),
    SME_12 = cms.vdouble( 0.102, 0.599, 0.0, 0.38, 0.0, 0.0 ),
    SME_13 = cms.vdouble( -1.286, 1.711, 0.0, 0.356, 0.0, 0.0 ),
    SME_21 = cms.vdouble( -0.529, 1.194, -0.358, 0.472, 0.086, 0.0 ),
    SME_22 = cms.vdouble( -1.207, 1.491, -0.251, 0.189, 0.243, 0.0 ),
    SME_31 = cms.vdouble( -1.594, 1.482, -0.317, 0.487, 0.097, 0.0 ),
    SME_32 = cms.vdouble( -0.901, 1.333, -0.47, 0.41, 0.073, 0.0 ),
    SME_41 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    SME_42 = cms.vdouble( -0.003, 0.005, 0.005, 0.608, 0.076, 0.0 ),
    SMB_10 = cms.vdouble( 1.387, -0.038, 0.0, 0.19, 0.0, 0.0 ),
    SMB_11 = cms.vdouble( 1.247, 0.72, -0.802, 0.229, -0.075, 0.0 ),
    SMB_12 = cms.vdouble( 2.128, -0.956, 0.0, 0.199, 0.0, 0.0 ),
    SMB_20 = cms.vdouble( 1.011, -0.052, 0.0, 0.188, 0.0, 0.0 ),
    SMB_21 = cms.vdouble( 1.043, -0.124, 0.0, 0.183, 0.0, 0.0 ),
    SMB_22 = cms.vdouble( 1.474, -0.758, 0.0, 0.185, 0.0, 0.0 ),
    SMB_30 = cms.vdouble( 0.505, -0.022, 0.0, 0.215, 0.0, 0.0 ),
    SMB_31 = cms.vdouble( 0.549, -0.145, 0.0, 0.207, 0.0, 0.0 ),
    SMB_32 = cms.vdouble( 0.67, -0.327, 0.0, 0.22, 0.0, 0.0 ),
    CSC_01_1_scale = cms.vdouble( -1.915329, 0.0 ),
    CSC_12_1_scale = cms.vdouble( -6.434242, 0.0 ),
    CSC_12_2_scale = cms.vdouble( -1.63622, 0.0 ),
    CSC_12_3_scale = cms.vdouble( -1.63622, 0.0 ),
    CSC_13_2_scale = cms.vdouble( -6.077936, 0.0 ),
    CSC_13_3_scale = cms.vdouble( -1.701268, 0.0 ),
    CSC_14_3_scale = cms.vdouble( -1.969563, 0.0 ),
    CSC_23_1_scale = cms.vdouble( -19.084285, 0.0 ),
    CSC_23_2_scale = cms.vdouble( -6.079917, 0.0 ),
    CSC_24_1_scale = cms.vdouble( -6.055701, 0.0 ),
    CSC_34_1_scale = cms.vdouble( -11.520507, 0.0 ),
    OL_1213_0_scale = cms.vdouble( -4.488158, 0.0 ),
    OL_1222_0_scale = cms.vdouble( -5.810449, 0.0 ),
    OL_1232_0_scale = cms.vdouble( -5.964634, 0.0 ),
    OL_2213_0_scale = cms.vdouble( -7.239789, 0.0 ),
    OL_2222_0_scale = cms.vdouble( -7.667231, 0.0 ),
    DT_12_1_scale = cms.vdouble( -3.692398, 0.0 ),
    DT_12_2_scale = cms.vdouble( -3.518165, 0.0 ),
    DT_13_1_scale = cms.vdouble( -4.520923, 0.0 ),
    DT_13_2_scale = cms.vdouble( -4.257687, 0.0 ),
    DT_14_1_scale = cms.vdouble( -5.644816, 0.0 ),
    DT_14_2_scale = cms.vdouble( -4.808546, 0.0 ),
    DT_23_1_scale = cms.vdouble( -5.320346, 0.0 ),
    DT_23_2_scale = cms.vdouble( -5.117625, 0.0 ),
    DT_24_1_scale = cms.vdouble( -7.490909, 0.0 ),
    DT_24_2_scale = cms.vdouble( -6.63094, 0.0 ),
    DT_34_1_scale = cms.vdouble( -13.783765, 0.0 ),
    DT_34_2_scale = cms.vdouble( -11.901897, 0.0 ),
    SMB_10_0_scale = cms.vdouble( 2.448566, 0.0 ),
    SMB_11_0_scale = cms.vdouble( 2.56363, 0.0 ),
    SMB_12_0_scale = cms.vdouble( 2.283221, 0.0 ),
    SMB_20_0_scale = cms.vdouble( 1.486168, 0.0 ),
    SMB_21_0_scale = cms.vdouble( 1.58384, 0.0 ),
    SMB_22_0_scale = cms.vdouble( 1.346681, 0.0 ),
    SMB_30_0_scale = cms.vdouble( -3.629838, 0.0 ),
    SMB_31_0_scale = cms.vdouble( -3.323768, 0.0 ),
    SMB_32_0_scale = cms.vdouble( -3.054156, 0.0 ),
    SME_11_0_scale = cms.vdouble( 1.325085, 0.0 ),
    SME_12_0_scale = cms.vdouble( 2.279181, 0.0 ),
    SME_13_0_scale = cms.vdouble( 0.104905, 0.0 ),
    SME_21_0_scale = cms.vdouble( -0.040862, 0.0 ),
    SME_22_0_scale = cms.vdouble( -3.457901, 0.0 )
)
fragment.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGeneratorFromL1T",
    GMTReadoutCollection = cms.InputTag( "" ),
    InputObjects = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 7 ),
    SetMinPtBarrelTo = cms.double( 3.5 ),
    SetMinPtEndcapTo = cms.double( 1.0 ),
    UseOfflineSeed = cms.untracked.bool( True ),
    UseUnassociatedL1 = cms.bool( False ),
    MatchDR = cms.vdouble( 0.3 ),
    EtaMatchingBins = cms.vdouble( 0.0, 2.5 ),
    CentralBxOnly = cms.bool( True ),
    MatchType = cms.uint32( 0 ),
    SortType = cms.uint32( 0 ),
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2OfflineMuonSeeds" ),
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' )
    )
)
fragment.hltL2Muons = cms.EDProducer( "L2MuonProducer",
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' )
    ),
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    SeedTransformerParameters = cms.PSet( 
      Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
      NMinRecHits = cms.uint32( 2 ),
      RescaleError = cms.double( 100.0 ),
      Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      UseSubRecHits = cms.bool( False ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      BWFilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        BWSeedType = cms.string( "fromGenerator" ),
        GEMRecSegmentLabel = cms.InputTag( "hltGemRecHits" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableGEMMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          ExcludeRPCFromFit = cms.bool( False ),
          Granularity = cms.int32( 0 ),
          MaxChi2 = cms.double( 25.0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 ),
          UseInvalidHits = cms.bool( True )
        ),
        EnableCSCMeasurement = cms.bool( True ),
        MaxChi2 = cms.double( 100.0 ),
        FitDirection = cms.string( "outsideIn" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        NumberOfSigma = cms.double( 3.0 ),
        EnableDTMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False ),
      FilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        GEMRecSegmentLabel = cms.InputTag( "hltGemRecHits" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableGEMMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          ExcludeRPCFromFit = cms.bool( False ),
          Granularity = cms.int32( 0 ),
          MaxChi2 = cms.double( 25.0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 ),
          UseInvalidHits = cms.bool( True )
        ),
        EnableCSCMeasurement = cms.bool( True ),
        MaxChi2 = cms.double( 1000.0 ),
        FitDirection = cms.string( "insideOut" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        NumberOfSigma = cms.double( 3.0 ),
        EnableDTMeasurement = cms.bool( True )
      ),
      SeedPosition = cms.string( "in" ),
      DoBackwardFilter = cms.bool( True ),
      DoRefit = cms.bool( False ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
        NMinRecHits = cms.uint32( 2 ),
        RescaleError = cms.double( 100.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        UseSubRecHits = cms.bool( False ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
      ),
      SeedPropagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" )
    ),
    DoSeedRefit = cms.bool( False ),
    TrackLoaderParameters = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      DoSmoothing = cms.bool( False ),
      VertexConstraint = cms.bool( True ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" )
      ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
    ),
    MuonTrajectoryBuilder = cms.string( "Exhaustive" )
)
fragment.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
fragment.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
fragment.hltSiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    ConditionsLabel = cms.string( "" ),
    onDemand = cms.bool( True ),
    DoAPVEmulatorCheck = cms.bool( False ),
    LegacyUnpacker = cms.bool( False ),
    HybridZeroSuppressed = cms.bool( False ),
    Clusterizer = cms.PSet( 
      ConditionsLabel = cms.string( "" ),
      ClusterThreshold = cms.double( 5.0 ),
      SeedThreshold = cms.double( 3.0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      ChannelThreshold = cms.double( 2.0 ),
      MaxAdjacentBad = cms.uint32( 0 ),
      setDetId = cms.bool( True ),
      MaxSequentialHoles = cms.uint32( 0 ),
      RemoveApvShots = cms.bool( True ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      MaxSequentialBad = cms.uint32( 1 )
    ),
    Algorithms = cms.PSet( 
      Use10bitsTruncation = cms.bool( False ),
      CommonModeNoiseSubtractionMode = cms.string( "Median" ),
      useCMMeanMap = cms.bool( False ),
      TruncateInSuppressor = cms.bool( True ),
      doAPVRestore = cms.bool( False ),
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      PedestalSubtractionFedMode = cms.bool( True )
    )
)
fragment.hltMeasurementTrackerEvent = cms.EDProducer( "MeasurementTrackerEventProducer",
    measurementTracker = cms.string( "hltESPMeasurementTracker" ),
    skipClusters = cms.InputTag( "" ),
    pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
    stripClusterProducer = cms.string( "hltSiStripRawToClustersFacility" ),
    Phase2TrackerCluster1DProducer = cms.string( "" ),
    vectorHits = cms.InputTag( "" ),
    vectorHitsRej = cms.InputTag( "" ),
    inactivePixelDetectorLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    badPixelFEDChannelCollectionLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    pixelCablingMapLabel = cms.string( "" ),
    inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
    switchOffPixelsIfEmpty = cms.bool( True )
)
fragment.hltIterL3OISeedsFromL2Muons = cms.EDProducer( "TSGForOIDNN",
    src = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    layersToTry = cms.int32( 2 ),
    fixedErrorRescaleFactorForHitless = cms.double( 2.0 ),
    hitsToTry = cms.int32( 1 ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    estimator = cms.string( "hltESPChi2MeasurementEstimator100" ),
    maxEtaForTOB = cms.double( 1.8 ),
    minEtaForTEC = cms.double( 0.7 ),
    debug = cms.untracked.bool( False ),
    maxSeeds = cms.uint32( 20 ),
    maxHitlessSeeds = cms.uint32( 5 ),
    maxHitSeeds = cms.uint32( 1 ),
    propagatorName = cms.string( "PropagatorWithMaterialParabolicMf" ),
    maxHitlessSeedsIP = cms.uint32( 5 ),
    maxHitlessSeedsMuS = cms.uint32( 0 ),
    maxHitDoubletSeeds = cms.uint32( 0 ),
    getStrategyFromDNN = cms.bool( True ),
    useRegressor = cms.bool( False ),
    dnnMetadataPath = cms.string( "RecoMuon/TrackerSeedGenerator/data/OIseeding/DNNclassifier_Run3_metadata.json" )
)
fragment.hltIterL3OITrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltIterL3OISeedsFromL2Muons" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTPSetMuonCkfTrajectoryBuilder" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    TrajectoryCleaner = cms.string( "muonSeededTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 500000 ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 )
)
fragment.hltIterL3OIMuCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltIterL3OITrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "iter10" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( False ),
    SimpleMagneticField = cms.string( "" ),
    Fitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    MeasurementTracker = cms.string( "hltESPMeasurementTracker" )
)
fragment.hltIterL3OIMuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltIterL3OIMuCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "Notused" ),
    ignoreVertices = cms.bool( True ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 100.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 3.40282346639E38 ),
        dr_par2 = cms.vdouble( 0.3, 0.3, 3.40282346639E38 ),
        dr_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dr_exp = cms.vint32( 4, 4, 2147483647 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 3.40282346639E38 )
      ),
      maxLostLayers = cms.vint32( 4, 3, 2 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dz_par2 = cms.vdouble( 0.35, 0.35, 3.40282346639E38 ),
        dz_exp = cms.vint32( 4, 4, 2147483647 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 3.40282346639E38 ),
      maxChi2n = cms.vdouble( 10.0, 1.0, 0.4 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 5, 5 )
    )
)
fragment.hltIterL3OIMuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIterL3OIMuCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIterL3OIMuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIterL3OIMuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltL3MuonsIterL3OI = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    TrackLoaderParameters = cms.PSet( 
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      DoSmoothing = cms.bool( True ),
      SmoothTkTrack = cms.untracked.bool( False ),
      VertexConstraint = cms.bool( False ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
      ),
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_2 = cms.double( 15.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        Quality_3 = cms.double( 7.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        Quality_1 = cms.double( 20.0 ),
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        LocChi2Cut = cms.double( 0.001 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        MinPt = cms.double( 1.0 ),
        MinP = cms.double( 2.5 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      tkTrajUseVertex = cms.bool( False ),
      MuonTrackingRegionBuilder = cms.PSet( 
        Rescale_Dz = cms.double( 4.0 ),
        Pt_fixed = cms.bool( False ),
        Eta_fixed = cms.bool( True ),
        Eta_min = cms.double( 0.1 ),
        DeltaZ = cms.double( 24.2 ),
        maxRegions = cms.int32( 2 ),
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        UseVertex = cms.bool( False ),
        Z_fixed = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Rescale_phi = cms.double( 3.0 ),
        DeltaEta = cms.double( 0.2 ),
        precise = cms.bool( True ),
        OnDemand = cms.int32( -1 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Pt_min = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        Phi_fixed = cms.bool( True ),
        DeltaR = cms.double( 0.025 ),
        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
        DeltaPhi = cms.double( 0.15 ),
        Phi_min = cms.double( 0.1 ),
        Rescale_eta = cms.double( 3.0 )
      ),
      TrackTransformer = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
      ),
      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      RefitRPCHits = cms.bool( True ),
      tkTrajVertex = cms.InputTag( "Notused" ),
      GlbRefitterParameters = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        RefitFlag = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        PropDirForCosmics = cms.bool( False ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        GEMRecHitLabel = cms.InputTag( "hltGemRecHits" ),
        HitThreshold = cms.int32( 1 ),
        Chi2CutGEM = cms.double( 1.0 ),
        DYTthrs = cms.vint32( 30, 15 ),
        TrackerSkipSystem = cms.int32( -1 ),
        RefitDirection = cms.string( "insideOut" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        TrackerSkipSection = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonHitsOption = cms.int32( 1 ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
      ),
      PCut = cms.double( 2.5 ),
      tkTrajMaxDXYBeamSpot = cms.double( 9999.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      tkTrajMaxChi2 = cms.double( 9999.0 ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      ScaleTECyFactor = cms.double( -1.0 ),
      tkTrajLabel = cms.InputTag( "hltIterL3OIMuonTrackSelectionHighPurity" )
    )
)
fragment.hltIterL3OIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI' )
)
fragment.hltIterL3OIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI' )
)
fragment.hltIterL3OIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltIterL3OIL3Muons" ),
    InputLinksObjects = cms.InputTag( "hltIterL3OIL3MuonsLinksCombination" ),
    MuonPtOption = cms.string( "Tracker" )
)
fragment.hltL2SelectorForL3IO = cms.EDProducer( "HLTMuonL2SelectorForL3IO",
    l2Src = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    l3OISrc = cms.InputTag( "hltIterL3OIL3MuonCandidates" ),
    InputLinks = cms.InputTag( "hltIterL3OIL3MuonsLinksCombination" ),
    applyL3Filters = cms.bool( False ),
    MinNhits = cms.int32( 1 ),
    MaxNormalizedChi2 = cms.double( 20.0 ),
    MinNmuonHits = cms.int32( 1 ),
    MaxPtDifference = cms.double( 0.3 )
)
fragment.hltIterL3MuonPixelTracksTrackingRegions = cms.EDProducer( "MuonTrackingRegionByPtEDProducer",
    DeltaR = cms.double( 0.025 ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    OnDemand = cms.int32( -1 ),
    vertexCollection = cms.InputTag( "notUsed" ),
    MeasurementTrackerName = cms.InputTag( "" ),
    UseVertex = cms.bool( False ),
    Rescale_Dz = cms.double( 4.0 ),
    Pt_fixed = cms.bool( True ),
    Z_fixed = cms.bool( True ),
    Pt_min = cms.double( 0.0 ),
    DeltaZ = cms.double( 24.2 ),
    ptRanges = cms.vdouble( 0.0, 15.0, 20.0, 1.0E64 ),
    deltaEtas = cms.vdouble( 0.2, 0.2, 0.2 ),
    deltaPhis = cms.vdouble( 0.75, 0.45, 0.225 ),
    maxRegions = cms.int32( 5 ),
    precise = cms.bool( True ),
    input = cms.InputTag( "hltL2SelectorForL3IO" )
)
fragment.hltPixelTracksInRegionL2 = cms.EDProducer( "TrackSelectorByRegion",
    tracks = cms.InputTag( "hltPixelTracks" ),
    regions = cms.InputTag( "hltIterL3MuonPixelTracksTrackingRegions" ),
    produceTrackCollection = cms.bool( True ),
    produceMask = cms.bool( False )
)
fragment.hltIter0IterL3MuonPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag( "hltPixelTracksInRegionL2" ),
    InputVertexCollection = cms.InputTag( "" ),
    originHalfLength = cms.double( 0.3 ),
    originRadius = cms.double( 0.1 ),
    useProtoTrackKinematics = cms.bool( False ),
    useEventsWithNoVertex = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    usePV = cms.bool( False ),
    includeFourthHit = cms.bool( True ),
    produceComplement = cms.bool( False ),
    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromProtoTracks" ) )
)
fragment.hltIter0IterL3MuonPixelSeedsFromPixelTracksFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
    src = cms.InputTag( "hltIter0IterL3MuonPixelSeedsFromPixelTracks" ),
    L1Muon = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L2Muon = cms.InputTag( "hltL2MuonCandidates" ),
    rejectAll = cms.bool( False ),
    isFromL1 = cms.bool( False ),
    mvaFileB = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter0_PatatrackSeeds_barrel_v3.xml" ),
    mvaFileE = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter0_PatatrackSeeds_endcap_v3.xml" ),
    mvaScaleMeanB = cms.vdouble( 4.332629261558539E-4, 4.689795312031938E-6, 7.644844964566431E-6, 6.580623848546099E-4, 0.00523266117445817, 5.6968993532947E-4, 0.20322471101222087, -0.005575351463397025, 0.18247595248098955, 1.5342398341020196E-4 ),
    mvaScaleStdB = cms.vdouble( 7.444819891335438E-4, 0.0014335177986615237, 0.003503839482232683, 0.07764362324530726, 0.8223406268068466, 0.6392468338330071, 0.2405783807668161, 0.2904161358810494, 0.21887441827342669, 0.27045195352036544 ),
    mvaScaleMeanE = cms.vdouble( 3.120747098810717E-4, 4.5298701434656295E-6, 1.2002076996572005E-5, 0.007900535887258366, -0.022166389143849694, 7.12338927507459E-4, 0.22819667672872926, -0.0039375694144792705, 0.19304371973554835, -1.2936058928324214E-5 ),
    mvaScaleStdE = cms.vdouble( 6.302274350028021E-4, 0.0013138279991871378, 0.004880335178644773, 0.32509543981045624, 0.9449952711981982, 0.279802349646327, 0.3193063648341999, 0.3334815828876066, 0.22528017441813106, 0.2822750719936266 ),
    doSort = cms.bool( False ),
    nSeedsMaxB = cms.int32( 99999 ),
    nSeedsMaxE = cms.int32( 99999 ),
    etaEdge = cms.double( 1.2 ),
    mvaCutB = cms.double( 0.04 ),
    mvaCutE = cms.double( 0.04 ),
    minL1Qual = cms.int32( 7 ),
    baseScore = cms.double( 0.5 )
)
fragment.hltIter0IterL3MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( True ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltIter0IterL3MuonPixelSeedsFromPixelTracksFiltered" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3MuonPSetGroupedCkfTrajectoryBuilderIT" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "none" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 100000 ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 )
)
fragment.hltIter0IterL3MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltIter0IterL3MuonCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "hltIter0" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( True ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "" ),
    MeasurementTracker = cms.string( "" )
)
fragment.hltIter0IterL3MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltIter0IterL3MuonCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltTrimmedPixelVertices" ),
    ignoreVertices = cms.bool( False ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 100.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 3.40282346639E38 ),
        dr_par2 = cms.vdouble( 0.3, 0.3, 3.40282346639E38 ),
        dr_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dr_exp = cms.vint32( 4, 4, 2147483647 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 3.40282346639E38 )
      ),
      maxLostLayers = cms.vint32( 1, 1, 1 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dz_par2 = cms.vdouble( 0.35, 0.35, 3.40282346639E38 ),
        dz_exp = cms.vint32( 4, 4, 2147483647 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 3.40282346639E38 ),
      maxChi2n = cms.vdouble( 1.2, 1.0, 0.7 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 3, 4 )
    )
)
fragment.hltIter0IterL3MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0IterL3MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0IterL3MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0IterL3MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltL3MuonsIterL3IO = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    TrackLoaderParameters = cms.PSet( 
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      DoSmoothing = cms.bool( False ),
      SmoothTkTrack = cms.untracked.bool( False ),
      VertexConstraint = cms.bool( False ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
      ),
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_2 = cms.double( 15.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        Quality_3 = cms.double( 7.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        Quality_1 = cms.double( 20.0 ),
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        LocChi2Cut = cms.double( 0.001 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        MinPt = cms.double( 1.0 ),
        MinP = cms.double( 2.5 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      tkTrajUseVertex = cms.bool( False ),
      MuonTrackingRegionBuilder = cms.PSet( 
        Rescale_Dz = cms.double( 4.0 ),
        Pt_fixed = cms.bool( True ),
        Eta_fixed = cms.bool( True ),
        Eta_min = cms.double( 0.1 ),
        DeltaZ = cms.double( 24.2 ),
        maxRegions = cms.int32( 2 ),
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        UseVertex = cms.bool( False ),
        Z_fixed = cms.bool( True ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Rescale_phi = cms.double( 3.0 ),
        DeltaEta = cms.double( 0.04 ),
        precise = cms.bool( True ),
        OnDemand = cms.int32( -1 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Pt_min = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        Phi_fixed = cms.bool( True ),
        DeltaR = cms.double( 0.025 ),
        input = cms.InputTag( "hltL2SelectorForL3IO" ),
        DeltaPhi = cms.double( 0.15 ),
        Phi_min = cms.double( 0.1 ),
        Rescale_eta = cms.double( 3.0 )
      ),
      TrackTransformer = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
      ),
      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      RefitRPCHits = cms.bool( True ),
      tkTrajVertex = cms.InputTag( "hltTrimmedPixelVertices" ),
      GlbRefitterParameters = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        RefitFlag = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        PropDirForCosmics = cms.bool( False ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        GEMRecHitLabel = cms.InputTag( "hltGemRecHits" ),
        HitThreshold = cms.int32( 1 ),
        Chi2CutGEM = cms.double( 1.0 ),
        DYTthrs = cms.vint32( 30, 15 ),
        TrackerSkipSystem = cms.int32( -1 ),
        RefitDirection = cms.string( "insideOut" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        TrackerSkipSection = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonHitsOption = cms.int32( 1 ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
      ),
      PCut = cms.double( 2.5 ),
      tkTrajMaxDXYBeamSpot = cms.double( 9999.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchToSeeds = cms.bool( True ),
      tkTrajMaxChi2 = cms.double( 9999.0 ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      ScaleTECyFactor = cms.double( -1.0 ),
      tkTrajLabel = cms.InputTag( "hltIter0IterL3MuonTrackSelectionHighPurity" )
    )
)
fragment.hltIterL3MuonsFromL2LinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI','hltL3MuonsIterL3IO' )
)
fragment.hltL1MuonsPt0 = cms.EDProducer( "HLTL1TMuonSelector",
    InputObjects = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MinPt = cms.double( -1.0 ),
    L1MaxEta = cms.double( 5.0 ),
    L1MinQuality = cms.uint32( 7 ),
    CentralBxOnly = cms.bool( True )
)
fragment.hltIterL3FromL1MuonPixelTracksTrackingRegions = cms.EDProducer( "L1MuonSeededTrackingRegionsEDProducer",
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 7 ),
    SetMinPtBarrelTo = cms.double( 3.5 ),
    SetMinPtEndcapTo = cms.double( 1.0 ),
    CentralBxOnly = cms.bool( True ),
    RegionPSet = cms.PSet( 
      vertexCollection = cms.InputTag( "notUsed" ),
      deltaEtas = cms.vdouble( 0.35, 0.35, 0.35, 0.35 ),
      zErrorVetex = cms.double( 0.2 ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      zErrorBeamSpot = cms.double( 24.2 ),
      maxNVertices = cms.int32( 1 ),
      maxNRegions = cms.int32( 5 ),
      nSigmaZVertex = cms.double( 3.0 ),
      nSigmaZBeamSpot = cms.double( 4.0 ),
      ptMin = cms.double( 0.0 ),
      mode = cms.string( "BeamSpotSigma" ),
      input = cms.InputTag( "hltL1MuonsPt0" ),
      ptRanges = cms.vdouble( 0.0, 10.0, 15.0, 20.0, 1.0E64 ),
      searchOpt = cms.bool( False ),
      deltaPhis = cms.vdouble( 1.0, 0.8, 0.6, 0.3 ),
      whereToUseMeasurementTracker = cms.string( "Never" ),
      originRadius = cms.double( 0.2 ),
      measurementTrackerName = cms.InputTag( "" ),
      precise = cms.bool( True )
    ),
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' )
    )
)
fragment.hltPixelTracksInRegionL1 = cms.EDProducer( "TrackSelectorByRegion",
    tracks = cms.InputTag( "hltPixelTracks" ),
    regions = cms.InputTag( "hltIterL3FromL1MuonPixelTracksTrackingRegions" ),
    produceTrackCollection = cms.bool( True ),
    produceMask = cms.bool( False )
)
fragment.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag( "hltPixelTracksInRegionL1" ),
    InputVertexCollection = cms.InputTag( "" ),
    originHalfLength = cms.double( 0.3 ),
    originRadius = cms.double( 0.1 ),
    useProtoTrackKinematics = cms.bool( False ),
    useEventsWithNoVertex = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    usePV = cms.bool( False ),
    includeFourthHit = cms.bool( True ),
    produceComplement = cms.bool( False ),
    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromProtoTracks" ) )
)
fragment.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracksFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
    src = cms.InputTag( "hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks" ),
    L1Muon = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L2Muon = cms.InputTag( "hltL2MuonCandidates" ),
    rejectAll = cms.bool( False ),
    isFromL1 = cms.bool( True ),
    mvaFileB = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter0FromL1_PatatrackSeeds_barrel_v3.xml" ),
    mvaFileE = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter0FromL1_PatatrackSeeds_endcap_v3.xml" ),
    mvaScaleMeanB = cms.vdouble( 3.999966523561405E-4, 1.5340202670472034E-5, 2.6710290157638425E-5, 5.978116313043455E-4, 0.0049135275917734636, 3.4305653488182246E-5, 0.24525118734715307, -0.0024635178849904426 ),
    mvaScaleStdB = cms.vdouble( 7.666933596884494E-4, 0.015685297920984408, 0.026294325262867256, 0.07665283880432934, 0.834879854164998, 0.5397258722194461, 0.2807075832224741, 0.32820882609116625 ),
    mvaScaleMeanE = cms.vdouble( 3.017047347441654E-4, 9.077959353128816E-5, 2.7101609045025927E-4, 0.004557390407735609, -0.020781128525626812, 9.286198943080588E-4, 0.26674085200387376, -0.002971698676536822 ),
    mvaScaleStdE = cms.vdouble( 8.125341035878315E-4, 0.19268436761240013, 0.579019516987623, 0.3222327708969556, 1.0567488273501275, 0.2648980106841699, 0.30889713721141826, 0.3593729790466801 ),
    doSort = cms.bool( False ),
    nSeedsMaxB = cms.int32( 99999 ),
    nSeedsMaxE = cms.int32( 99999 ),
    etaEdge = cms.double( 1.2 ),
    mvaCutB = cms.double( 0.04 ),
    mvaCutE = cms.double( 0.04 ),
    minL1Qual = cms.int32( 7 ),
    baseScore = cms.double( 0.5 )
)
fragment.hltIter0IterL3FromL1MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( True ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracksFiltered" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter0IterL3FromL1MuonPSetGroupedCkfTrajectoryBuilderIT" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "none" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 100000 ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 )
)
fragment.hltIter0IterL3FromL1MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltIter0IterL3FromL1MuonCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "hltIter0" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( True ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "" ),
    MeasurementTracker = cms.string( "" )
)
fragment.hltIter0IterL3FromL1MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltIter0IterL3FromL1MuonCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltTrimmedPixelVertices" ),
    ignoreVertices = cms.bool( False ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 100.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 3.40282346639E38 ),
        dr_par2 = cms.vdouble( 0.3, 0.3, 3.40282346639E38 ),
        dr_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dr_exp = cms.vint32( 4, 4, 2147483647 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 3.40282346639E38 )
      ),
      maxLostLayers = cms.vint32( 1, 1, 1 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 0.4, 0.4, 3.40282346639E38 ),
        dz_par2 = cms.vdouble( 0.35, 0.35, 3.40282346639E38 ),
        dz_exp = cms.vint32( 4, 4, 2147483647 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 3.40282346639E38, 3.40282346639E38, 3.40282346639E38 ),
      maxChi2n = cms.vdouble( 1.2, 1.0, 0.7 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 3, 4 )
    )
)
fragment.hltIter0IterL3FromL1MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0IterL3FromL1MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0IterL3FromL1MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0IterL3FromL1MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltIter3IterL3FromL1MuonClustersRefRemoval = cms.EDProducer( "TrackClusterRemover",
    trajectories = cms.InputTag( "hltIter0IterL3FromL1MuonTrackSelectionHighPurity" ),
    trackClassifier = cms.InputTag( '','QualityMasks' ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    TrackQuality = cms.string( "highPurity" ),
    maxChi2 = cms.double( 16.0 ),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    overrideTrkQuals = cms.InputTag( "" )
)
fragment.hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    src = cms.InputTag( "hltMeasurementTrackerEvent" ),
    clustersToSkip = cms.InputTag( "hltIter3IterL3FromL1MuonClustersRefRemoval" ),
    phase2clustersToSkip = cms.InputTag( "" )
)
fragment.hltIter3IterL3FromL1MuonPixelLayersAndRegions = cms.EDProducer( "PixelInactiveAreaTrackingRegionsSeedingLayersProducer",
    RegionPSet = cms.PSet( 
      vertexCollection = cms.InputTag( "hltTrimmedPixelVertices" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      zErrorBeamSpot = cms.double( 15.0 ),
      extraPhi = cms.double( 0.0 ),
      extraEta = cms.double( 0.0 ),
      maxNVertices = cms.int32( 3 ),
      nSigmaZVertex = cms.double( 3.0 ),
      nSigmaZBeamSpot = cms.double( 4.0 ),
      ptMin = cms.double( 1.2 ),
      operationMode = cms.string( "VerticesFixed" ),
      searchOpt = cms.bool( False ),
      whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
      originRadius = cms.double( 0.015 ),
      measurementTrackerName = cms.InputTag( "hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent" ),
      precise = cms.bool( True ),
      zErrorVertex = cms.double( 0.03 )
    ),
    inactivePixelDetectorLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    badPixelFEDChannelCollectionLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    ignoreSingleFPixPanelModules = cms.bool( True ),
    debug = cms.untracked.bool( False ),
    createPlottingFiles = cms.untracked.bool( False ),
    layerList = cms.vstring( 'BPix1+BPix2',
      'BPix1+BPix3',
      'BPix1+BPix4',
      'BPix2+BPix3',
      'BPix2+BPix4',
      'BPix3+BPix4',
      'BPix1+FPix1_pos',
      'BPix1+FPix1_neg',
      'BPix1+FPix2_pos',
      'BPix1+FPix2_neg',
      'BPix1+FPix3_pos',
      'BPix1+FPix3_neg',
      'BPix2+FPix1_pos',
      'BPix2+FPix1_neg',
      'BPix2+FPix2_pos',
      'BPix2+FPix2_neg',
      'BPix3+FPix1_pos',
      'BPix3+FPix1_neg',
      'FPix1_pos+FPix2_pos',
      'FPix1_neg+FPix2_neg',
      'FPix1_pos+FPix3_pos',
      'FPix1_neg+FPix3_neg',
      'FPix2_pos+FPix3_pos',
      'FPix2_neg+FPix3_neg' ),
    BPix = cms.PSet( 
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      skipClusters = cms.InputTag( "hltIter3IterL3FromL1MuonClustersRefRemoval" ),
      useErrorsFromParam = cms.bool( True ),
      hitErrorRZ = cms.double( 0.006 ),
      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    FPix = cms.PSet( 
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      skipClusters = cms.InputTag( "hltIter3IterL3FromL1MuonClustersRefRemoval" ),
      useErrorsFromParam = cms.bool( True ),
      hitErrorRZ = cms.double( 0.0036 ),
      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    TIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet(  ),
    MTEC = cms.PSet(  )
)
fragment.hltIter3IterL3FromL1MuonTrackingRegions = cms.EDProducer( "L1MuonSeededTrackingRegionsEDProducer",
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 7 ),
    SetMinPtBarrelTo = cms.double( 3.5 ),
    SetMinPtEndcapTo = cms.double( 1.0 ),
    CentralBxOnly = cms.bool( True ),
    RegionPSet = cms.PSet( 
      vertexCollection = cms.InputTag( "hltTrimmedPixelVertices" ),
      deltaEtas = cms.vdouble( 0.175, 0.175, 0.175, 0.175 ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      zErrorBeamSpot = cms.double( 15.0 ),
      maxNVertices = cms.int32( 3 ),
      maxNRegions = cms.int32( 3 ),
      nSigmaZVertex = cms.double( 3.0 ),
      nSigmaZBeamSpot = cms.double( 4.0 ),
      ptMin = cms.double( 1.2 ),
      mode = cms.string( "VerticesFixed" ),
      input = cms.InputTag( "hltL1MuonsPt0" ),
      ptRanges = cms.vdouble( 0.0, 10.0, 15.0, 20.0, 1.0E64 ),
      searchOpt = cms.bool( False ),
      deltaPhis = cms.vdouble( 0.5, 0.4, 0.3, 0.15 ),
      whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
      originRadius = cms.double( 0.015 ),
      measurementTrackerName = cms.InputTag( "hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent" ),
      precise = cms.bool( True )
    ),
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' )
    )
)
fragment.hltIter3IterL3FromL1MuonPixelClusterCheck = cms.EDProducer( "ClusterCheckerEDProducer",
    doClusterCheck = cms.bool( False ),
    MaxNumberOfStripClusters = cms.uint32( 50000 ),
    ClusterCollectionLabel = cms.InputTag( "hltMeasurementTrackerEvent" ),
    MaxNumberOfPixelClusters = cms.uint32( 40000 ),
    PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
    cut = cms.string( "" ),
    DontCountDetsAboveNClusters = cms.uint32( 0 ),
    silentClusterCheck = cms.untracked.bool( False )
)
fragment.hltIter3IterL3FromL1MuonPixelHitDoublets = cms.EDProducer( "HitPairEDProducer",
    seedingLayers = cms.InputTag( "hltIter3IterL3FromL1MuonPixelLayersAndRegions" ),
    trackingRegions = cms.InputTag( "hltIter3IterL3FromL1MuonTrackingRegions" ),
    trackingRegionsSeedingLayers = cms.InputTag( "" ),
    clusterCheck = cms.InputTag( "hltIter3IterL3FromL1MuonPixelClusterCheck" ),
    produceSeedingHitSets = cms.bool( True ),
    produceIntermediateHitDoublets = cms.bool( False ),
    maxElement = cms.uint32( 0 ),
    maxElementTotal = cms.uint32( 50000000 ),
    putEmptyIfMaxElementReached = cms.bool( False ),
    layerPairs = cms.vuint32( 0 )
)
fragment.hltIter3IterL3FromL1MuonPixelSeeds = cms.EDProducer( "SeedCreatorFromRegionConsecutiveHitsEDProducer",
    seedingHitSets = cms.InputTag( "hltIter3IterL3FromL1MuonPixelHitDoublets" ),
    propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
    SeedMomentumForBOFF = cms.double( 5.0 ),
    OriginTransverseErrorMultiplier = cms.double( 1.0 ),
    MinOneOverPtError = cms.double( 1.0 ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    magneticField = cms.string( "ParabolicMf" ),
    forceKinematicWithRegionDirection = cms.bool( False ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
)
fragment.hltIter3IterL3FromL1MuonPixelSeedsFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
    src = cms.InputTag( "hltIter3IterL3FromL1MuonPixelSeeds" ),
    L1Muon = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L2Muon = cms.InputTag( "hltL2MuonCandidates" ),
    rejectAll = cms.bool( False ),
    isFromL1 = cms.bool( True ),
    mvaFileB = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter3FromL1_DoubletSeeds_barrel_v1.xml" ),
    mvaFileE = cms.FileInPath( "RecoMuon/TrackerSeedGenerator/data/xgb_Run3_Iter3FromL1_DoubletSeeds_endcap_v1.xml" ),
    mvaScaleMeanB = cms.vdouble( 0.006826621711798213, 1.340471761359199E-5, 2.5827749083302998E-6, 3.8329754175309627E-4, -0.006327854398161656, 0.0017211841076523692, 0.2760538806332439, -0.010429922003892818 ),
    mvaScaleStdB = cms.vdouble( 0.006225819995879627, 7.4048803387083885E-6, 3.6347963283736586E-6, 0.062213478665703675, 0.828854421408699, 0.3714730344087147, 0.42155116686695293, 0.38566415759730355 ),
    mvaScaleMeanE = cms.vdouble( 0.0013243955281318262, 7.150658575633707E-6, 1.0493070182976E-5, -0.004802713888821372, -0.022186379498012398, 8.335525228198972E-4, 0.2915475574025415, -0.01200308471140653 ),
    mvaScaleStdE = cms.vdouble( 0.0013768261827517547, 7.80116971559064E-6, 8.819635719472336E-5, 0.27824938208607475, 1.798678366076454, 0.16556388679148643, 0.48300543536161705, 0.401204958844809 ),
    doSort = cms.bool( False ),
    nSeedsMaxB = cms.int32( 99999 ),
    nSeedsMaxE = cms.int32( 99999 ),
    etaEdge = cms.double( 1.2 ),
    mvaCutB = cms.double( 0.1 ),
    mvaCutE = cms.double( 0.1 ),
    minL1Qual = cms.int32( 7 ),
    baseScore = cms.double( 0.5 )
)
fragment.hltIter3IterL3FromL1MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTrackerEvent = cms.InputTag( "hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltIter3IterL3FromL1MuonPixelSeedsFiltered" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter2GroupedCkfTrajectoryBuilderIT" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 100000 ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 )
)
fragment.hltIter3IterL3FromL1MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltIter3IterL3FromL1MuonCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "hltIter3IterL3FromL1Muon" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( True ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "" ),
    MeasurementTracker = cms.string( "" )
)
fragment.hltIter3IterL3FromL1MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltIter3IterL3FromL1MuonCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltTrimmedPixelVertices" ),
    ignoreVertices = cms.bool( False ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 15.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
        dr_par2 = cms.vdouble( 3.40282346639E38, 0.3, 0.3 ),
        dr_par1 = cms.vdouble( 3.40282346639E38, 0.4, 0.4 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
      ),
      maxLostLayers = cms.vint32( 1, 1, 1 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 3.40282346639E38, 0.4, 0.4 ),
        dz_par2 = cms.vdouble( 3.40282346639E38, 0.35, 0.35 ),
        dz_exp = cms.vint32( 4, 4, 4 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 9999.0, 25.0, 16.0 ),
      maxChi2n = cms.vdouble( 1.2, 1.0, 0.7 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 3, 3 )
    )
)
fragment.hltIter3IterL3FromL1MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter3IterL3FromL1MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter3IterL3FromL1MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter3IterL3FromL1MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltIter03IterL3FromL1MuonMerged = cms.EDProducer( "TrackListMerger",
    copyExtras = cms.untracked.bool( True ),
    copyMVA = cms.bool( False ),
    TrackProducers = cms.VInputTag( 'hltIter0IterL3FromL1MuonTrackSelectionHighPurity','hltIter3IterL3FromL1MuonTrackSelectionHighPurity' ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    MinPT = cms.double( 0.05 ),
    MinFound = cms.int32( 3 ),
    Epsilon = cms.double( -0.001 ),
    ShareFrac = cms.double( 0.19 ),
    allowFirstHitShare = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    LostHitPenalty = cms.double( 20.0 ),
    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
    newQuality = cms.string( "confirmed" ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( False ),
        tLists = cms.vint32( 0, 1 )
      )
    ),
    hasSelector = cms.vint32( 0, 0 ),
    selectedTrackQuals = cms.VInputTag( 'hltIter0IterL3FromL1MuonTrackSelectionHighPurity','hltIter3IterL3FromL1MuonTrackSelectionHighPurity' ),
    writeOnlyTrkQuals = cms.bool( False ),
    makeReKeyedSeeds = cms.untracked.bool( False ),
    trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" )
)
fragment.hltIterL3MuonMerged = cms.EDProducer( "TrackListMerger",
    copyExtras = cms.untracked.bool( True ),
    copyMVA = cms.bool( False ),
    TrackProducers = cms.VInputTag( 'hltIterL3OIMuonTrackSelectionHighPurity','hltIter0IterL3MuonTrackSelectionHighPurity' ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    MinPT = cms.double( 0.05 ),
    MinFound = cms.int32( 3 ),
    Epsilon = cms.double( -0.001 ),
    ShareFrac = cms.double( 0.19 ),
    allowFirstHitShare = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    LostHitPenalty = cms.double( 20.0 ),
    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
    newQuality = cms.string( "confirmed" ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( False ),
        tLists = cms.vint32( 0, 1 )
      )
    ),
    hasSelector = cms.vint32( 0, 0 ),
    selectedTrackQuals = cms.VInputTag( 'hltIterL3OIMuonTrackSelectionHighPurity','hltIter0IterL3MuonTrackSelectionHighPurity' ),
    writeOnlyTrkQuals = cms.bool( False ),
    makeReKeyedSeeds = cms.untracked.bool( False ),
    trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" )
)
fragment.hltIterL3MuonAndMuonFromL1Merged = cms.EDProducer( "TrackListMerger",
    copyExtras = cms.untracked.bool( True ),
    copyMVA = cms.bool( False ),
    TrackProducers = cms.VInputTag( 'hltIterL3MuonMerged','hltIter03IterL3FromL1MuonMerged' ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    MinPT = cms.double( 0.05 ),
    MinFound = cms.int32( 3 ),
    Epsilon = cms.double( -0.001 ),
    ShareFrac = cms.double( 0.19 ),
    allowFirstHitShare = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    LostHitPenalty = cms.double( 20.0 ),
    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
    newQuality = cms.string( "confirmed" ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( False ),
        tLists = cms.vint32( 0, 1 )
      )
    ),
    hasSelector = cms.vint32( 0, 0 ),
    selectedTrackQuals = cms.VInputTag( 'hltIterL3MuonMerged','hltIter03IterL3FromL1MuonMerged' ),
    writeOnlyTrkQuals = cms.bool( False ),
    makeReKeyedSeeds = cms.untracked.bool( False ),
    trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" )
)
fragment.hltIterL3GlbMuon = cms.EDProducer( "L3MuonProducer",
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
        'SteppingHelixPropagatorAny',
        'hltESPSmartPropagator',
        'hltESPSteppingHelixPropagatorOpposite' )
    ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    TrackLoaderParameters = cms.PSet( 
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      DoSmoothing = cms.bool( True ),
      SmoothTkTrack = cms.untracked.bool( False ),
      VertexConstraint = cms.bool( False ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" )
      ),
      PutTkTrackIntoEvent = cms.untracked.bool( False ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      PtCut = cms.double( 1.0 ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Chi2Cut_3 = cms.double( 200.0 ),
        DeltaDCut_2 = cms.double( 10.0 ),
        Eta_threshold = cms.double( 1.2 ),
        Quality_2 = cms.double( 15.0 ),
        DeltaDCut_1 = cms.double( 40.0 ),
        Quality_3 = cms.double( 7.0 ),
        DeltaDCut_3 = cms.double( 15.0 ),
        Quality_1 = cms.double( 20.0 ),
        Pt_threshold1 = cms.double( 0.0 ),
        DeltaRCut_2 = cms.double( 0.2 ),
        DeltaRCut_1 = cms.double( 0.1 ),
        Pt_threshold2 = cms.double( 9.99999999E8 ),
        Chi2Cut_1 = cms.double( 50.0 ),
        Chi2Cut_2 = cms.double( 50.0 ),
        DeltaRCut_3 = cms.double( 1.0 ),
        LocChi2Cut = cms.double( 0.001 ),
        Propagator = cms.string( "hltESPSmartPropagator" ),
        MinPt = cms.double( 1.0 ),
        MinP = cms.double( 2.5 )
      ),
      ScaleTECxFactor = cms.double( -1.0 ),
      tkTrajUseVertex = cms.bool( False ),
      MuonTrackingRegionBuilder = cms.PSet( 
        Rescale_Dz = cms.double( 4.0 ),
        Pt_fixed = cms.bool( False ),
        Eta_fixed = cms.bool( True ),
        Eta_min = cms.double( 0.1 ),
        DeltaZ = cms.double( 24.2 ),
        maxRegions = cms.int32( 2 ),
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        UseVertex = cms.bool( False ),
        Z_fixed = cms.bool( False ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        Rescale_phi = cms.double( 3.0 ),
        DeltaEta = cms.double( 0.2 ),
        precise = cms.bool( True ),
        OnDemand = cms.int32( -1 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Pt_min = cms.double( 3.0 ),
        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
        Phi_fixed = cms.bool( True ),
        DeltaR = cms.double( 0.025 ),
        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
        DeltaPhi = cms.double( 0.15 ),
        Phi_min = cms.double( 0.1 ),
        Rescale_eta = cms.double( 3.0 )
      ),
      TrackTransformer = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
      ),
      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      RefitRPCHits = cms.bool( True ),
      tkTrajVertex = cms.InputTag( "Notused" ),
      GlbRefitterParameters = cms.PSet( 
        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        RefitFlag = cms.bool( True ),
        SkipStation = cms.int32( -1 ),
        Chi2CutRPC = cms.double( 1.0 ),
        PropDirForCosmics = cms.bool( False ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        GEMRecHitLabel = cms.InputTag( "hltGemRecHits" ),
        HitThreshold = cms.int32( 1 ),
        Chi2CutGEM = cms.double( 1.0 ),
        DYTthrs = cms.vint32( 30, 15 ),
        TrackerSkipSystem = cms.int32( -1 ),
        RefitDirection = cms.string( "insideOut" ),
        Chi2CutCSC = cms.double( 150.0 ),
        Chi2CutDT = cms.double( 10.0 ),
        RefitRPCHits = cms.bool( True ),
        TrackerSkipSection = cms.int32( -1 ),
        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
        DoPredictionsOnly = cms.bool( False ),
        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
        MuonHitsOption = cms.int32( 1 ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
      ),
      PCut = cms.double( 2.5 ),
      tkTrajMaxDXYBeamSpot = cms.double( 9999.0 ),
      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      tkTrajMaxChi2 = cms.double( 9999.0 ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
      ScaleTECyFactor = cms.double( -1.0 ),
      tkTrajLabel = cms.InputTag( "hltIterL3MuonAndMuonFromL1Merged" )
    )
)
fragment.hltIterL3MuonsNoID = cms.EDProducer( "MuonIdProducer",
    MuonCaloCompatibility = cms.PSet( 
      delta_eta = cms.double( 0.02 ),
      delta_phi = cms.double( 0.02 ),
      allSiPMHO = cms.bool( False ),
      MuonTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_muons_lowPt_3_1_norm.root" ),
      PionTemplateFileName = cms.FileInPath( "RecoMuon/MuonIdentification/data/MuID_templates_pions_lowPt_3_1_norm.root" )
    ),
    TrackAssociatorParameters = cms.PSet( 
      useMuon = cms.bool( True ),
      truthMatch = cms.bool( False ),
      usePreshower = cms.bool( False ),
      dRPreshowerPreselection = cms.double( 0.2 ),
      muonMaxDistanceSigmaY = cms.double( 0.0 ),
      useEcal = cms.bool( False ),
      muonMaxDistanceSigmaX = cms.double( 0.0 ),
      dRMuon = cms.double( 9999.0 ),
      dREcal = cms.double( 9999.0 ),
      CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
      DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
      EBRecHitCollectionLabel = cms.InputTag( "Notused" ),
      useGEM = cms.bool( True ),
      GEMSegmentCollectionLabel = cms.InputTag( "hltGemSegments" ),
      CaloTowerCollectionLabel = cms.InputTag( "Notused" ),
      propagateAllDirections = cms.bool( True ),
      muonMaxDistanceY = cms.double( 5.0 ),
      useHO = cms.bool( False ),
      muonMaxDistanceX = cms.double( 5.0 ),
      trajectoryUncertaintyTolerance = cms.double( -1.0 ),
      useHcal = cms.bool( False ),
      HBHERecHitCollectionLabel = cms.InputTag( "Notused" ),
      accountForTrajectoryChangeCalo = cms.bool( False ),
      dREcalPreselection = cms.double( 0.05 ),
      useCalo = cms.bool( False ),
      dRMuonPreselection = cms.double( 0.2 ),
      EERecHitCollectionLabel = cms.InputTag( "Notused" ),
      dRHcal = cms.double( 9999.0 ),
      dRHcalPreselection = cms.double( 0.2 ),
      HORecHitCollectionLabel = cms.InputTag( "Notused" )
    ),
    CaloExtractorPSet = cms.PSet( 
      DR_Veto_H = cms.double( 0.1 ),
      CenterConeOnCalIntersection = cms.bool( False ),
      NoiseTow_EE = cms.double( 0.15 ),
      Noise_EB = cms.double( 0.025 ),
      Noise_HE = cms.double( 0.2 ),
      DR_Veto_E = cms.double( 0.07 ),
      NoiseTow_EB = cms.double( 0.04 ),
      Noise_EE = cms.double( 0.1 ),
      UseRecHitsFlag = cms.bool( False ),
      DR_Max = cms.double( 1.0 ),
      DepositLabel = cms.untracked.string( "Cal" ),
      Noise_HO = cms.double( 0.2 ),
      DR_Veto_HO = cms.double( 0.1 ),
      Threshold_H = cms.double( 0.5 ),
      PrintTimeReport = cms.untracked.bool( False ),
      Threshold_E = cms.double( 0.2 ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      ComponentName = cms.string( "CaloExtractorByAssociator" ),
      Threshold_HO = cms.double( 0.5 ),
      DepositInstanceLabels = cms.vstring( 'ecal',
        'hcal',
        'ho' ),
      ServiceParameters = cms.PSet( 
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False ),
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' )
      ),
      TrackAssociatorParameters = cms.PSet( 
        useMuon = cms.bool( False ),
        truthMatch = cms.bool( False ),
        usePreshower = cms.bool( False ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        useEcal = cms.bool( False ),
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        dRMuon = cms.double( 9999.0 ),
        dREcal = cms.double( 1.0 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EBRecHitCollectionLabel = cms.InputTag( "Notused" ),
        CaloTowerCollectionLabel = cms.InputTag( "Notused" ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        muonMaxDistanceX = cms.double( 5.0 ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        useHcal = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "Notused" ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        dREcalPreselection = cms.double( 1.0 ),
        useCalo = cms.bool( True ),
        dRMuonPreselection = cms.double( 0.2 ),
        EERecHitCollectionLabel = cms.InputTag( "Notused" ),
        dRHcal = cms.double( 1.0 ),
        dRHcalPreselection = cms.double( 1.0 ),
        HORecHitCollectionLabel = cms.InputTag( "Notused" )
      ),
      Noise_HB = cms.double( 0.2 )
    ),
    TrackExtractorPSet = cms.PSet( 
      Diff_z = cms.double( 0.2 ),
      inputTrackCollection = cms.InputTag( "hltIter03IterL3FromL1MuonMerged" ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      BeamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
      DR_Veto = cms.double( 0.01 ),
      Pt_Min = cms.double( -1.0 ),
      DR_Max = cms.double( 1.0 ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      Diff_r = cms.double( 0.1 ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" ),
      ComponentName = cms.string( "TrackExtractor" )
    ),
    JetExtractorPSet = cms.PSet( 
      JetCollectionLabel = cms.InputTag( "Notused" ),
      DR_Veto = cms.double( 0.1 ),
      DR_Max = cms.double( 1.0 ),
      ExcludeMuonVeto = cms.bool( True ),
      PrintTimeReport = cms.untracked.bool( False ),
      PropagatorName = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      ComponentName = cms.string( "JetExtractor" ),
      ServiceParameters = cms.PSet( 
        RPCLayers = cms.bool( False ),
        UseMuonNavigation = cms.untracked.bool( False ),
        Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' )
      ),
      TrackAssociatorParameters = cms.PSet( 
        useMuon = cms.bool( False ),
        truthMatch = cms.bool( False ),
        usePreshower = cms.bool( False ),
        dRPreshowerPreselection = cms.double( 0.2 ),
        muonMaxDistanceSigmaY = cms.double( 0.0 ),
        useEcal = cms.bool( False ),
        muonMaxDistanceSigmaX = cms.double( 0.0 ),
        dRMuon = cms.double( 9999.0 ),
        dREcal = cms.double( 0.5 ),
        CSCSegmentCollectionLabel = cms.InputTag( "hltCscSegments" ),
        DTRecSegment4DCollectionLabel = cms.InputTag( "hltDt4DSegments" ),
        EBRecHitCollectionLabel = cms.InputTag( "Notused" ),
        CaloTowerCollectionLabel = cms.InputTag( "Notused" ),
        propagateAllDirections = cms.bool( True ),
        muonMaxDistanceY = cms.double( 5.0 ),
        useHO = cms.bool( False ),
        muonMaxDistanceX = cms.double( 5.0 ),
        trajectoryUncertaintyTolerance = cms.double( -1.0 ),
        useHcal = cms.bool( False ),
        HBHERecHitCollectionLabel = cms.InputTag( "Notused" ),
        accountForTrajectoryChangeCalo = cms.bool( False ),
        dREcalPreselection = cms.double( 0.5 ),
        useCalo = cms.bool( True ),
        dRMuonPreselection = cms.double( 0.2 ),
        EERecHitCollectionLabel = cms.InputTag( "Notused" ),
        dRHcal = cms.double( 0.5 ),
        dRHcalPreselection = cms.double( 0.5 ),
        HORecHitCollectionLabel = cms.InputTag( "Notused" )
      ),
      Threshold = cms.double( 5.0 )
    ),
    trackDepositName = cms.string( "tracker" ),
    ecalDepositName = cms.string( "ecal" ),
    hcalDepositName = cms.string( "hcal" ),
    hoDepositName = cms.string( "ho" ),
    jetDepositName = cms.string( "jets" ),
    TimingFillerParameters = cms.PSet( 
      DTTimingParameters = cms.PSet( 
        HitError = cms.double( 6.0 ),
        MatchParameters = cms.PSet( 
          TightMatchDT = cms.bool( False ),
          DTradius = cms.double( 0.01 ),
          TightMatchCSC = cms.bool( True ),
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" )
        ),
        debug = cms.bool( False ),
        DoWireCorr = cms.bool( False ),
        RequireBothProjections = cms.bool( False ),
        DTTimeOffset = cms.double( 2.7 ),
        PruneCut = cms.double( 10000.0 ),
        DTsegments = cms.InputTag( "hltDt4DSegments" ),
        UseSegmentT0 = cms.bool( False ),
        HitsMin = cms.int32( 5 ),
        DropTheta = cms.bool( True ),
        ServiceParameters = cms.PSet( 
          RPCLayers = cms.bool( True ),
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' )
        )
      ),
      UseCSC = cms.bool( True ),
      CSCTimingParameters = cms.PSet( 
        MatchParameters = cms.PSet( 
          TightMatchDT = cms.bool( False ),
          DTradius = cms.double( 0.01 ),
          TightMatchCSC = cms.bool( True ),
          CSCsegments = cms.InputTag( "hltCscSegments" ),
          DTsegments = cms.InputTag( "hltDt4DSegments" )
        ),
        debug = cms.bool( False ),
        CSCWireTimeOffset = cms.double( 0.0 ),
        CSCStripError = cms.double( 7.0 ),
        CSCTimeOffset = cms.double( 0.0 ),
        CSCWireError = cms.double( 8.6 ),
        PruneCut = cms.double( 100.0 ),
        CSCsegments = cms.InputTag( "hltCscSegments" ),
        UseStripTime = cms.bool( True ),
        CSCStripTimeOffset = cms.double( 0.0 ),
        UseWireTime = cms.bool( True ),
        ServiceParameters = cms.PSet( 
          RPCLayers = cms.bool( True ),
          Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny' )
        )
      ),
      ErrorDT = cms.double( 6.0 ),
      EcalEnergyCut = cms.double( 0.4 ),
      UseECAL = cms.bool( True ),
      ErrorEB = cms.double( 2.085 ),
      UseDT = cms.bool( True ),
      ErrorEE = cms.double( 6.95 ),
      ErrorCSC = cms.double( 7.4 )
    ),
    ShowerDigiFillerParameters = cms.PSet( 
      cscDigiCollectionLabel = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
      digiMaxDistanceX = cms.double( 25.0 ),
      dtDigiCollectionLabel = cms.InputTag( "hltMuonDTDigis" )
    ),
    TrackerKinkFinderParameters = cms.PSet( 
      usePosition = cms.bool( False ),
      diagonalOnly = cms.bool( False )
    ),
    fillEnergy = cms.bool( False ),
    storeCrossedHcalRecHits = cms.bool( False ),
    maxAbsPullX = cms.double( 4.0 ),
    maxAbsEta = cms.double( 3.0 ),
    minPt = cms.double( 2.0 ),
    inputCollectionTypes = cms.vstring( 'inner tracks',
      'links',
      'outer tracks' ),
    addExtraSoftMuons = cms.bool( False ),
    fillGlobalTrackRefits = cms.bool( False ),
    debugWithTruthMatching = cms.bool( False ),
    inputCollectionLabels = cms.VInputTag( 'hltIterL3MuonAndMuonFromL1Merged','hltIterL3GlbMuon','hltL2Muons:UpdatedAtVtx' ),
    fillCaloCompatibility = cms.bool( False ),
    maxAbsPullY = cms.double( 9999.0 ),
    maxAbsDy = cms.double( 9999.0 ),
    minP = cms.double( 0.0 ),
    minPCaloMuon = cms.double( 1.0E9 ),
    maxAbsDx = cms.double( 3.0 ),
    fillIsolation = cms.bool( False ),
    writeIsoDeposits = cms.bool( False ),
    minNumberOfMatches = cms.int32( 1 ),
    fillMatching = cms.bool( True ),
    fillShowerDigis = cms.bool( False ),
    ptThresholdToFillCandidateP4WithGlobalFit = cms.double( 200.0 ),
    sigmaThresholdToFillCandidateP4WithGlobalFit = cms.double( 2.0 ),
    fillGlobalTrackQuality = cms.bool( False ),
    globalTrackQualityInputTag = cms.InputTag( "" ),
    selectHighPurity = cms.bool( False ),
    pvInputTag = cms.InputTag( "" ),
    fillTrackerKink = cms.bool( False ),
    minCaloCompatibility = cms.double( 0.6 ),
    runArbitrationCleaner = cms.bool( False ),
    arbitrationCleanerOptions = cms.PSet( 
      OverlapDTheta = cms.double( 0.02 ),
      Overlap = cms.bool( True ),
      Clustering = cms.bool( True ),
      ME1a = cms.bool( True ),
      ClusterDTheta = cms.double( 0.02 ),
      ClusterDPhi = cms.double( 0.6 ),
      OverlapDPhi = cms.double( 0.0786 )
    ),
    arbitrateTrackerMuons = cms.bool( True )
)
fragment.hltIterL3Muons = cms.EDProducer( "MuonIDFilterProducerForHLT",
    inputMuonCollection = cms.InputTag( "hltIterL3MuonsNoID" ),
    applyTriggerIdLoose = cms.bool( True ),
    typeMuon = cms.uint32( 0 ),
    allowedTypeMask = cms.uint32( 0 ),
    requiredTypeMask = cms.uint32( 0 ),
    minNMuonHits = cms.int32( 0 ),
    minNMuonStations = cms.int32( 0 ),
    minNTrkLayers = cms.int32( 0 ),
    minTrkHits = cms.int32( 0 ),
    minPixLayer = cms.int32( 0 ),
    minPixHits = cms.int32( 0 ),
    minPt = cms.double( 0.0 ),
    maxNormalizedChi2 = cms.double( 9999.0 )
)
fragment.hltL3MuonsIterL3Links = cms.EDProducer( "MuonLinksProducer",
    inputCollection = cms.InputTag( "hltIterL3Muons" )
)
fragment.hltIterL3MuonTracks = cms.EDProducer( "HLTMuonTrackSelector",
    track = cms.InputTag( "hltIterL3MuonAndMuonFromL1Merged" ),
    muon = cms.InputTag( "hltIterL3Muons" ),
    originalMVAVals = cms.InputTag( "none" ),
    copyMVA = cms.bool( False ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltIterL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducerFromMuons",
    InputObjects = cms.InputTag( "hltIterL3Muons" ),
    DisplacedReconstruction = cms.bool( False )
)
fragment.hltIter0PFLowPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag( "hltPixelTracks" ),
    InputVertexCollection = cms.InputTag( "hltTrimmedPixelVertices" ),
    originHalfLength = cms.double( 0.3 ),
    originRadius = cms.double( 0.1 ),
    useProtoTrackKinematics = cms.bool( False ),
    useEventsWithNoVertex = cms.bool( True ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
    usePV = cms.bool( False ),
    includeFourthHit = cms.bool( True ),
    produceComplement = cms.bool( False ),
    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromProtoTracks" ) )
)
fragment.hltIter0PFlowCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltIter0PFLowPixelSeedsFromPixelTracks" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter0GroupedCkfTrajectoryBuilderIT" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 100000 ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 )
)
fragment.hltIter0PFlowCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltIter0PFlowCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "hltIter0" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( True ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "" ),
    MeasurementTracker = cms.string( "" )
)
fragment.hltIter0PFlowTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltTrimmedPixelVertices" ),
    ignoreVertices = cms.bool( False ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 15.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
        dr_par2 = cms.vdouble( 3.40282346639E38, 0.6, 0.6 ),
        dr_par1 = cms.vdouble( 3.40282346639E38, 0.8, 0.8 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
      ),
      maxLostLayers = cms.vint32( 1, 1, 1 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 3.40282346639E38, 0.75, 0.75 ),
        dz_par2 = cms.vdouble( 3.40282346639E38, 0.5, 0.5 ),
        dz_exp = cms.vint32( 4, 4, 4 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 9999.0, 25.0, 16.0 ),
      maxChi2n = cms.vdouble( 1.2, 1.0, 0.7 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 3, 3 )
    )
)
fragment.hltIter0PFlowTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0PFlowTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0PFlowTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltDoubletRecoveryClustersRefRemoval = cms.EDProducer( "TrackClusterRemover",
    trajectories = cms.InputTag( "hltIter0PFlowTrackSelectionHighPurity" ),
    trackClassifier = cms.InputTag( '','QualityMasks' ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    oldClusterRemovalInfo = cms.InputTag( "" ),
    TrackQuality = cms.string( "highPurity" ),
    maxChi2 = cms.double( 16.0 ),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32( 0 ),
    overrideTrkQuals = cms.InputTag( "" )
)
fragment.hltDoubletRecoveryMaskedMeasurementTrackerEvent = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    src = cms.InputTag( "hltMeasurementTrackerEvent" ),
    clustersToSkip = cms.InputTag( "hltDoubletRecoveryClustersRefRemoval" ),
    phase2clustersToSkip = cms.InputTag( "" )
)
fragment.hltDoubletRecoveryPixelLayersAndRegions = cms.EDProducer( "PixelInactiveAreaTrackingRegionsSeedingLayersProducer",
    RegionPSet = cms.PSet( 
      vertexCollection = cms.InputTag( "hltTrimmedPixelVertices" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      zErrorBeamSpot = cms.double( 15.0 ),
      extraPhi = cms.double( 0.0 ),
      extraEta = cms.double( 0.0 ),
      maxNVertices = cms.int32( 3 ),
      nSigmaZVertex = cms.double( 3.0 ),
      nSigmaZBeamSpot = cms.double( 4.0 ),
      ptMin = cms.double( 1.2 ),
      operationMode = cms.string( "VerticesFixed" ),
      searchOpt = cms.bool( False ),
      whereToUseMeasurementTracker = cms.string( "ForSiStrips" ),
      originRadius = cms.double( 0.015 ),
      measurementTrackerName = cms.InputTag( "hltDoubletRecoveryMaskedMeasurementTrackerEvent" ),
      precise = cms.bool( True ),
      zErrorVertex = cms.double( 0.03 )
    ),
    inactivePixelDetectorLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    badPixelFEDChannelCollectionLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    ignoreSingleFPixPanelModules = cms.bool( True ),
    debug = cms.untracked.bool( False ),
    createPlottingFiles = cms.untracked.bool( False ),
    layerList = cms.vstring( 'BPix1+BPix2',
      'BPix2+FPix1_pos',
      'BPix2+FPix1_neg',
      'FPix1_pos+FPix2_pos',
      'FPix1_neg+FPix2_neg',
      'BPix1+FPix2_neg',
      'BPix2+FPix2_neg',
      'FPix2_neg+FPix3_neg',
      'BPix2+BPix3' ),
    BPix = cms.PSet( 
      hitErrorRPhi = cms.double( 0.0027 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      skipClusters = cms.InputTag( "hltDoubletRecoveryClustersRefRemoval" ),
      useErrorsFromParam = cms.bool( True ),
      hitErrorRZ = cms.double( 0.006 ),
      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    FPix = cms.PSet( 
      hitErrorRPhi = cms.double( 0.0051 ),
      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
      skipClusters = cms.InputTag( "hltDoubletRecoveryClustersRefRemoval" ),
      useErrorsFromParam = cms.bool( True ),
      hitErrorRZ = cms.double( 0.0036 ),
      HitProducer = cms.string( "hltSiPixelRecHits" )
    ),
    TIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    TEC = cms.PSet(  ),
    MTIB = cms.PSet(  ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet(  ),
    MTEC = cms.PSet(  )
)
fragment.hltDoubletRecoveryPFlowPixelClusterCheck = cms.EDProducer( "ClusterCheckerEDProducer",
    doClusterCheck = cms.bool( False ),
    MaxNumberOfStripClusters = cms.uint32( 50000 ),
    ClusterCollectionLabel = cms.InputTag( "hltMeasurementTrackerEvent" ),
    MaxNumberOfPixelClusters = cms.uint32( 40000 ),
    PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
    cut = cms.string( "" ),
    DontCountDetsAboveNClusters = cms.uint32( 0 ),
    silentClusterCheck = cms.untracked.bool( False )
)
fragment.hltDoubletRecoveryPFlowPixelHitDoublets = cms.EDProducer( "HitPairEDProducer",
    seedingLayers = cms.InputTag( "" ),
    trackingRegions = cms.InputTag( "" ),
    trackingRegionsSeedingLayers = cms.InputTag( "hltDoubletRecoveryPixelLayersAndRegions" ),
    clusterCheck = cms.InputTag( "hltDoubletRecoveryPFlowPixelClusterCheck" ),
    produceSeedingHitSets = cms.bool( True ),
    produceIntermediateHitDoublets = cms.bool( False ),
    maxElement = cms.uint32( 0 ),
    maxElementTotal = cms.uint32( 50000000 ),
    putEmptyIfMaxElementReached = cms.bool( False ),
    layerPairs = cms.vuint32( 0 )
)
fragment.hltDoubletRecoveryPFlowPixelSeeds = cms.EDProducer( "SeedCreatorFromRegionConsecutiveHitsEDProducer",
    seedingHitSets = cms.InputTag( "hltDoubletRecoveryPFlowPixelHitDoublets" ),
    propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
    SeedMomentumForBOFF = cms.double( 5.0 ),
    OriginTransverseErrorMultiplier = cms.double( 1.0 ),
    MinOneOverPtError = cms.double( 1.0 ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    magneticField = cms.string( "ParabolicMf" ),
    forceKinematicWithRegionDirection = cms.bool( False ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
)
fragment.hltDoubletRecoveryPFlowCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    MeasurementTrackerEvent = cms.InputTag( "hltDoubletRecoveryMaskedMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltDoubletRecoveryPFlowPixelSeeds" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter2GroupedCkfTrajectoryBuilderIT" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
    maxNSeeds = cms.uint32( 100000 ),
    maxSeedsBeforeCleaning = cms.uint32( 1000 )
)
fragment.hltDoubletRecoveryPFlowCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltDoubletRecoveryPFlowCkfTrackCandidates" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "hltDoubletRecovery" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltDoubletRecoveryMaskedMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( True ),
    SimpleMagneticField = cms.string( "ParabolicMf" ),
    Fitter = cms.string( "hltESPFittingSmootherIT" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    NavigationSchool = cms.string( "" ),
    MeasurementTracker = cms.string( "" )
)
fragment.hltDoubletRecoveryPFlowTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
    src = cms.InputTag( "hltDoubletRecoveryPFlowCtfWithMaterialTracks" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    vertices = cms.InputTag( "hltTrimmedPixelVertices" ),
    ignoreVertices = cms.bool( False ),
    qualityCuts = cms.vdouble( -0.7, 0.1, 0.7 ),
    mva = cms.PSet( 
      minPixelHits = cms.vint32( 0, 0, 0 ),
      maxDzWrtBS = cms.vdouble( 3.40282346639E38, 24.0, 15.0 ),
      dr_par = cms.PSet( 
        d0err = cms.vdouble( 0.003, 0.003, 0.003 ),
        dr_par2 = cms.vdouble( 3.40282346639E38, 0.3, 0.3 ),
        dr_par1 = cms.vdouble( 3.40282346639E38, 0.4, 0.4 ),
        dr_exp = cms.vint32( 4, 4, 4 ),
        d0err_par = cms.vdouble( 0.001, 0.001, 0.001 )
      ),
      maxLostLayers = cms.vint32( 1, 1, 1 ),
      min3DLayers = cms.vint32( 0, 0, 0 ),
      dz_par = cms.PSet( 
        dz_par1 = cms.vdouble( 3.40282346639E38, 0.4, 0.4 ),
        dz_par2 = cms.vdouble( 3.40282346639E38, 0.35, 0.35 ),
        dz_exp = cms.vint32( 4, 4, 4 )
      ),
      minNVtxTrk = cms.int32( 3 ),
      maxDz = cms.vdouble( 0.5, 0.2, 3.40282346639E38 ),
      minNdof = cms.vdouble( 1.0E-5, 1.0E-5, 1.0E-5 ),
      maxChi2 = cms.vdouble( 9999.0, 25.0, 16.0 ),
      maxChi2n = cms.vdouble( 1.2, 1.0, 0.7 ),
      maxDr = cms.vdouble( 0.5, 0.03, 3.40282346639E38 ),
      minLayers = cms.vint32( 3, 3, 3 )
    )
)
fragment.hltDoubletRecoveryPFlowTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltDoubletRecoveryPFlowCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltDoubletRecoveryPFlowTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltDoubletRecoveryPFlowTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
fragment.hltMergedTracks = cms.EDProducer( "TrackListMerger",
    copyExtras = cms.untracked.bool( True ),
    copyMVA = cms.bool( False ),
    TrackProducers = cms.VInputTag( 'hltIter0PFlowTrackSelectionHighPurity','hltDoubletRecoveryPFlowTrackSelectionHighPurity' ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    MinPT = cms.double( 0.05 ),
    MinFound = cms.int32( 3 ),
    Epsilon = cms.double( -0.001 ),
    ShareFrac = cms.double( 0.19 ),
    allowFirstHitShare = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    LostHitPenalty = cms.double( 20.0 ),
    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
    newQuality = cms.string( "confirmed" ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( False ),
        tLists = cms.vint32( 0, 1 )
      )
    ),
    hasSelector = cms.vint32( 0, 0 ),
    selectedTrackQuals = cms.VInputTag( 'hltIter0PFlowTrackSelectionHighPurity','hltDoubletRecoveryPFlowTrackSelectionHighPurity' ),
    writeOnlyTrkQuals = cms.bool( False ),
    makeReKeyedSeeds = cms.untracked.bool( False ),
    trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" )
)
fragment.hltPFMuonMerging = cms.EDProducer( "TrackListMerger",
    copyExtras = cms.untracked.bool( True ),
    copyMVA = cms.bool( False ),
    TrackProducers = cms.VInputTag( 'hltIterL3MuonTracks','hltMergedTracks' ),
    MaxNormalizedChisq = cms.double( 1000.0 ),
    MinPT = cms.double( 0.05 ),
    MinFound = cms.int32( 3 ),
    Epsilon = cms.double( -0.001 ),
    ShareFrac = cms.double( 0.19 ),
    allowFirstHitShare = cms.bool( True ),
    FoundHitBonus = cms.double( 5.0 ),
    LostHitPenalty = cms.double( 20.0 ),
    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
    newQuality = cms.string( "confirmed" ),
    setsToMerge = cms.VPSet( 
      cms.PSet(  pQual = cms.bool( False ),
        tLists = cms.vint32( 0, 1 )
      )
    ),
    hasSelector = cms.vint32( 0, 0 ),
    selectedTrackQuals = cms.VInputTag( 'hltIterL3MuonTracks','hltMergedTracks' ),
    writeOnlyTrkQuals = cms.bool( False ),
    makeReKeyedSeeds = cms.untracked.bool( False ),
    trackAlgoPriorityOrder = cms.string( "hltESPTrackAlgoPriorityOrder" )
)
fragment.hltVerticesPF = cms.EDProducer( "PrimaryVertexProducer",
    vertexCollections = cms.VPSet( 
      cms.PSet(  chi2cutoff = cms.double( 3.0 ),
        label = cms.string( "" ),
        useBeamConstraint = cms.bool( False ),
        minNdof = cms.double( 0.0 ),
        maxDistanceToBeam = cms.double( 1.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" )
      ),
      cms.PSet(  chi2cutoff = cms.double( 3.0 ),
        label = cms.string( "WithBS" ),
        useBeamConstraint = cms.bool( True ),
        minNdof = cms.double( 0.0 ),
        maxDistanceToBeam = cms.double( 1.0 ),
        algorithm = cms.string( "AdaptiveVertexFitter" )
      )
    ),
    verbose = cms.untracked.bool( False ),
    TkFilterParameters = cms.PSet( 
      maxEta = cms.double( 100.0 ),
      minPt = cms.double( 0.0 ),
      minSiliconLayersWithHits = cms.int32( 5 ),
      minPixelLayersWithHits = cms.int32( 2 ),
      maxNormalizedChi2 = cms.double( 20.0 ),
      trackQuality = cms.string( "any" ),
      algorithm = cms.string( "filter" ),
      maxD0Significance = cms.double( 999.0 )
    ),
    beamSpotLabel = cms.InputTag( "hltOnlineBeamSpot" ),
    TrackLabel = cms.InputTag( "hltPFMuonMerging" ),
    TrackTimeResosLabel = cms.InputTag( "dummy_default" ),
    TrackTimesLabel = cms.InputTag( "dummy_default" ),
    trackMTDTimeQualityVMapTag = cms.InputTag( "dummy_default" ),
    TkClusParameters = cms.PSet( 
      TkDAClusParameters = cms.PSet( 
        zmerge = cms.double( 0.01 ),
        Tstop = cms.double( 0.5 ),
        d0CutOff = cms.double( 999.0 ),
        dzCutOff = cms.double( 4.0 ),
        vertexSize = cms.double( 0.15 ),
        coolingFactor = cms.double( 0.6 ),
        Tpurge = cms.double( 2.0 ),
        Tmin = cms.double( 2.4 ),
        uniquetrkweight = cms.double( 0.9 )
      ),
      algorithm = cms.string( "DA_vect" )
    ),
    isRecoveryIteration = cms.bool( False ),
    recoveryVtxCollection = cms.InputTag( "" ),
    useMVACut = cms.bool( False ),
    minTrackTimeQuality = cms.double( 0.8 )
)
fragment.hltVerticesPFSelector = cms.EDFilter( "PrimaryVertexObjectFilter",
    src = cms.InputTag( "hltVerticesPF" ),
    filterParams = cms.PSet( 
      maxZ = cms.double( 24.0 ),
      minNdof = cms.double( 4.0 ),
      maxRho = cms.double( 2.0 )
    ),
    filter = cms.bool( False )
)
fragment.hltVerticesPFFilter = cms.EDFilter( "VertexSelector",
    src = cms.InputTag( "hltVerticesPFSelector" ),
    cut = cms.string( "!isFake" ),
    filter = cms.bool( True ),
    throwOnMissing = cms.untracked.bool( True )
)
fragment.hltFEDSelectorOnlineMetaData = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1022 )
)
fragment.hltL1sIsolatedBunch = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_IsolatedBunch" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreZeroBiasIsolatedBunches = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sL1ZeroBiasFirstBunchAfterTrain = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_FirstBunchAfterTrain" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreZeroBiasFirstBXAfterTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sL1ZeroBiasFirstCollisionAfterAbortGap = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_FirstCollisionInOrbit" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreZeroBiasFirstCollisionAfterAbortGap = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sL1ZeroBiasFirstCollisionInTrainNOTFirstCollisionInOrbit = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_FirstCollisionInTrain AND (NOT L1_FirstCollisionInOrbit)" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreZeroBiasFirstCollisionInTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sL1ZeroBiasLastBunchInTrain = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_LastCollisionInTrain" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreZeroBiasLastCollisionInTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTTForBeamSpot = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT120er OR L1_HTT160er OR L1_HTT200er OR L1_HTT255er OR L1_HTT280er OR L1_HTT320er OR L1_HTT360er OR L1_ETT2000 OR L1_HTT400er OR L1_HTT450er OR L1_SingleJet120 OR L1_SingleJet140er2p5 OR L1_SingleJet160er2p5 OR L1_SingleJet180 OR L1_SingleJet200 OR L1_DoubleJet40er2p5 OR L1_DoubleJet100er2p5 OR L1_DoubleJet120er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreHT300Beamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltAK4CaloJets = cms.EDProducer( "FastjetJetProducer",
    useMassDropTagger = cms.bool( False ),
    useFiltering = cms.bool( False ),
    useDynamicFiltering = cms.bool( False ),
    useTrimming = cms.bool( False ),
    usePruning = cms.bool( False ),
    useCMSBoostedTauSeedingAlgorithm = cms.bool( False ),
    useKtPruning = cms.bool( False ),
    useConstituentSubtraction = cms.bool( False ),
    useSoftDrop = cms.bool( False ),
    correctShape = cms.bool( False ),
    UseOnlyVertexTracks = cms.bool( False ),
    UseOnlyOnePV = cms.bool( False ),
    muCut = cms.double( -1.0 ),
    yCut = cms.double( -1.0 ),
    rFilt = cms.double( -1.0 ),
    rFiltFactor = cms.double( -1.0 ),
    trimPtFracMin = cms.double( -1.0 ),
    zcut = cms.double( -1.0 ),
    rcut_factor = cms.double( -1.0 ),
    csRho_EtaMax = cms.double( -1.0 ),
    csRParam = cms.double( -1.0 ),
    beta = cms.double( -1.0 ),
    R0 = cms.double( -1.0 ),
    gridMaxRapidity = cms.double( -1.0 ),
    gridSpacing = cms.double( -1.0 ),
    DzTrVtxMax = cms.double( 0.0 ),
    DxyTrVtxMax = cms.double( 0.0 ),
    MaxVtxZ = cms.double( 15.0 ),
    subjetPtMin = cms.double( -1.0 ),
    muMin = cms.double( -1.0 ),
    muMax = cms.double( -1.0 ),
    yMin = cms.double( -1.0 ),
    yMax = cms.double( -1.0 ),
    dRMin = cms.double( -1.0 ),
    dRMax = cms.double( -1.0 ),
    maxDepth = cms.int32( -1 ),
    nFilt = cms.int32( -1 ),
    MinVtxNdof = cms.int32( 5 ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    srcPVs = cms.InputTag( "NotUsed" ),
    jetType = cms.string( "CaloJet" ),
    jetAlgorithm = cms.string( "AntiKt" ),
    rParam = cms.double( 0.4 ),
    inputEtMin = cms.double( 0.3 ),
    inputEMin = cms.double( 0.0 ),
    jetPtMin = cms.double( 1.0 ),
    doPVCorrection = cms.bool( False ),
    doAreaFastjet = cms.bool( False ),
    doRhoFastjet = cms.bool( False ),
    doPUOffsetCorr = cms.bool( False ),
    puPtMin = cms.double( 10.0 ),
    nSigmaPU = cms.double( 1.0 ),
    radiusPU = cms.double( 0.4 ),
    subtractorName = cms.string( "" ),
    useExplicitGhosts = cms.bool( False ),
    doAreaDiskApprox = cms.bool( True ),
    voronoiRfact = cms.double( 0.9 ),
    Rho_EtaMax = cms.double( 4.4 ),
    Ghost_EtaMax = cms.double( 6.0 ),
    Active_Area_Repeats = cms.int32( 5 ),
    GhostArea = cms.double( 0.01 ),
    restrictInputs = cms.bool( False ),
    maxInputs = cms.uint32( 1 ),
    writeCompound = cms.bool( False ),
    writeJetsWithConst = cms.bool( False ),
    doFastJetNonUniform = cms.bool( False ),
    useDeterministicSeed = cms.bool( True ),
    minSeed = cms.uint32( 14327 ),
    verbosity = cms.int32( 0 ),
    puWidth = cms.double( 0.0 ),
    nExclude = cms.uint32( 0 ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    puCenters = cms.vdouble(  ),
    applyWeight = cms.bool( False ),
    srcWeights = cms.InputTag( "" ),
    minimumTowersFraction = cms.double( 0.0 ),
    jetCollInstanceName = cms.string( "" ),
    sumRecHits = cms.bool( False )
)
fragment.hltAK4CaloJetsIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
    min_N90 = cms.int32( -2 ),
    min_N90hits = cms.int32( 2 ),
    min_EMF = cms.double( 1.0E-6 ),
    max_EMF = cms.double( 999.0 ),
    jetsInput = cms.InputTag( "hltAK4CaloJets" ),
    JetIDParams = cms.PSet( 
      hfRecHitsColl = cms.InputTag( "hltHfreco" ),
      hoRecHitsColl = cms.InputTag( "hltHoreco" ),
      ebRecHitsColl = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
      hbheRecHitsColl = cms.InputTag( "hltHbhereco" ),
      useRecHits = cms.bool( True ),
      eeRecHitsColl = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' )
    )
)
fragment.hltFixedGridRhoFastjetAllCalo = cms.EDProducer( "FixedGridRhoProducerFastjet",
    maxRapidity = cms.double( 5.0 ),
    gridSpacing = cms.double( 0.55 ),
    pfCandidatesTag = cms.InputTag( "hltTowerMakerForAll" )
)
fragment.hltAK4CaloFastJetCorrector = cms.EDProducer( "L1FastjetCorrectorProducer",
    level = cms.string( "L1FastJet" ),
    algorithm = cms.string( "AK4CaloHLT" ),
    srcRho = cms.InputTag( "hltFixedGridRhoFastjetAllCalo" )
)
fragment.hltAK4CaloRelativeCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L2Relative" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
fragment.hltAK4CaloAbsoluteCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L3Absolute" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
fragment.hltAK4CaloResidualCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L2L3Residual" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
fragment.hltAK4CaloCorrector = cms.EDProducer( "ChainedJetCorrectorProducer",
    correctors = cms.VInputTag( 'hltAK4CaloFastJetCorrector','hltAK4CaloRelativeCorrector','hltAK4CaloAbsoluteCorrector','hltAK4CaloResidualCorrector' )
)
fragment.hltAK4CaloJetsCorrected = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltAK4CaloJets" ),
    correctors = cms.VInputTag( 'hltAK4CaloCorrector' ),
    verbose = cms.untracked.bool( False )
)
fragment.hltAK4CaloJetsCorrectedIDPassed = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltAK4CaloJetsIDPassed" ),
    correctors = cms.VInputTag( 'hltAK4CaloCorrector' ),
    verbose = cms.untracked.bool( False )
)
fragment.hltHtMht = cms.EDProducer( "HLTHtMhtProducer",
    usePt = cms.bool( False ),
    excludePFMuons = cms.bool( False ),
    minNJetHt = cms.int32( 0 ),
    minNJetMht = cms.int32( 0 ),
    minPtJetHt = cms.double( 40.0 ),
    minPtJetMht = cms.double( 30.0 ),
    maxEtaJetHt = cms.double( 2.5 ),
    maxEtaJetMht = cms.double( 5.0 ),
    jetsLabel = cms.InputTag( "hltAK4CaloJetsCorrected" ),
    pfCandidatesLabel = cms.InputTag( "" )
)
fragment.hltHT300 = cms.EDFilter( "HLTHtMhtFilter",
    saveTags = cms.bool( True ),
    htLabels = cms.VInputTag( 'hltHtMht' ),
    mhtLabels = cms.VInputTag( 'hltHtMht' ),
    minHt = cms.vdouble( 300.0 ),
    minMht = cms.vdouble( 0.0 ),
    minMeff = cms.vdouble( 0.0 ),
    meffSlope = cms.vdouble( 1.0 )
)
fragment.hltL1sV0SingleJet3OR = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet60 OR L1_SingleJet200 OR L1_DoubleJet120er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreIsoTrackHB = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPixelTracksQuadruplets = cms.EDProducer( "TrackWithVertexSelector",
    src = cms.InputTag( "hltPixelTracks" ),
    copyExtras = cms.untracked.bool( False ),
    copyTrajectories = cms.untracked.bool( False ),
    numberOfValidHits = cms.uint32( 0 ),
    numberOfValidPixelHits = cms.uint32( 4 ),
    numberOfLostHits = cms.uint32( 999 ),
    normalizedChi2 = cms.double( 999999.0 ),
    ptMin = cms.double( 0.0 ),
    ptMax = cms.double( 999999.0 ),
    etaMin = cms.double( -999.0 ),
    etaMax = cms.double( 999.0 ),
    dzMax = cms.double( 999.0 ),
    d0Max = cms.double( 999.0 ),
    ptErrorCut = cms.double( 999999.0 ),
    quality = cms.string( "loose" ),
    useVtx = cms.bool( False ),
    nVertices = cms.uint32( 0 ),
    vertexTag = cms.InputTag( "hltTrimmedPixelVertices" ),
    timesTag = cms.InputTag( "" ),
    timeResosTag = cms.InputTag( "" ),
    vtxFallback = cms.bool( False ),
    zetaVtx = cms.double( 999999.0 ),
    rhoVtx = cms.double( 999999.0 ),
    nSigmaDtVertex = cms.double( 0.0 )
)
fragment.hltIsolPixelTrackProdHB = cms.EDProducer( "IsolatedPixelTrackCandidateL1TProducer",
    L1eTauJetsSource = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    tauAssociationCone = cms.double( 0.0 ),
    tauUnbiasCone = cms.double( 1.2 ),
    PixelTracksSources = cms.VInputTag( 'hltPixelTracksQuadruplets' ),
    ExtrapolationConeSize = cms.double( 1.0 ),
    PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
    L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet3OR" ),
    MaxVtxDXYSeed = cms.double( 101.0 ),
    MaxVtxDXYIsol = cms.double( 101.0 ),
    VertexLabel = cms.InputTag( "hltTrimmedPixelVertices" ),
    MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
    minPTrack = cms.double( 5.0 ),
    maxPTrackForIsolation = cms.double( 3.0 ),
    EBEtaBoundary = cms.double( 1.479 )
)
fragment.hltIsolPixelTrackL2FilterHB = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolPixelTrackProdHB" ),
    MaxPtNearby = cms.double( 2.0 ),
    MinEnergyTrack = cms.double( 12.0 ),
    MinPtTrack = cms.double( 3.5 ),
    MaxEtaTrack = cms.double( 1.15 ),
    MinEtaTrack = cms.double( 0.0 ),
    filterTrackEnergy = cms.bool( True ),
    NMaxTrackCandidates = cms.int32( 10 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltIsolEcalPixelTrackProdHB = cms.EDProducer( "IsolatedEcalPixelTrackCandidateProducer",
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
    EBRecHitSource = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    EERecHitSource = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    EBHitEnergyThreshold = cms.double( 0.1 ),
    EBHitCountEnergyThreshold = cms.double( 0.5 ),
    EEHitEnergyThreshold0 = cms.double( -41.0664 ),
    EEHitEnergyThreshold1 = cms.double( 68.795 ),
    EEHitEnergyThreshold2 = cms.double( -38.143 ),
    EEHitEnergyThreshold3 = cms.double( 7.043 ),
    EEFacHitCountEnergyThreshold = cms.double( 10.0 ),
    EcalConeSizeEta0 = cms.double( 0.09 ),
    EcalConeSizeEta1 = cms.double( 0.14 )
)
fragment.hltEcalIsolPixelTrackL2FilterHB = cms.EDFilter( "HLTEcalPixelIsolTrackFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolEcalPixelTrackProdHB" ),
    MaxEnergyInEB = cms.double( 2.0 ),
    MaxEnergyInEE = cms.double( 4.0 ),
    MaxEnergyOutEB = cms.double( 1.2 ),
    MaxEnergyOutEE = cms.double( 2.0 ),
    NMaxTrackCandidates = cms.int32( 10 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltHcalITIPTCorrectorHB = cms.EDProducer( "IPTCorrector",
    corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
    associationCone = cms.double( 0.2 )
)
fragment.hltIsolPixelTrackL3FilterHB = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltHcalITIPTCorrectorHB" ),
    MaxPtNearby = cms.double( 2.0 ),
    MinEnergyTrack = cms.double( 18.0 ),
    MinPtTrack = cms.double( 20.0 ),
    MaxEtaTrack = cms.double( 1.15 ),
    MinEtaTrack = cms.double( 0.0 ),
    filterTrackEnergy = cms.bool( True ),
    NMaxTrackCandidates = cms.int32( 999 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltPreIsoTrackHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltIsolPixelTrackProdHE = cms.EDProducer( "IsolatedPixelTrackCandidateL1TProducer",
    L1eTauJetsSource = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    tauAssociationCone = cms.double( 0.0 ),
    tauUnbiasCone = cms.double( 1.2 ),
    PixelTracksSources = cms.VInputTag( 'hltPixelTracksQuadruplets' ),
    ExtrapolationConeSize = cms.double( 1.0 ),
    PixelIsolationConeSizeAtEC = cms.double( 40.0 ),
    L1GTSeedLabel = cms.InputTag( "hltL1sV0SingleJet3OR" ),
    MaxVtxDXYSeed = cms.double( 101.0 ),
    MaxVtxDXYIsol = cms.double( 101.0 ),
    VertexLabel = cms.InputTag( "hltTrimmedPixelVertices" ),
    MagFieldRecordName = cms.string( "VolumeBasedMagneticField" ),
    minPTrack = cms.double( 5.0 ),
    maxPTrackForIsolation = cms.double( 3.0 ),
    EBEtaBoundary = cms.double( 1.479 )
)
fragment.hltIsolPixelTrackL2FilterHE = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolPixelTrackProdHE" ),
    MaxPtNearby = cms.double( 2.0 ),
    MinEnergyTrack = cms.double( 12.0 ),
    MinPtTrack = cms.double( 3.5 ),
    MaxEtaTrack = cms.double( 2.2 ),
    MinEtaTrack = cms.double( 1.1 ),
    filterTrackEnergy = cms.bool( True ),
    NMaxTrackCandidates = cms.int32( 5 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltIsolEcalPixelTrackProdHE = cms.EDProducer( "IsolatedEcalPixelTrackCandidateProducer",
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
    EBRecHitSource = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    EERecHitSource = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    EBHitEnergyThreshold = cms.double( 0.1 ),
    EBHitCountEnergyThreshold = cms.double( 0.5 ),
    EEHitEnergyThreshold0 = cms.double( -41.0664 ),
    EEHitEnergyThreshold1 = cms.double( 68.795 ),
    EEHitEnergyThreshold2 = cms.double( -38.143 ),
    EEHitEnergyThreshold3 = cms.double( 7.043 ),
    EEFacHitCountEnergyThreshold = cms.double( 10.0 ),
    EcalConeSizeEta0 = cms.double( 0.09 ),
    EcalConeSizeEta1 = cms.double( 0.14 )
)
fragment.hltEcalIsolPixelTrackL2FilterHE = cms.EDFilter( "HLTEcalPixelIsolTrackFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolEcalPixelTrackProdHE" ),
    MaxEnergyInEB = cms.double( 2.0 ),
    MaxEnergyInEE = cms.double( 4.0 ),
    MaxEnergyOutEB = cms.double( 1.2 ),
    MaxEnergyOutEE = cms.double( 2.0 ),
    NMaxTrackCandidates = cms.int32( 10 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltHcalITIPTCorrectorHE = cms.EDProducer( "IPTCorrector",
    corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
    associationCone = cms.double( 0.2 )
)
fragment.hltIsolPixelTrackL3FilterHE = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltHcalITIPTCorrectorHE" ),
    MaxPtNearby = cms.double( 2.0 ),
    MinEnergyTrack = cms.double( 18.0 ),
    MinPtTrack = cms.double( 20.0 ),
    MaxEtaTrack = cms.double( 2.2 ),
    MinEtaTrack = cms.double( 1.1 ),
    filterTrackEnergy = cms.bool( True ),
    NMaxTrackCandidates = cms.int32( 999 ),
    DropMultiL2Event = cms.bool( False )
)
fragment.hltL1sSingleMuCosmics = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuCosmics" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMuCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1MuCosmicsL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuCosmics" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen_er1p4_NotBptxOR_3BX OR L1_SingleMuOpen_er1p1_NotBptxOR_3BX" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL2Mu10NoVertexNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltDt4DSegmentsMeanTimer = cms.EDProducer( "DTRecSegment4DProducer",
    Reco4DAlgoName = cms.string( "DTMeantimerPatternReco4D" ),
    Reco4DAlgoConfig = cms.PSet( 
      Reco2DAlgoConfig = cms.PSet( 
        AlphaMaxPhi = cms.double( 1.0 ),
        debug = cms.untracked.bool( False ),
        segmCleanerMode = cms.int32( 2 ),
        AlphaMaxTheta = cms.double( 0.9 ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
        MaxChi2 = cms.double( 4.0 ),
        recAlgoConfig = cms.PSet( 
          maxTime = cms.double( 420.0 ),
          debug = cms.untracked.bool( False ),
          stepTwoFromDigi = cms.bool( False ),
          tTrigModeConfig = cms.PSet( 
            debug = cms.untracked.bool( False ),
            tofCorrType = cms.int32( 0 ),
            tTrigLabel = cms.string( "" ),
            wirePropCorrType = cms.int32( 0 ),
            doTOFCorrection = cms.bool( True ),
            vPropWire = cms.double( 24.4 ),
            doT0Correction = cms.bool( True ),
            doWirePropCorrection = cms.bool( True ),
            t0Label = cms.string( "" )
          ),
          useUncertDB = cms.bool( True ),
          doVdriftCorr = cms.bool( True ),
          minTime = cms.double( -3.0 ),
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          readLegacyTTrigDB = cms.bool( True ),
          readLegacyVDriftDB = cms.bool( True )
        ),
        MaxAllowedHits = cms.uint32( 50 ),
        nUnSharedHitsMin = cms.int32( 2 ),
        nSharedHitsMax = cms.int32( 2 ),
        performT0SegCorrection = cms.bool( False ),
        perform_delta_rejecting = cms.bool( False )
      ),
      Reco2DAlgoName = cms.string( "DTMeantimerPatternReco" ),
      debug = cms.untracked.bool( False ),
      segmCleanerMode = cms.int32( 2 ),
      AllDTRecHits = cms.bool( True ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      recAlgo = cms.string( "DTLinearDriftFromDBAlgo" ),
      recAlgoConfig = cms.PSet( 
        maxTime = cms.double( 420.0 ),
        debug = cms.untracked.bool( False ),
        stepTwoFromDigi = cms.bool( False ),
        tTrigModeConfig = cms.PSet( 
          debug = cms.untracked.bool( False ),
          tofCorrType = cms.int32( 0 ),
          tTrigLabel = cms.string( "" ),
          wirePropCorrType = cms.int32( 0 ),
          doTOFCorrection = cms.bool( True ),
          vPropWire = cms.double( 24.4 ),
          doT0Correction = cms.bool( True ),
          doWirePropCorrection = cms.bool( True ),
          t0Label = cms.string( "" )
        ),
        useUncertDB = cms.bool( True ),
        doVdriftCorr = cms.bool( True ),
        minTime = cms.double( -3.0 ),
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        readLegacyTTrigDB = cms.bool( True ),
        readLegacyVDriftDB = cms.bool( True )
      ),
      nUnSharedHitsMin = cms.int32( 2 ),
      nSharedHitsMax = cms.int32( 2 ),
      performT0SegCorrection = cms.bool( False ),
      perform_delta_rejecting = cms.bool( False )
    ),
    debug = cms.untracked.bool( False ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" )
)
fragment.hltL2CosmicOfflineMuonSeeds = cms.EDProducer( "CosmicMuonSeedGenerator",
    EnableDTMeasurement = cms.bool( True ),
    EnableCSCMeasurement = cms.bool( True ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegmentsMeanTimer" ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    MaxSeeds = cms.int32( 1000 ),
    MaxDTChi2 = cms.double( 300.0 ),
    MaxCSCChi2 = cms.double( 300.0 ),
    ForcePointDown = cms.bool( False )
)
fragment.hltL2CosmicMuonSeeds = cms.EDProducer( "L2MuonSeedGeneratorFromL1T",
    GMTReadoutCollection = cms.InputTag( "" ),
    InputObjects = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 1 ),
    SetMinPtBarrelTo = cms.double( 3.5 ),
    SetMinPtEndcapTo = cms.double( 1.0 ),
    UseOfflineSeed = cms.untracked.bool( True ),
    UseUnassociatedL1 = cms.bool( False ),
    MatchDR = cms.vdouble( 0.3 ),
    EtaMatchingBins = cms.vdouble( 0.0, 2.5 ),
    CentralBxOnly = cms.bool( True ),
    MatchType = cms.uint32( 0 ),
    SortType = cms.uint32( 0 ),
    OfflineSeedLabel = cms.untracked.InputTag( "hltL2CosmicOfflineMuonSeeds" ),
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny' )
    )
)
fragment.hltL2CosmicMuons = cms.EDProducer( "L2MuonProducer",
    ServiceParameters = cms.PSet( 
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True ),
      Propagators = cms.untracked.vstring( 'hltESPFastSteppingHelixPropagatorAny',
        'hltESPFastSteppingHelixPropagatorOpposite' )
    ),
    InputObjects = cms.InputTag( "hltL2CosmicMuonSeeds" ),
    SeedTransformerParameters = cms.PSet( 
      Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
      NMinRecHits = cms.uint32( 2 ),
      RescaleError = cms.double( 100.0 ),
      Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
      UseSubRecHits = cms.bool( False ),
      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      BWFilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegmentsMeanTimer" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        BWSeedType = cms.string( "fromGenerator" ),
        GEMRecSegmentLabel = cms.InputTag( "hltGemRecHits" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableGEMMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          ExcludeRPCFromFit = cms.bool( False ),
          Granularity = cms.int32( 0 ),
          MaxChi2 = cms.double( 25.0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 ),
          UseInvalidHits = cms.bool( True )
        ),
        EnableCSCMeasurement = cms.bool( True ),
        MaxChi2 = cms.double( 100.0 ),
        FitDirection = cms.string( "outsideIn" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        NumberOfSigma = cms.double( 3.0 ),
        EnableDTMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False ),
      FilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegmentsMeanTimer" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        GEMRecSegmentLabel = cms.InputTag( "hltGemRecHits" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableGEMMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          ExcludeRPCFromFit = cms.bool( False ),
          Granularity = cms.int32( 0 ),
          MaxChi2 = cms.double( 25.0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 ),
          UseInvalidHits = cms.bool( True )
        ),
        EnableCSCMeasurement = cms.bool( True ),
        MaxChi2 = cms.double( 1000.0 ),
        FitDirection = cms.string( "insideOut" ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        NumberOfSigma = cms.double( 3.0 ),
        EnableDTMeasurement = cms.bool( True )
      ),
      SeedPosition = cms.string( "in" ),
      DoBackwardFilter = cms.bool( True ),
      DoRefit = cms.bool( False ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "hltESPKFFittingSmootherForL2Muon" ),
        NMinRecHits = cms.uint32( 2 ),
        RescaleError = cms.double( 100.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
        UseSubRecHits = cms.bool( False ),
        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" )
      ),
      SeedPropagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" )
    ),
    DoSeedRefit = cms.bool( False ),
    TrackLoaderParameters = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
      DoSmoothing = cms.bool( False ),
      VertexConstraint = cms.bool( True ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" )
      ),
      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" )
    ),
    MuonTrajectoryBuilder = cms.string( "StandAloneMuonTrajectoryBuilder" )
)
fragment.hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL2CosmicMuons" )
)
fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltL1sSingleMuOpenNotBptxOR = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen_NotBptxOR" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL2Mu10NoVertexNoBPTX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sMuOpenNotBptxORL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpenNotBptxOR" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL2fL1sMuOpenNotBptxORL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMuOpenNotBptxORL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 0 ),
    MinNhits = cms.vint32( 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltPreL2Mu45NoVertex3StaNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered45Sta3 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 3 ),
    MinNhits = cms.vint32( 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 45.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltPreL2Mu40NoVertex3StaNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered40Sta3 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    AbsEtaBins = cms.vdouble( 5.0 ),
    MinNstations = cms.vint32( 3 ),
    MinNhits = cms.vint32( 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 40.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltL1sCDC = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_CDC_SingleMu_3_er1p2_TOP120_DPHI2p618_3p142" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreCDCL2cosmic10er1p0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sCDCL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sCDC" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( False ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations10er1p0 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sCDCL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.0 ),
    AbsEtaBins = cms.vdouble( 0.9, 1.5, 2.1, 5.0 ),
    MinNstations = cms.vint32( 0, 2, 0, 2 ),
    MinNhits = cms.vint32( 0, 1, 0, 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltPreCDCL2cosmic5p5er1p0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations5p5er1p0 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
    saveTags = cms.bool( True ),
    BeamSpotTag = cms.InputTag( "hltOnlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed" ),
    PreviousCandTag = cms.InputTag( "hltL1fL1sCDCL1Filtered0" ),
    SeedMapTag = cms.InputTag( "hltL2CosmicMuons" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 1.0 ),
    AbsEtaBins = cms.vdouble( 0.9, 1.5, 2.1, 5.0 ),
    MinNstations = cms.vint32( 0, 2, 0, 2 ),
    MinNhits = cms.vint32( 0, 1, 0, 1 ),
    CutOnChambers = cms.bool( False ),
    MinNchambers = cms.vint32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MinDr = cms.double( -1.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinDxySig = cms.double( -1.0 ),
    MinPt = cms.double( 5.5 ),
    NSigmaPt = cms.double( 0.0 ),
    MatchToPreviousCand = cms.bool( True )
)
fragment.hltPrePPSMaxTracksPerArm1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltCTPPSPixelDigis = cms.EDProducer( "CTPPSPixelRawToDigi",
    isRun3 = cms.bool( True ),
    includeErrors = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    mappingLabel = cms.string( "RPix" )
)
fragment.hltCTPPSPixelClusters = cms.EDProducer( "CTPPSPixelClusterProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    tag = cms.InputTag( "hltCTPPSPixelDigis" ),
    SeedADCThreshold = cms.int32( 2 ),
    ADCThreshold = cms.int32( 2 ),
    ElectronADCGain = cms.double( 135.0 ),
    VCaltoElectronGain = cms.int32( 50 ),
    VCaltoElectronOffset = cms.int32( -411 ),
    doSingleCalibration = cms.bool( False )
)
fragment.hltCTPPSPixelRecHits = cms.EDProducer( "CTPPSPixelRecHitProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    RPixClusterTag = cms.InputTag( "hltCTPPSPixelClusters" )
)
fragment.hltCTPPSPixelLocalTracks = cms.EDProducer( "CTPPSPixelLocalTrackProducer",
    tag = cms.InputTag( "hltCTPPSPixelRecHits" ),
    patternFinderAlgorithm = cms.string( "RPixRoadFinder" ),
    trackFinderAlgorithm = cms.string( "RPixPlaneCombinatoryTracking" ),
    trackMinNumberOfPoints = cms.uint32( 3 ),
    verbosity = cms.untracked.int32( 0 ),
    maximumChi2OverNDF = cms.double( 5.0 ),
    maximumXLocalDistanceFromTrack = cms.double( 0.2 ),
    maximumYLocalDistanceFromTrack = cms.double( 0.3 ),
    maxHitPerPlane = cms.int32( 20 ),
    maxHitPerRomanPot = cms.int32( 60 ),
    maxTrackPerRomanPot = cms.int32( 10 ),
    maxTrackPerPattern = cms.int32( 5 ),
    numberOfPlanesPerPot = cms.int32( 6 ),
    roadRadius = cms.double( 1.0 ),
    minRoadSize = cms.int32( 3 ),
    maxRoadSize = cms.int32( 20 ),
    roadRadiusBadPot = cms.double( 0.5 )
)
fragment.hltPPSExpCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 1 ),
    do_express = cms.bool( True ),
    triggerType = cms.int32( 91 )
)
fragment.hltPPSCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 579, 581, 582, 583, 586, 587, 588, 589, 1462, 1463 )
)
fragment.hltPrePPSMaxTracksPerRP4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPPSPrCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 4 ),
    do_express = cms.bool( False ),
    triggerType = cms.int32( 91 )
)
fragment.hltPrePPSRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreSpecialHLTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreAlCaLumiPixelsCountsRandomHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sZeroBiasOrZeroBiasCopy = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_ZeroBias_copy" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaLumiPixelsCountsZeroBiasVdM = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_ZeroBias_copy OR L1_AlwaysTrue OR L1_BptxOR" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreAlCaLumiPixelsCountsZeroBiasGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltAlcaPixelClusterCountsGated = cms.EDProducer( "AlcaPCCEventProducer",
    pixelClusterLabel = cms.InputTag( "hltSiPixelClusters" ),
    trigstring = cms.untracked.string( "alcaPCCEvent" ),
    savePerROCInfo = cms.bool( False )
)
fragment.hltL1sSingleMuOpen = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMuOpen = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1MuOpenL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpen" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltPreL1SingleMuOpenDT = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1MuOpenL1FilteredDT = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMuOpen" ),
    MaxEta = cms.double( 1.25 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sSingleMu3 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMu3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sMu3L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu3" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sSingleMu5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMu5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sMu5L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu5" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sSingleMu7 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMu7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1fL1sMu7L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sSingleMu7" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 1 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sDoubleMu0 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu0" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDoubleMu0L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
    saveTags = cms.bool( True ),
    CandTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    PreviousCandTag = cms.InputTag( "hltL1sDoubleMu0" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MaxDeltaR = cms.double( 0.3 ),
    MinN = cms.int32( 2 ),
    CentralBxOnly = cms.bool( True ),
    SelectQualities = cms.vint32(  )
)
fragment.hltL1sSingleJet8erHE = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet8erHE" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet8erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet10erHE = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet10erHE" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet10erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet12erHE = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet12erHE" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet12erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet35 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet35" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet35 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet200 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet200" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet200 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG8er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG8er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG8er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG10er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG10er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG10er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG15er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG15er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG15er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG26er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG26er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG26er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG28er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG28er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG28er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG28er2p1 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG28er2p1" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG28er2p1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG28er1p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG28er1p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG28er1p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG34er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG34er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG34er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG36er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG36er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG36er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG38er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG38er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG38er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG40er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG40er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG40er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG42er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG42er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG42er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleEG45er2p5 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG45er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG45er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sL1SingleEG50 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG50" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleEG50 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet60 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet60" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet60 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet90 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet90" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet90 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet120 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet120" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet120 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleJet180 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet180" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleJet180 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT120er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT120er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT120er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT160er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT160er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT160er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT200er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT200er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT200er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT255er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT255er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT255er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT280er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT280er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT280er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT320er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT320er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT320er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT360er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT360er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT360er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT400er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT400er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT400er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sHTT450er = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT450er" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1HTT450er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sETM120 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM120" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1ETM120 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sETM150 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM150" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1ETM150 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sEXTHCALLaserMon1 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HCAL_LaserMon_Trig" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1EXTHCALLaserMon1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sEXTHCALLaserMon4 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_HCAL_LaserMon_Veto" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1EXTHCALLaserMon4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sMuShowerOneNominal = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuShower_Nominal OR L1_SingleMuShower_Tight" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreCscClusterCosmic = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltCSCrechitClusters = cms.EDProducer( "CSCrechitClusterProducer",
    nRechitMin = cms.int32( 50 ),
    rParam = cms.double( 0.4 ),
    nStationThres = cms.int32( 10 ),
    recHitLabel = cms.InputTag( "hltCsc2DRecHits" )
)
fragment.hltCscClusterCosmic = cms.EDFilter( "HLTMuonRecHitClusterFilter",
    ClusterTag = cms.InputTag( "hltCSCrechitClusters" ),
    MinN = cms.int32( 1 ),
    MinSize = cms.int32( 50 ),
    MinSizeMinusMB1 = cms.int32( -1 ),
    MinSizeRegionCutEtas = cms.vdouble( -1.0, -1.0, 1.9, 1.9 ),
    MaxSizeRegionCutEtas = cms.vdouble( 1.9, 1.9, -1.0, -1.0 ),
    MinSizeRegionCutNstations = cms.vint32( -1, 1, -1, 1 ),
    MaxSizeRegionCutNstations = cms.vint32( 1, -1, 1, -1 ),
    MinSizeRegionCutClusterSize = cms.vint32( -1, -1, -1, -1 ),
    Max_nMB1 = cms.int32( -1 ),
    Max_nMB2 = cms.int32( -1 ),
    Max_nME11 = cms.int32( -1 ),
    Max_nME12 = cms.int32( -1 ),
    Max_nME41 = cms.int32( -1 ),
    Max_nME42 = cms.int32( -1 ),
    MinNstation = cms.int32( 0 ),
    MinAvgStation = cms.double( 0.0 ),
    MinTime = cms.double( -999.0 ),
    MaxTime = cms.double( 999.0 ),
    MinEta = cms.double( -1.0 ),
    MaxEta = cms.double( -1.0 ),
    MaxTimeSpread = cms.double( -1.0 )
)
fragment.hltL1sHTTForBeamSpotHT60 = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_HTT120er OR L1_HTT160er OR L1_HTT200er OR L1_HTT255er OR L1_HTT280er OR L1_HTT320er OR L1_HTT360er OR L1_ETT2000 OR L1_HTT400er OR L1_HTT450er OR L1_SingleJet120 OR L1_SingleJet140er2p5 OR L1_SingleJet160er2p5 OR L1_SingleJet180 OR L1_SingleJet200 OR L1_DoubleJet40er2p5 OR L1_DoubleJet100er2p5 OR L1_DoubleJet120er2p5" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreHT60Beamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltHT60 = cms.EDFilter( "HLTHtMhtFilter",
    saveTags = cms.bool( True ),
    htLabels = cms.VInputTag( 'hltHtMht' ),
    mhtLabels = cms.VInputTag( 'hltHtMht' ),
    minHt = cms.vdouble( 60.0 ),
    minMht = cms.vdouble( 0.0 ),
    minMeff = cms.vdouble( 0.0 ),
    meffSlope = cms.vdouble( 1.0 )
)
fragment.hltL1sZeroBiasOrMinBias = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_AlwaysTrue OR L1_MinimumBiasHF0_AND_BptxAND" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreHT300BeamspotPixelClustersWP2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPixelActivityFilterWP2 = cms.EDFilter( "HLTPixelActivityFilter",
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltSiPixelClusters" ),
    minClusters = cms.uint32( 0 ),
    maxClusters = cms.uint32( 0 ),
    minClustersBPix = cms.uint32( 10 ),
    maxClustersBPix = cms.uint32( 0 ),
    minClustersFPix = cms.uint32( 0 ),
    maxClustersFPix = cms.uint32( 0 ),
    minLayersBPix = cms.uint32( 0 ),
    maxLayersBPix = cms.uint32( 0 ),
    minLayersFPix = cms.uint32( 0 ),
    maxLayersFPix = cms.uint32( 0 )
)
fragment.hltPrePixelClustersWP2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPrePixelClustersWP1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPixelActivityFilterWP1 = cms.EDFilter( "HLTPixelActivityFilter",
    saveTags = cms.bool( True ),
    inputTag = cms.InputTag( "hltSiPixelClusters" ),
    minClusters = cms.uint32( 0 ),
    maxClusters = cms.uint32( 0 ),
    minClustersBPix = cms.uint32( 25 ),
    maxClustersBPix = cms.uint32( 0 ),
    minClustersFPix = cms.uint32( 0 ),
    maxClustersFPix = cms.uint32( 0 ),
    minLayersBPix = cms.uint32( 0 ),
    maxLayersBPix = cms.uint32( 0 ),
    minLayersFPix = cms.uint32( 0 ),
    maxLayersFPix = cms.uint32( 0 )
)
fragment.hltL1sBptxOR = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_BptxPlus OR L1_BptxMinus OR L1_ZeroBias OR L1_BptxOR" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreBptxOR = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL1sSingleMuCosmicsEMTF = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuCosmics_EMTF" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreL1SingleMuCosmicsEMTF = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreL1SingleMuCosmicsCosmicTracking = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltFullSiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
    ProductLabel = cms.InputTag( "rawDataCollector" ),
    ConditionsLabel = cms.string( "" ),
    onDemand = cms.bool( False ),
    DoAPVEmulatorCheck = cms.bool( False ),
    LegacyUnpacker = cms.bool( False ),
    HybridZeroSuppressed = cms.bool( False ),
    Clusterizer = cms.PSet( 
      ConditionsLabel = cms.string( "" ),
      ClusterThreshold = cms.double( 5.0 ),
      SeedThreshold = cms.double( 3.0 ),
      Algorithm = cms.string( "ThreeThresholdAlgorithm" ),
      ChannelThreshold = cms.double( 2.0 ),
      MaxAdjacentBad = cms.uint32( 0 ),
      setDetId = cms.bool( True ),
      MaxSequentialHoles = cms.uint32( 0 ),
      RemoveApvShots = cms.bool( True ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      MaxSequentialBad = cms.uint32( 1 )
    ),
    Algorithms = cms.PSet( 
      Use10bitsTruncation = cms.bool( False ),
      CommonModeNoiseSubtractionMode = cms.string( "Median" ),
      useCMMeanMap = cms.bool( False ),
      TruncateInSuppressor = cms.bool( True ),
      doAPVRestore = cms.bool( False ),
      SiStripFedZeroSuppressionMode = cms.uint32( 4 ),
      PedestalSubtractionFedMode = cms.bool( True )
    )
)
fragment.hltFullMeasurementTrackerEvent = cms.EDProducer( "MeasurementTrackerEventProducer",
    measurementTracker = cms.string( "hltESPMeasurementTracker" ),
    skipClusters = cms.InputTag( "" ),
    pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
    stripClusterProducer = cms.string( "hltFullSiStripRawToClustersFacility" ),
    Phase2TrackerCluster1DProducer = cms.string( "" ),
    vectorHits = cms.InputTag( "" ),
    vectorHitsRej = cms.InputTag( "" ),
    inactivePixelDetectorLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    badPixelFEDChannelCollectionLabels = cms.VInputTag( 'hltSiPixelDigiErrors' ),
    pixelCablingMapLabel = cms.string( "" ),
    inactiveStripDetectorLabels = cms.VInputTag( 'hltSiStripExcludedFEDListProducer' ),
    switchOffPixelsIfEmpty = cms.bool( True )
)
fragment.hltGlobalSiStripMatchedRecHitsFull = cms.EDProducer( "SiStripRecHitConverter",
    ClusterProducer = cms.InputTag( "hltFullSiStripRawToClustersFacility" ),
    rphiRecHits = cms.string( "rphiRecHit" ),
    stereoRecHits = cms.string( "stereoRecHit" ),
    matchedRecHits = cms.string( "matchedRecHit" ),
    useSiStripQuality = cms.bool( False ),
    MaskBadAPVFibers = cms.bool( False ),
    doMatching = cms.bool( True ),
    StripCPE = cms.ESInputTag( "hltESPStripCPEfromTrackAngle","hltESPStripCPEfromTrackAngle" ),
    Matcher = cms.ESInputTag( "SiStripRecHitMatcherESProducer","StandardMatcher" ),
    siStripQualityLabel = cms.ESInputTag( "","" )
)
fragment.hltSimpleCosmicBONSeedingLayers = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'MTOB4+MTOB5+MTOB6',
      'MTOB3+MTOB5+MTOB6',
      'MTOB3+MTOB4+MTOB5',
      'MTOB3+MTOB4+MTOB6',
      'TOB2+MTOB4+MTOB5',
      'TOB2+MTOB3+MTOB5',
      'TEC7_pos+TEC8_pos+TEC9_pos',
      'TEC6_pos+TEC7_pos+TEC8_pos',
      'TEC5_pos+TEC6_pos+TEC7_pos',
      'TEC4_pos+TEC5_pos+TEC6_pos',
      'TEC3_pos+TEC4_pos+TEC5_pos',
      'TEC2_pos+TEC3_pos+TEC4_pos',
      'TEC1_pos+TEC2_pos+TEC3_pos',
      'TEC7_neg+TEC8_neg+TEC9_neg',
      'TEC6_neg+TEC7_neg+TEC8_neg',
      'TEC5_neg+TEC6_neg+TEC7_neg',
      'TEC4_neg+TEC5_neg+TEC6_neg',
      'TEC3_neg+TEC4_neg+TEC5_neg',
      'TEC2_neg+TEC3_neg+TEC4_neg',
      'TEC1_neg+TEC2_neg+TEC3_neg',
      'TEC6_pos+TEC8_pos+TEC9_pos',
      'TEC5_pos+TEC7_pos+TEC8_pos',
      'TEC4_pos+TEC6_pos+TEC7_pos',
      'TEC3_pos+TEC5_pos+TEC6_pos',
      'TEC2_pos+TEC4_pos+TEC5_pos',
      'TEC1_pos+TEC3_pos+TEC4_pos',
      'TEC6_pos+TEC7_pos+TEC9_pos',
      'TEC5_pos+TEC6_pos+TEC8_pos',
      'TEC4_pos+TEC5_pos+TEC7_pos',
      'TEC3_pos+TEC4_pos+TEC6_pos',
      'TEC2_pos+TEC3_pos+TEC5_pos',
      'TEC1_pos+TEC2_pos+TEC4_pos',
      'TEC6_neg+TEC8_neg+TEC9_neg',
      'TEC5_neg+TEC7_neg+TEC8_neg',
      'TEC4_neg+TEC6_neg+TEC7_neg',
      'TEC3_neg+TEC5_neg+TEC6_neg',
      'TEC2_neg+TEC4_neg+TEC5_neg',
      'TEC1_neg+TEC3_neg+TEC4_neg',
      'TEC6_neg+TEC7_neg+TEC9_neg',
      'TEC5_neg+TEC6_neg+TEC8_neg',
      'TEC4_neg+TEC5_neg+TEC7_neg',
      'TEC3_neg+TEC4_neg+TEC6_neg',
      'TEC2_neg+TEC3_neg+TEC5_neg',
      'TEC1_neg+TEC2_neg+TEC4_neg',
      'MTOB6+TEC1_pos+TEC2_pos',
      'MTOB6+TEC1_neg+TEC2_neg',
      'MTOB6+MTOB5+TEC1_pos',
      'MTOB6+MTOB5+TEC1_neg' ),
    BPix = cms.PSet(  ),
    FPix = cms.PSet(  ),
    TIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TID = cms.PSet(  ),
    TOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TEC = cms.PSet( 
      useSimpleRphiHitsCleaner = cms.bool( False ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      minRing = cms.int32( 5 ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      useRingSlector = cms.bool( False ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      maxRing = cms.int32( 7 ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTEC = cms.PSet(  )
)
fragment.hltSimpleCosmicBONSeeds = cms.EDProducer( "SimpleCosmicBONSeeder",
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    ClusterCheckPSet = cms.PSet( 
      MaxNumberOfPixelClusters = cms.uint32( 1000 ),
      DontCountDetsAboveNClusters = cms.uint32( 20 ),
      ClusterCollectionLabel = cms.InputTag( "hltFullSiStripRawToClustersFacility" ),
      MaxNumberOfStripClusters = cms.uint32( 300 ),
      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
      doClusterCheck = cms.bool( True )
    ),
    maxTriplets = cms.int32( 50000 ),
    maxSeeds = cms.int32( 20000 ),
    RegionPSet = cms.PSet( 
      originZPosition = cms.double( 0.0 ),
      ptMin = cms.double( 0.5 ),
      originHalfLength = cms.double( 90.0 ),
      pMin = cms.double( 1.0 ),
      originRadius = cms.double( 150.0 )
    ),
    TripletsSrc = cms.InputTag( "hltSimpleCosmicBONSeedingLayers" ),
    TripletsDebugLevel = cms.untracked.uint32( 0 ),
    seedOnMiddle = cms.bool( False ),
    rescaleError = cms.double( 1.0 ),
    ClusterChargeCheck = cms.PSet( 
      Thresholds = cms.PSet( 
        TOB = cms.int32( 0 ),
        TIB = cms.int32( 0 ),
        TID = cms.int32( 0 ),
        TEC = cms.int32( 0 )
      ),
      matchedRecHitsUseAnd = cms.bool( True ),
      checkCharge = cms.bool( False )
    ),
    HitsPerModuleCheck = cms.PSet( 
      Thresholds = cms.PSet( 
        TOB = cms.int32( 20 ),
        TIB = cms.int32( 20 ),
        TID = cms.int32( 20 ),
        TEC = cms.int32( 20 )
      ),
      checkHitsPerModule = cms.bool( True )
    ),
    minimumGoodHitsInSeed = cms.int32( 3 ),
    writeTriplets = cms.bool( False ),
    helixDebugLevel = cms.untracked.uint32( 0 ),
    seedDebugLevel = cms.untracked.uint32( 0 ),
    PositiveYOnly = cms.bool( False ),
    NegativeYOnly = cms.bool( False )
)
fragment.hltCombinatorialcosmicseedingtripletsP5 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'MTOB4+MTOB5+MTOB6',
      'MTOB3+MTOB5+MTOB6',
      'MTOB3+MTOB4+MTOB5',
      'TOB2+MTOB4+MTOB5',
      'MTOB3+MTOB4+MTOB6',
      'TOB2+MTOB4+MTOB6' ),
    BPix = cms.PSet(  ),
    FPix = cms.PSet(  ),
    TIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TID = cms.PSet(  ),
    TOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TEC = cms.PSet( 
      useSimpleRphiHitsCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      minRing = cms.int32( 5 ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      useRingSlector = cms.bool( False ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      maxRing = cms.int32( 7 ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTEC = cms.PSet(  )
)
fragment.hltCombinatorialcosmicseedingpairsTOBP5 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'MTOB5+MTOB6',
      'MTOB4+MTOB5' ),
    BPix = cms.PSet(  ),
    FPix = cms.PSet(  ),
    TIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TID = cms.PSet(  ),
    TOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) )
    ),
    TEC = cms.PSet( 
      useSimpleRphiHitsCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      minRing = cms.int32( 5 ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      useRingSlector = cms.bool( False ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      maxRing = cms.int32( 7 ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTIB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet( 
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTEC = cms.PSet(  )
)
fragment.hltCombinatorialcosmicseedingpairsTECposP5 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'TEC1_pos+TEC2_pos',
      'TEC2_pos+TEC3_pos',
      'TEC3_pos+TEC4_pos',
      'TEC4_pos+TEC5_pos',
      'TEC5_pos+TEC6_pos',
      'TEC6_pos+TEC7_pos',
      'TEC7_pos+TEC8_pos',
      'TEC8_pos+TEC9_pos' ),
    BPix = cms.PSet(  ),
    FPix = cms.PSet(  ),
    TIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    TEC = cms.PSet( 
      useSimpleRphiHitsCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      minRing = cms.int32( 5 ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      useRingSlector = cms.bool( True ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      maxRing = cms.int32( 7 ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTIB = cms.PSet(  ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet(  ),
    MTEC = cms.PSet(  )
)
fragment.hltCombinatorialcosmicseedingpairsTECnegP5 = cms.EDProducer( "SeedingLayersEDProducer",
    layerList = cms.vstring( 'TEC1_neg+TEC2_neg',
      'TEC2_neg+TEC3_neg',
      'TEC3_neg+TEC4_neg',
      'TEC4_neg+TEC5_neg',
      'TEC5_neg+TEC6_neg',
      'TEC6_neg+TEC7_neg',
      'TEC7_neg+TEC8_neg',
      'TEC8_neg+TEC9_neg' ),
    BPix = cms.PSet(  ),
    FPix = cms.PSet(  ),
    TIB = cms.PSet(  ),
    TID = cms.PSet(  ),
    TOB = cms.PSet(  ),
    TEC = cms.PSet( 
      useSimpleRphiHitsCleaner = cms.bool( True ),
      TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
      minRing = cms.int32( 5 ),
      matchedRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','matchedRecHit' ),
      useRingSlector = cms.bool( True ),
      clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
      maxRing = cms.int32( 7 ),
      rphiRecHits = cms.InputTag( 'hltGlobalSiStripMatchedRecHitsFull','rphiRecHit' )
    ),
    MTIB = cms.PSet(  ),
    MTID = cms.PSet(  ),
    MTOB = cms.PSet(  ),
    MTEC = cms.PSet(  )
)
fragment.hltCombinatorialcosmicseedfinderP5 = cms.EDProducer( "CtfSpecialSeedGenerator",
    SeedMomentum = cms.double( 5.0 ),
    ErrorRescaling = cms.double( 50.0 ),
    UseScintillatorsConstraint = cms.bool( False ),
    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
    SeedsFromPositiveY = cms.bool( True ),
    SeedsFromNegativeY = cms.bool( False ),
    CheckHitsAreOnDifferentLayers = cms.bool( False ),
    SetMomentum = cms.bool( True ),
    requireBOFF = cms.bool( True ),
    maxSeeds = cms.int32( 10000 ),
    doClusterCheck = cms.bool( True ),
    MaxNumberOfStripClusters = cms.uint32( 300 ),
    ClusterCollectionLabel = cms.InputTag( "hltFullSiStripRawToClustersFacility" ),
    MaxNumberOfPixelClusters = cms.uint32( 1000 ),
    PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
    cut = cms.string( "strip < 400000 && pixel < 40000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + 0.1*strip)" ),
    DontCountDetsAboveNClusters = cms.uint32( 20 ),
    Charges = cms.vint32( -1 ),
    RegionFactoryPSet = cms.PSet( 
      RegionPSet = cms.PSet( 
        ptMin = cms.double( 0.9 ),
        originXPos = cms.double( 0.0 ),
        originYPos = cms.double( 0.0 ),
        originZPos = cms.double( 0.0 ),
        originHalfLength = cms.double( 21.2 ),
        originRadius = cms.double( 0.2 ),
        precise = cms.bool( True ),
        useMultipleScattering = cms.bool( False )
      ),
      ComponentName = cms.string( "GlobalRegionProducer" )
    ),
    UpperScintillatorParameters = cms.PSet( 
      WidthInX = cms.double( 100.0 ),
      GlobalX = cms.double( 0.0 ),
      GlobalY = cms.double( 300.0 ),
      GlobalZ = cms.double( 50.0 ),
      LenghtInZ = cms.double( 100.0 )
    ),
    LowerScintillatorParameters = cms.PSet( 
      WidthInX = cms.double( 100.0 ),
      GlobalX = cms.double( 0.0 ),
      GlobalY = cms.double( -100.0 ),
      GlobalZ = cms.double( 50.0 ),
      LenghtInZ = cms.double( 100.0 )
    ),
    OrderedHitsFactoryPSets = cms.VPSet( 
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingtripletsP5" ),
        NavigationDirection = cms.string( "outsideIn" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericTripletGenerator" )
      ),
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingpairsTOBP5" ),
        NavigationDirection = cms.string( "outsideIn" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericPairGenerator" )
      ),
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingpairsTECposP5" ),
        NavigationDirection = cms.string( "outsideIn" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericPairGenerator" )
      ),
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingpairsTECposP5" ),
        NavigationDirection = cms.string( "insideOut" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericPairGenerator" )
      ),
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingpairsTECnegP5" ),
        NavigationDirection = cms.string( "outsideIn" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericPairGenerator" )
      ),
      cms.PSet(  LayerSrc = cms.InputTag( "hltCombinatorialcosmicseedingpairsTECnegP5" ),
        NavigationDirection = cms.string( "insideOut" ),
        PropagationDirection = cms.string( "alongMomentum" ),
        ComponentName = cms.string( "GenericPairGenerator" )
      )
    )
)
fragment.hltCombinedP5SeedsForCTF = cms.EDProducer( "SeedCombiner",
    seedCollections = cms.VInputTag( 'hltCombinatorialcosmicseedfinderP5','hltSimpleCosmicBONSeeds' ),
    clusterRemovalInfos = cms.VInputTag(  )
)
fragment.hltCkfTrackCandidatesP5 = cms.EDProducer( "CkfTrackCandidateMaker",
    cleanTrajectoryAfterInOut = cms.bool( True ),
    doSeedingRegionRebuilding = cms.bool( True ),
    onlyPixelHitsForSeedCleaner = cms.bool( False ),
    reverseTrajectories = cms.bool( False ),
    useHitsSplitting = cms.bool( True ),
    MeasurementTrackerEvent = cms.InputTag( "hltFullMeasurementTrackerEvent" ),
    src = cms.InputTag( "hltCombinedP5SeedsForCTF" ),
    clustersToSkip = cms.InputTag( "" ),
    phase2clustersToSkip = cms.InputTag( "" ),
    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTGroupedCkfTrajectoryBuilderP5" ) ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      numberMeasurementsForFit = cms.int32( 4 ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    numHitsForSeedCleaner = cms.int32( 4 ),
    NavigationSchool = cms.string( "CosmicNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHitsP5" ),
    maxNSeeds = cms.uint32( 500000 ),
    maxSeedsBeforeCleaning = cms.uint32( 5000 )
)
fragment.hltCtfWithMaterialTracksCosmics = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    src = cms.InputTag( "hltCkfTrackCandidatesP5" ),
    clusterRemovalInfo = cms.InputTag( "" ),
    AlgorithmName = cms.string( "ctf" ),
    GeometricInnerState = cms.bool( True ),
    reMatchSplitHits = cms.bool( False ),
    usePropagatorForPCA = cms.bool( False ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    MeasurementTrackerEvent = cms.InputTag( "hltFullMeasurementTrackerEvent" ),
    useSimpleMF = cms.bool( False ),
    SimpleMagneticField = cms.string( "" ),
    Fitter = cms.string( "hltESFittingSmootherRKP5" ),
    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
    TTRHBuilder = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    MeasurementTracker = cms.string( "hltESPMeasurementTracker" )
)
fragment.hltCtfWithMaterialTracksP5 = cms.EDProducer( "CosmicTrackSelector",
    src = cms.InputTag( "hltCtfWithMaterialTracksCosmics" ),
    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False ),
    keepAllTracks = cms.bool( False ),
    chi2n_par = cms.double( 10.0 ),
    max_d0 = cms.double( 110.0 ),
    max_z0 = cms.double( 300.0 ),
    min_pt = cms.double( 1.0 ),
    max_eta = cms.double( 2.0 ),
    min_nHit = cms.uint32( 5 ),
    min_nPixelHit = cms.uint32( 0 ),
    minNumberLayers = cms.uint32( 0 ),
    minNumber3DLayers = cms.uint32( 0 ),
    maxNumberLostLayers = cms.uint32( 999 ),
    qualityBit = cms.string( "" )
)
fragment.hltCtfWithMaterialTracksP5TrackCountFilter = cms.EDFilter( "TrackCountFilter",
    src = cms.InputTag( "hltCtfWithMaterialTracksP5" ),
    minNumber = cms.uint32( 1 )
)
fragment.hltPreL1SingleMuCosmicsPointingCosmicTracking = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltL2CosmicsMuonTrackerPointingFilter = cms.EDFilter( "HLTMuonPointingFilter",
    SALabel = cms.InputTag( 'hltL2CosmicMuons','UpdatedAtVtx' ),
    PropagatorName = cms.string( "SteppingHelixPropagatorAny" ),
    radius = cms.double( 90.0 ),
    maxZ = cms.double( 280.0 ),
    PixHits = cms.uint32( 0 ),
    TkLayers = cms.uint32( 0 ),
    MuonHits = cms.uint32( 0 )
)
fragment.hltPreL1FatEvents = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreRandomHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBiasHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreZeroBiasGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltBXGateFilter = cms.EDFilter( "BunchCrossingFilter",
    bunches = cms.vuint32( 2 )
)
fragment.hltL1sZeroBiasCopyOrAlwaysTrue = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias_copy OR L1_AlwaysTrue" ),
    L1ObjectMapInputTag = cms.InputTag( "hltGtStage2ObjectMap" ),
    L1GlobalInputTag = cms.InputTag( "hltGtStage2Digis" ),
    L1MuonInputTag = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MuonShowerInputTag = cms.InputTag( 'hltGtStage2Digis','MuonShower' ),
    L1EGammaInputTag = cms.InputTag( 'hltGtStage2Digis','EGamma' ),
    L1JetInputTag = cms.InputTag( 'hltGtStage2Digis','Jet' ),
    L1TauInputTag = cms.InputTag( 'hltGtStage2Digis','Tau' ),
    L1EtSumInputTag = cms.InputTag( 'hltGtStage2Digis','EtSum' ),
    L1EtSumZdcInputTag = cms.InputTag( 'hltGtStage2Digis','EtSumZDC' )
)
fragment.hltPreSpecialZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
fragment.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
fragment.hltL1TGlobalSummary = cms.EDAnalyzer( "L1TGlobalSummary",
    AlgInputTag = cms.InputTag( "hltGtStage2Digis" ),
    ExtInputTag = cms.InputTag( "hltGtStage2Digis" ),
    MinBx = cms.int32( 0 ),
    MaxBx = cms.int32( 0 ),
    DumpTrigResults = cms.bool( False ),
    DumpRecord = cms.bool( False ),
    DumpTrigSummary = cms.bool( True ),
    ReadPrescalesFromFile = cms.bool( False ),
    psFileName = cms.string( "prescale_L1TGlobal.csv" ),
    psColumn = cms.int32( 0 )
)
fragment.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
    reportBy = cms.untracked.string( "job" ),
    resetBy = cms.untracked.string( "never" ),
    serviceBy = cms.untracked.string( "never" ),
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 )
)
fragment.hltDatasetAlCaLumiPixelsCountsExpress = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4 / 100',
      'AlCa_LumiPixelsCounts_Random_v10' )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsExpress = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaLumiPixelsCountsPrompt = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_Random_v10',
      'AlCa_LumiPixelsCounts_ZeroBias_v12' )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPrompt = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
      'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaLumiPixelsCountsGated = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_ZeroBiasGated_v5' )
)
fragment.hltPreDatasetAlCaLumiPixelsCountsGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaP0 = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_EcalEtaEBonly_v25',
      'AlCa_EcalEtaEEonly_v25',
      'AlCa_EcalPi0EBonly_v25',
      'AlCa_EcalPi0EEonly_v25' )
)
fragment.hltPreDatasetAlCaP0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaPPSExpress = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
      'HLT_PPSMaxTracksPerRP4_v9' )
)
fragment.hltPreDatasetAlCaPPSExpress = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaPPSPrompt = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
      'HLT_PPSMaxTracksPerRP4_v9' )
)
fragment.hltPreDatasetAlCaPPSPrompt = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetAlCaPhiSym = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_EcalPhiSym_v20' )
)
fragment.hltPreDatasetAlCaPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetCommissioning = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_IsoTrackHB_v14',
      'HLT_IsoTrackHE_v14',
      'HLT_L1SingleMuCosmics_EMTF_v4' )
)
fragment.hltPreDatasetCommissioning = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetCosmics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_L1SingleMu3_v5',
      'HLT_L1SingleMu5_v5',
      'HLT_L1SingleMu7_v5',
      'HLT_L1SingleMuCosmics_v8',
      'HLT_L1SingleMuOpen_DT_v6',
      'HLT_L1SingleMuOpen_v6' )
)
fragment.hltPreDatasetCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetDQMGPUvsCPU = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'DQM_EcalReconstruction_v12',
      'DQM_HcalReconstruction_v10',
      'DQM_PixelReconstruction_v12' )
)
fragment.hltPreDatasetDQMGPUvsCPU = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetDQMOnlineBeamspot = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_HT300_Beamspot_v23',
      'HLT_HT60_Beamspot_v22',
      'HLT_ZeroBias_Beamspot_v16' )
)
fragment.hltPreDatasetDQMOnlineBeamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetDQMPPSRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSRandom_v1' )
)
fragment.hltPreDatasetDQMPPSRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetEcalLaser = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4' )
)
fragment.hltPreDatasetEcalLaser = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetEventDisplay = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_BptxOR_v6',
      'HLT_L1ETM120_v4',
      'HLT_L1ETM150_v4',
      'HLT_L1HTT120er_v4',
      'HLT_L1HTT160er_v4',
      'HLT_L1HTT200er_v4',
      'HLT_L1HTT255er_v4',
      'HLT_L1HTT280er_v4',
      'HLT_L1HTT320er_v4',
      'HLT_L1HTT360er_v4',
      'HLT_L1HTT400er_v4',
      'HLT_L1HTT450er_v4',
      'HLT_L1SingleEG10er2p5_v4',
      'HLT_L1SingleEG15er2p5_v4',
      'HLT_L1SingleEG26er2p5_v4',
      'HLT_L1SingleEG28er1p5_v4',
      'HLT_L1SingleEG28er2p1_v4',
      'HLT_L1SingleEG28er2p5_v4',
      'HLT_L1SingleEG34er2p5_v4',
      'HLT_L1SingleEG36er2p5_v4',
      'HLT_L1SingleEG38er2p5_v4',
      'HLT_L1SingleEG40er2p5_v4',
      'HLT_L1SingleEG42er2p5_v4',
      'HLT_L1SingleEG45er2p5_v4',
      'HLT_L1SingleEG50_v4',
      'HLT_L1SingleEG8er2p5_v4',
      'HLT_L1SingleJet120_v4',
      'HLT_L1SingleJet180_v4',
      'HLT_L1SingleJet60_v4',
      'HLT_L1SingleJet90_v4',
      'HLT_L1SingleMu7_v5',
      'HLT_Physics_v14 / 10' )
)
fragment.hltPreDatasetEventDisplay = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetExpressAlignment = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_HT300_Beamspot_PixelClusters_WP2_v7',
      'HLT_HT300_Beamspot_v23',
      'HLT_HT60_Beamspot_v22',
      'HLT_PixelClusters_WP2_v4',
      'HLT_ZeroBias_Beamspot_v16' )
)
fragment.hltPreDatasetExpressAlignment = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetExpressCosmics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_L1SingleMuCosmics_v8',
      'HLT_L1SingleMuOpen_DT_v6',
      'HLT_L1SingleMuOpen_v6',
      'HLT_Random_v3' )
)
fragment.hltPreDatasetExpressCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetExpressPhysics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_BptxOR_v6',
      'HLT_L1SingleEG10er2p5_v4',
      'HLT_L1SingleEG15er2p5_v4',
      'HLT_L1SingleEG26er2p5_v4',
      'HLT_L1SingleEG28er1p5_v4',
      'HLT_L1SingleEG28er2p1_v4',
      'HLT_L1SingleEG28er2p5_v4',
      'HLT_L1SingleEG34er2p5_v4',
      'HLT_L1SingleEG36er2p5_v4',
      'HLT_L1SingleEG38er2p5_v4',
      'HLT_L1SingleEG40er2p5_v4',
      'HLT_L1SingleEG42er2p5_v4',
      'HLT_L1SingleEG45er2p5_v4',
      'HLT_L1SingleEG50_v4',
      'HLT_L1SingleEG8er2p5_v4',
      'HLT_L1SingleJet60_v4',
      'HLT_Physics_v14 / 2',
      'HLT_PixelClusters_WP1_v4',
      'HLT_PixelClusters_WP2_v4',
      'HLT_Random_v3',
      'HLT_ZeroBias_Alignment_v8',
      'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
      'HLT_ZeroBias_IsolatedBunches_v12',
      'HLT_ZeroBias_v13 / 2' )
)
fragment.hltPreDatasetExpressPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetHLTMonitor = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_L1SingleMuCosmics_CosmicTracking_v1',
      'HLT_L1SingleMuCosmics_PointingCosmicTracking_v1' )
)
fragment.hltPreDatasetHLTMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetHLTPhysics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Physics_v14' )
)
fragment.hltPreDatasetHLTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetHcalNZS = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_HcalNZS_v21',
      'HLT_HcalPhiSym_v23' )
)
fragment.hltPreDatasetHcalNZS = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetL1Accept = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'DST_Physics_v16',
      'DST_ZeroBias_v11' )
)
fragment.hltPreDatasetL1Accept = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetMinimumBias = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_BptxOR_v6',
      'HLT_L1ETM120_v4',
      'HLT_L1ETM150_v4',
      'HLT_L1EXT_HCAL_LaserMon1_v5',
      'HLT_L1EXT_HCAL_LaserMon4_v5',
      'HLT_L1HTT120er_v4',
      'HLT_L1HTT160er_v4',
      'HLT_L1HTT200er_v4',
      'HLT_L1HTT255er_v4',
      'HLT_L1HTT280er_v4',
      'HLT_L1HTT320er_v4',
      'HLT_L1HTT360er_v4',
      'HLT_L1HTT400er_v4',
      'HLT_L1HTT450er_v4',
      'HLT_L1SingleEG10er2p5_v4',
      'HLT_L1SingleEG15er2p5_v4',
      'HLT_L1SingleEG26er2p5_v4',
      'HLT_L1SingleEG28er1p5_v4',
      'HLT_L1SingleEG28er2p1_v4',
      'HLT_L1SingleEG28er2p5_v4',
      'HLT_L1SingleEG34er2p5_v4',
      'HLT_L1SingleEG36er2p5_v4',
      'HLT_L1SingleEG38er2p5_v4',
      'HLT_L1SingleEG40er2p5_v4',
      'HLT_L1SingleEG42er2p5_v4',
      'HLT_L1SingleEG45er2p5_v4',
      'HLT_L1SingleEG50_v4',
      'HLT_L1SingleEG8er2p5_v4',
      'HLT_L1SingleJet10erHE_v5',
      'HLT_L1SingleJet120_v4',
      'HLT_L1SingleJet12erHE_v5',
      'HLT_L1SingleJet180_v4',
      'HLT_L1SingleJet200_v5',
      'HLT_L1SingleJet35_v5',
      'HLT_L1SingleJet60_v4',
      'HLT_L1SingleJet8erHE_v5',
      'HLT_L1SingleJet90_v4' )
)
fragment.hltPreDatasetMinimumBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetMuonShower = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_CscCluster_Cosmic_v4' )
)
fragment.hltPreDatasetMuonShower = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetNoBPTX = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_CDC_L2cosmic_10_er1p0_v10',
      'HLT_CDC_L2cosmic_5p5_er1p0_v10',
      'HLT_L2Mu10_NoVertex_NoBPTX3BX_v14',
      'HLT_L2Mu10_NoVertex_NoBPTX_v15',
      'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14',
      'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13' )
)
fragment.hltPreDatasetNoBPTX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetOnlineMonitor = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'DQM_Random_v1',
      'DQM_ZeroBias_v3',
      'HLT_BptxOR_v6',
      'HLT_CDC_L2cosmic_10_er1p0_v10',
      'HLT_CDC_L2cosmic_5p5_er1p0_v10',
      'HLT_HcalNZS_v21',
      'HLT_HcalPhiSym_v23',
      'HLT_IsoTrackHB_v14',
      'HLT_IsoTrackHE_v14',
      'HLT_L1DoubleMu0_v5',
      'HLT_L1ETM120_v4',
      'HLT_L1ETM150_v4',
      'HLT_L1FatEvents_v5',
      'HLT_L1HTT120er_v4',
      'HLT_L1HTT160er_v4',
      'HLT_L1HTT200er_v4',
      'HLT_L1HTT255er_v4',
      'HLT_L1HTT280er_v4',
      'HLT_L1HTT320er_v4',
      'HLT_L1HTT360er_v4',
      'HLT_L1HTT400er_v4',
      'HLT_L1HTT450er_v4',
      'HLT_L1SingleEG10er2p5_v4',
      'HLT_L1SingleEG15er2p5_v4',
      'HLT_L1SingleEG26er2p5_v4',
      'HLT_L1SingleEG28er1p5_v4',
      'HLT_L1SingleEG28er2p1_v4',
      'HLT_L1SingleEG28er2p5_v4',
      'HLT_L1SingleEG34er2p5_v4',
      'HLT_L1SingleEG36er2p5_v4',
      'HLT_L1SingleEG38er2p5_v4',
      'HLT_L1SingleEG40er2p5_v4',
      'HLT_L1SingleEG42er2p5_v4',
      'HLT_L1SingleEG45er2p5_v4',
      'HLT_L1SingleEG50_v4',
      'HLT_L1SingleEG8er2p5_v4',
      'HLT_L1SingleJet120_v4',
      'HLT_L1SingleJet180_v4',
      'HLT_L1SingleJet200_v5',
      'HLT_L1SingleJet35_v5',
      'HLT_L1SingleJet60_v4',
      'HLT_L1SingleJet90_v4',
      'HLT_L1SingleMuCosmics_v8',
      'HLT_L1SingleMuOpen_v6',
      'HLT_L2Mu10_NoVertex_NoBPTX3BX_v14',
      'HLT_L2Mu10_NoVertex_NoBPTX_v15',
      'HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14',
      'HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13',
      'HLT_Physics_v14',
      'HLT_PixelClusters_WP1_v4',
      'HLT_PixelClusters_WP2_v4',
      'HLT_Random_v3',
      'HLT_ZeroBias_Alignment_v8',
      'HLT_ZeroBias_FirstBXAfterTrain_v10',
      'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
      'HLT_ZeroBias_FirstCollisionInTrain_v11',
      'HLT_ZeroBias_IsolatedBunches_v12',
      'HLT_ZeroBias_LastCollisionInTrain_v10',
      'HLT_ZeroBias_v13' )
)
fragment.hltPreDatasetOnlineMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetRPCMonitor = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_RPCMuonNormalisation_v23' )
)
fragment.hltPreDatasetRPCMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetTestEnablesEcalHcal = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4',
      'HLT_HcalCalibration_v6' )
)
fragment.hltPreDatasetTestEnablesEcalHcal = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetTestEnablesEcalHcalDQM = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4',
      'HLT_HcalCalibration_v6' )
)
fragment.hltPreDatasetTestEnablesEcalHcalDQM = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetVRRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Random_HighRate_v1' )
)
fragment.hltPreDatasetVRRandom0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetVRRandom15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetZeroBias = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Random_v3',
      'HLT_ZeroBias_Alignment_v8',
      'HLT_ZeroBias_FirstBXAfterTrain_v10',
      'HLT_ZeroBias_FirstCollisionAfterAbortGap_v12',
      'HLT_ZeroBias_FirstCollisionInTrain_v11',
      'HLT_ZeroBias_IsolatedBunches_v12',
      'HLT_ZeroBias_LastCollisionInTrain_v10',
      'HLT_ZeroBias_v13' )
)
fragment.hltPreDatasetZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetSpecialRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Random_HighRate_v1' )
)
fragment.hltPreDatasetSpecialRandom0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialRandom19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetSpecialZeroBias = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_SpecialZeroBias_v6',
      'HLT_ZeroBias_Gated_v4',
      'HLT_ZeroBias_HighRate_v4' )
)
fragment.hltPreDatasetSpecialZeroBias0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias20 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 20 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias21 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 21 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias22 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 22 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias23 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 23 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias24 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 24 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias25 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 25 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias26 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 26 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias27 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 27 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias28 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 28 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias29 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 29 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias30 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 30 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialZeroBias31 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 31 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltDatasetSpecialHLTPhysics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_SpecialHLTPhysics_v7' )
)
fragment.hltPreDatasetSpecialHLTPhysics0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
fragment.hltPreDatasetSpecialHLTPhysics19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)

fragment.HLTL1UnpackerSequence = cms.Sequence( fragment.hltGtStage2Digis + fragment.hltGtStage2ObjectMap )
fragment.HLTBeamSpot = cms.Sequence( fragment.hltOnlineMetaDataDigis + fragment.hltOnlineBeamSpot )
fragment.HLTBeginSequence = cms.Sequence( fragment.hltTriggerType + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence( fragment.hltEcalDigisLegacy + fragment.hltEcalDigisSoA + fragment.hltEcalDigis + fragment.hltEcalUncalibRecHitSoA + fragment.hltEcalUncalibRecHit + fragment.hltEcalDetIdToBeRecovered + fragment.hltEcalRecHit )
fragment.HLTPreshowerSequence = cms.Sequence( fragment.hltEcalPreshowerDigis + fragment.hltEcalPreshowerRecHit )
fragment.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence( fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + fragment.HLTPreshowerSequence )
fragment.HLTEndSequence = cms.Sequence( fragment.hltBoolEnd )
fragment.HLTFEDSelectorsForRPCMonitor = cms.Sequence( fragment.hltFEDSelectorTCDS + fragment.hltFEDSelectorL1 + fragment.hltFEDSelectorDT + fragment.hltFEDSelectorRPC + fragment.hltFEDSelectorCSC + fragment.hltFEDSelectorGEM + fragment.hltFEDSelectorTwinMux + fragment.hltFEDSelectorOMTF + fragment.hltFEDSelectorCPPF )
fragment.HLTBeginSequenceRandom = cms.Sequence( fragment.hltRandomEventsFilter + fragment.hltGtStage2Digis )
fragment.HLTDoLocalPixelSequence = cms.Sequence( fragment.hltOnlineBeamSpotDevice + fragment.hltSiPixelClustersSoA + fragment.hltSiPixelClusters + fragment.hltSiPixelDigiErrors + fragment.hltSiPixelRecHitsSoA + fragment.hltSiPixelRecHits )
fragment.HLTDoLocalPixelSequenceSerialSync = cms.Sequence( fragment.hltOnlineBeamSpotDeviceSerialSync + fragment.hltSiPixelClustersSoASerialSync + fragment.hltSiPixelClustersSerialSync + fragment.hltSiPixelDigiErrorsSerialSync + fragment.hltSiPixelRecHitsSoASerialSync + fragment.hltSiPixelRecHitsSerialSync )
fragment.HLTRecoPixelTracksSequence = cms.Sequence( fragment.hltPixelTracksSoA + fragment.hltPixelTracks )
fragment.HLTRecopixelvertexingSequence = cms.Sequence( fragment.HLTRecoPixelTracksSequence + fragment.hltPixelVerticesSoA + fragment.hltPixelVertices + fragment.hltTrimmedPixelVertices )
fragment.HLTRecoPixelTracksSequenceSerialSync = cms.Sequence( fragment.hltPixelTracksSoASerialSync + fragment.hltPixelTracksSerialSync )
fragment.HLTRecopixelvertexingSequenceSerialSync = cms.Sequence( fragment.HLTRecoPixelTracksSequenceSerialSync + fragment.hltPixelVerticesSoASerialSync + fragment.hltPixelVerticesSerialSync + fragment.hltTrimmedPixelVerticesSerialSync )
fragment.HLTDQMPixelReconstruction = cms.Sequence( fragment.hltSiPixelRecHitsSoAMonitorCPU + fragment.hltSiPixelRecHitsSoAMonitorGPU + fragment.hltSiPixelRecHitsSoACompareGPUvsCPU + fragment.hltPixelTracksSoAMonitorCPU + fragment.hltPixelTracksSoAMonitorGPU + fragment.hltPixelTracksSoACompareGPUvsCPU + fragment.hltPixelVerticesSoAMonitorCPU + fragment.hltPixelVerticesSoAMonitorGPU + fragment.hltPixelVerticesSoACompareGPUvsCPU )
fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequenceSerialSync = cms.Sequence( fragment.hltEcalDigisLegacy + fragment.hltEcalDigisSoASerialSync + fragment.hltEcalDigisSerialSync + fragment.hltEcalUncalibRecHitSoASerialSync + fragment.hltEcalUncalibRecHitSerialSync + fragment.hltEcalDetIdToBeRecovered + fragment.hltEcalRecHitSerialSync )
fragment.HLTDoLocalHcalSequence = cms.Sequence( fragment.hltHcalDigis + fragment.hltHcalDigisSoA + fragment.hltHbheRecoSoA + fragment.hltHbhereco + fragment.hltHfprereco + fragment.hltHfreco + fragment.hltHoreco )
fragment.HLTDoLocalHcalSequenceSerialSync = cms.Sequence( fragment.hltHcalDigis + fragment.hltHcalDigisSoASerialSync + fragment.hltHbheRecoSoASerialSync + fragment.hltHbherecoSerialSync + fragment.hltHfprereco + fragment.hltHfreco + fragment.hltHoreco )
fragment.HLTPFHcalClustering = cms.Sequence( fragment.hltParticleFlowRecHitHBHESoA + fragment.hltParticleFlowRecHitHBHE + fragment.hltParticleFlowClusterHBHESoA + fragment.hltParticleFlowClusterHBHE + fragment.hltParticleFlowClusterHCAL )
fragment.HLTPFHcalClusteringSerialSync = cms.Sequence( fragment.hltParticleFlowRecHitHBHESoASerialSync + fragment.hltParticleFlowRecHitHBHESerialSync + fragment.hltParticleFlowClusterHBHESoASerialSync + fragment.hltParticleFlowClusterHBHESerialSync + fragment.hltParticleFlowClusterHCALSerialSync )
fragment.HLTBeginSequenceCalibration = cms.Sequence( fragment.hltCalibrationEventsFilter + fragment.hltGtStage2Digis )
fragment.HLTBeginSequenceNZS = cms.Sequence( fragment.hltTriggerType + fragment.hltL1EventNumberNZS + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTBeginSequenceL1Fat = cms.Sequence( fragment.hltTriggerType + fragment.hltL1EventNumberL1Fat + fragment.HLTL1UnpackerSequence + fragment.HLTBeamSpot )
fragment.HLTDoCaloSequencePF = cms.Sequence( fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + fragment.HLTDoLocalHcalSequence + fragment.hltTowerMakerForAll )
fragment.HLTAK4CaloJetsPrePFRecoSequence = cms.Sequence( fragment.HLTDoCaloSequencePF + fragment.hltAK4CaloJetsPF )
fragment.HLTPreAK4PFJetsRecoSequence = cms.Sequence( fragment.HLTAK4CaloJetsPrePFRecoSequence + fragment.hltAK4CaloJetsPFEt5 )
fragment.HLTMuonLocalRecoSequence = cms.Sequence( fragment.hltMuonDTDigis + fragment.hltDt1DRecHits + fragment.hltDt4DSegments + fragment.hltMuonCSCDigis + fragment.hltCsc2DRecHits + fragment.hltCscSegments + fragment.hltMuonRPCDigisCPPF + fragment.hltOmtfDigis + fragment.hltMuonRPCDigisTwinMux + fragment.hltMuonRPCDigis + fragment.hltRpcRecHits + fragment.hltMuonGEMDigis + fragment.hltGemRecHits + fragment.hltGemSegments )
fragment.HLTL2muonrecoNocandSequence = cms.Sequence( fragment.HLTMuonLocalRecoSequence + fragment.hltL2OfflineMuonSeeds + fragment.hltL2MuonSeeds + fragment.hltL2Muons )
fragment.HLTL2muonrecoSequence = cms.Sequence( fragment.HLTL2muonrecoNocandSequence + fragment.hltL2MuonCandidates )
fragment.HLTDoLocalStripSequence = cms.Sequence( fragment.hltSiStripExcludedFEDListProducer + fragment.hltSiStripRawToClustersFacility + fragment.hltMeasurementTrackerEvent )
fragment.HLTIterL3OImuonTkCandidateSequence = cms.Sequence( fragment.hltIterL3OISeedsFromL2Muons + fragment.hltIterL3OITrackCandidates + fragment.hltIterL3OIMuCtfWithMaterialTracks + fragment.hltIterL3OIMuonTrackCutClassifier + fragment.hltIterL3OIMuonTrackSelectionHighPurity + fragment.hltL3MuonsIterL3OI )
fragment.HLTIterL3MuonRecopixelvertexingSequence = cms.Sequence( fragment.HLTRecopixelvertexingSequence + fragment.hltIterL3MuonPixelTracksTrackingRegions + fragment.hltPixelTracksInRegionL2 )
fragment.HLTIterativeTrackingIteration0ForIterL3Muon = cms.Sequence( fragment.hltIter0IterL3MuonPixelSeedsFromPixelTracks + fragment.hltIter0IterL3MuonPixelSeedsFromPixelTracksFiltered + fragment.hltIter0IterL3MuonCkfTrackCandidates + fragment.hltIter0IterL3MuonCtfWithMaterialTracks + fragment.hltIter0IterL3MuonTrackCutClassifier + fragment.hltIter0IterL3MuonTrackSelectionHighPurity )
fragment.HLTIterL3IOmuonTkCandidateSequence = cms.Sequence( fragment.HLTIterL3MuonRecopixelvertexingSequence + fragment.HLTIterativeTrackingIteration0ForIterL3Muon + fragment.hltL3MuonsIterL3IO )
fragment.HLTIterL3OIAndIOFromL2muonTkCandidateSequence = cms.Sequence( fragment.HLTIterL3OImuonTkCandidateSequence + fragment.hltIterL3OIL3MuonsLinksCombination + fragment.hltIterL3OIL3Muons + fragment.hltIterL3OIL3MuonCandidates + fragment.hltL2SelectorForL3IO + fragment.HLTIterL3IOmuonTkCandidateSequence + fragment.hltIterL3MuonsFromL2LinksCombination )
fragment.HLTRecopixelvertexingSequenceForIterL3FromL1Muon = cms.Sequence( fragment.HLTRecopixelvertexingSequence + fragment.hltIterL3FromL1MuonPixelTracksTrackingRegions + fragment.hltPixelTracksInRegionL1 )
fragment.HLTIterativeTrackingIteration0ForIterL3FromL1Muon = cms.Sequence( fragment.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks + fragment.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracksFiltered + fragment.hltIter0IterL3FromL1MuonCkfTrackCandidates + fragment.hltIter0IterL3FromL1MuonCtfWithMaterialTracks + fragment.hltIter0IterL3FromL1MuonTrackCutClassifier + fragment.hltIter0IterL3FromL1MuonTrackSelectionHighPurity )
fragment.HLTIterativeTrackingIteration3ForIterL3FromL1Muon = cms.Sequence( fragment.hltIter3IterL3FromL1MuonClustersRefRemoval + fragment.hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent + fragment.hltIter3IterL3FromL1MuonPixelLayersAndRegions + fragment.hltIter3IterL3FromL1MuonTrackingRegions + fragment.hltIter3IterL3FromL1MuonPixelClusterCheck + fragment.hltIter3IterL3FromL1MuonPixelHitDoublets + fragment.hltIter3IterL3FromL1MuonPixelSeeds + fragment.hltIter3IterL3FromL1MuonPixelSeedsFiltered + fragment.hltIter3IterL3FromL1MuonCkfTrackCandidates + fragment.hltIter3IterL3FromL1MuonCtfWithMaterialTracks + fragment.hltIter3IterL3FromL1MuonTrackCutClassifier + fragment.hltIter3IterL3FromL1MuonTrackSelectionHighPurity )
fragment.HLTIterL3IOmuonFromL1TkCandidateSequence = cms.Sequence( fragment.HLTRecopixelvertexingSequenceForIterL3FromL1Muon + fragment.HLTIterativeTrackingIteration0ForIterL3FromL1Muon + fragment.HLTIterativeTrackingIteration3ForIterL3FromL1Muon )
fragment.HLTIterL3muonTkCandidateSequence = cms.Sequence( fragment.HLTDoLocalPixelSequence + fragment.HLTDoLocalStripSequence + fragment.HLTIterL3OIAndIOFromL2muonTkCandidateSequence + fragment.hltL1MuonsPt0 + fragment.HLTIterL3IOmuonFromL1TkCandidateSequence )
fragment.HLTL3muonrecoNocandSequence = cms.Sequence( fragment.HLTIterL3muonTkCandidateSequence + fragment.hltIter03IterL3FromL1MuonMerged + fragment.hltIterL3MuonMerged + fragment.hltIterL3MuonAndMuonFromL1Merged + fragment.hltIterL3GlbMuon + fragment.hltIterL3MuonsNoID + fragment.hltIterL3Muons + fragment.hltL3MuonsIterL3Links + fragment.hltIterL3MuonTracks )
fragment.HLTL3muonrecoSequence = cms.Sequence( fragment.HLTL3muonrecoNocandSequence + fragment.hltIterL3MuonCandidates )
fragment.HLTIterativeTrackingIteration0 = cms.Sequence( fragment.hltIter0PFLowPixelSeedsFromPixelTracks + fragment.hltIter0PFlowCkfTrackCandidates + fragment.hltIter0PFlowCtfWithMaterialTracks + fragment.hltIter0PFlowTrackCutClassifier + fragment.hltIter0PFlowTrackSelectionHighPurity )
fragment.HLTIterativeTrackingDoubletRecovery = cms.Sequence( fragment.hltDoubletRecoveryClustersRefRemoval + fragment.hltDoubletRecoveryMaskedMeasurementTrackerEvent + fragment.hltDoubletRecoveryPixelLayersAndRegions + fragment.hltDoubletRecoveryPFlowPixelClusterCheck + fragment.hltDoubletRecoveryPFlowPixelHitDoublets + fragment.hltDoubletRecoveryPFlowPixelSeeds + fragment.hltDoubletRecoveryPFlowCkfTrackCandidates + fragment.hltDoubletRecoveryPFlowCtfWithMaterialTracks + fragment.hltDoubletRecoveryPFlowTrackCutClassifier + fragment.hltDoubletRecoveryPFlowTrackSelectionHighPurity )
fragment.HLTIterativeTrackingIter02 = cms.Sequence( fragment.HLTIterativeTrackingIteration0 + fragment.HLTIterativeTrackingDoubletRecovery + fragment.hltMergedTracks )
fragment.HLTTrackingForBeamSpot = cms.Sequence( fragment.HLTPreAK4PFJetsRecoSequence + fragment.HLTL2muonrecoSequence + fragment.HLTL3muonrecoSequence + fragment.HLTDoLocalPixelSequence + fragment.HLTRecopixelvertexingSequence + fragment.HLTDoLocalStripSequence + fragment.HLTIterativeTrackingIter02 + fragment.hltPFMuonMerging )
fragment.HLTDoCaloSequence = cms.Sequence( fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + fragment.HLTDoLocalHcalSequence + fragment.hltTowerMakerForAll )
fragment.HLTAK4CaloJetsReconstructionSequence = cms.Sequence( fragment.HLTDoCaloSequence + fragment.hltAK4CaloJets + fragment.hltAK4CaloJetsIDPassed )
fragment.HLTAK4CaloCorrectorProducersSequence = cms.Sequence( fragment.hltAK4CaloFastJetCorrector + fragment.hltAK4CaloRelativeCorrector + fragment.hltAK4CaloAbsoluteCorrector + fragment.hltAK4CaloResidualCorrector + fragment.hltAK4CaloCorrector )
fragment.HLTAK4CaloJetsCorrectionSequence = cms.Sequence( fragment.hltFixedGridRhoFastjetAllCalo + fragment.HLTAK4CaloCorrectorProducersSequence + fragment.hltAK4CaloJetsCorrected + fragment.hltAK4CaloJetsCorrectedIDPassed )
fragment.HLTAK4CaloJetsSequence = cms.Sequence( fragment.HLTAK4CaloJetsReconstructionSequence + fragment.HLTAK4CaloJetsCorrectionSequence )
fragment.HLTMuonLocalRecoMeanTimerSequence = cms.Sequence( fragment.hltMuonDTDigis + fragment.hltDt1DRecHits + fragment.hltDt4DSegmentsMeanTimer + fragment.hltMuonCSCDigis + fragment.hltCsc2DRecHits + fragment.hltCscSegments + fragment.hltMuonRPCDigisCPPF + fragment.hltOmtfDigis + fragment.hltMuonRPCDigisTwinMux + fragment.hltMuonRPCDigis + fragment.hltRpcRecHits + fragment.hltMuonGEMDigis + fragment.hltGemRecHits + fragment.hltGemSegments )
fragment.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence = cms.Sequence( fragment.HLTMuonLocalRecoMeanTimerSequence + fragment.hltL2CosmicOfflineMuonSeeds + fragment.hltL2CosmicMuonSeeds + fragment.hltL2CosmicMuons )
fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer = cms.Sequence( fragment.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence + fragment.hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed )
fragment.HLTPPSPixelRecoSequence = cms.Sequence( fragment.hltCTPPSPixelDigis + fragment.hltCTPPSPixelClusters + fragment.hltCTPPSPixelRecHits + fragment.hltCTPPSPixelLocalTracks )
fragment.HLTDoLocalStripFullSequence = cms.Sequence( fragment.hltSiStripExcludedFEDListProducer + fragment.hltFullSiStripRawToClustersFacility + fragment.hltFullMeasurementTrackerEvent + fragment.hltGlobalSiStripMatchedRecHitsFull )
fragment.HLTCTFCosmicsSequence = cms.Sequence( fragment.hltSimpleCosmicBONSeedingLayers + fragment.hltSimpleCosmicBONSeeds + fragment.hltCombinatorialcosmicseedingtripletsP5 + fragment.hltCombinatorialcosmicseedingpairsTOBP5 + fragment.hltCombinatorialcosmicseedingpairsTECposP5 + fragment.hltCombinatorialcosmicseedingpairsTECnegP5 + fragment.hltCombinatorialcosmicseedfinderP5 + fragment.hltCombinedP5SeedsForCTF + fragment.hltCkfTrackCandidatesP5 + fragment.hltCtfWithMaterialTracksCosmics + fragment.hltCtfWithMaterialTracksP5 )
fragment.HLTDatasetPathBeginSequence = cms.Sequence( fragment.hltGtStage2Digis )

fragment.HLTriggerFirstPath = cms.Path( fragment.hltGetRaw + fragment.hltPSetMap + fragment.hltBoolFalse )
fragment.Status_OnCPU = cms.Path( fragment.hltBackend + ~fragment.hltStatusOnGPUFilter )
fragment.Status_OnGPU = cms.Path( fragment.hltBackend + fragment.hltStatusOnGPUFilter )
fragment.AlCa_EcalPhiSym_v20 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasIorAlwaysTrueIorIsolatedBunch + fragment.hltPreAlCaEcalPhiSym + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltEcalPhiSymFilter + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.AlCa_EcalEtaEBonly_v25 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sAlCaEcalPi0Eta + fragment.hltPreAlCaEcalEtaEBonly + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltSimple3x3Clusters + fragment.hltAlCaEtaRecHitsFilterEBonlyRegional + fragment.hltAlCaEtaEBUncalibrator + fragment.hltAlCaEtaEBRechitsToDigis + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.AlCa_EcalEtaEEonly_v25 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sAlCaEcalPi0Eta + fragment.hltPreAlCaEcalEtaEEonly + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltSimple3x3Clusters + fragment.hltAlCaEtaRecHitsFilterEEonlyRegional + fragment.hltAlCaEtaEEUncalibrator + fragment.hltAlCaEtaEERechitsToDigis + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.AlCa_EcalPi0EBonly_v25 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sAlCaEcalPi0Eta + fragment.hltPreAlCaEcalPi0EBonly + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltSimple3x3Clusters + fragment.hltAlCaPi0RecHitsFilterEBonlyRegional + fragment.hltAlCaPi0EBUncalibrator + fragment.hltAlCaPi0EBRechitsToDigis + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.AlCa_EcalPi0EEonly_v25 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sAlCaEcalPi0Eta + fragment.hltPreAlCaEcalPi0EEonly + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltSimple3x3Clusters + fragment.hltAlCaPi0RecHitsFilterEEonlyRegional + fragment.hltAlCaPi0EEUncalibrator + fragment.hltAlCaPi0EERechitsToDigis + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.AlCa_RPCMuonNormalisation_v23 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5IorSingleMu14erIorSingleMu16er + fragment.hltPreAlCaRPCMuonNormalisation + fragment.hltRPCMuonNormaL1Filtered0 + fragment.HLTFEDSelectorsForRPCMonitor + fragment.HLTEndSequence )
fragment.AlCa_LumiPixelsCounts_Random_v10 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreAlCaLumiPixelsCountsRandom + fragment.HLTBeamSpot + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltAlcaPixelClusterCounts + fragment.HLTEndSequence )
fragment.AlCa_LumiPixelsCounts_ZeroBias_v12 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreAlCaLumiPixelsCountsZeroBias + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltAlcaPixelClusterCounts + fragment.HLTEndSequence )
fragment.DQM_PixelReconstruction_v12 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDQMPixelReconstruction + fragment.hltPreDQMPixelReconstruction + fragment.hltBackend + fragment.hltStatusOnGPUFilter + fragment.HLTDoLocalPixelSequence + fragment.HLTDoLocalPixelSequenceSerialSync + fragment.HLTRecopixelvertexingSequence + fragment.HLTRecopixelvertexingSequenceSerialSync + fragment.HLTDQMPixelReconstruction + fragment.HLTEndSequence )
fragment.DQM_EcalReconstruction_v12 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDQMEcalReconstruction + fragment.hltPreDQMEcalReconstruction + fragment.hltBackend + fragment.hltStatusOnGPUFilter + fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + fragment.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequenceSerialSync + fragment.HLTEndSequence )
fragment.DQM_HcalReconstruction_v10 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDQMHcalReconstruction + fragment.hltPreDQMHcalReconstruction + fragment.hltBackend + fragment.hltStatusOnGPUFilter + fragment.HLTDoLocalHcalSequence + fragment.HLTDoLocalHcalSequenceSerialSync + fragment.HLTPFHcalClustering + fragment.HLTPFHcalClusteringSerialSync + fragment.HLTEndSequence )
fragment.DQM_Random_v1 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreDQMRandom + fragment.HLTEndSequence )
fragment.DQM_ZeroBias_v3 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreDQMZeroBias + fragment.HLTEndSequence )
fragment.DST_ZeroBias_v11 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreDSTZeroBias + fragment.hltFEDSelectorL1 + fragment.hltFEDSelectorL1uGTTest + fragment.hltFEDSelectorTCDS + fragment.HLTEndSequence )
fragment.DST_Physics_v16 = cms.Path( fragment.HLTBeginSequence + fragment.hltPreDSTPhysics + fragment.hltFEDSelectorL1 + fragment.hltFEDSelectorL1uGTTest + fragment.hltFEDSelectorTCDS + fragment.HLTEndSequence )
fragment.HLT_EcalCalibration_v4 = cms.Path( fragment.HLTBeginSequenceCalibration + fragment.hltPreEcalCalibration + fragment.hltEcalCalibrationRaw + fragment.HLTEndSequence )
fragment.HLT_HcalCalibration_v6 = cms.Path( fragment.HLTBeginSequenceCalibration + fragment.hltPreHcalCalibration + fragment.hltHcalCalibrationRaw + fragment.HLTEndSequence )
fragment.HLT_HcalNZS_v21 = cms.Path( fragment.HLTBeginSequenceNZS + fragment.hltL1sHcalNZS + fragment.hltPreHcalNZS + fragment.HLTEndSequence )
fragment.HLT_HcalPhiSym_v23 = cms.Path( fragment.HLTBeginSequenceNZS + fragment.hltL1sSingleEGorSingleorDoubleMu + fragment.hltPreHcalPhiSym + fragment.HLTEndSequence )
fragment.HLT_Random_v3 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreRandom + fragment.HLTEndSequence )
fragment.HLT_Physics_v14 = cms.Path( fragment.HLTBeginSequenceL1Fat + fragment.hltPrePhysics + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_v13 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreZeroBias + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_Alignment_v8 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreZeroBiasAlignment + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_Beamspot_v16 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPreZeroBiasBeamspot + fragment.HLTTrackingForBeamSpot + fragment.hltVerticesPF + fragment.hltVerticesPFSelector + fragment.hltVerticesPFFilter + fragment.hltFEDSelectorOnlineMetaData + fragment.hltFEDSelectorTCDS + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_IsolatedBunches_v12 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sIsolatedBunch + fragment.hltPreZeroBiasIsolatedBunches + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_FirstBXAfterTrain_v10 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBiasFirstBunchAfterTrain + fragment.hltPreZeroBiasFirstBXAfterTrain + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_FirstCollisionAfterAbortGap_v12 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBiasFirstCollisionAfterAbortGap + fragment.hltPreZeroBiasFirstCollisionAfterAbortGap + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_FirstCollisionInTrain_v11 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBiasFirstCollisionInTrainNOTFirstCollisionInOrbit + fragment.hltPreZeroBiasFirstCollisionInTrain + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_LastCollisionInTrain_v10 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1ZeroBiasLastBunchInTrain + fragment.hltPreZeroBiasLastCollisionInTrain + fragment.HLTEndSequence )
fragment.HLT_HT300_Beamspot_v23 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTTForBeamSpot + fragment.hltPreHT300Beamspot + fragment.HLTAK4CaloJetsSequence + fragment.hltHtMht + fragment.hltHT300 + fragment.HLTTrackingForBeamSpot + fragment.hltVerticesPF + fragment.hltVerticesPFSelector + fragment.hltVerticesPFFilter + fragment.hltFEDSelectorOnlineMetaData + fragment.hltFEDSelectorTCDS + fragment.HLTEndSequence )
fragment.HLT_IsoTrackHB_v14 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0SingleJet3OR + fragment.hltPreIsoTrackHB + fragment.HLTDoLocalPixelSequence + fragment.HLTRecopixelvertexingSequence + fragment.hltPixelTracksQuadruplets + fragment.hltIsolPixelTrackProdHB + fragment.hltIsolPixelTrackL2FilterHB + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltIsolEcalPixelTrackProdHB + fragment.hltEcalIsolPixelTrackL2FilterHB + fragment.HLTDoLocalStripSequence + fragment.hltIter0PFLowPixelSeedsFromPixelTracks + fragment.hltIter0PFlowCkfTrackCandidates + fragment.hltIter0PFlowCtfWithMaterialTracks + fragment.hltHcalITIPTCorrectorHB + fragment.hltIsolPixelTrackL3FilterHB + fragment.HLTEndSequence )
fragment.HLT_IsoTrackHE_v14 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sV0SingleJet3OR + fragment.hltPreIsoTrackHE + fragment.HLTDoLocalPixelSequence + fragment.HLTRecopixelvertexingSequence + fragment.hltPixelTracksQuadruplets + fragment.hltIsolPixelTrackProdHE + fragment.hltIsolPixelTrackL2FilterHE + fragment.HLTDoFullUnpackingEgammaEcalSequence + fragment.hltIsolEcalPixelTrackProdHE + fragment.hltEcalIsolPixelTrackL2FilterHE + fragment.HLTDoLocalStripSequence + fragment.hltIter0PFLowPixelSeedsFromPixelTracks + fragment.hltIter0PFlowCkfTrackCandidates + fragment.hltIter0PFlowCtfWithMaterialTracks + fragment.hltHcalITIPTCorrectorHE + fragment.hltIsolPixelTrackL3FilterHE + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuCosmics_v8 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuCosmics + fragment.hltPreL1SingleMuCosmics + fragment.hltL1MuCosmicsL1Filtered0 + fragment.HLTEndSequence )
fragment.HLT_L2Mu10_NoVertex_NoBPTX3BX_v14 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + fragment.hltPreL2Mu10NoVertexNoBPTX3BX + fragment.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 + fragment.HLTEndSequence )
fragment.HLT_L2Mu10_NoVertex_NoBPTX_v15 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenNotBptxOR + fragment.hltPreL2Mu10NoVertexNoBPTX + fragment.hltL1fL1sMuOpenNotBptxORL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sMuOpenNotBptxORL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 + fragment.HLTEndSequence )
fragment.HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + fragment.hltPreL2Mu45NoVertex3StaNoBPTX3BX + fragment.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered45Sta3 + fragment.HLTEndSequence )
fragment.HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + fragment.hltPreL2Mu40NoVertex3StaNoBPTX3BX + fragment.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered40Sta3 + fragment.HLTEndSequence )
fragment.HLT_CDC_L2cosmic_10_er1p0_v10 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCDC + fragment.hltPreCDCL2cosmic10er1p0 + fragment.hltL1fL1sCDCL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations10er1p0 + fragment.HLTEndSequence )
fragment.HLT_CDC_L2cosmic_5p5_er1p0_v10 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sCDC + fragment.hltPreCDCL2cosmic5p5er1p0 + fragment.hltL1fL1sCDCL1Filtered0 + fragment.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + fragment.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations5p5er1p0 + fragment.HLTEndSequence )
fragment.HLT_PPSMaxTracksPerArm1_v9 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPrePPSMaxTracksPerArm1 + fragment.HLTPPSPixelRecoSequence + fragment.hltPPSExpCalFilter + fragment.hltPPSCalibrationRaw + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.HLT_PPSMaxTracksPerRP4_v9 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBias + fragment.hltPrePPSMaxTracksPerRP4 + fragment.HLTPPSPixelRecoSequence + fragment.hltPPSPrCalFilter + fragment.hltPPSCalibrationRaw + fragment.hltFEDSelectorL1 + fragment.HLTEndSequence )
fragment.HLT_PPSRandom_v1 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPrePPSRandom + fragment.hltPPSCalibrationRaw + fragment.HLTEndSequence )
fragment.HLT_SpecialHLTPhysics_v7 = cms.Path( fragment.HLTBeginSequence + fragment.hltPreSpecialHLTPhysics + fragment.HLTEndSequence )
fragment.AlCa_LumiPixelsCounts_RandomHighRate_v4 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreAlCaLumiPixelsCountsRandomHighRate + fragment.HLTBeamSpot + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltAlcaPixelClusterCounts + fragment.HLTEndSequence )
fragment.AlCa_LumiPixelsCounts_ZeroBiasVdM_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrZeroBiasCopy + fragment.hltPreAlCaLumiPixelsCountsZeroBiasVdM + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltAlcaPixelClusterCounts + fragment.HLTEndSequence )
fragment.AlCa_LumiPixelsCounts_ZeroBiasGated_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR + fragment.hltPreAlCaLumiPixelsCountsZeroBiasGated + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltAlcaPixelClusterCountsGated + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuOpen_v6 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpen + fragment.hltPreL1SingleMuOpen + fragment.hltL1MuOpenL1Filtered0 + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuOpen_DT_v6 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuOpen + fragment.hltPreL1SingleMuOpenDT + fragment.hltL1MuOpenL1FilteredDT + fragment.HLTEndSequence )
fragment.HLT_L1SingleMu3_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu3 + fragment.hltPreL1SingleMu3 + fragment.hltL1fL1sMu3L1Filtered0 + fragment.HLTEndSequence )
fragment.HLT_L1SingleMu5_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu5 + fragment.hltPreL1SingleMu5 + fragment.hltL1fL1sMu5L1Filtered0 + fragment.HLTEndSequence )
fragment.HLT_L1SingleMu7_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMu7 + fragment.hltPreL1SingleMu7 + fragment.hltL1fL1sMu7L1Filtered0 + fragment.HLTEndSequence )
fragment.HLT_L1DoubleMu0_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sDoubleMu0 + fragment.hltPreL1DoubleMu0 + fragment.hltDoubleMu0L1Filtered + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet8erHE_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet8erHE + fragment.hltPreL1SingleJet8erHE + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet10erHE_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet10erHE + fragment.hltPreL1SingleJet10erHE + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet12erHE_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet12erHE + fragment.hltPreL1SingleJet12erHE + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet35_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet35 + fragment.hltPreL1SingleJet35 + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet200_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet200 + fragment.hltPreL1SingleJet200 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG8er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG8er2p5 + fragment.hltPreL1SingleEG8er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG10er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG10er2p5 + fragment.hltPreL1SingleEG10er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG15er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG15er2p5 + fragment.hltPreL1SingleEG15er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG26er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG26er2p5 + fragment.hltPreL1SingleEG26er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG28er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG28er2p5 + fragment.hltPreL1SingleEG28er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG28er2p1_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG28er2p1 + fragment.hltPreL1SingleEG28er2p1 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG28er1p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG28er1p5 + fragment.hltPreL1SingleEG28er1p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG34er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG34er2p5 + fragment.hltPreL1SingleEG34er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG36er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG36er2p5 + fragment.hltPreL1SingleEG36er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG38er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG38er2p5 + fragment.hltPreL1SingleEG38er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG40er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG40er2p5 + fragment.hltPreL1SingleEG40er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG42er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG42er2p5 + fragment.hltPreL1SingleEG42er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG45er2p5_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleEG45er2p5 + fragment.hltPreL1SingleEG45er2p5 + fragment.HLTEndSequence )
fragment.HLT_L1SingleEG50_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sL1SingleEG50 + fragment.hltPreL1SingleEG50 + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet60_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet60 + fragment.hltPreL1SingleJet60 + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet90_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet90 + fragment.hltPreL1SingleJet90 + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet120_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet120 + fragment.hltPreL1SingleJet120 + fragment.HLTEndSequence )
fragment.HLT_L1SingleJet180_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleJet180 + fragment.hltPreL1SingleJet180 + fragment.HLTEndSequence )
fragment.HLT_L1HTT120er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT120er + fragment.hltPreL1HTT120er + fragment.HLTEndSequence )
fragment.HLT_L1HTT160er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT160er + fragment.hltPreL1HTT160er + fragment.HLTEndSequence )
fragment.HLT_L1HTT200er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT200er + fragment.hltPreL1HTT200er + fragment.HLTEndSequence )
fragment.HLT_L1HTT255er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT255er + fragment.hltPreL1HTT255er + fragment.HLTEndSequence )
fragment.HLT_L1HTT280er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT280er + fragment.hltPreL1HTT280er + fragment.HLTEndSequence )
fragment.HLT_L1HTT320er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT320er + fragment.hltPreL1HTT320er + fragment.HLTEndSequence )
fragment.HLT_L1HTT360er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT360er + fragment.hltPreL1HTT360er + fragment.HLTEndSequence )
fragment.HLT_L1HTT400er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT400er + fragment.hltPreL1HTT400er + fragment.HLTEndSequence )
fragment.HLT_L1HTT450er_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTT450er + fragment.hltPreL1HTT450er + fragment.HLTEndSequence )
fragment.HLT_L1ETM120_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sETM120 + fragment.hltPreL1ETM120 + fragment.HLTEndSequence )
fragment.HLT_L1ETM150_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sETM150 + fragment.hltPreL1ETM150 + fragment.HLTEndSequence )
fragment.HLT_L1EXT_HCAL_LaserMon1_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEXTHCALLaserMon1 + fragment.hltPreL1EXTHCALLaserMon1 + fragment.HLTEndSequence )
fragment.HLT_L1EXT_HCAL_LaserMon4_v5 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sEXTHCALLaserMon4 + fragment.hltPreL1EXTHCALLaserMon4 + fragment.HLTEndSequence )
fragment.HLT_CscCluster_Cosmic_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sMuShowerOneNominal + fragment.hltPreCscClusterCosmic + fragment.HLTMuonLocalRecoSequence + fragment.hltCSCrechitClusters + fragment.hltCscClusterCosmic + fragment.HLTEndSequence )
fragment.HLT_HT60_Beamspot_v22 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sHTTForBeamSpotHT60 + fragment.hltPreHT60Beamspot + fragment.HLTAK4CaloJetsSequence + fragment.hltHtMht + fragment.hltHT60 + fragment.HLTTrackingForBeamSpot + fragment.hltVerticesPF + fragment.hltVerticesPFSelector + fragment.hltVerticesPFFilter + fragment.hltFEDSelectorOnlineMetaData + fragment.hltFEDSelectorTCDS + fragment.HLTEndSequence )
fragment.HLT_HT300_Beamspot_PixelClusters_WP2_v7 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrMinBias + fragment.hltPreHT300BeamspotPixelClustersWP2 + fragment.hltPixelTrackerHVOn + fragment.HLTAK4CaloJetsSequence + fragment.hltHtMht + fragment.hltHT300 + fragment.HLTDoLocalPixelSequence + fragment.hltPixelActivityFilterWP2 + fragment.HLTTrackingForBeamSpot + fragment.hltVerticesPF + fragment.hltVerticesPFSelector + fragment.hltVerticesPFFilter + fragment.HLTEndSequence )
fragment.HLT_PixelClusters_WP2_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrMinBias + fragment.hltPrePixelClustersWP2 + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltPixelActivityFilterWP2 + fragment.HLTEndSequence )
fragment.HLT_PixelClusters_WP1_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrMinBias + fragment.hltPrePixelClustersWP1 + fragment.hltPixelTrackerHVOn + fragment.HLTDoLocalPixelSequence + fragment.hltPixelActivityFilterWP1 + fragment.HLTEndSequence )
fragment.HLT_BptxOR_v6 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sBptxOR + fragment.hltPreBptxOR + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuCosmics_EMTF_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuCosmicsEMTF + fragment.hltPreL1SingleMuCosmicsEMTF + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuCosmics_CosmicTracking_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuCosmics + fragment.hltPreL1SingleMuCosmicsCosmicTracking + fragment.hltL1MuCosmicsL1Filtered0 + fragment.HLTDoLocalPixelSequence + fragment.HLTDoLocalStripFullSequence + fragment.HLTCTFCosmicsSequence + fragment.hltCtfWithMaterialTracksP5TrackCountFilter + fragment.HLTEndSequence )
fragment.HLT_L1SingleMuCosmics_PointingCosmicTracking_v1 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sSingleMuCosmics + fragment.hltPreL1SingleMuCosmicsPointingCosmicTracking + fragment.hltL1MuCosmicsL1Filtered0 + fragment.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence + fragment.hltL2CosmicsMuonTrackerPointingFilter + fragment.HLTDoLocalPixelSequence + fragment.HLTDoLocalStripFullSequence + fragment.HLTCTFCosmicsSequence + fragment.hltCtfWithMaterialTracksP5TrackCountFilter + fragment.HLTEndSequence )
fragment.HLT_L1FatEvents_v5 = cms.Path( fragment.HLTBeginSequenceL1Fat + fragment.hltPreL1FatEvents + fragment.HLTEndSequence )
fragment.HLT_Random_HighRate_v1 = cms.Path( fragment.HLTBeginSequenceRandom + fragment.hltPreRandomHighRate + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_HighRate_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrZeroBiasCopy + fragment.hltPreZeroBiasHighRate + fragment.HLTEndSequence )
fragment.HLT_ZeroBias_Gated_v4 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR + fragment.hltPreZeroBiasGated + fragment.hltBXGateFilter + fragment.HLTEndSequence )
fragment.HLT_SpecialZeroBias_v6 = cms.Path( fragment.HLTBeginSequence + fragment.hltL1sZeroBiasCopyOrAlwaysTrue + fragment.hltPreSpecialZeroBias + fragment.HLTEndSequence )
fragment.HLTriggerFinalPath = cms.Path( fragment.hltGtStage2Digis + fragment.hltTriggerSummaryAOD + fragment.hltTriggerSummaryRAW + fragment.hltBoolFalse )
fragment.HLTAnalyzerEndpath = cms.EndPath( fragment.hltGtStage2Digis + fragment.hltL1TGlobalSummary + fragment.hltTrigReport )
fragment.Dataset_AlCaLumiPixelsCountsExpress = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsExpress + fragment.hltPreDatasetAlCaLumiPixelsCountsExpress )
fragment.Dataset_AlCaLumiPixelsCountsPrompt = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPrompt + fragment.hltPreDatasetAlCaLumiPixelsCountsPrompt )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate0 )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate1 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate1 )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate2 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate2 )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate3 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate3 )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate4 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate4 )
fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate5 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsPromptHighRate + fragment.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate5 )
fragment.Dataset_AlCaLumiPixelsCountsGated = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaLumiPixelsCountsGated + fragment.hltPreDatasetAlCaLumiPixelsCountsGated )
fragment.Dataset_AlCaP0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaP0 + fragment.hltPreDatasetAlCaP0 )
fragment.Dataset_AlCaPPSExpress = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaPPSExpress + fragment.hltPreDatasetAlCaPPSExpress )
fragment.Dataset_AlCaPPSPrompt = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaPPSPrompt + fragment.hltPreDatasetAlCaPPSPrompt )
fragment.Dataset_AlCaPhiSym = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetAlCaPhiSym + fragment.hltPreDatasetAlCaPhiSym )
fragment.Dataset_Commissioning = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetCommissioning + fragment.hltPreDatasetCommissioning )
fragment.Dataset_Cosmics = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetCosmics + fragment.hltPreDatasetCosmics )
fragment.Dataset_DQMGPUvsCPU = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetDQMGPUvsCPU + fragment.hltPreDatasetDQMGPUvsCPU )
fragment.Dataset_DQMOnlineBeamspot = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetDQMOnlineBeamspot + fragment.hltPreDatasetDQMOnlineBeamspot )
fragment.Dataset_DQMPPSRandom = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetDQMPPSRandom + fragment.hltPreDatasetDQMPPSRandom )
fragment.Dataset_EcalLaser = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetEcalLaser + fragment.hltPreDatasetEcalLaser )
fragment.Dataset_EventDisplay = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetEventDisplay + fragment.hltPreDatasetEventDisplay )
fragment.Dataset_ExpressAlignment = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetExpressAlignment + fragment.hltPreDatasetExpressAlignment )
fragment.Dataset_ExpressCosmics = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetExpressCosmics + fragment.hltPreDatasetExpressCosmics )
fragment.Dataset_ExpressPhysics = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetExpressPhysics + fragment.hltPreDatasetExpressPhysics )
fragment.Dataset_HLTMonitor = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetHLTMonitor + fragment.hltPreDatasetHLTMonitor )
fragment.Dataset_HLTPhysics = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetHLTPhysics + fragment.hltPreDatasetHLTPhysics )
fragment.Dataset_HcalNZS = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetHcalNZS + fragment.hltPreDatasetHcalNZS )
fragment.Dataset_L1Accept = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetL1Accept + fragment.hltPreDatasetL1Accept )
fragment.Dataset_MinimumBias = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetMinimumBias + fragment.hltPreDatasetMinimumBias )
fragment.Dataset_MuonShower = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetMuonShower + fragment.hltPreDatasetMuonShower )
fragment.Dataset_NoBPTX = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetNoBPTX + fragment.hltPreDatasetNoBPTX )
fragment.Dataset_OnlineMonitor = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetOnlineMonitor + fragment.hltPreDatasetOnlineMonitor )
fragment.Dataset_RPCMonitor = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetRPCMonitor + fragment.hltPreDatasetRPCMonitor )
fragment.Dataset_TestEnablesEcalHcal = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetTestEnablesEcalHcal + fragment.hltPreDatasetTestEnablesEcalHcal )
fragment.Dataset_TestEnablesEcalHcalDQM = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetTestEnablesEcalHcalDQM + fragment.hltPreDatasetTestEnablesEcalHcalDQM )
fragment.Dataset_VRRandom0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom0 )
fragment.Dataset_VRRandom1 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom1 )
fragment.Dataset_VRRandom2 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom2 )
fragment.Dataset_VRRandom3 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom3 )
fragment.Dataset_VRRandom4 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom4 )
fragment.Dataset_VRRandom5 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom5 )
fragment.Dataset_VRRandom6 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom6 )
fragment.Dataset_VRRandom7 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom7 )
fragment.Dataset_VRRandom8 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom8 )
fragment.Dataset_VRRandom9 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom9 )
fragment.Dataset_VRRandom10 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom10 )
fragment.Dataset_VRRandom11 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom11 )
fragment.Dataset_VRRandom12 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom12 )
fragment.Dataset_VRRandom13 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom13 )
fragment.Dataset_VRRandom14 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom14 )
fragment.Dataset_VRRandom15 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetVRRandom + fragment.hltPreDatasetVRRandom15 )
fragment.Dataset_ZeroBias = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetZeroBias + fragment.hltPreDatasetZeroBias )
fragment.Dataset_SpecialRandom0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom0 )
fragment.Dataset_SpecialRandom1 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom1 )
fragment.Dataset_SpecialRandom2 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom2 )
fragment.Dataset_SpecialRandom3 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom3 )
fragment.Dataset_SpecialRandom4 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom4 )
fragment.Dataset_SpecialRandom5 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom5 )
fragment.Dataset_SpecialRandom6 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom6 )
fragment.Dataset_SpecialRandom7 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom7 )
fragment.Dataset_SpecialRandom8 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom8 )
fragment.Dataset_SpecialRandom9 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom9 )
fragment.Dataset_SpecialRandom10 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom10 )
fragment.Dataset_SpecialRandom11 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom11 )
fragment.Dataset_SpecialRandom12 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom12 )
fragment.Dataset_SpecialRandom13 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom13 )
fragment.Dataset_SpecialRandom14 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom14 )
fragment.Dataset_SpecialRandom15 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom15 )
fragment.Dataset_SpecialRandom16 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom16 )
fragment.Dataset_SpecialRandom17 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom17 )
fragment.Dataset_SpecialRandom18 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom18 )
fragment.Dataset_SpecialRandom19 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialRandom + fragment.hltPreDatasetSpecialRandom19 )
fragment.Dataset_SpecialZeroBias0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias0 )
fragment.Dataset_SpecialZeroBias1 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias1 )
fragment.Dataset_SpecialZeroBias2 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias2 )
fragment.Dataset_SpecialZeroBias3 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias3 )
fragment.Dataset_SpecialZeroBias4 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias4 )
fragment.Dataset_SpecialZeroBias5 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias5 )
fragment.Dataset_SpecialZeroBias6 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias6 )
fragment.Dataset_SpecialZeroBias7 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias7 )
fragment.Dataset_SpecialZeroBias8 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias8 )
fragment.Dataset_SpecialZeroBias9 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias9 )
fragment.Dataset_SpecialZeroBias10 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias10 )
fragment.Dataset_SpecialZeroBias11 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias11 )
fragment.Dataset_SpecialZeroBias12 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias12 )
fragment.Dataset_SpecialZeroBias13 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias13 )
fragment.Dataset_SpecialZeroBias14 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias14 )
fragment.Dataset_SpecialZeroBias15 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias15 )
fragment.Dataset_SpecialZeroBias16 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias16 )
fragment.Dataset_SpecialZeroBias17 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias17 )
fragment.Dataset_SpecialZeroBias18 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias18 )
fragment.Dataset_SpecialZeroBias19 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias19 )
fragment.Dataset_SpecialZeroBias20 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias20 )
fragment.Dataset_SpecialZeroBias21 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias21 )
fragment.Dataset_SpecialZeroBias22 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias22 )
fragment.Dataset_SpecialZeroBias23 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias23 )
fragment.Dataset_SpecialZeroBias24 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias24 )
fragment.Dataset_SpecialZeroBias25 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias25 )
fragment.Dataset_SpecialZeroBias26 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias26 )
fragment.Dataset_SpecialZeroBias27 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias27 )
fragment.Dataset_SpecialZeroBias28 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias28 )
fragment.Dataset_SpecialZeroBias29 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias29 )
fragment.Dataset_SpecialZeroBias30 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias30 )
fragment.Dataset_SpecialZeroBias31 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialZeroBias + fragment.hltPreDatasetSpecialZeroBias31 )
fragment.Dataset_SpecialHLTPhysics0 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics0 )
fragment.Dataset_SpecialHLTPhysics1 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics1 )
fragment.Dataset_SpecialHLTPhysics2 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics2 )
fragment.Dataset_SpecialHLTPhysics3 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics3 )
fragment.Dataset_SpecialHLTPhysics4 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics4 )
fragment.Dataset_SpecialHLTPhysics5 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics5 )
fragment.Dataset_SpecialHLTPhysics6 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics6 )
fragment.Dataset_SpecialHLTPhysics7 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics7 )
fragment.Dataset_SpecialHLTPhysics8 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics8 )
fragment.Dataset_SpecialHLTPhysics9 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics9 )
fragment.Dataset_SpecialHLTPhysics10 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics10 )
fragment.Dataset_SpecialHLTPhysics11 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics11 )
fragment.Dataset_SpecialHLTPhysics12 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics12 )
fragment.Dataset_SpecialHLTPhysics13 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics13 )
fragment.Dataset_SpecialHLTPhysics14 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics14 )
fragment.Dataset_SpecialHLTPhysics15 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics15 )
fragment.Dataset_SpecialHLTPhysics16 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics16 )
fragment.Dataset_SpecialHLTPhysics17 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics17 )
fragment.Dataset_SpecialHLTPhysics18 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics18 )
fragment.Dataset_SpecialHLTPhysics19 = cms.Path( fragment.HLTDatasetPathBeginSequence + fragment.hltDatasetSpecialHLTPhysics + fragment.hltPreDatasetSpecialHLTPhysics19 )


fragment.schedule = cms.Schedule( *(fragment.HLTriggerFirstPath, fragment.Status_OnCPU, fragment.Status_OnGPU, fragment.AlCa_EcalPhiSym_v20, fragment.AlCa_EcalEtaEBonly_v25, fragment.AlCa_EcalEtaEEonly_v25, fragment.AlCa_EcalPi0EBonly_v25, fragment.AlCa_EcalPi0EEonly_v25, fragment.AlCa_RPCMuonNormalisation_v23, fragment.AlCa_LumiPixelsCounts_Random_v10, fragment.AlCa_LumiPixelsCounts_ZeroBias_v12, fragment.DQM_PixelReconstruction_v12, fragment.DQM_EcalReconstruction_v12, fragment.DQM_HcalReconstruction_v10, fragment.DQM_Random_v1, fragment.DQM_ZeroBias_v3, fragment.DST_ZeroBias_v11, fragment.DST_Physics_v16, fragment.HLT_EcalCalibration_v4, fragment.HLT_HcalCalibration_v6, fragment.HLT_HcalNZS_v21, fragment.HLT_HcalPhiSym_v23, fragment.HLT_Random_v3, fragment.HLT_Physics_v14, fragment.HLT_ZeroBias_v13, fragment.HLT_ZeroBias_Alignment_v8, fragment.HLT_ZeroBias_Beamspot_v16, fragment.HLT_ZeroBias_IsolatedBunches_v12, fragment.HLT_ZeroBias_FirstBXAfterTrain_v10, fragment.HLT_ZeroBias_FirstCollisionAfterAbortGap_v12, fragment.HLT_ZeroBias_FirstCollisionInTrain_v11, fragment.HLT_ZeroBias_LastCollisionInTrain_v10, fragment.HLT_HT300_Beamspot_v23, fragment.HLT_IsoTrackHB_v14, fragment.HLT_IsoTrackHE_v14, fragment.HLT_L1SingleMuCosmics_v8, fragment.HLT_L2Mu10_NoVertex_NoBPTX3BX_v14, fragment.HLT_L2Mu10_NoVertex_NoBPTX_v15, fragment.HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13, fragment.HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14, fragment.HLT_CDC_L2cosmic_10_er1p0_v10, fragment.HLT_CDC_L2cosmic_5p5_er1p0_v10, fragment.HLT_PPSMaxTracksPerArm1_v9, fragment.HLT_PPSMaxTracksPerRP4_v9, fragment.HLT_PPSRandom_v1, fragment.HLT_SpecialHLTPhysics_v7, fragment.AlCa_LumiPixelsCounts_RandomHighRate_v4, fragment.AlCa_LumiPixelsCounts_ZeroBiasVdM_v4, fragment.AlCa_LumiPixelsCounts_ZeroBiasGated_v5, fragment.HLT_L1SingleMuOpen_v6, fragment.HLT_L1SingleMuOpen_DT_v6, fragment.HLT_L1SingleMu3_v5, fragment.HLT_L1SingleMu5_v5, fragment.HLT_L1SingleMu7_v5, fragment.HLT_L1DoubleMu0_v5, fragment.HLT_L1SingleJet8erHE_v5, fragment.HLT_L1SingleJet10erHE_v5, fragment.HLT_L1SingleJet12erHE_v5, fragment.HLT_L1SingleJet35_v5, fragment.HLT_L1SingleJet200_v5, fragment.HLT_L1SingleEG8er2p5_v4, fragment.HLT_L1SingleEG10er2p5_v4, fragment.HLT_L1SingleEG15er2p5_v4, fragment.HLT_L1SingleEG26er2p5_v4, fragment.HLT_L1SingleEG28er2p5_v4, fragment.HLT_L1SingleEG28er2p1_v4, fragment.HLT_L1SingleEG28er1p5_v4, fragment.HLT_L1SingleEG34er2p5_v4, fragment.HLT_L1SingleEG36er2p5_v4, fragment.HLT_L1SingleEG38er2p5_v4, fragment.HLT_L1SingleEG40er2p5_v4, fragment.HLT_L1SingleEG42er2p5_v4, fragment.HLT_L1SingleEG45er2p5_v4, fragment.HLT_L1SingleEG50_v4, fragment.HLT_L1SingleJet60_v4, fragment.HLT_L1SingleJet90_v4, fragment.HLT_L1SingleJet120_v4, fragment.HLT_L1SingleJet180_v4, fragment.HLT_L1HTT120er_v4, fragment.HLT_L1HTT160er_v4, fragment.HLT_L1HTT200er_v4, fragment.HLT_L1HTT255er_v4, fragment.HLT_L1HTT280er_v4, fragment.HLT_L1HTT320er_v4, fragment.HLT_L1HTT360er_v4, fragment.HLT_L1HTT400er_v4, fragment.HLT_L1HTT450er_v4, fragment.HLT_L1ETM120_v4, fragment.HLT_L1ETM150_v4, fragment.HLT_L1EXT_HCAL_LaserMon1_v5, fragment.HLT_L1EXT_HCAL_LaserMon4_v5, fragment.HLT_CscCluster_Cosmic_v4, fragment.HLT_HT60_Beamspot_v22, fragment.HLT_HT300_Beamspot_PixelClusters_WP2_v7, fragment.HLT_PixelClusters_WP2_v4, fragment.HLT_PixelClusters_WP1_v4, fragment.HLT_BptxOR_v6, fragment.HLT_L1SingleMuCosmics_EMTF_v4, fragment.HLT_L1SingleMuCosmics_CosmicTracking_v1, fragment.HLT_L1SingleMuCosmics_PointingCosmicTracking_v1, fragment.HLT_L1FatEvents_v5, fragment.HLT_Random_HighRate_v1, fragment.HLT_ZeroBias_HighRate_v4, fragment.HLT_ZeroBias_Gated_v4, fragment.HLT_SpecialZeroBias_v6, fragment.HLTriggerFinalPath, fragment.HLTAnalyzerEndpath, fragment.Dataset_AlCaLumiPixelsCountsExpress, fragment.Dataset_AlCaLumiPixelsCountsPrompt, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate0, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate1, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate2, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate3, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate4, fragment.Dataset_AlCaLumiPixelsCountsPromptHighRate5, fragment.Dataset_AlCaLumiPixelsCountsGated, fragment.Dataset_AlCaP0, fragment.Dataset_AlCaPPSExpress, fragment.Dataset_AlCaPPSPrompt, fragment.Dataset_AlCaPhiSym, fragment.Dataset_Commissioning, fragment.Dataset_Cosmics, fragment.Dataset_DQMGPUvsCPU, fragment.Dataset_DQMOnlineBeamspot, fragment.Dataset_DQMPPSRandom, fragment.Dataset_EcalLaser, fragment.Dataset_EventDisplay, fragment.Dataset_ExpressAlignment, fragment.Dataset_ExpressCosmics, fragment.Dataset_ExpressPhysics, fragment.Dataset_HLTMonitor, fragment.Dataset_HLTPhysics, fragment.Dataset_HcalNZS, fragment.Dataset_L1Accept, fragment.Dataset_MinimumBias, fragment.Dataset_MuonShower, fragment.Dataset_NoBPTX, fragment.Dataset_OnlineMonitor, fragment.Dataset_RPCMonitor, fragment.Dataset_TestEnablesEcalHcal, fragment.Dataset_TestEnablesEcalHcalDQM, fragment.Dataset_VRRandom0, fragment.Dataset_VRRandom1, fragment.Dataset_VRRandom2, fragment.Dataset_VRRandom3, fragment.Dataset_VRRandom4, fragment.Dataset_VRRandom5, fragment.Dataset_VRRandom6, fragment.Dataset_VRRandom7, fragment.Dataset_VRRandom8, fragment.Dataset_VRRandom9, fragment.Dataset_VRRandom10, fragment.Dataset_VRRandom11, fragment.Dataset_VRRandom12, fragment.Dataset_VRRandom13, fragment.Dataset_VRRandom14, fragment.Dataset_VRRandom15, fragment.Dataset_ZeroBias, fragment.Dataset_SpecialRandom0, fragment.Dataset_SpecialRandom1, fragment.Dataset_SpecialRandom2, fragment.Dataset_SpecialRandom3, fragment.Dataset_SpecialRandom4, fragment.Dataset_SpecialRandom5, fragment.Dataset_SpecialRandom6, fragment.Dataset_SpecialRandom7, fragment.Dataset_SpecialRandom8, fragment.Dataset_SpecialRandom9, fragment.Dataset_SpecialRandom10, fragment.Dataset_SpecialRandom11, fragment.Dataset_SpecialRandom12, fragment.Dataset_SpecialRandom13, fragment.Dataset_SpecialRandom14, fragment.Dataset_SpecialRandom15, fragment.Dataset_SpecialRandom16, fragment.Dataset_SpecialRandom17, fragment.Dataset_SpecialRandom18, fragment.Dataset_SpecialRandom19, fragment.Dataset_SpecialZeroBias0, fragment.Dataset_SpecialZeroBias1, fragment.Dataset_SpecialZeroBias2, fragment.Dataset_SpecialZeroBias3, fragment.Dataset_SpecialZeroBias4, fragment.Dataset_SpecialZeroBias5, fragment.Dataset_SpecialZeroBias6, fragment.Dataset_SpecialZeroBias7, fragment.Dataset_SpecialZeroBias8, fragment.Dataset_SpecialZeroBias9, fragment.Dataset_SpecialZeroBias10, fragment.Dataset_SpecialZeroBias11, fragment.Dataset_SpecialZeroBias12, fragment.Dataset_SpecialZeroBias13, fragment.Dataset_SpecialZeroBias14, fragment.Dataset_SpecialZeroBias15, fragment.Dataset_SpecialZeroBias16, fragment.Dataset_SpecialZeroBias17, fragment.Dataset_SpecialZeroBias18, fragment.Dataset_SpecialZeroBias19, fragment.Dataset_SpecialZeroBias20, fragment.Dataset_SpecialZeroBias21, fragment.Dataset_SpecialZeroBias22, fragment.Dataset_SpecialZeroBias23, fragment.Dataset_SpecialZeroBias24, fragment.Dataset_SpecialZeroBias25, fragment.Dataset_SpecialZeroBias26, fragment.Dataset_SpecialZeroBias27, fragment.Dataset_SpecialZeroBias28, fragment.Dataset_SpecialZeroBias29, fragment.Dataset_SpecialZeroBias30, fragment.Dataset_SpecialZeroBias31, fragment.Dataset_SpecialHLTPhysics0, fragment.Dataset_SpecialHLTPhysics1, fragment.Dataset_SpecialHLTPhysics2, fragment.Dataset_SpecialHLTPhysics3, fragment.Dataset_SpecialHLTPhysics4, fragment.Dataset_SpecialHLTPhysics5, fragment.Dataset_SpecialHLTPhysics6, fragment.Dataset_SpecialHLTPhysics7, fragment.Dataset_SpecialHLTPhysics8, fragment.Dataset_SpecialHLTPhysics9, fragment.Dataset_SpecialHLTPhysics10, fragment.Dataset_SpecialHLTPhysics11, fragment.Dataset_SpecialHLTPhysics12, fragment.Dataset_SpecialHLTPhysics13, fragment.Dataset_SpecialHLTPhysics14, fragment.Dataset_SpecialHLTPhysics15, fragment.Dataset_SpecialHLTPhysics16, fragment.Dataset_SpecialHLTPhysics17, fragment.Dataset_SpecialHLTPhysics18, fragment.Dataset_SpecialHLTPhysics19, ))


# dummify hltGetConditions in cff's
if 'hltGetConditions' in fragment.__dict__ and 'HLTriggerFirstPath' in fragment.__dict__ :
    fragment.hltDummyConditions = cms.EDFilter( "HLTBool",
        result = cms.bool( True )
    )
    fragment.HLTriggerFirstPath.replace(fragment.hltGetConditions,fragment.hltDummyConditions)

# add specific customizations
from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
fragment = customizeHLTforAll(fragment,"Special")

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
fragment = customizeHLTforCMSSW(fragment,"Special")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(fragment)

