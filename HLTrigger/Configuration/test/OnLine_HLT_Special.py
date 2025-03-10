# hltGetConfiguration /dev/CMSSW_15_0_0/Special --full --data --type Special --unprescale --process HLTSpecial --globaltag auto:run3_hlt_Special --input file:RelVal_Raw_Special_DATA.root

# /dev/CMSSW_15_0_0/Special/V8 (CMSSW_15_0_0)

import FWCore.ParameterSet.Config as cms

process = cms.Process( "HLTSpecial" )

process.load("Configuration.StandardSequences.Accelerators_cff")

process.HLTConfigVersion = cms.PSet(
  tableName = cms.string("/dev/CMSSW_15_0_0/Special/V8")
)

process.HLTGroupedCkfTrajectoryBuilderP5 = cms.PSet( 
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
process.HLTCkfBaseTrajectoryFilterP5 = cms.PSet( 
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
process.HLTCkfBaseTrajectoryFilter_block = cms.PSet( 
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
process.HLTIter4PSetTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter0GroupedCkfTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter4PSetTrajectoryFilterIT = cms.PSet( 
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
process.HLTPSetPvClusterComparerForIT = cms.PSet( 
  track_chi2_max = cms.double( 20.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 ),
  track_pt_min = cms.double( 1.0 )
)
process.HLTPSetMuonCkfTrajectoryBuilder = cms.PSet( 
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
process.HLTIter0HighPtTkMuPSetTrajectoryFilterIT = cms.PSet( 
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
process.HLTPSetPvClusterComparerForBTag = cms.PSet( 
  track_chi2_max = cms.double( 20.0 ),
  track_pt_max = cms.double( 20.0 ),
  track_prob_min = cms.double( -1.0 ),
  track_pt_min = cms.double( 0.1 )
)
process.HLTIter2GroupedCkfTrajectoryBuilderIT = cms.PSet( 
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
process.HLTSiStripClusterChargeCutTight = cms.PSet(  value = cms.double( 1945.0 ) )
process.HLTPSetMuonTrackingRegionBuilder8356 = cms.PSet( 
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
process.HLTIter2PSetTrajectoryFilterIT = cms.PSet( 
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
process.HLTPSetMuTrackJpsiTrajectoryBuilder = cms.PSet( 
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
process.HLTPSetTrajectoryBuilderForGsfElectrons = cms.PSet( 
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
process.HLTSiStripClusterChargeCutNone = cms.PSet(  value = cms.double( -1.0 ) )
process.HLTPSetMuonCkfTrajectoryFilter = cms.PSet( 
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
process.HLTIter1PSetTrajectoryFilterIT = cms.PSet( 
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
process.HLTSeedFromProtoTracks = cms.PSet( 
  TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "ParabolicMf" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  MinOneOverPtError = cms.double( 1.0 )
)
process.HLTPSetMuTrackJpsiTrajectoryFilter = cms.PSet( 
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
process.HLTIter0PSetTrajectoryFilterIT = cms.PSet( 
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
process.HLTSeedFromConsecutiveHitsCreator = cms.PSet( 
  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
  SeedMomentumForBOFF = cms.double( 5.0 ),
  propagator = cms.string( "PropagatorWithMaterial" ),
  forceKinematicWithRegionDirection = cms.bool( False ),
  magneticField = cms.string( "" ),
  OriginTransverseErrorMultiplier = cms.double( 1.0 ),
  ComponentName = cms.string( "SeedFromConsecutiveHitsCreator" ),
  MinOneOverPtError = cms.double( 1.0 )
)
process.HLTSiStripClusterChargeCutForHI = cms.PSet(  value = cms.double( 2069.0 ) )
process.HLTPSetTrajectoryFilterForElectrons = cms.PSet( 
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
process.HLTIter0HighPtTkMuPSetTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter1GroupedCkfTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter0IterL3MuonPSetGroupedCkfTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter0IterL3FromL1MuonGroupedCkfTrajectoryFilterIT = cms.PSet( 
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
process.HLTIter0IterL3FromL1MuonPSetGroupedCkfTrajectoryBuilderIT = cms.PSet( 
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
process.HLTIter0IterL3MuonGroupedCkfTrajectoryFilterIT = cms.PSet( 
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
process.HLTPSetCkfBaseTrajectoryFilter_block = cms.PSet( 
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
process.HLTSiStripClusterChargeCutLoose = cms.PSet(  value = cms.double( 1620.0 ) )
process.HLTPSetInitialStepTrajectoryFilterShapePreSplittingPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryFilterBasePreSplittingForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryBuilderPreSplittingForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryFilterPreSplittingForFullTrackingPPOnAA = cms.PSet( 
  ComponentType = cms.string( "CompositeTrajectoryFilter" ),
  filters = cms.VPSet( 
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterBasePreSplittingForFullTrackingPPOnAA" )    ),
    cms.PSet(  refToPSet_ = cms.string( "HLTPSetInitialStepTrajectoryFilterShapePreSplittingPPOnAA" )    )
  )
)
process.HLTPSetInitialStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtQuadStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtQuadStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetHighPtTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetHighPtTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetDetachedQuadStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetDetachedTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetPixelPairStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetPixelPairStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetMixedTripletStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetPixelLessStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetPixelLessStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetTobTecStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetTobTecStepInOutTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetTobTecStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetJetCoreStepTrajectoryFilterForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetJetCoreStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetPixelPairStepTrajectoryFilterInOutForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetMixedTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetDetachedQuadStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetDetachedTripletStepTrajectoryBuilderForFullTrackingPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
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
process.HLTPSetInitialStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtQuadStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
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
process.HLTPSetLowPtQuadStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
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
process.HLTPSetHighPtTripletStepTrajectoryFilterForDmesonPPOnAA = cms.PSet( 
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
process.HLTPSetHighPtTripletStepTrajectoryBuilderForDmesonPPOnAA = cms.PSet( 
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
process.streams = cms.PSet( 
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
  PhysicsSpecialMinimumBias0 = cms.vstring( 'SpecialMinimumBias0' ),
  PhysicsSpecialMinimumBias1 = cms.vstring( 'SpecialMinimumBias1' ),
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
process.datasets = cms.PSet( 
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
  SpecialMinimumBias0 = cms.vstring( 'HLT_L1MinimumBiasHF0ANDBptxAND_v1',
    'HLT_PixelClusters_WP2_HighRate_v1' ),
  SpecialMinimumBias1 = cms.vstring( 'HLT_L1MinimumBiasHF0ANDBptxAND_v1',
    'HLT_PixelClusters_WP2_HighRate_v1' ),
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

process.CSCChannelMapperESSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "CSCChannelMapperRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.CSCINdexerESSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "CSCIndexerRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.GlobalParametersRcdSource = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "L1TGlobalParametersRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.GlobalTag = cms.ESSource( "PoolDBESSource",
    DBParameters = cms.PSet( 
      connectionRetrialTimeOut = cms.untracked.int32( 60 ),
      idleConnectionCleanupPeriod = cms.untracked.int32( 10 ),
      enableReadOnlySessionOnUpdateConnection = cms.untracked.bool( False ),
      enablePoolAutomaticCleanUp = cms.untracked.bool( False ),
      messageLevel = cms.untracked.int32( 0 ),
      authenticationPath = cms.untracked.string( "." ),
      connectionRetrialPeriod = cms.untracked.int32( 10 ),
      connectionTimeOut = cms.untracked.int32( 0 ),
      enableConnectionSharing = cms.untracked.bool( True )
    ),
    connect = cms.string( "frontier://FrontierProd/CMS_CONDITIONS" ),
    globaltag = cms.string( "None" ),
    snapshotTime = cms.string( "" ),
    toGet = cms.VPSet( 
      cms.PSet(  refreshTime = cms.uint64( 2 ),
        record = cms.string( "BeamSpotOnlineLegacyObjectsRcd" )
      ),
      cms.PSet(  refreshTime = cms.uint64( 2 ),
        record = cms.string( "BeamSpotOnlineHLTObjectsRcd" )
      ),
      cms.PSet(  refreshTime = cms.uint64( 40 ),
        record = cms.string( "LHCInfoPerLSRcd" )
      ),
      cms.PSet(  refreshTime = cms.uint64( 40 ),
        record = cms.string( "LHCInfoPerFillRcd" )
      )
    ),
    DumpStat = cms.untracked.bool( False ),
    ReconnectEachRun = cms.untracked.bool( True ),
    RefreshAlways = cms.untracked.bool( False ),
    RefreshEachRun = cms.untracked.bool( True ),
    RefreshOpenIOVs = cms.untracked.bool( False ),
    pfnPostfix = cms.untracked.string( "" ),
    pfnPrefix = cms.untracked.string( "" )
)
process.HcalTimeSlewEP = cms.ESSource( "HcalTimeSlewEP",
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
process.HepPDTESSource = cms.ESSource( "HepPDTESSource",
    pdtFileName = cms.FileInPath( "SimGeneral/HepPDTESSource/data/pythiaparticle.tbl" )
)
process.eegeom = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "EcalMappingRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.es_hardcode = cms.ESSource( "HcalHardcodeCalibrations",
    fromDDD = cms.untracked.bool( False ),
    toGet = cms.untracked.vstring( 'GainWidths' )
)
process.hltESSBTagRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "JetTagComputerRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSEcalSeverityLevel = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "EcalSeverityLevelAlgoRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSHcalSeverityLevel = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSPFRecHitHCALParamsRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "PFRecHitHCALParamsRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSPFRecHitHCALTopologyRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "PFRecHitHCALTopologyRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.hltESSTfGraphRecord = cms.ESSource( "EmptyESSource",
    recordName = cms.string( "TfGraphRecord" ),
    iovIsRunNotTime = cms.bool( True ),
    firstValid = cms.vuint32( 1 )
)
process.ppsPixelTopologyESSource = cms.ESSource( "PPSPixelTopologyESSource",
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

process.AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "anyDirection" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
process.CSCChannelMapperESProducer = cms.ESProducer( "CSCChannelMapperESProducer",
  AlgoName = cms.string( "CSCChannelMapperPostls1" )
)
process.CSCGeometryESModule = cms.ESProducer( "CSCGeometryESModule",
  fromDDD = cms.bool( False ),
  fromDD4hep = cms.bool( False ),
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  useRealWireGeometry = cms.bool( True ),
  useOnlyWiresInME1a = cms.bool( False ),
  useGangedStripsInME1a = cms.bool( False ),
  useCentreTIOffsets = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  debugV = cms.untracked.bool( False )
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
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" ),
  MapAuto = cms.untracked.bool( False ),
  SkipHE = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.CaloTowerGeometryFromDBEP = cms.ESProducer( "CaloTowerGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.CaloTowerTopologyEP = cms.ESProducer( "CaloTowerTopologyEP",
  appendToDataLabel = cms.string( "" )
)
process.CastorDbProducer = cms.ESProducer( "CastorDbProducer",
  dump = cms.untracked.vstring(  ),
  appendToDataLabel = cms.string( "" )
)
process.ClusterShapeHitFilterESProducer = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "ClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutNone" ) ),
  appendToDataLabel = cms.string( "" )
)
process.DTGeometryESModule = cms.ESProducer( "DTGeometryESModule",
  fromDDD = cms.bool( False ),
  fromDD4hep = cms.bool( False ),
  DDDetector = cms.ESInputTag( "","" ),
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" ),
  attribute = cms.string( "MuStructure" ),
  value = cms.string( "MuonBarrelDT" ),
  applyAlignment = cms.bool( True )
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
process.EcalLaserCorrectionService = cms.ESProducer( "EcalLaserCorrectionService",
  maxExtrapolationTimeInSec = cms.uint32( 0 ),
  appendToDataLabel = cms.string( "" )
)
process.EcalPreshowerGeometryFromDBEP = cms.ESProducer( "EcalPreshowerGeometryFromDBEP",
  applyAlignment = cms.bool( True )
)
process.GEMGeometryESModule = cms.ESProducer( "GEMGeometryESModule",
  fromDDD = cms.bool( False ),
  fromDD4hep = cms.bool( False ),
  applyAlignment = cms.bool( False ),
  alignmentsLabel = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.GlobalParameters = cms.ESProducer( "StableParametersTrivialProducer",
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
process.HcalGeometryFromDBEP = cms.ESProducer( "HcalGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.HcalTopologyIdealEP = cms.ESProducer( "HcalTopologyIdealEP",
  Exclude = cms.untracked.string( "" ),
  MergePosition = cms.untracked.bool( True ),
  appendToDataLabel = cms.string( "" )
)
process.MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.MaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForHI" ),
  Mass = cms.double( 0.139 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.MaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.OppositeMaterialPropagatorForHI = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialOppositeForHI" ),
  Mass = cms.double( 0.139 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.OppositeMaterialPropagatorParabolicMF = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialParabolicMfOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.OppositePropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStepOpposite" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( 0.1 )
)
process.ParametrizedMagneticFieldProducer = cms.ESProducer( "AutoParametrizedMagneticFieldProducer",
  version = cms.string( "Parabolic" ),
  label = cms.untracked.string( "ParabolicMf" ),
  valueOverride = cms.int32( -1 )
)
process.PropagatorWithMaterialForLoopers = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 4.0 ),
  ComponentName = cms.string( "PropagatorWithMaterialForLoopers" ),
  Mass = cms.double( 0.1396 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.PropagatorWithMaterialForMixedStep = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "PropagatorWithMaterialForMixedStep" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( 0.1 )
)
process.RPCGeometryESModule = cms.ESProducer( "RPCGeometryESModule",
  fromDDD = cms.untracked.bool( False ),
  fromDD4hep = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.SiPixelTemplateStoreESProducer = cms.ESProducer( "SiPixelTemplateStoreESProducer",
  appendToDataLabel = cms.string( "" )
)
process.SiStripClusterizerConditionsESProducer = cms.ESProducer( "SiStripClusterizerConditionsESProducer",
  QualityLabel = cms.string( "" ),
  Label = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.SiStripGainESProducer = cms.ESProducer( "SiStripGainESProducer",
  appendToDataLabel = cms.string( "" ),
  printDebug = cms.untracked.bool( False ),
  AutomaticNormalization = cms.bool( False ),
  APVGain = cms.VPSet( 
    cms.PSet(  NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" ),
      Record = cms.string( "SiStripApvGainRcd" )
    ),
    cms.PSet(  NormalizationFactor = cms.untracked.double( 1.0 ),
      Label = cms.untracked.string( "" ),
      Record = cms.string( "SiStripApvGain2Rcd" )
    )
  )
)
process.SiStripQualityESProducer = cms.ESProducer( "SiStripQualityESProducer",
  appendToDataLabel = cms.string( "" ),
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
  ),
  ReduceGranularity = cms.bool( False ),
  ThresholdForReducedGranularity = cms.double( 0.3 ),
  PrintDebugOutput = cms.bool( False ),
  UseEmptyRunInfo = cms.bool( False )
)
process.SiStripRecHitMatcherESProducer = cms.ESProducer( "SiStripRecHitMatcherESProducer",
  ComponentName = cms.string( "StandardMatcher" ),
  NSigmaInside = cms.double( 3.0 ),
  PreFilter = cms.bool( False )
)
process.SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 ),
  appendToDataLabel = cms.string( "" )
)
process.SimpleSecondaryVertex3TrkComputer = cms.ESProducer( "SimpleSecondaryVertexESProducer",
  use3d = cms.bool( True ),
  useSignificance = cms.bool( True ),
  unBoost = cms.bool( False ),
  minTracks = cms.uint32( 3 ),
  minVertices = cms.uint32( 1 ),
  appendToDataLabel = cms.string( "" )
)
process.SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.TrackerAdditionalParametersPerDetESModule = cms.ESProducer( "TrackerAdditionalParametersPerDetESModule",
  appendToDataLabel = cms.string( "" )
)
process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
  appendToDataLabel = cms.string( "" ),
  fromDDD = cms.bool( False ),
  applyAlignment = cms.bool( True ),
  alignmentsLabel = cms.string( "" )
)
process.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
  fromDDD = cms.bool( False ),
  fromDD4hep = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" ),
  appendToDataLabel = cms.string( "" )
)
process.VolumeBasedMagneticFieldESProducer = cms.ESProducer( "VolumeBasedMagneticFieldESProducerFromDB",
  label = cms.untracked.string( "" ),
  debugBuilder = cms.untracked.bool( False ),
  valueOverride = cms.int32( -1 )
)
process.ZdcGeometryFromDBEP = cms.ESProducer( "ZdcGeometryFromDBEP",
  applyAlignment = cms.bool( False )
)
process.caloDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "CaloDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
process.cosmicsNavigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "CosmicNavigationSchool" ),
  PluginName = cms.string( "" ),
  SimpleMagneticField = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.ctppsGeometryESModule = cms.ESProducer( "CTPPSGeometryESModule",
  verbosity = cms.untracked.uint32( 1 ),
  buildMisalignedGeometry = cms.bool( False ),
  isRun2 = cms.bool( False ),
  dbTag = cms.string( "" ),
  compactViewTag = cms.string( "" ),
  fromPreprocessedDB = cms.untracked.bool( True ),
  fromDD4hep = cms.untracked.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.ctppsInterpolatedOpticalFunctionsESSource = cms.ESProducer( "CTPPSInterpolatedOpticalFunctionsESSource",
  lhcInfoLabel = cms.string( "" ),
  lhcInfoPerFillLabel = cms.string( "" ),
  lhcInfoPerLSLabel = cms.string( "" ),
  opticsLabel = cms.string( "" ),
  useNewLHCInfo = cms.bool( True ),
  appendToDataLabel = cms.string( "" )
)
process.ecalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "EcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.02 ),
  nEta = cms.int32( 300 ),
  nPhi = cms.int32( 360 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
process.ecalElectronicsMappingHostESProducer = cms.ESProducer( "EcalElectronicsMappingHostESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.ecalMultifitConditionsHostESProducer = cms.ESProducer( "EcalMultifitConditionsHostESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.ecalSeverityLevel = cms.ESProducer( "EcalSeverityLevelESProducer",
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
process.hcalChannelPropertiesESProd = cms.ESProducer( "HcalChannelPropertiesEP" )
process.hcalDDDRecConstants = cms.ESProducer( "HcalDDDRecConstantsESModule",
  appendToDataLabel = cms.string( "" )
)
process.hcalDDDSimConstants = cms.ESProducer( "HcalDDDSimConstantsESModule",
  appendToDataLabel = cms.string( "" )
)
process.hcalDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HcalDetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 70 ),
  nPhi = cms.int32( 72 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
process.hcalMahiConditionsESProducer = cms.ESProducer( "HcalMahiConditionsESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
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
process.hcalRecoParamWithPulseShapeESProducer = cms.ESProducer( "HcalRecoParamWithPulseShapeESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hcalSiPMCharacteristicsESProducer = cms.ESProducer( "HcalSiPMCharacteristicsESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hcal_db_producer = cms.ESProducer( "HcalDbProducer" )
process.hltBoostedDoubleSecondaryVertexAK8Computer = cms.ESProducer( "CandidateBoostedDoubleSecondaryVertexESProducer",
  useCondDB = cms.bool( False ),
  gbrForestLabel = cms.string( "" ),
  weightFile = cms.FileInPath( "RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT_v4.weights.xml.gz" ),
  useGBRForest = cms.bool( True ),
  useAdaBoost = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.hltCombinedSecondaryVertex = cms.ESProducer( "CombinedSecondaryVertexESProducer",
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
process.hltCombinedSecondaryVertexV2 = cms.ESProducer( "CombinedSecondaryVertexESProducer",
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
process.hltDisplacedDijethltESPPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
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
process.hltDisplacedDijethltESPTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
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
process.hltESChi2MeasurementEstimatorForP5 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 100.0 ),
  nSigma = cms.double( 4.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 100000.0 ),
  ComponentName = cms.string( "hltESChi2MeasurementEstimatorForP5" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESFittingSmootherRKP5 = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "hltESPAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPBwdAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "hltESPBwdAnalyticalPropagator" ),
  SimpleMagneticField = cms.string( "" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPBwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPBwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.hltESPChi2ChargeLooseMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeMeasurementEstimator2000 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeMeasurementEstimator9ForHI = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2ChargeTightMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPChi2MeasurementEstimator100 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 40.0 ),
  nSigma = cms.double( 4.0 ),
  MaxDisplacement = cms.double( 0.5 ),
  MaxSagitta = cms.double( 2.0 ),
  MinimalTolerance = cms.double( 0.5 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1.0E12 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator100" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPChi2MeasurementEstimator16 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 16.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator16" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPChi2MeasurementEstimator30 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator30" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPChi2MeasurementEstimator9 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 9.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPChi2MeasurementEstimator9" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPCloseComponentsMerger5D = cms.ESProducer( "CloseComponentsMergerESProducer5D",
  ComponentName = cms.string( "hltESPCloseComponentsMerger5D" ),
  MaxComponents = cms.int32( 12 ),
  DistanceMeasure = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPDetachedQuadStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPDetachedQuadStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedQuadStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPDetachedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPDetachedTripletStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPDetachedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPDetachedTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.13 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPDisplacedDijethltPromptTrackCountingESProducer = cms.ESProducer( "PromptTrackCountingESProducer",
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
process.hltESPDisplacedDijethltPromptTrackCountingESProducerLong = cms.ESProducer( "PromptTrackCountingESProducer",
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
process.hltESPDisplacedDijethltPromptTrackCountingESProducerShortSig5 = cms.ESProducer( "PromptTrackCountingESProducer",
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
process.hltESPDisplacedDijethltTrackCounting2D1st = cms.ESProducer( "TrackCountingESProducer",
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
process.hltESPDisplacedDijethltTrackCounting2D1stLoose = cms.ESProducer( "TrackCountingESProducer",
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
process.hltESPDisplacedDijethltTrackCounting2D2ndLong = cms.ESProducer( "TrackCountingESProducer",
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
process.hltESPDummyDetLayerGeometry = cms.ESProducer( "DetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPDummyDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
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
process.hltESPFastSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPFittingSmootherIT = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPFittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPFlexibleKFFittingSmoother" ),
  standardFitter = cms.string( "hltESPKFFittingSmootherWithOutliersRejectionAndRK" ),
  looperFitter = cms.string( "hltESPKFFittingSmootherForLoopers" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPFwdElectronPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPFwdElectronPropagator" ),
  Mass = cms.double( 5.11E-4 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( False ),
  ptMin = cms.double( -1.0 )
)
process.hltESPGlobalDetLayerGeometry = cms.ESProducer( "GlobalDetLayerGeometryESProducer",
  ComponentName = cms.string( "hltESPGlobalDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPGlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
process.hltESPGsfElectronFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPGsfTrajectoryFitter = cms.ESProducer( "GsfTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPGsfTrajectoryFitter" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  GeometricalPropagator = cms.string( "hltESPAnalyticalPropagator" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPGsfTrajectorySmoother = cms.ESProducer( "GsfTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPGsfTrajectorySmoother" ),
  MaterialEffectsUpdator = cms.string( "hltESPElectronMaterialEffects" ),
  GeometricalPropagator = cms.string( "hltESPBwdAnalyticalPropagator" ),
  Merger = cms.string( "hltESPCloseComponentsMerger5D" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  ErrorRescaling = cms.double( 100.0 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPHighPtTripletStepChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPInitialStepChi2ChargeMeasurementEstimator30 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPInitialStepChi2MeasurementEstimator36 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 36.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPInitialStepChi2MeasurementEstimator36" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFFittingSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFFittingSmootherWithOutliersRejectionAndRK = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitter" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPKFTrajectoryFitterForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmoother" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "hltESPFastSteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFTrajectorySmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
  Propagator = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKFUpdator = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "hltESPKFUpdator" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPKullbackLeiblerDistance5D = cms.ESProducer( "DistanceBetweenComponentsESProducer5D",
  DistanceMeasure = cms.string( "KullbackLeibler" ),
  ComponentName = cms.string( "hltESPKullbackLeiblerDistance5D" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPL3MuKFTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
  Propagator = cms.string( "hltESPSmartPropagatorAny" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPLowPtQuadStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPLowPtQuadStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtQuadStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPLowPtStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPLowPtTripletStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPLowPtTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPLowPtTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.16 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPMeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
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
process.hltESPMixedStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPMixedStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPMixedStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPMixedTripletStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPMixedTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPMixedTripletStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPMuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
process.hltESPMuonTransientTrackingRecHitBuilder = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPPFRecHitHCALParams = cms.ESProducer( "PFRecHitHCALParamsESProducer@alpaka",
  energyThresholdsHB = cms.vdouble( 0.1, 0.2, 0.3, 0.3 ),
  energyThresholdsHE = cms.vdouble( 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltESPPFRecHitHCALTopology = cms.ESProducer( "PFRecHitHCALTopologyESProducer@alpaka",
  usePFThresholdsFromDB = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltESPPixelCPEFastParamsHIonPhase1 = cms.ESProducer( "PixelCPEFastParamsESProducerAlpakaHIonPhase1@alpaka",
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
process.hltESPPixelCPEFastParamsPhase1 = cms.ESProducer( "PixelCPEFastParamsESProducerAlpakaPhase1@alpaka",
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
process.hltESPPixelCPEGeneric = cms.ESProducer( "PixelCPEGenericESProducer",
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
process.hltESPPixelCPETemplateReco = cms.ESProducer( "PixelCPETemplateRecoESProducer",
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
process.hltESPPixelLessStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPPixelLessStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPPixelLessStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPPixelLessStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelLessStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.11 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPPixelPairStepChi2ChargeMeasurementEstimator9 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPPixelPairStepChi2MeasurementEstimator25 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  MaxChi2 = cms.double( 25.0 ),
  nSigma = cms.double( 3.0 ),
  MaxDisplacement = cms.double( 100.0 ),
  MaxSagitta = cms.double( -1.0 ),
  MinimalTolerance = cms.double( 10.0 ),
  MinPtForHitRecoveryInGluedDet = cms.double( 1000000.0 ),
  ComponentName = cms.string( "hltESPPixelPairStepChi2MeasurementEstimator25" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPPixelPairTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPPixelPairTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPRKTrajectoryFitter" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPRKTrajectorySmoother" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPGlobalDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPRungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  SimpleMagneticField = cms.string( "" ),
  MaxDPhi = cms.double( 1.6 ),
  ComponentName = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Mass = cms.double( 0.105 ),
  PropagationDirection = cms.string( "alongMomentum" ),
  useRungeKutta = cms.bool( True ),
  ptMin = cms.double( -1.0 )
)
process.hltESPSiPixelCablingSoA = cms.ESProducer( "SiPixelCablingSoAESProducer@alpaka",
  CablingMapLabel = cms.string( "" ),
  UseQualityInfo = cms.bool( False ),
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltESPSiPixelGainCalibrationForHLTSoA = cms.ESProducer( "SiPixelGainCalibrationForHLTSoAESProducer@alpaka",
  appendToDataLabel = cms.string( "" ),
  alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltESPSmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "hltESPSteppingHelixPropagatorAlong" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPSmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPSmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "hltESPSmartPropagatorAnyOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPSoftLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPSteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPSteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
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
process.hltESPStripCPEfromTrackAngle = cms.ESProducer( "StripCPEESProducer",
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
process.hltESPTTRHBWithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBWithTrackAngle" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTTRHBuilderAngleAndTemplate = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderAngleAndTemplate" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "hltESPStripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "hltESPPixelCPETemplateReco" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderPixelOnly" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "hltESPTTRHBuilderWithoutAngle4PixelTriplets" ),
  ComputeCoarseLocalPositionFromDisk = cms.bool( False ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" ),
  Phase2StripCPE = cms.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepChi2ChargeMeasurementEstimator16 = cms.ESProducer( "Chi2ChargeMeasurementEstimatorESProducer",
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
process.hltESPTobTecStepClusterShapeHitFilter = cms.ESProducer( "ClusterShapeHitFilterESProducer",
  PixelShapeFile = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_noL1.par" ),
  PixelShapeFileL1 = cms.string( "RecoTracker/PixelLowPtUtilities/data/pixelShapePhase1_loose.par" ),
  ComponentName = cms.string( "hltESPTobTecStepClusterShapeHitFilter" ),
  isPhase2 = cms.bool( False ),
  doPixelShapeCut = cms.bool( True ),
  doStripShapeCut = cms.bool( True ),
  clusterChargeCut = cms.PSet(  refToPSet_ = cms.string( "HLTSiStripClusterChargeCutTight" ) ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPTobTecStepFittingSmootherForLoopers = cms.ESProducer( "KFFittingSmootherESProducer",
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
process.hltESPTobTecStepFlexibleKFFittingSmoother = cms.ESProducer( "FlexibleKFFittingSmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepFlexibleKFFittingSmoother" ),
  standardFitter = cms.string( "hltESPTobTecStepFitterSmoother" ),
  looperFitter = cms.string( "hltESPTobTecStepFitterSmootherForLoopers" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepRKTrajectoryFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKFitter" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepRKTrajectoryFitterForLoopers = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKFitterForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepRKTrajectorySmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKSmoother" ),
  Propagator = cms.string( "PropagatorWithMaterialParabolicMf" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepRKTrajectorySmootherForLoopers = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTobTecStepRKSmootherForLoopers" ),
  Propagator = cms.string( "PropagatorWithMaterialForLoopers" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 7 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTobTecStepTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTobTecStepTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.09 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPTrackAlgoPriorityOrder = cms.ESProducer( "TrackAlgoPriorityOrderESProducer",
  ComponentName = cms.string( "hltESPTrackAlgoPriorityOrder" ),
  algoOrder = cms.vstring(  ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTrackSelectionTfCKF = cms.ESProducer( "TfGraphDefProducer",
  ComponentName = cms.string( "hltESPTrackSelectionTfCKF" ),
  FileName = cms.FileInPath( "RecoTracker/FinalTrackSelectors/data/TrackTfClassifier/CKF_Run3_12_5_0_pre5.pb" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer",
  usePhase2Stacks = cms.bool( False ),
  trackerGeometryLabel = cms.untracked.string( "" ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.5 ),
  ValidHitBonus = cms.double( 100.0 ),
  MissingHitPenalty = cms.double( 0.0 ),
  allowSharedFirstHit = cms.bool( False )
)
process.hltESPTrajectoryCleanerBySharedHitsP5 = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "hltESPTrajectoryCleanerBySharedHitsP5" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.19 ),
  ValidHitBonus = cms.double( 5.0 ),
  MissingHitPenalty = cms.double( 20.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.hltESPTrajectoryFitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltESPTrajectoryFitterRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltESPTrajectorySmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltESPTrajectorySmootherRK" ),
  Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" ),
  Updator = cms.string( "hltESPKFUpdator" ),
  Estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
  RecoGeometry = cms.string( "hltESPDummyDetLayerGeometry" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
process.hltOnlineBeamSpotESProducer = cms.ESProducer( "OnlineBeamSpotESProducer",
  timeThreshold = cms.int32( 48 ),
  sigmaZThreshold = cms.double( 2.0 ),
  sigmaXYThreshold = cms.double( 4.0 ),
  appendToDataLabel = cms.string( "" )
)
process.hltPixelTracksCleanerBySharedHits = cms.ESProducer( "PixelTrackCleanerBySharedHitsESProducer",
  ComponentName = cms.string( "hltPixelTracksCleanerBySharedHits" ),
  useQuadrupletAlgo = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.hltTrackCleaner = cms.ESProducer( "TrackCleanerESProducer",
  ComponentName = cms.string( "hltTrackCleaner" ),
  appendToDataLabel = cms.string( "" )
)
process.hoDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "HODetIdAssociator" ),
  etaBinSize = cms.double( 0.087 ),
  nEta = cms.int32( 30 ),
  nPhi = cms.int32( 72 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
process.multipleScatteringParametrisationMakerESProducer = cms.ESProducer( "MultipleScatteringParametrisationMakerESProducer",
  appendToDataLabel = cms.string( "" )
)
process.muonDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "MuonDetIdAssociator" ),
  etaBinSize = cms.double( 0.125 ),
  nEta = cms.int32( 48 ),
  nPhi = cms.int32( 48 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( True ),
  includeGEM = cms.bool( True ),
  includeME0 = cms.bool( False )
)
process.muonSeededTrajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "muonSeededTrajectoryCleanerBySharedHits" ),
  ComponentType = cms.string( "TrajectoryCleanerBySharedHits" ),
  fractionShared = cms.double( 0.1 ),
  ValidHitBonus = cms.double( 1000.0 ),
  MissingHitPenalty = cms.double( 1.0 ),
  allowSharedFirstHit = cms.bool( True )
)
process.navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" ),
  PluginName = cms.string( "" ),
  SimpleMagneticField = cms.string( "ParabolicMf" ),
  appendToDataLabel = cms.string( "" )
)
process.preshowerDetIdAssociator = cms.ESProducer( "DetIdAssociatorESProducer",
  ComponentName = cms.string( "PreshowerDetIdAssociator" ),
  etaBinSize = cms.double( 0.1 ),
  nEta = cms.int32( 60 ),
  nPhi = cms.int32( 30 ),
  hcalRegion = cms.int32( 2 ),
  includeBadChambers = cms.bool( False ),
  includeGEM = cms.bool( False ),
  includeME0 = cms.bool( False )
)
process.siPixelGainCalibrationForHLTGPU = cms.ESProducer( "SiPixelGainCalibrationForHLTGPUESProducer",
  appendToDataLabel = cms.string( "" )
)
process.siPixelQualityESProducer = cms.ESProducer( "SiPixelQualityESProducer",
  siPixelQualityFromDbLabel = cms.string( "" ),
  ListOfRecordToMerge = cms.VPSet( 
    cms.PSet(  record = cms.string( "SiPixelQualityFromDbRcd" ),
      tag = cms.string( "" )
    ),
    cms.PSet(  record = cms.string( "SiPixelDetVOffRcd" ),
      tag = cms.string( "" )
    )
  ),
  appendToDataLabel = cms.string( "" )
)
process.siPixelROCsStatusAndMappingWrapperESProducer = cms.ESProducer( "SiPixelROCsStatusAndMappingWrapperESProducer",
  ComponentName = cms.string( "" ),
  CablingMapLabel = cms.string( "" ),
  UseQualityInfo = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
process.siPixelTemplateDBObjectESProducer = cms.ESProducer( "SiPixelTemplateDBObjectESProducer" )
process.siStripBackPlaneCorrectionDepESProducer = cms.ESProducer( "SiStripBackPlaneCorrectionDepESProducer",
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
process.siStripLorentzAngleDepESProducer = cms.ESProducer( "SiStripLorentzAngleDepESProducer",
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
process.sistripconn = cms.ESProducer( "SiStripConnectivity" )
process.trackerTopology = cms.ESProducer( "TrackerTopologyEP",
  appendToDataLabel = cms.string( "" )
)
process.zdcTopologyEP = cms.ESProducer( "ZdcTopologyEP",
  appendToDataLabel = cms.string( "" )
)

process.FastTimerService = cms.Service( "FastTimerService",
    printEventSummary = cms.untracked.bool( False ),
    printRunSummary = cms.untracked.bool( True ),
    printJobSummary = cms.untracked.bool( True ),
    writeJSONSummary = cms.untracked.bool( False ),
    jsonFileName = cms.untracked.string( "resources.json" ),
    enableDQM = cms.untracked.bool( True ),
    enableDQMbyModule = cms.untracked.bool( False ),
    enableDQMbyPath = cms.untracked.bool( False ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    enableDQMTransitions = cms.untracked.bool( False ),
    dqmTimeRange = cms.untracked.double( 2000.0 ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    dqmMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmMemoryResolution = cms.untracked.double( 5000.0 ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    dqmPathMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmPathMemoryResolution = cms.untracked.double( 5000.0 ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    dqmModuleMemoryRange = cms.untracked.double( 100000.0 ),
    dqmModuleMemoryResolution = cms.untracked.double( 500.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
)
process.MessageLogger = cms.Service( "MessageLogger",
    suppressWarning = cms.untracked.vstring( 'hltL3MuonsIOHit',
      'hltL3MuonsOIHit',
      'hltL3MuonsOIState',
      'hltLightPFTracks',
      'hltPixelTracks',
      'hltPixelTracksForHighMult',
      'hltSiPixelClusters',
      'hltSiPixelDigis' ),
    suppressFwkInfo = cms.untracked.vstring(  ),
    suppressInfo = cms.untracked.vstring(  ),
    suppressDebug = cms.untracked.vstring(  ),
    debugModules = cms.untracked.vstring(  ),
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
      enableStatistics = cms.untracked.bool( False )
    ),
    suppressError = cms.untracked.vstring( 'hltL3TkTracksFromL2IOHit',
      'hltL3TkTracksFromL2OIHit',
      'hltL3TkTracksFromL2OIState' )
)
process.ThroughputService = cms.Service( "ThroughputService",
    eventRange = cms.untracked.uint32( 10000 ),
    eventResolution = cms.untracked.uint32( 1 ),
    printEventSummary = cms.untracked.bool( False ),
    enableDQM = cms.untracked.bool( True ),
    dqmPathByProcesses = cms.untracked.bool( True ),
    dqmPath = cms.untracked.string( "HLT/Throughput" ),
    timeRange = cms.untracked.double( 60000.0 ),
    timeResolution = cms.untracked.double( 5.828 )
)

process.hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
process.hltPSetMap = cms.EDProducer( "ParameterSetBlobProducer" )
process.hltBoolFalse = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
process.hltBackend = cms.EDProducer( "AlpakaBackendProducer@alpaka"
)
process.hltStatusOnGPUFilter = cms.EDFilter( "AlpakaBackendFilter",
    producer = cms.InputTag( 'hltBackend','backend' ),
    backends = cms.vstring( 'CudaAsync',
      'ROCmAsync' )
)
process.hltTriggerType = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 1 )
)
process.hltGtStage2Digis = cms.EDProducer( "L1TRawToDigi",
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
process.hltGtStage2ObjectMap = cms.EDProducer( "L1TGlobalProducer",
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
process.hltOnlineMetaDataDigis = cms.EDProducer( "OnlineMetaDataRawToDigi",
    onlineMetaDataInputLabel = cms.InputTag( "rawDataCollector" )
)
process.hltOnlineBeamSpot = cms.EDProducer( "BeamSpotOnlineProducer",
    changeToCMSCoordinates = cms.bool( False ),
    maxZ = cms.double( 40.0 ),
    setSigmaZ = cms.double( 0.0 ),
    beamMode = cms.untracked.uint32( 11 ),
    src = cms.InputTag( "" ),
    gtEvmLabel = cms.InputTag( "" ),
    maxRadius = cms.double( 2.0 ),
    useTransientRecord = cms.bool( True )
)
process.hltL1sZeroBiasIorAlwaysTrueIorIsolatedBunch = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaEcalPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltEcalDigisLegacy = cms.EDProducer( "EcalRawToDigi",
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
process.hltEcalDigisSoA = cms.EDProducer( "EcalRawToDigiPortable@alpaka",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    maxChannelsEB = cms.uint32( 61200 ),
    maxChannelsEE = cms.uint32( 14648 ),
    digisLabelEB = cms.string( "ebDigis" ),
    digisLabelEE = cms.string( "eeDigis" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltEcalDigis = cms.EDProducer( "EcalDigisFromPortableProducer",
    digisInLabelEB = cms.InputTag( 'hltEcalDigisSoA','ebDigis' ),
    digisInLabelEE = cms.InputTag( 'hltEcalDigisSoA','eeDigis' ),
    digisOutLabelEB = cms.string( "ebDigis" ),
    digisOutLabelEE = cms.string( "eeDigis" ),
    produceDummyIntegrityCollections = cms.bool( False )
)
process.hltEcalUncalibRecHitSoA = cms.EDProducer( "EcalUncalibRecHitProducerPortable@alpaka",
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
process.hltEcalUncalibRecHit = cms.EDProducer( "EcalUncalibRecHitSoAToLegacy",
    inputCollectionEB = cms.InputTag( 'hltEcalUncalibRecHitSoA','EcalUncalibRecHitsEB' ),
    outputLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    isPhase2 = cms.bool( False ),
    inputCollectionEE = cms.InputTag( 'hltEcalUncalibRecHitSoA','EcalUncalibRecHitsEE' ),
    outputLabelEE = cms.string( "EcalUncalibRecHitsEE" )
)
process.hltEcalDetIdToBeRecovered = cms.EDProducer( "EcalDetIdToBeRecoveredProducer",
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
process.hltEcalRecHit = cms.EDProducer( "EcalRecHitProducer",
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
process.hltEcalPreshowerDigis = cms.EDProducer( "ESRawToDigi",
    sourceTag = cms.InputTag( "rawDataCollector" ),
    debugMode = cms.untracked.bool( False ),
    InstanceES = cms.string( "" ),
    LookupTable = cms.FileInPath( "EventFilter/ESDigiToRaw/data/ES_lookup_table.dat" ),
    ESdigiCollection = cms.string( "" )
)
process.hltEcalPreshowerRecHit = cms.EDProducer( "ESRecHitProducer",
    ESrechitCollection = cms.string( "EcalRecHitsES" ),
    ESdigiCollection = cms.InputTag( "hltEcalPreshowerDigis" ),
    algo = cms.string( "ESRecHitWorker" ),
    ESRecoAlgo = cms.int32( 0 )
)
process.hltEcalPhiSymFilter = cms.EDFilter( "HLTEcalPhiSymFilter",
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
process.hltFEDSelectorL1 = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1404 )
)
process.hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
process.hltL1sAlCaEcalPi0Eta = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaEcalEtaEBonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltSimple3x3Clusters = cms.EDProducer( "EgammaHLTNxNClusterProducer",
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
process.hltAlCaEtaRecHitsFilterEBonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
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
process.hltAlCaEtaEBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
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
process.hltAlCaEtaEBRechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "barrel" ),
    digisIn = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    digisOut = cms.string( "etaEBDigis" ),
    recHits = cms.InputTag( 'hltAlCaEtaEBUncalibrator','etaEcalRecHitsEB' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "etaEBSrFlags" )
)
process.hltPreAlCaEcalEtaEEonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltAlCaEtaRecHitsFilterEEonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
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
process.hltAlCaEtaEEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
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
process.hltAlCaEtaEERechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "endcap" ),
    digisIn = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    digisOut = cms.string( "etaEEDigis" ),
    recHits = cms.InputTag( 'hltAlCaEtaEEUncalibrator','etaEcalRecHitsEE' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "etaEESrFlags" )
)
process.hltPreAlCaEcalPi0EBonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltAlCaPi0RecHitsFilterEBonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
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
process.hltAlCaPi0EBUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
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
process.hltAlCaPi0EBRechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "barrel" ),
    digisIn = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    digisOut = cms.string( "pi0EBDigis" ),
    recHits = cms.InputTag( 'hltAlCaPi0EBUncalibrator','pi0EcalRecHitsEB' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "pi0EBSrFlags" )
)
process.hltPreAlCaEcalPi0EEonly = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltAlCaPi0RecHitsFilterEEonlyRegional = cms.EDFilter( "HLTRegionalEcalResonanceFilter",
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
process.hltAlCaPi0EEUncalibrator = cms.EDProducer( "EcalRecalibRecHitProducer",
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
process.hltAlCaPi0EERechitsToDigis = cms.EDProducer( "HLTRechitsToDigis",
    region = cms.string( "endcap" ),
    digisIn = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    digisOut = cms.string( "pi0EEDigis" ),
    recHits = cms.InputTag( 'hltAlCaPi0EEUncalibrator','pi0EcalRecHitsEE' ),
    srFlagsIn = cms.InputTag( "hltEcalDigisLegacy" ),
    srFlagsOut = cms.string( "pi0EESrFlags" )
)
process.hltL1sSingleMu5IorSingleMu14erIorSingleMu16er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaRPCMuonNormalisation = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltRPCMuonNormaL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltFEDSelectorTCDS = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1024, 1025 )
)
process.hltFEDSelectorDT = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1369, 1370, 1371 )
)
process.hltFEDSelectorRPC = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 790, 791, 792, 793, 794, 795, 821 )
)
process.hltFEDSelectorCSC = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 831, 832, 833, 834, 835, 836, 837, 838, 839, 841, 842, 843, 844, 845, 846, 847, 848, 849, 851, 852, 853, 854, 855, 856, 857, 858, 859, 861, 862, 863, 864, 865, 866, 867, 868, 869 )
)
process.hltFEDSelectorGEM = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478 )
)
process.hltFEDSelectorTwinMux = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1390, 1391, 1393, 1394, 1395 )
)
process.hltFEDSelectorOMTF = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1380, 1381 )
)
process.hltFEDSelectorCPPF = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1386 )
)
process.hltRandomEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 3 )
)
process.hltPreAlCaLumiPixelsCountsRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPixelTrackerHVOn = cms.EDFilter( "DetectorStateFilter",
    DebugOn = cms.untracked.bool( False ),
    DetectorType = cms.untracked.string( "pixel" ),
    acceptedCombinations = cms.untracked.vstring(  ),
    DcsStatusLabel = cms.untracked.InputTag( "" ),
    DCSRecordLabel = cms.untracked.InputTag( "hltOnlineMetaDataDigis" )
)
process.hltOnlineBeamSpotDevice = cms.EDProducer( "BeamSpotDeviceProducer@alpaka",
    src = cms.InputTag( "hltOnlineBeamSpot" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltSiPixelClustersSoA = cms.EDProducer( "SiPixelRawToClusterPhase1@alpaka",
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
process.hltSiPixelClusters = cms.EDProducer( "SiPixelDigisClustersFromSoAAlpakaPhase1",
    src = cms.InputTag( "hltSiPixelClustersSoA" ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    produceDigis = cms.bool( False ),
    storeDigis = cms.bool( False )
)
process.hltSiPixelDigiErrors = cms.EDProducer( "SiPixelDigiErrorsFromSoAAlpaka",
    digiErrorSoASrc = cms.InputTag( "hltSiPixelClustersSoA" ),
    fmtErrorsSoASrc = cms.InputTag( "hltSiPixelClustersSoA" ),
    CablingMapLabel = cms.string( "" ),
    UsePhase1 = cms.bool( True ),
    ErrorList = cms.vint32( 29 ),
    UserErrorList = cms.vint32( 40 )
)
process.hltSiPixelRecHitsSoA = cms.EDProducer( "SiPixelRecHitAlpakaPhase1@alpaka",
    beamSpot = cms.InputTag( "hltOnlineBeamSpotDevice" ),
    src = cms.InputTag( "hltSiPixelClustersSoA" ),
    CPE = cms.string( "PixelCPEFastParams" ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltSiPixelRecHits = cms.EDProducer( "SiPixelRecHitFromSoAAlpakaPhase1",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    src = cms.InputTag( "hltSiPixelClusters" )
)
process.hltAlcaPixelClusterCounts = cms.EDProducer( "AlcaPCCEventProducer",
    pixelClusterLabel = cms.InputTag( "hltSiPixelClusters" ),
    trigstring = cms.untracked.string( "alcaPCCEvent" ),
    savePerROCInfo = cms.bool( True )
)
process.hltL1sZeroBias = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaLumiPixelsCountsZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sDQMPixelReconstruction = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreDQMPixelReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltOnlineBeamSpotDeviceSerialSync = cms.EDProducer( "alpaka_serial_sync::BeamSpotDeviceProducer",
    src = cms.InputTag( "hltOnlineBeamSpot" )
)
process.hltSiPixelClustersSoASerialSync = cms.EDProducer( "alpaka_serial_sync::SiPixelRawToClusterPhase1",
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
process.hltSiPixelClustersSerialSync = cms.EDProducer( "SiPixelDigisClustersFromSoAAlpakaPhase1",
    src = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    clusterThreshold_layer1 = cms.int32( 4000 ),
    clusterThreshold_otherLayers = cms.int32( 4000 ),
    produceDigis = cms.bool( False ),
    storeDigis = cms.bool( False )
)
process.hltSiPixelDigiErrorsSerialSync = cms.EDProducer( "SiPixelDigiErrorsFromSoAAlpaka",
    digiErrorSoASrc = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    fmtErrorsSoASrc = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    CablingMapLabel = cms.string( "" ),
    UsePhase1 = cms.bool( True ),
    ErrorList = cms.vint32( 29 ),
    UserErrorList = cms.vint32( 40 )
)
process.hltSiPixelRecHitsSoASerialSync = cms.EDProducer( "alpaka_serial_sync::SiPixelRecHitAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpotDeviceSerialSync" ),
    src = cms.InputTag( "hltSiPixelClustersSoASerialSync" ),
    CPE = cms.string( "PixelCPEFastParams" )
)
process.hltSiPixelRecHitsSerialSync = cms.EDProducer( "SiPixelRecHitFromSoAAlpakaPhase1",
    pixelRecHitSrc = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    src = cms.InputTag( "hltSiPixelClustersSerialSync" )
)
process.hltPixelTracksSoA = cms.EDProducer( "CAHitNtupletAlpakaPhase1@alpaka",
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
process.hltPixelTracks = cms.EDProducer( "PixelTrackProducerFromSoAAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    trackSrc = cms.InputTag( "hltPixelTracksSoA" ),
    pixelRecHitLegacySrc = cms.InputTag( "hltSiPixelRecHits" ),
    minNumberOfHits = cms.int32( 0 ),
    minQuality = cms.string( "loose" )
)
process.hltPixelVerticesSoA = cms.EDProducer( "PixelVertexProducerAlpakaPhase1@alpaka",
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
process.hltPixelVertices = cms.EDProducer( "PixelVertexProducerFromSoAAlpaka",
    TrackCollection = cms.InputTag( "hltPixelTracks" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    src = cms.InputTag( "hltPixelVerticesSoA" )
)
process.hltTrimmedPixelVertices = cms.EDProducer( "PixelVertexCollectionTrimmer",
    src = cms.InputTag( "hltPixelVertices" ),
    maxVtx = cms.uint32( 100 ),
    fractionSumPt2 = cms.double( 0.3 ),
    minSumPt2 = cms.double( 0.0 ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) )
)
process.hltPixelTracksSoASerialSync = cms.EDProducer( "alpaka_serial_sync::CAHitNtupletAlpakaPhase1",
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
process.hltPixelTracksSerialSync = cms.EDProducer( "PixelTrackProducerFromSoAAlpakaPhase1",
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    trackSrc = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    pixelRecHitLegacySrc = cms.InputTag( "hltSiPixelRecHitsSerialSync" ),
    minNumberOfHits = cms.int32( 0 ),
    minQuality = cms.string( "loose" )
)
process.hltPixelVerticesSoASerialSync = cms.EDProducer( "alpaka_serial_sync::PixelVertexProducerAlpakaPhase1",
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
process.hltPixelVerticesSerialSync = cms.EDProducer( "PixelVertexProducerFromSoAAlpaka",
    TrackCollection = cms.InputTag( "hltPixelTracksSerialSync" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    src = cms.InputTag( "hltPixelVerticesSoASerialSync" )
)
process.hltTrimmedPixelVerticesSerialSync = cms.EDProducer( "PixelVertexCollectionTrimmer",
    src = cms.InputTag( "hltPixelVerticesSerialSync" ),
    maxVtx = cms.uint32( 100 ),
    fractionSumPt2 = cms.double( 0.3 ),
    minSumPt2 = cms.double( 0.0 ),
    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) )
)
process.hltSiPixelRecHitsSoAMonitorCPU = cms.EDProducer( "SiPixelPhase1MonitorRecHitsSoAAlpaka",
    pixelHitsSrc = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    TopFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsCPU" )
)
process.hltSiPixelRecHitsSoAMonitorGPU = cms.EDProducer( "SiPixelPhase1MonitorRecHitsSoAAlpaka",
    pixelHitsSrc = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    TopFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsGPU" )
)
process.hltSiPixelRecHitsSoACompareGPUvsCPU = cms.EDProducer( "SiPixelPhase1CompareRecHits",
    pixelHitsReferenceSoA = cms.InputTag( "hltSiPixelRecHitsSoASerialSync" ),
    pixelHitsTargetSoA = cms.InputTag( "hltSiPixelRecHitsSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU" ),
    minD2cut = cms.double( 1.0E-4 )
)
process.hltPixelTracksSoAMonitorCPU = cms.EDProducer( "SiPixelPhase1MonitorTrackSoAAlpaka",
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackCPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" )
)
process.hltPixelTracksSoAMonitorGPU = cms.EDProducer( "SiPixelPhase1MonitorTrackSoAAlpaka",
    pixelTrackSrc = cms.InputTag( "hltPixelTracksSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackGPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" )
)
process.hltPixelTracksSoACompareGPUvsCPU = cms.EDProducer( "SiPixelPhase1CompareTracks",
    pixelTrackReferenceSoA = cms.InputTag( "hltPixelTracksSoASerialSync" ),
    pixelTrackTargetSoA = cms.InputTag( "hltPixelTracksSoA" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU" ),
    useQualityCut = cms.bool( True ),
    minQuality = cms.string( "loose" ),
    deltaR2cut = cms.double( 4.0E-4 )
)
process.hltPixelVerticesSoAMonitorCPU = cms.EDProducer( "SiPixelMonitorVertexSoAAlpaka",
    pixelVertexSrc = cms.InputTag( "hltPixelVerticesSoASerialSync" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexCPU" )
)
process.hltPixelVerticesSoAMonitorGPU = cms.EDProducer( "SiPixelMonitorVertexSoAAlpaka",
    pixelVertexSrc = cms.InputTag( "hltPixelVerticesSoA" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexGPU" )
)
process.hltPixelVerticesSoACompareGPUvsCPU = cms.EDProducer( "SiPixelCompareVertices",
    pixelVertexReferenceSoA = cms.InputTag( "hltPixelVerticesSoASerialSync" ),
    pixelVertexTargetSoA = cms.InputTag( "hltPixelVerticesSoA" ),
    beamSpotSrc = cms.InputTag( "hltOnlineBeamSpot" ),
    topFolderName = cms.string( "SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU" ),
    dzCut = cms.double( 1.0 )
)
process.hltL1sDQMEcalReconstruction = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreDQMEcalReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltEcalDigisSoASerialSync = cms.EDProducer( "alpaka_serial_sync::EcalRawToDigiPortable",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    FEDs = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    maxChannelsEB = cms.uint32( 61200 ),
    maxChannelsEE = cms.uint32( 14648 ),
    digisLabelEB = cms.string( "ebDigis" ),
    digisLabelEE = cms.string( "eeDigis" )
)
process.hltEcalDigisSerialSync = cms.EDProducer( "EcalDigisFromPortableProducer",
    digisInLabelEB = cms.InputTag( 'hltEcalDigisSoASerialSync','ebDigis' ),
    digisInLabelEE = cms.InputTag( 'hltEcalDigisSoASerialSync','eeDigis' ),
    digisOutLabelEB = cms.string( "ebDigis" ),
    digisOutLabelEE = cms.string( "eeDigis" ),
    produceDummyIntegrityCollections = cms.bool( False )
)
process.hltEcalUncalibRecHitSoASerialSync = cms.EDProducer( "alpaka_serial_sync::EcalUncalibRecHitProducerPortable",
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
process.hltEcalUncalibRecHitSerialSync = cms.EDProducer( "EcalUncalibRecHitSoAToLegacy",
    inputCollectionEB = cms.InputTag( 'hltEcalUncalibRecHitSoASerialSync','EcalUncalibRecHitsEB' ),
    outputLabelEB = cms.string( "EcalUncalibRecHitsEB" ),
    isPhase2 = cms.bool( False ),
    inputCollectionEE = cms.InputTag( 'hltEcalUncalibRecHitSoASerialSync','EcalUncalibRecHitsEE' ),
    outputLabelEE = cms.string( "EcalUncalibRecHitsEE" )
)
process.hltEcalRecHitSerialSync = cms.EDProducer( "EcalRecHitProducer",
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
process.hltL1sDQMHcalReconstruction = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreDQMHcalReconstruction = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
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
process.hltHcalDigisSoA = cms.EDProducer( "HcalDigisSoAProducer@alpaka",
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
process.hltHbheRecoSoA = cms.EDProducer( "HBHERecHitProducerPortable@alpaka",
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
process.hltHbhereco = cms.EDProducer( "HcalRecHitSoAToLegacy",
    src = cms.InputTag( "hltHbheRecoSoA" )
)
process.hltHfprereco = cms.EDProducer( "HFPreReconstructor",
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    forceSOI = cms.int32( -1 ),
    soiShift = cms.int32( 0 ),
    dropZSmarkedPassed = cms.bool( True ),
    tsFromDB = cms.bool( False ),
    sumAllTimeSlices = cms.bool( False )
)
process.hltHfreco = cms.EDProducer( "HFPhase1Reconstructor",
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
process.hltHoreco = cms.EDProducer( "HcalHitReconstructor",
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
process.hltHcalDigisSoASerialSync = cms.EDProducer( "alpaka_serial_sync::HcalDigisSoAProducer",
    hbheDigisLabel = cms.InputTag( "hltHcalDigis" ),
    qie11DigiLabel = cms.InputTag( "hltHcalDigis" ),
    digisLabelF01HE = cms.string( "f01HEDigis" ),
    digisLabelF5HB = cms.string( "f5HBDigis" ),
    digisLabelF3HB = cms.string( "f3HBDigis" ),
    maxChannelsF01HE = cms.uint32( 10000 ),
    maxChannelsF5HB = cms.uint32( 10000 ),
    maxChannelsF3HB = cms.uint32( 10000 )
)
process.hltHbheRecoSoASerialSync = cms.EDProducer( "alpaka_serial_sync::HBHERecHitProducerPortable",
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
process.hltHbherecoSerialSync = cms.EDProducer( "HcalRecHitSoAToLegacy",
    src = cms.InputTag( "hltHbheRecoSoASerialSync" )
)
process.hltParticleFlowRecHitHBHESoA = cms.EDProducer( "PFRecHitSoAProducerHCAL@alpaka",
    producers = cms.VPSet( 
      cms.PSet(  src = cms.InputTag( "hltHbheRecoSoA" ),
        params = cms.ESInputTag( "hltESPPFRecHitHCALParams","" )
      )
    ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    synchronise = cms.untracked.bool( False ),
    alpaka = cms.untracked.PSet(  backend = cms.untracked.string( "" ) )
)
process.hltParticleFlowRecHitHBHE = cms.EDProducer( "LegacyPFRecHitProducer",
    src = cms.InputTag( "hltParticleFlowRecHitHBHESoA" )
)
process.hltParticleFlowClusterHBHESoA = cms.EDProducer( "PFClusterSoAProducer@alpaka",
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
process.hltParticleFlowClusterHBHE = cms.EDProducer( "LegacyPFClusterProducer",
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
process.hltParticleFlowClusterHCAL = cms.EDProducer( "PFMultiDepthClusterProducer",
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
process.hltParticleFlowRecHitHBHESoASerialSync = cms.EDProducer( "alpaka_serial_sync::PFRecHitSoAProducerHCAL",
    producers = cms.VPSet( 
      cms.PSet(  src = cms.InputTag( "hltHbheRecoSoASerialSync" ),
        params = cms.ESInputTag( "hltESPPFRecHitHCALParams","" )
      )
    ),
    topology = cms.ESInputTag( "hltESPPFRecHitHCALTopology","" ),
    synchronise = cms.untracked.bool( False )
)
process.hltParticleFlowRecHitHBHESerialSync = cms.EDProducer( "LegacyPFRecHitProducer",
    src = cms.InputTag( "hltParticleFlowRecHitHBHESoASerialSync" )
)
process.hltParticleFlowClusterHBHESoASerialSync = cms.EDProducer( "alpaka_serial_sync::PFClusterSoAProducer",
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
process.hltParticleFlowClusterHBHESerialSync = cms.EDProducer( "LegacyPFClusterProducer",
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
process.hltParticleFlowClusterHCALSerialSync = cms.EDProducer( "PFMultiDepthClusterProducer",
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
process.hltPreDQMRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDQMZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDSTZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltFEDSelectorL1uGTTest = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1405 )
)
process.hltPreDSTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltCalibrationEventsFilter = cms.EDFilter( "HLTTriggerTypeFilter",
    SelectedTriggerType = cms.int32( 2 )
)
process.hltPreEcalCalibration = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltEcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 1024 )
)
process.hltPreHcalCalibration = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltHcalCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 1024, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199 )
)
process.hltL1EventNumberNZS = cms.EDFilter( "HLTL1NumberFilter",
    rawInput = cms.InputTag( "rawDataCollector" ),
    period = cms.uint32( 4096 ),
    invert = cms.bool( False ),
    fedId = cms.int32( 1024 ),
    useTCDSEventNumber = cms.bool( False )
)
process.hltL1sHcalNZS = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreHcalNZS = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEGorSingleorDoubleMu = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreHcalPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1EventNumberL1Fat = cms.EDFilter( "HLTL1NumberFilter",
    rawInput = cms.InputTag( "rawDataCollector" ),
    period = cms.uint32( 107 ),
    invert = cms.bool( False ),
    fedId = cms.int32( 1024 ),
    useTCDSEventNumber = cms.bool( True )
)
process.hltPrePhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreZeroBiasAlignment = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreZeroBiasBeamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
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
process.hltAK4CaloJetsPF = cms.EDProducer( "FastjetJetProducer",
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
process.hltAK4CaloJetsPFEt5 = cms.EDFilter( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltAK4CaloJetsPF" ),
    filter = cms.bool( False ),
    etMin = cms.double( 5.0 )
)
process.hltMuonDTDigis = cms.EDProducer( "DTuROSRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    debug = cms.untracked.bool( False )
)
process.hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
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
process.hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
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
process.hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
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
process.hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
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
process.hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
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
process.hltMuonRPCDigisCPPF = cms.EDProducer( "RPCAMCRawToDigi",
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
process.hltOmtfDigis = cms.EDProducer( "OmtfUnpacker",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    skipRpc = cms.bool( False ),
    skipCsc = cms.bool( False ),
    skipDt = cms.bool( False ),
    skipMuon = cms.bool( False ),
    useRpcConnectionFile = cms.bool( False ),
    rpcConnectionFile = cms.string( "" ),
    outputTag = cms.string( "" )
)
process.hltMuonRPCDigisTwinMux = cms.EDProducer( "RPCTwinMuxRawToDigi",
    inputTag = cms.InputTag( "rawDataCollector" ),
    calculateCRC = cms.bool( True ),
    fillCounters = cms.bool( True ),
    bxMin = cms.int32( -2 ),
    bxMax = cms.int32( 2 )
)
process.hltMuonRPCDigis = cms.EDProducer( "RPCDigiMerger",
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
process.hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    maskSource = cms.string( "File" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    deadSource = cms.string( "File" ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" )
)
process.hltMuonGEMDigis = cms.EDProducer( "GEMRawToDigiModule",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    useDBEMap = cms.bool( True ),
    keepDAQStatus = cms.bool( False ),
    readMultiBX = cms.bool( False ),
    ge21Off = cms.bool( True ),
    fedIdStart = cms.uint32( 1467 ),
    fedIdEnd = cms.uint32( 1478 )
)
process.hltGemRecHits = cms.EDProducer( "GEMRecHitProducer",
    recAlgoConfig = cms.PSet(  ),
    recAlgo = cms.string( "GEMRecHitStandardAlgo" ),
    gemDigiLabel = cms.InputTag( "hltMuonGEMDigis" ),
    applyMasking = cms.bool( True ),
    ge21Off = cms.bool( False )
)
process.hltGemSegments = cms.EDProducer( "GEMSegmentProducer",
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
process.hltL2OfflineMuonSeeds = cms.EDProducer( "MuonSeedGenerator",
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
process.hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGeneratorFromL1T",
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
process.hltL2Muons = cms.EDProducer( "L2MuonProducer",
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
process.hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
process.hltSiStripExcludedFEDListProducer = cms.EDProducer( "SiStripExcludedFEDListProducer",
    ProductLabel = cms.InputTag( "rawDataCollector" )
)
process.hltSiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
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
process.hltMeasurementTrackerEvent = cms.EDProducer( "MeasurementTrackerEventProducer",
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
process.hltIterL3OISeedsFromL2Muons = cms.EDProducer( "TSGForOIDNN",
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
process.hltIterL3OITrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltIterL3OIMuCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltIterL3OIMuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltIterL3OIMuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIterL3OIMuCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIterL3OIMuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIterL3OIMuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltL3MuonsIterL3OI = cms.EDProducer( "L3MuonProducer",
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
process.hltIterL3OIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI' )
)
process.hltIterL3OIL3Muons = cms.EDProducer( "L3TrackCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI' )
)
process.hltIterL3OIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltIterL3OIL3Muons" ),
    InputLinksObjects = cms.InputTag( "hltIterL3OIL3MuonsLinksCombination" ),
    MuonPtOption = cms.string( "Tracker" )
)
process.hltL2SelectorForL3IO = cms.EDProducer( "HLTMuonL2SelectorForL3IO",
    l2Src = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    l3OISrc = cms.InputTag( "hltIterL3OIL3MuonCandidates" ),
    InputLinks = cms.InputTag( "hltIterL3OIL3MuonsLinksCombination" ),
    applyL3Filters = cms.bool( False ),
    MinNhits = cms.int32( 1 ),
    MaxNormalizedChi2 = cms.double( 20.0 ),
    MinNmuonHits = cms.int32( 1 ),
    MaxPtDifference = cms.double( 0.3 )
)
process.hltIterL3MuonPixelTracksTrackingRegions = cms.EDProducer( "MuonTrackingRegionByPtEDProducer",
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
process.hltPixelTracksInRegionL2 = cms.EDProducer( "TrackSelectorByRegion",
    tracks = cms.InputTag( "hltPixelTracks" ),
    regions = cms.InputTag( "hltIterL3MuonPixelTracksTrackingRegions" ),
    produceTrackCollection = cms.bool( True ),
    produceMask = cms.bool( False )
)
process.hltIter0IterL3MuonPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
process.hltIter0IterL3MuonPixelSeedsFromPixelTracksFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
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
process.hltIter0IterL3MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltIter0IterL3MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltIter0IterL3MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltIter0IterL3MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0IterL3MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0IterL3MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0IterL3MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltL3MuonsIterL3IO = cms.EDProducer( "L3MuonProducer",
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
process.hltIterL3MuonsFromL2LinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
    labels = cms.VInputTag( 'hltL3MuonsIterL3OI','hltL3MuonsIterL3IO' )
)
process.hltL1MuonsPt0 = cms.EDProducer( "HLTL1TMuonSelector",
    InputObjects = cms.InputTag( 'hltGtStage2Digis','Muon' ),
    L1MinPt = cms.double( -1.0 ),
    L1MaxEta = cms.double( 5.0 ),
    L1MinQuality = cms.uint32( 7 ),
    CentralBxOnly = cms.bool( True )
)
process.hltIterL3FromL1MuonPixelTracksTrackingRegions = cms.EDProducer( "L1MuonSeededTrackingRegionsEDProducer",
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
process.hltPixelTracksInRegionL1 = cms.EDProducer( "TrackSelectorByRegion",
    tracks = cms.InputTag( "hltPixelTracks" ),
    regions = cms.InputTag( "hltIterL3FromL1MuonPixelTracksTrackingRegions" ),
    produceTrackCollection = cms.bool( True ),
    produceMask = cms.bool( False )
)
process.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
process.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracksFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
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
process.hltIter0IterL3FromL1MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltIter0IterL3FromL1MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltIter0IterL3FromL1MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltIter0IterL3FromL1MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0IterL3FromL1MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0IterL3FromL1MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0IterL3FromL1MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltIter3IterL3FromL1MuonClustersRefRemoval = cms.EDProducer( "TrackClusterRemover",
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
process.hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    src = cms.InputTag( "hltMeasurementTrackerEvent" ),
    clustersToSkip = cms.InputTag( "hltIter3IterL3FromL1MuonClustersRefRemoval" ),
    phase2clustersToSkip = cms.InputTag( "" )
)
process.hltIter3IterL3FromL1MuonPixelLayersAndRegions = cms.EDProducer( "PixelInactiveAreaTrackingRegionsSeedingLayersProducer",
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
process.hltIter3IterL3FromL1MuonTrackingRegions = cms.EDProducer( "L1MuonSeededTrackingRegionsEDProducer",
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
process.hltIter3IterL3FromL1MuonPixelClusterCheck = cms.EDProducer( "ClusterCheckerEDProducer",
    doClusterCheck = cms.bool( False ),
    MaxNumberOfStripClusters = cms.uint32( 50000 ),
    ClusterCollectionLabel = cms.InputTag( "hltMeasurementTrackerEvent" ),
    MaxNumberOfPixelClusters = cms.uint32( 40000 ),
    PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
    cut = cms.string( "" ),
    DontCountDetsAboveNClusters = cms.uint32( 0 ),
    silentClusterCheck = cms.untracked.bool( False )
)
process.hltIter3IterL3FromL1MuonPixelHitDoublets = cms.EDProducer( "HitPairEDProducer",
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
process.hltIter3IterL3FromL1MuonPixelSeeds = cms.EDProducer( "SeedCreatorFromRegionConsecutiveHitsEDProducer",
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
process.hltIter3IterL3FromL1MuonPixelSeedsFiltered = cms.EDProducer( "MuonHLTSeedMVAClassifier",
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
process.hltIter3IterL3FromL1MuonCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltIter3IterL3FromL1MuonCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltIter3IterL3FromL1MuonTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltIter3IterL3FromL1MuonTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter3IterL3FromL1MuonCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter3IterL3FromL1MuonTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter3IterL3FromL1MuonTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltIter03IterL3FromL1MuonMerged = cms.EDProducer( "TrackListMerger",
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
process.hltIterL3MuonMerged = cms.EDProducer( "TrackListMerger",
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
process.hltIterL3MuonAndMuonFromL1Merged = cms.EDProducer( "TrackListMerger",
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
process.hltIterL3GlbMuon = cms.EDProducer( "L3MuonProducer",
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
process.hltIterL3MuonsNoID = cms.EDProducer( "MuonIdProducer",
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
process.hltIterL3Muons = cms.EDProducer( "MuonIDFilterProducerForHLT",
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
process.hltL3MuonsIterL3Links = cms.EDProducer( "MuonLinksProducer",
    inputCollection = cms.InputTag( "hltIterL3Muons" )
)
process.hltIterL3MuonTracks = cms.EDProducer( "HLTMuonTrackSelector",
    track = cms.InputTag( "hltIterL3MuonAndMuonFromL1Merged" ),
    muon = cms.InputTag( "hltIterL3Muons" ),
    originalMVAVals = cms.InputTag( "none" ),
    copyMVA = cms.bool( False ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltIterL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducerFromMuons",
    InputObjects = cms.InputTag( "hltIterL3Muons" ),
    DisplacedReconstruction = cms.bool( False )
)
process.hltIter0PFLowPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
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
process.hltIter0PFlowCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltIter0PFlowCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltIter0PFlowTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltIter0PFlowTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltIter0PFlowTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltIter0PFlowTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltDoubletRecoveryClustersRefRemoval = cms.EDProducer( "TrackClusterRemover",
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
process.hltDoubletRecoveryMaskedMeasurementTrackerEvent = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
    src = cms.InputTag( "hltMeasurementTrackerEvent" ),
    clustersToSkip = cms.InputTag( "hltDoubletRecoveryClustersRefRemoval" ),
    phase2clustersToSkip = cms.InputTag( "" )
)
process.hltDoubletRecoveryPixelLayersAndRegions = cms.EDProducer( "PixelInactiveAreaTrackingRegionsSeedingLayersProducer",
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
process.hltDoubletRecoveryPFlowPixelClusterCheck = cms.EDProducer( "ClusterCheckerEDProducer",
    doClusterCheck = cms.bool( False ),
    MaxNumberOfStripClusters = cms.uint32( 50000 ),
    ClusterCollectionLabel = cms.InputTag( "hltMeasurementTrackerEvent" ),
    MaxNumberOfPixelClusters = cms.uint32( 40000 ),
    PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
    cut = cms.string( "" ),
    DontCountDetsAboveNClusters = cms.uint32( 0 ),
    silentClusterCheck = cms.untracked.bool( False )
)
process.hltDoubletRecoveryPFlowPixelHitDoublets = cms.EDProducer( "HitPairEDProducer",
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
process.hltDoubletRecoveryPFlowPixelSeeds = cms.EDProducer( "SeedCreatorFromRegionConsecutiveHitsEDProducer",
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
process.hltDoubletRecoveryPFlowCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltDoubletRecoveryPFlowCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
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
process.hltDoubletRecoveryPFlowTrackCutClassifier = cms.EDProducer( "TrackCutClassifier",
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
process.hltDoubletRecoveryPFlowTrackSelectionHighPurity = cms.EDProducer( "TrackCollectionFilterCloner",
    originalSource = cms.InputTag( "hltDoubletRecoveryPFlowCtfWithMaterialTracks" ),
    originalMVAVals = cms.InputTag( 'hltDoubletRecoveryPFlowTrackCutClassifier','MVAValues' ),
    originalQualVals = cms.InputTag( 'hltDoubletRecoveryPFlowTrackCutClassifier','QualityMasks' ),
    minQuality = cms.string( "highPurity" ),
    copyExtras = cms.untracked.bool( True ),
    copyTrajectories = cms.untracked.bool( False )
)
process.hltMergedTracks = cms.EDProducer( "TrackListMerger",
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
process.hltPFMuonMerging = cms.EDProducer( "TrackListMerger",
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
process.hltVerticesPF = cms.EDProducer( "PrimaryVertexProducer",
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
process.hltVerticesPFSelector = cms.EDFilter( "PrimaryVertexObjectFilter",
    src = cms.InputTag( "hltVerticesPF" ),
    filterParams = cms.PSet( 
      maxZ = cms.double( 24.0 ),
      minNdof = cms.double( 4.0 ),
      maxRho = cms.double( 2.0 )
    ),
    filter = cms.bool( False )
)
process.hltVerticesPFFilter = cms.EDFilter( "VertexSelector",
    src = cms.InputTag( "hltVerticesPFSelector" ),
    cut = cms.string( "!isFake" ),
    filter = cms.bool( True ),
    throwOnMissing = cms.untracked.bool( True )
)
process.hltFEDSelectorOnlineMetaData = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 1022 )
)
process.hltL1sIsolatedBunch = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreZeroBiasIsolatedBunches = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1ZeroBiasFirstBunchAfterTrain = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreZeroBiasFirstBXAfterTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1ZeroBiasFirstCollisionAfterAbortGap = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreZeroBiasFirstCollisionAfterAbortGap = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1ZeroBiasFirstCollisionInTrainNOTFirstCollisionInOrbit = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreZeroBiasFirstCollisionInTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1ZeroBiasLastBunchInTrain = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreZeroBiasLastCollisionInTrain = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTTForBeamSpot = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreHT300Beamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltAK4CaloJets = cms.EDProducer( "FastjetJetProducer",
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
process.hltAK4CaloJetsIDPassed = cms.EDProducer( "HLTCaloJetIDProducer",
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
process.hltFixedGridRhoFastjetAllCalo = cms.EDProducer( "FixedGridRhoProducerFastjet",
    maxRapidity = cms.double( 5.0 ),
    gridSpacing = cms.double( 0.55 ),
    pfCandidatesTag = cms.InputTag( "hltTowerMakerForAll" )
)
process.hltAK4CaloFastJetCorrector = cms.EDProducer( "L1FastjetCorrectorProducer",
    level = cms.string( "L1FastJet" ),
    algorithm = cms.string( "AK4CaloHLT" ),
    srcRho = cms.InputTag( "hltFixedGridRhoFastjetAllCalo" )
)
process.hltAK4CaloRelativeCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L2Relative" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
process.hltAK4CaloAbsoluteCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L3Absolute" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
process.hltAK4CaloResidualCorrector = cms.EDProducer( "LXXXCorrectorProducer",
    level = cms.string( "L2L3Residual" ),
    algorithm = cms.string( "AK4CaloHLT" )
)
process.hltAK4CaloCorrector = cms.EDProducer( "ChainedJetCorrectorProducer",
    correctors = cms.VInputTag( 'hltAK4CaloFastJetCorrector','hltAK4CaloRelativeCorrector','hltAK4CaloAbsoluteCorrector','hltAK4CaloResidualCorrector' )
)
process.hltAK4CaloJetsCorrected = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltAK4CaloJets" ),
    correctors = cms.VInputTag( 'hltAK4CaloCorrector' ),
    verbose = cms.untracked.bool( False )
)
process.hltAK4CaloJetsCorrectedIDPassed = cms.EDProducer( "CorrectedCaloJetProducer",
    src = cms.InputTag( "hltAK4CaloJetsIDPassed" ),
    correctors = cms.VInputTag( 'hltAK4CaloCorrector' ),
    verbose = cms.untracked.bool( False )
)
process.hltHtMht = cms.EDProducer( "HLTHtMhtProducer",
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
process.hltHT300 = cms.EDFilter( "HLTHtMhtFilter",
    saveTags = cms.bool( True ),
    htLabels = cms.VInputTag( 'hltHtMht' ),
    mhtLabels = cms.VInputTag( 'hltHtMht' ),
    minHt = cms.vdouble( 300.0 ),
    minMht = cms.vdouble( 0.0 ),
    minMeff = cms.vdouble( 0.0 ),
    meffSlope = cms.vdouble( 1.0 )
)
process.hltL1sV0SingleJet3OR = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreIsoTrackHB = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPixelTracksQuadruplets = cms.EDProducer( "TrackWithVertexSelector",
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
process.hltIsolPixelTrackProdHB = cms.EDProducer( "IsolatedPixelTrackCandidateL1TProducer",
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
process.hltIsolPixelTrackL2FilterHB = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
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
process.hltIsolEcalPixelTrackProdHB = cms.EDProducer( "IsolatedEcalPixelTrackCandidateProducer",
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
process.hltEcalIsolPixelTrackL2FilterHB = cms.EDFilter( "HLTEcalPixelIsolTrackFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolEcalPixelTrackProdHB" ),
    MaxEnergyInEB = cms.double( 2.0 ),
    MaxEnergyInEE = cms.double( 4.0 ),
    MaxEnergyOutEB = cms.double( 1.2 ),
    MaxEnergyOutEE = cms.double( 2.0 ),
    NMaxTrackCandidates = cms.int32( 10 ),
    DropMultiL2Event = cms.bool( False )
)
process.hltHcalITIPTCorrectorHB = cms.EDProducer( "IPTCorrector",
    corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHB" ),
    associationCone = cms.double( 0.2 )
)
process.hltIsolPixelTrackL3FilterHB = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
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
process.hltPreIsoTrackHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltIsolPixelTrackProdHE = cms.EDProducer( "IsolatedPixelTrackCandidateL1TProducer",
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
process.hltIsolPixelTrackL2FilterHE = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
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
process.hltIsolEcalPixelTrackProdHE = cms.EDProducer( "IsolatedEcalPixelTrackCandidateProducer",
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
process.hltEcalIsolPixelTrackL2FilterHE = cms.EDFilter( "HLTEcalPixelIsolTrackFilter",
    saveTags = cms.bool( True ),
    candTag = cms.InputTag( "hltIsolEcalPixelTrackProdHE" ),
    MaxEnergyInEB = cms.double( 2.0 ),
    MaxEnergyInEE = cms.double( 4.0 ),
    MaxEnergyOutEB = cms.double( 1.2 ),
    MaxEnergyOutEE = cms.double( 2.0 ),
    NMaxTrackCandidates = cms.int32( 10 ),
    DropMultiL2Event = cms.bool( False )
)
process.hltHcalITIPTCorrectorHE = cms.EDProducer( "IPTCorrector",
    corTracksLabel = cms.InputTag( "hltIter0PFlowCtfWithMaterialTracks" ),
    filterLabel = cms.InputTag( "hltIsolPixelTrackL2FilterHE" ),
    associationCone = cms.double( 0.2 )
)
process.hltIsolPixelTrackL3FilterHE = cms.EDFilter( "HLTPixelIsolTrackL1TFilter",
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
process.hltL1sSingleMuCosmics = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMuCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1MuCosmicsL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL2Mu10NoVertexNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltDt4DSegmentsMeanTimer = cms.EDProducer( "DTRecSegment4DProducer",
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
process.hltL2CosmicOfflineMuonSeeds = cms.EDProducer( "CosmicMuonSeedGenerator",
    EnableDTMeasurement = cms.bool( True ),
    EnableCSCMeasurement = cms.bool( True ),
    DTRecSegmentLabel = cms.InputTag( "hltDt4DSegmentsMeanTimer" ),
    CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
    MaxSeeds = cms.int32( 1000 ),
    MaxDTChi2 = cms.double( 300.0 ),
    MaxCSCChi2 = cms.double( 300.0 ),
    ForcePointDown = cms.bool( False )
)
process.hltL2CosmicMuonSeeds = cms.EDProducer( "L2MuonSeedGeneratorFromL1T",
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
process.hltL2CosmicMuons = cms.EDProducer( "L2MuonProducer",
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
process.hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL2CosmicMuons" )
)
process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltL1sSingleMuOpenNotBptxOR = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL2Mu10NoVertexNoBPTX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sMuOpenNotBptxORL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL2fL1sMuOpenNotBptxORL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltPreL2Mu45NoVertex3StaNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered45Sta3 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltPreL2Mu40NoVertex3StaNoBPTX3BX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered40Sta3 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltL1sCDC = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreCDCL2cosmic10er1p0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sCDCL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations10er1p0 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltPreCDCL2cosmic5p5er1p0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations5p5er1p0 = cms.EDFilter( "HLTMuonL2FromL1TPreFilter",
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
process.hltPrePPSMaxTracksPerArm1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltCTPPSPixelDigis = cms.EDProducer( "CTPPSPixelRawToDigi",
    isRun3 = cms.bool( True ),
    includeErrors = cms.bool( True ),
    inputLabel = cms.InputTag( "rawDataCollector" ),
    mappingLabel = cms.string( "RPix" )
)
process.hltCTPPSPixelClusters = cms.EDProducer( "CTPPSPixelClusterProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    tag = cms.InputTag( "hltCTPPSPixelDigis" ),
    SeedADCThreshold = cms.int32( 2 ),
    ADCThreshold = cms.int32( 2 ),
    ElectronADCGain = cms.double( 135.0 ),
    VCaltoElectronGain = cms.int32( 50 ),
    VCaltoElectronOffset = cms.int32( -411 ),
    doSingleCalibration = cms.bool( False )
)
process.hltCTPPSPixelRecHits = cms.EDProducer( "CTPPSPixelRecHitProducer",
    RPixVerbosity = cms.untracked.int32( 0 ),
    RPixClusterTag = cms.InputTag( "hltCTPPSPixelClusters" )
)
process.hltCTPPSPixelLocalTracks = cms.EDProducer( "CTPPSPixelLocalTrackProducer",
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
process.hltPPSExpCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 1 ),
    do_express = cms.bool( True ),
    triggerType = cms.int32( 91 )
)
process.hltPPSCalibrationRaw = cms.EDProducer( "EvFFEDSelector",
    inputTag = cms.InputTag( "rawDataCollector" ),
    fedList = cms.vuint32( 579, 581, 582, 583, 586, 587, 588, 589, 1462, 1463 )
)
process.hltPrePPSMaxTracksPerRP4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPPSPrCalFilter = cms.EDFilter( "HLTPPSCalFilter",
    pixelLocalTrackInputTag = cms.InputTag( "hltCTPPSPixelLocalTracks" ),
    minTracks = cms.int32( 1 ),
    maxTracks = cms.int32( 4 ),
    do_express = cms.bool( False ),
    triggerType = cms.int32( 91 )
)
process.hltPrePPSRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreSpecialHLTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreAlCaLumiPixelsCountsRandomHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sZeroBiasOrZeroBiasCopy = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaLumiPixelsCountsZeroBiasVdM = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreAlCaLumiPixelsCountsZeroBiasGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltAlcaPixelClusterCountsGated = cms.EDProducer( "AlcaPCCEventProducer",
    pixelClusterLabel = cms.InputTag( "hltSiPixelClusters" ),
    trigstring = cms.untracked.string( "alcaPCCEvent" ),
    savePerROCInfo = cms.bool( False )
)
process.hltL1sSingleMuOpen = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMuOpen = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1MuOpenL1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltPreL1SingleMuOpenDT = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1MuOpenL1FilteredDT = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sSingleMu3 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMu3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sMu3L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sSingleMu5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMu5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sMu5L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sSingleMu7 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMu7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1fL1sMu7L1Filtered0 = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sDoubleMu0 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1DoubleMu0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDoubleMu0L1Filtered = cms.EDFilter( "HLTMuonL1TFilter",
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
process.hltL1sSingleJet8erHE = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet8erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet10erHE = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet10erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet12erHE = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet12erHE = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet35 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet35 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet200 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet200 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG8er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG8er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG10er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG10er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG15er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG15er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG26er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG26er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG28er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG28er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG28er2p1 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG28er2p1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG28er1p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG28er1p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG34er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG34er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG36er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG36er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG38er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG38er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG40er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG40er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG42er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG42er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleEG45er2p5 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG45er2p5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1SingleEG50 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleEG50 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet60 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet60 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet90 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet90 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet120 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet120 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleJet180 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleJet180 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT120er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT120er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT160er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT160er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT200er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT200er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT255er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT255er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT280er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT280er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT320er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT320er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT360er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT360er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT400er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT400er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sHTT450er = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1HTT450er = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sETM120 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1ETM120 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sETM150 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1ETM150 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sEXTHCALLaserMon1 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1EXTHCALLaserMon1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sEXTHCALLaserMon4 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1EXTHCALLaserMon4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sL1MinimumBiasHF0ANDBptxAND = cms.EDFilter( "HLTL1TSeed",
    saveTags = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MinimumBiasHF0_AND_BptxAND" ),
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
process.hltPreL1MinimumBiasHF0ANDBptxAND = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sMuShowerOneNominal = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreCscClusterCosmic = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltCSCrechitClusters = cms.EDProducer( "CSCrechitClusterProducer",
    nRechitMin = cms.int32( 50 ),
    rParam = cms.double( 0.4 ),
    nStationThres = cms.int32( 10 ),
    recHitLabel = cms.InputTag( "hltCsc2DRecHits" )
)
process.hltCscClusterCosmic = cms.EDFilter( "HLTMuonRecHitClusterFilter",
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
process.hltL1sHTTForBeamSpotHT60 = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreHT60Beamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltHT60 = cms.EDFilter( "HLTHtMhtFilter",
    saveTags = cms.bool( True ),
    htLabels = cms.VInputTag( 'hltHtMht' ),
    mhtLabels = cms.VInputTag( 'hltHtMht' ),
    minHt = cms.vdouble( 60.0 ),
    minMht = cms.vdouble( 0.0 ),
    minMeff = cms.vdouble( 0.0 ),
    meffSlope = cms.vdouble( 1.0 )
)
process.hltL1sZeroBiasOrMinBias = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreHT300BeamspotPixelClustersWP2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPixelActivityFilterWP2 = cms.EDFilter( "HLTPixelActivityFilter",
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
process.hltPrePixelClustersWP2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPrePixelClustersWP2HighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPrePixelClustersWP1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPixelActivityFilterWP1 = cms.EDFilter( "HLTPixelActivityFilter",
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
process.hltL1sBptxOR = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreBptxOR = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL1sSingleMuCosmicsEMTF = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreL1SingleMuCosmicsEMTF = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreL1SingleMuCosmicsCosmicTracking = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltFullSiStripRawToClustersFacility = cms.EDProducer( "SiStripClusterizerFromRaw",
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
process.hltFullMeasurementTrackerEvent = cms.EDProducer( "MeasurementTrackerEventProducer",
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
process.hltGlobalSiStripMatchedRecHitsFull = cms.EDProducer( "SiStripRecHitConverter",
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
process.hltSimpleCosmicBONSeedingLayers = cms.EDProducer( "SeedingLayersEDProducer",
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
process.hltSimpleCosmicBONSeeds = cms.EDProducer( "SimpleCosmicBONSeeder",
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
process.hltCombinatorialcosmicseedingtripletsP5 = cms.EDProducer( "SeedingLayersEDProducer",
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
process.hltCombinatorialcosmicseedingpairsTOBP5 = cms.EDProducer( "SeedingLayersEDProducer",
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
process.hltCombinatorialcosmicseedingpairsTECposP5 = cms.EDProducer( "SeedingLayersEDProducer",
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
process.hltCombinatorialcosmicseedingpairsTECnegP5 = cms.EDProducer( "SeedingLayersEDProducer",
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
process.hltCombinatorialcosmicseedfinderP5 = cms.EDProducer( "CtfSpecialSeedGenerator",
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
process.hltCombinedP5SeedsForCTF = cms.EDProducer( "SeedCombiner",
    seedCollections = cms.VInputTag( 'hltCombinatorialcosmicseedfinderP5','hltSimpleCosmicBONSeeds' ),
    clusterRemovalInfos = cms.VInputTag(  )
)
process.hltCkfTrackCandidatesP5 = cms.EDProducer( "CkfTrackCandidateMaker",
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
process.hltCtfWithMaterialTracksCosmics = cms.EDProducer( "TrackProducer",
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
process.hltCtfWithMaterialTracksP5 = cms.EDProducer( "CosmicTrackSelector",
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
process.hltCtfWithMaterialTracksP5TrackCountFilter = cms.EDFilter( "TrackCountFilter",
    src = cms.InputTag( "hltCtfWithMaterialTracksP5" ),
    minNumber = cms.uint32( 1 )
)
process.hltPreL1SingleMuCosmicsPointingCosmicTracking = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltL2CosmicsMuonTrackerPointingFilter = cms.EDFilter( "HLTMuonPointingFilter",
    SALabel = cms.InputTag( 'hltL2CosmicMuons','UpdatedAtVtx' ),
    PropagatorName = cms.string( "SteppingHelixPropagatorAny" ),
    radius = cms.double( 90.0 ),
    maxZ = cms.double( 280.0 ),
    PixHits = cms.uint32( 0 ),
    TkLayers = cms.uint32( 0 ),
    MuonHits = cms.uint32( 0 )
)
process.hltPreL1FatEvents = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreRandomHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreZeroBiasHighRate = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreZeroBiasGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltBXGateFilter = cms.EDFilter( "BunchCrossingFilter",
    bunches = cms.vuint32( 2 )
)
process.hltL1sZeroBiasCopyOrAlwaysTrue = cms.EDFilter( "HLTL1TSeed",
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
process.hltPreSpecialZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    throw = cms.bool( False ),
    processName = cms.string( "@" ),
    moduleLabelPatternsToMatch = cms.vstring( 'hlt*' ),
    moduleLabelPatternsToSkip = cms.vstring(  )
)
process.hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
process.hltL1TGlobalSummary = cms.EDAnalyzer( "L1TGlobalSummary",
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
process.hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','@currentProcess' ),
    reportBy = cms.untracked.string( "job" ),
    resetBy = cms.untracked.string( "never" ),
    serviceBy = cms.untracked.string( "never" ),
    ReferencePath = cms.untracked.string( "HLTriggerFinalPath" ),
    ReferenceRate = cms.untracked.double( 100.0 )
)
process.hltDatasetAlCaLumiPixelsCountsExpress = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4 / 100',
      'AlCa_LumiPixelsCounts_Random_v10' )
)
process.hltPreDatasetAlCaLumiPixelsCountsExpress = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaLumiPixelsCountsPrompt = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_Random_v10',
      'AlCa_LumiPixelsCounts_ZeroBias_v12' )
)
process.hltPreDatasetAlCaLumiPixelsCountsPrompt = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaLumiPixelsCountsPromptHighRate = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_RandomHighRate_v4',
      'AlCa_LumiPixelsCounts_ZeroBiasVdM_v4' )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaLumiPixelsCountsGated = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_LumiPixelsCounts_ZeroBiasGated_v5' )
)
process.hltPreDatasetAlCaLumiPixelsCountsGated = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaP0 = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetAlCaP0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaPPSExpress = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
      'HLT_PPSMaxTracksPerRP4_v9' )
)
process.hltPreDatasetAlCaPPSExpress = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaPPSPrompt = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSMaxTracksPerArm1_v9',
      'HLT_PPSMaxTracksPerRP4_v9' )
)
process.hltPreDatasetAlCaPPSPrompt = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetAlCaPhiSym = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_EcalPhiSym_v20' )
)
process.hltPreDatasetAlCaPhiSym = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetCommissioning = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_IsoTrackHB_v14',
      'HLT_IsoTrackHE_v14',
      'HLT_L1SingleMuCosmics_EMTF_v4' )
)
process.hltPreDatasetCommissioning = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetCosmics = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetDQMGPUvsCPU = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'DQM_EcalReconstruction_v12',
      'DQM_HcalReconstruction_v10',
      'DQM_PixelReconstruction_v12' )
)
process.hltPreDatasetDQMGPUvsCPU = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetDQMOnlineBeamspot = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_HT300_Beamspot_v23',
      'HLT_HT60_Beamspot_v22',
      'HLT_ZeroBias_Beamspot_v16' )
)
process.hltPreDatasetDQMOnlineBeamspot = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetDQMPPSRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_PPSRandom_v1' )
)
process.hltPreDatasetDQMPPSRandom = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetEcalLaser = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4' )
)
process.hltPreDatasetEcalLaser = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetEventDisplay = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetEventDisplay = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetExpressAlignment = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetExpressAlignment = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetExpressCosmics = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetExpressCosmics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetExpressPhysics = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetExpressPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetHLTMonitor = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_L1SingleMuCosmics_CosmicTracking_v1',
      'HLT_L1SingleMuCosmics_PointingCosmicTracking_v1' )
)
process.hltPreDatasetHLTMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetHLTPhysics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Physics_v14' )
)
process.hltPreDatasetHLTPhysics = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetHcalNZS = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_HcalNZS_v21',
      'HLT_HcalPhiSym_v23' )
)
process.hltPreDatasetHcalNZS = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetL1Accept = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'DST_Physics_v16',
      'DST_ZeroBias_v11' )
)
process.hltPreDatasetL1Accept = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetMinimumBias = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetMinimumBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetMuonShower = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_CscCluster_Cosmic_v4' )
)
process.hltPreDatasetMuonShower = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetNoBPTX = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetNoBPTX = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetOnlineMonitor = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetOnlineMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetRPCMonitor = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'AlCa_RPCMuonNormalisation_v23' )
)
process.hltPreDatasetRPCMonitor = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetTestEnablesEcalHcal = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4',
      'HLT_HcalCalibration_v6' )
)
process.hltPreDatasetTestEnablesEcalHcal = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetTestEnablesEcalHcalDQM = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_EcalCalibration_v4',
      'HLT_HcalCalibration_v6' )
)
process.hltPreDatasetTestEnablesEcalHcalDQM = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetVRRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Random_HighRate_v1' )
)
process.hltPreDatasetVRRandom0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetVRRandom15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetZeroBias = cms.EDFilter( "TriggerResultsFilter",
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
process.hltPreDatasetZeroBias = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetSpecialRandom = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_Random_HighRate_v1' )
)
process.hltPreDatasetSpecialRandom0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialRandom19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetSpecialZeroBias = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_SpecialZeroBias_v6',
      'HLT_ZeroBias_Gated_v4',
      'HLT_ZeroBias_HighRate_v4' )
)
process.hltPreDatasetSpecialZeroBias0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias20 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 20 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias21 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 21 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias22 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 22 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias23 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 23 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias24 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 24 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias25 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 25 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias26 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 26 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias27 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 27 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias28 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 28 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias29 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 29 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias30 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 30 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialZeroBias31 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 31 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetSpecialHLTPhysics = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_SpecialHLTPhysics_v7' )
)
process.hltPreDatasetSpecialHLTPhysics0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics2 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 2 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics3 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 3 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics4 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 4 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics5 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 5 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics6 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 6 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics7 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 7 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics8 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 8 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics9 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 9 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics10 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 10 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics11 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 11 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics12 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 12 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics13 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 13 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics14 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 14 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics15 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 15 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics16 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 16 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics17 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 17 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics18 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 18 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialHLTPhysics19 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 19 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltDatasetSpecialMinimumBias = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( True ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "" ),
    l1tIgnoreMaskAndPrescale = cms.bool( False ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( 'HLT_L1MinimumBiasHF0ANDBptxAND_v1',
      'HLT_PixelClusters_WP2_HighRate_v1' )
)
process.hltPreDatasetSpecialMinimumBias0 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 0 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)
process.hltPreDatasetSpecialMinimumBias1 = cms.EDFilter( "HLTPrescaler",
    offset = cms.uint32( 1 ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtStage2Digis" )
)

process.hltOutputALCALumiPixelsCountsExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsExpress.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsExpress' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsGated = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsGated.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsGated' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCountsGated_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPrompt = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPrompt.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPrompt' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate0' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate2 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate2.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate3 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate3.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate4 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate4.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCALumiPixelsCountsPromptHighRate5 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCALumiPixelsCountsPromptHighRate5.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaLumiPixelsCountsPromptHighRate5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlcaPixelClusterCounts_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCAP0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAP0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaP0' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltAlCaEtaEBRechitsToDigis_*_*',
      'keep *_hltAlCaEtaEERechitsToDigis_*_*',
      'keep *_hltAlCaEtaRecHitsFilterEEonlyRegional_etaEcalRecHitsES_*',
      'keep *_hltAlCaPi0EBRechitsToDigis_*_*',
      'keep *_hltAlCaPi0EERechitsToDigis_*_*',
      'keep *_hltAlCaPi0RecHitsFilterEEonlyRegional_pi0EcalRecHitsES_*',
      'keep *_hltFEDSelectorL1_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCAPHISYM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPHISYM.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaPhiSym' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalPhiSymFilter_*_*',
      'keep *_hltFEDSelectorL1_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputALCAPPSExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPPSExpress.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaPPSExpress' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorL1_*_*',
      'keep *_hltPPSCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputALCAPPSPrompt = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputALCAPPSPrompt.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_AlCaPPSPrompt' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorL1_*_*',
      'keep *_hltPPSCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputCalibration.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_TestEnablesEcalHcal' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQM = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQM.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_OnlineMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltDoubletRecoveryPFlowTrackSelectionHighPurity_*_*',
      'keep *_hltEcalRecHit_*_*',
      'keep *_hltEgammaCandidates_*_*',
      'keep *_hltEgammaGsfTracks_*_*',
      'keep *_hltGlbTrkMuonCandsNoVtx_*_*',
      'keep *_hltHbhereco_*_*',
      'keep *_hltHfreco_*_*',
      'keep *_hltHoreco_*_*',
      'keep *_hltIter0PFlowCtfWithMaterialTracks_*_*',
      'keep *_hltIter0PFlowTrackSelectionHighPurity_*_*',
      'keep *_hltL3NoFiltersNoVtxMuonCandidates_*_*',
      'keep *_hltMergedTracks_*_*',
      'keep *_hltOnlineBeamSpot_*_*',
      'keep *_hltParticleNetDiscriminatorsJetTags_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltPixelVertices_*_*',
      'keep *_hltSiPixelClusters_*_*',
      'keep *_hltSiStripRawToClustersFacility_*_*',
      'keep *_hltTrimmedPixelVertices_*_*',
      'keep *_hltVerticesPFFilter_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQMCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMCalibration.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_TestEnablesEcalHcalDQM' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep *_hltHcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQMEventDisplay = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMEventDisplay.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_EventDisplay' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputDQMGPUvsCPU = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMGPUvsCPU.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_DQMGPUvsCPU' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalDigisSerialSync_*_*',
      'keep *_hltEcalDigis_*_*',
      'keep *_hltEcalUncalibRecHitSerialSync_*_*',
      'keep *_hltEcalUncalibRecHit_*_*',
      'keep *_hltHbherecoSerialSync_*_*',
      'keep *_hltHbhereco_*_*',
      'keep *_hltParticleFlowClusterHCALSerialSync_*_*',
      'keep *_hltParticleFlowClusterHCAL_*_*',
      'keep *_hltSiPixelDigiErrorsSerialSync_*_*',
      'keep *_hltSiPixelDigiErrors_*_*' )
)
process.hltOutputDQMOnlineBeamspot = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMOnlineBeamspot.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_DQMOnlineBeamspot' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorOnlineMetaData_*_*',
      'keep *_hltFEDSelectorTCDS_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep recoTracks_hltPFMuonMerging_*_*',
      'keep recoVertexs_hltVerticesPFFilter_*_*' )
)
process.hltOutputDQMPPSRandom = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputDQMPPSRandom.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_DQMPPSRandom' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltPPSCalibrationRaw_*_*' )
)
process.hltOutputEcalCalibration = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputEcalCalibration.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_EcalLaser' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltEcalCalibrationRaw_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputExpress = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpress.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_ExpressPhysics' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputExpressAlignment = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpressAlignment.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_ExpressAlignment' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputExpressCosmics = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputExpressCosmics.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_ExpressCosmics' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputHLTMonitor = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputHLTMonitor.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_HLTMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltCtfWithMaterialTracksP5_*_*',
      'keep *_hltDeepBLifetimeTagInfosPF_*_*',
      'keep *_hltDisplacedhltIter4PFlowTrackSelectionHighPurity_*_*',
      'keep *_hltDoubletRecoveryPFlowTrackSelectionHighPurity_*_*',
      'keep *_hltEcalRecHit_*_*',
      'keep *_hltEgammaGsfTracks_*_*',
      'keep *_hltFullSiStripRawToClustersFacility_*_*',
      'keep *_hltHbhereco_*_*',
      'keep *_hltHfreco_*_*',
      'keep *_hltHoreco_*_*',
      'keep *_hltIter2MergedForDisplaced_*_*',
      'keep *_hltMergedTracks_*_*',
      'keep *_hltOnlineBeamSpot_*_*',
      'keep *_hltPFJetForBtag_*_*',
      'keep *_hltPFJetForPNetAK8_*_*',
      'keep *_hltPFMuonMerging_*_*',
      'keep *_hltParticleNetDiscriminatorsJetTagsAK8_*_*',
      'keep *_hltParticleNetDiscriminatorsJetTags_*_*',
      'keep *_hltParticleNetJetTagInfos_*_*',
      'keep *_hltPixelTracks_*_*',
      'keep *_hltPixelVertices_*_*',
      'keep *_hltSiPixelClusters_*_*',
      'keep *_hltSiStripRawToClustersFacility_*_*',
      'keep *_hltVerticesPFFilter_*_*',
      'keep *_hltVerticesPFSelector_*_*',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputNanoDST = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputNanoDST.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_L1Accept' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorL1_*_*',
      'keep *_hltFEDSelectorL1uGTTest_*_*',
      'keep *_hltFEDSelectorTCDS_*_*',
      'keep edmTriggerResults_*_*_*' )
)
process.hltOutputPhysicsCommissioning = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsCommissioning.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_Commissioning',
  'Dataset_Cosmics',
  'Dataset_HLTPhysics',
  'Dataset_HcalNZS',
  'Dataset_MinimumBias',
  'Dataset_MuonShower',
  'Dataset_NoBPTX',
  'Dataset_ZeroBias' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics0' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics10 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics10.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics10' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics11 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics11.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics11' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics12 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics12.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics12' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics13 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics13.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics13' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics14 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics14.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics14' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics15 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics15.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics15' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics16 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics16.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics16' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics17 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics17.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics17' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics18 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics18.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics18' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics19 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics19.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics19' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics2 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics2.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics2' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics3 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics3.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics4 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics4.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics4' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics5 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics5.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics6 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics6.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics6' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics7 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics7.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics7' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics8 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics8.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics8' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialHLTPhysics9 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialHLTPhysics9.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialHLTPhysics9' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialMinimumBias0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialMinimumBias0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialMinimumBias0' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialMinimumBias1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialMinimumBias1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialMinimumBias1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom0',
  'Dataset_SpecialRandom1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom2',
  'Dataset_SpecialRandom3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom2 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom2.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom4',
  'Dataset_SpecialRandom5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom3 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom3.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom6',
  'Dataset_SpecialRandom7' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom4 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom4.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom8',
  'Dataset_SpecialRandom9' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom5 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom5.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom10',
  'Dataset_SpecialRandom11' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom6 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom6.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom12',
  'Dataset_SpecialRandom13' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom7 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom7.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom14',
  'Dataset_SpecialRandom15' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom8 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom8.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom16',
  'Dataset_SpecialRandom17' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialRandom9 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialRandom9.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialRandom18',
  'Dataset_SpecialRandom19' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias0',
  'Dataset_SpecialZeroBias1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias2',
  'Dataset_SpecialZeroBias3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias10 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias10.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias20',
  'Dataset_SpecialZeroBias21' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias11 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias11.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias22',
  'Dataset_SpecialZeroBias23' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias12 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias12.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias24',
  'Dataset_SpecialZeroBias25' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias13 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias13.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias26',
  'Dataset_SpecialZeroBias27' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias14 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias14.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias28',
  'Dataset_SpecialZeroBias29' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias15 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias15.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias30',
  'Dataset_SpecialZeroBias31' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias2 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias2.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias4',
  'Dataset_SpecialZeroBias5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias3 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias3.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias6',
  'Dataset_SpecialZeroBias7' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias4 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias4.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias8',
  'Dataset_SpecialZeroBias9' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias5 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias5.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias10',
  'Dataset_SpecialZeroBias11' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias6 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias6.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias12',
  'Dataset_SpecialZeroBias13' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias7 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias7.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias14',
  'Dataset_SpecialZeroBias15' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias8 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias8.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias16',
  'Dataset_SpecialZeroBias17' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsSpecialZeroBias9 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsSpecialZeroBias9.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_SpecialZeroBias18',
  'Dataset_SpecialZeroBias19' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom0 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom0.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom0',
  'Dataset_VRRandom1' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom1 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom1.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom2',
  'Dataset_VRRandom3' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom2 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom2.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom4',
  'Dataset_VRRandom5' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom3 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom3.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom6',
  'Dataset_VRRandom7' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom4 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom4.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom8',
  'Dataset_VRRandom9' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom5 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom5.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom10',
  'Dataset_VRRandom11' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom6 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom6.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom12',
  'Dataset_VRRandom13' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputPhysicsVRRandom7 = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputPhysicsVRRandom7.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_VRRandom14',
  'Dataset_VRRandom15' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep FEDRawDataCollection_rawDataCollector_*_*',
      'keep GlobalObjectMapRecord_hltGtStage2ObjectMap_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)
process.hltOutputRPCMON = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "outputRPCMON.root" ),
    compressionAlgorithm = cms.untracked.string( "ZSTD" ),
    compressionLevel = cms.untracked.int32( 3 ),
    fastCloning = cms.untracked.bool( False ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string( "" ),
        dataTier = cms.untracked.string( "RAW" )
    ),
    SelectEvents = cms.untracked.PSet(  SelectEvents = cms.vstring( 'Dataset_RPCMonitor' ) ),
    outputCommands = cms.untracked.vstring( 'drop *',
      'keep *_hltFEDSelectorCPPF_*_*',
      'keep *_hltFEDSelectorCSC_*_*',
      'keep *_hltFEDSelectorDT_*_*',
      'keep *_hltFEDSelectorGEM_*_*',
      'keep *_hltFEDSelectorL1_*_*',
      'keep *_hltFEDSelectorOMTF_*_*',
      'keep *_hltFEDSelectorRPC_*_*',
      'keep *_hltFEDSelectorTCDS_*_*',
      'keep *_hltFEDSelectorTwinMux_*_*',
      'keep edmTriggerResults_*_*_*',
      'keep triggerTriggerEvent_*_*_*' )
)

process.HLTL1UnpackerSequence = cms.Sequence( process.hltGtStage2Digis + process.hltGtStage2ObjectMap )
process.HLTBeamSpot = cms.Sequence( process.hltOnlineMetaDataDigis + process.hltOnlineBeamSpot )
process.HLTBeginSequence = cms.Sequence( process.hltTriggerType + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence = cms.Sequence( process.hltEcalDigisLegacy + process.hltEcalDigisSoA + process.hltEcalDigis + process.hltEcalUncalibRecHitSoA + process.hltEcalUncalibRecHit + process.hltEcalDetIdToBeRecovered + process.hltEcalRecHit )
process.HLTPreshowerSequence = cms.Sequence( process.hltEcalPreshowerDigis + process.hltEcalPreshowerRecHit )
process.HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence( process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + process.HLTPreshowerSequence )
process.HLTEndSequence = cms.Sequence( process.hltBoolEnd )
process.HLTFEDSelectorsForRPCMonitor = cms.Sequence( process.hltFEDSelectorTCDS + process.hltFEDSelectorL1 + process.hltFEDSelectorDT + process.hltFEDSelectorRPC + process.hltFEDSelectorCSC + process.hltFEDSelectorGEM + process.hltFEDSelectorTwinMux + process.hltFEDSelectorOMTF + process.hltFEDSelectorCPPF )
process.HLTBeginSequenceRandom = cms.Sequence( process.hltRandomEventsFilter + process.hltGtStage2Digis )
process.HLTDoLocalPixelSequence = cms.Sequence( process.hltOnlineBeamSpotDevice + process.hltSiPixelClustersSoA + process.hltSiPixelClusters + process.hltSiPixelDigiErrors + process.hltSiPixelRecHitsSoA + process.hltSiPixelRecHits )
process.HLTDoLocalPixelSequenceSerialSync = cms.Sequence( process.hltOnlineBeamSpotDeviceSerialSync + process.hltSiPixelClustersSoASerialSync + process.hltSiPixelClustersSerialSync + process.hltSiPixelDigiErrorsSerialSync + process.hltSiPixelRecHitsSoASerialSync + process.hltSiPixelRecHitsSerialSync )
process.HLTRecoPixelTracksSequence = cms.Sequence( process.hltPixelTracksSoA + process.hltPixelTracks )
process.HLTRecopixelvertexingSequence = cms.Sequence( process.HLTRecoPixelTracksSequence + process.hltPixelVerticesSoA + process.hltPixelVertices + process.hltTrimmedPixelVertices )
process.HLTRecoPixelTracksSequenceSerialSync = cms.Sequence( process.hltPixelTracksSoASerialSync + process.hltPixelTracksSerialSync )
process.HLTRecopixelvertexingSequenceSerialSync = cms.Sequence( process.HLTRecoPixelTracksSequenceSerialSync + process.hltPixelVerticesSoASerialSync + process.hltPixelVerticesSerialSync + process.hltTrimmedPixelVerticesSerialSync )
process.HLTDQMPixelReconstruction = cms.Sequence( process.hltSiPixelRecHitsSoAMonitorCPU + process.hltSiPixelRecHitsSoAMonitorGPU + process.hltSiPixelRecHitsSoACompareGPUvsCPU + process.hltPixelTracksSoAMonitorCPU + process.hltPixelTracksSoAMonitorGPU + process.hltPixelTracksSoACompareGPUvsCPU + process.hltPixelVerticesSoAMonitorCPU + process.hltPixelVerticesSoAMonitorGPU + process.hltPixelVerticesSoACompareGPUvsCPU )
process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequenceSerialSync = cms.Sequence( process.hltEcalDigisLegacy + process.hltEcalDigisSoASerialSync + process.hltEcalDigisSerialSync + process.hltEcalUncalibRecHitSoASerialSync + process.hltEcalUncalibRecHitSerialSync + process.hltEcalDetIdToBeRecovered + process.hltEcalRecHitSerialSync )
process.HLTDoLocalHcalSequence = cms.Sequence( process.hltHcalDigis + process.hltHcalDigisSoA + process.hltHbheRecoSoA + process.hltHbhereco + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
process.HLTDoLocalHcalSequenceSerialSync = cms.Sequence( process.hltHcalDigis + process.hltHcalDigisSoASerialSync + process.hltHbheRecoSoASerialSync + process.hltHbherecoSerialSync + process.hltHfprereco + process.hltHfreco + process.hltHoreco )
process.HLTPFHcalClustering = cms.Sequence( process.hltParticleFlowRecHitHBHESoA + process.hltParticleFlowRecHitHBHE + process.hltParticleFlowClusterHBHESoA + process.hltParticleFlowClusterHBHE + process.hltParticleFlowClusterHCAL )
process.HLTPFHcalClusteringSerialSync = cms.Sequence( process.hltParticleFlowRecHitHBHESoASerialSync + process.hltParticleFlowRecHitHBHESerialSync + process.hltParticleFlowClusterHBHESoASerialSync + process.hltParticleFlowClusterHBHESerialSync + process.hltParticleFlowClusterHCALSerialSync )
process.HLTBeginSequenceCalibration = cms.Sequence( process.hltCalibrationEventsFilter + process.hltGtStage2Digis )
process.HLTBeginSequenceNZS = cms.Sequence( process.hltTriggerType + process.hltL1EventNumberNZS + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTBeginSequenceL1Fat = cms.Sequence( process.hltTriggerType + process.hltL1EventNumberL1Fat + process.HLTL1UnpackerSequence + process.HLTBeamSpot )
process.HLTDoCaloSequencePF = cms.Sequence( process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + process.HLTDoLocalHcalSequence + process.hltTowerMakerForAll )
process.HLTAK4CaloJetsPrePFRecoSequence = cms.Sequence( process.HLTDoCaloSequencePF + process.hltAK4CaloJetsPF )
process.HLTPreAK4PFJetsRecoSequence = cms.Sequence( process.HLTAK4CaloJetsPrePFRecoSequence + process.hltAK4CaloJetsPFEt5 )
process.HLTMuonLocalRecoSequence = cms.Sequence( process.hltMuonDTDigis + process.hltDt1DRecHits + process.hltDt4DSegments + process.hltMuonCSCDigis + process.hltCsc2DRecHits + process.hltCscSegments + process.hltMuonRPCDigisCPPF + process.hltOmtfDigis + process.hltMuonRPCDigisTwinMux + process.hltMuonRPCDigis + process.hltRpcRecHits + process.hltMuonGEMDigis + process.hltGemRecHits + process.hltGemSegments )
process.HLTL2muonrecoNocandSequence = cms.Sequence( process.HLTMuonLocalRecoSequence + process.hltL2OfflineMuonSeeds + process.hltL2MuonSeeds + process.hltL2Muons )
process.HLTL2muonrecoSequence = cms.Sequence( process.HLTL2muonrecoNocandSequence + process.hltL2MuonCandidates )
process.HLTDoLocalStripSequence = cms.Sequence( process.hltSiStripExcludedFEDListProducer + process.hltSiStripRawToClustersFacility + process.hltMeasurementTrackerEvent )
process.HLTIterL3OImuonTkCandidateSequence = cms.Sequence( process.hltIterL3OISeedsFromL2Muons + process.hltIterL3OITrackCandidates + process.hltIterL3OIMuCtfWithMaterialTracks + process.hltIterL3OIMuonTrackCutClassifier + process.hltIterL3OIMuonTrackSelectionHighPurity + process.hltL3MuonsIterL3OI )
process.HLTIterL3MuonRecopixelvertexingSequence = cms.Sequence( process.HLTRecopixelvertexingSequence + process.hltIterL3MuonPixelTracksTrackingRegions + process.hltPixelTracksInRegionL2 )
process.HLTIterativeTrackingIteration0ForIterL3Muon = cms.Sequence( process.hltIter0IterL3MuonPixelSeedsFromPixelTracks + process.hltIter0IterL3MuonPixelSeedsFromPixelTracksFiltered + process.hltIter0IterL3MuonCkfTrackCandidates + process.hltIter0IterL3MuonCtfWithMaterialTracks + process.hltIter0IterL3MuonTrackCutClassifier + process.hltIter0IterL3MuonTrackSelectionHighPurity )
process.HLTIterL3IOmuonTkCandidateSequence = cms.Sequence( process.HLTIterL3MuonRecopixelvertexingSequence + process.HLTIterativeTrackingIteration0ForIterL3Muon + process.hltL3MuonsIterL3IO )
process.HLTIterL3OIAndIOFromL2muonTkCandidateSequence = cms.Sequence( process.HLTIterL3OImuonTkCandidateSequence + process.hltIterL3OIL3MuonsLinksCombination + process.hltIterL3OIL3Muons + process.hltIterL3OIL3MuonCandidates + process.hltL2SelectorForL3IO + process.HLTIterL3IOmuonTkCandidateSequence + process.hltIterL3MuonsFromL2LinksCombination )
process.HLTRecopixelvertexingSequenceForIterL3FromL1Muon = cms.Sequence( process.HLTRecopixelvertexingSequence + process.hltIterL3FromL1MuonPixelTracksTrackingRegions + process.hltPixelTracksInRegionL1 )
process.HLTIterativeTrackingIteration0ForIterL3FromL1Muon = cms.Sequence( process.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracks + process.hltIter0IterL3FromL1MuonPixelSeedsFromPixelTracksFiltered + process.hltIter0IterL3FromL1MuonCkfTrackCandidates + process.hltIter0IterL3FromL1MuonCtfWithMaterialTracks + process.hltIter0IterL3FromL1MuonTrackCutClassifier + process.hltIter0IterL3FromL1MuonTrackSelectionHighPurity )
process.HLTIterativeTrackingIteration3ForIterL3FromL1Muon = cms.Sequence( process.hltIter3IterL3FromL1MuonClustersRefRemoval + process.hltIter3IterL3FromL1MuonMaskedMeasurementTrackerEvent + process.hltIter3IterL3FromL1MuonPixelLayersAndRegions + process.hltIter3IterL3FromL1MuonTrackingRegions + process.hltIter3IterL3FromL1MuonPixelClusterCheck + process.hltIter3IterL3FromL1MuonPixelHitDoublets + process.hltIter3IterL3FromL1MuonPixelSeeds + process.hltIter3IterL3FromL1MuonPixelSeedsFiltered + process.hltIter3IterL3FromL1MuonCkfTrackCandidates + process.hltIter3IterL3FromL1MuonCtfWithMaterialTracks + process.hltIter3IterL3FromL1MuonTrackCutClassifier + process.hltIter3IterL3FromL1MuonTrackSelectionHighPurity )
process.HLTIterL3IOmuonFromL1TkCandidateSequence = cms.Sequence( process.HLTRecopixelvertexingSequenceForIterL3FromL1Muon + process.HLTIterativeTrackingIteration0ForIterL3FromL1Muon + process.HLTIterativeTrackingIteration3ForIterL3FromL1Muon )
process.HLTIterL3muonTkCandidateSequence = cms.Sequence( process.HLTDoLocalPixelSequence + process.HLTDoLocalStripSequence + process.HLTIterL3OIAndIOFromL2muonTkCandidateSequence + process.hltL1MuonsPt0 + process.HLTIterL3IOmuonFromL1TkCandidateSequence )
process.HLTL3muonrecoNocandSequence = cms.Sequence( process.HLTIterL3muonTkCandidateSequence + process.hltIter03IterL3FromL1MuonMerged + process.hltIterL3MuonMerged + process.hltIterL3MuonAndMuonFromL1Merged + process.hltIterL3GlbMuon + process.hltIterL3MuonsNoID + process.hltIterL3Muons + process.hltL3MuonsIterL3Links + process.hltIterL3MuonTracks )
process.HLTL3muonrecoSequence = cms.Sequence( process.HLTL3muonrecoNocandSequence + process.hltIterL3MuonCandidates )
process.HLTIterativeTrackingIteration0 = cms.Sequence( process.hltIter0PFLowPixelSeedsFromPixelTracks + process.hltIter0PFlowCkfTrackCandidates + process.hltIter0PFlowCtfWithMaterialTracks + process.hltIter0PFlowTrackCutClassifier + process.hltIter0PFlowTrackSelectionHighPurity )
process.HLTIterativeTrackingDoubletRecovery = cms.Sequence( process.hltDoubletRecoveryClustersRefRemoval + process.hltDoubletRecoveryMaskedMeasurementTrackerEvent + process.hltDoubletRecoveryPixelLayersAndRegions + process.hltDoubletRecoveryPFlowPixelClusterCheck + process.hltDoubletRecoveryPFlowPixelHitDoublets + process.hltDoubletRecoveryPFlowPixelSeeds + process.hltDoubletRecoveryPFlowCkfTrackCandidates + process.hltDoubletRecoveryPFlowCtfWithMaterialTracks + process.hltDoubletRecoveryPFlowTrackCutClassifier + process.hltDoubletRecoveryPFlowTrackSelectionHighPurity )
process.HLTIterativeTrackingIter02 = cms.Sequence( process.HLTIterativeTrackingIteration0 + process.HLTIterativeTrackingDoubletRecovery + process.hltMergedTracks )
process.HLTTrackingForBeamSpot = cms.Sequence( process.HLTPreAK4PFJetsRecoSequence + process.HLTL2muonrecoSequence + process.HLTL3muonrecoSequence + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingSequence + process.HLTDoLocalStripSequence + process.HLTIterativeTrackingIter02 + process.hltPFMuonMerging )
process.HLTDoCaloSequence = cms.Sequence( process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + process.HLTDoLocalHcalSequence + process.hltTowerMakerForAll )
process.HLTAK4CaloJetsReconstructionSequence = cms.Sequence( process.HLTDoCaloSequence + process.hltAK4CaloJets + process.hltAK4CaloJetsIDPassed )
process.HLTAK4CaloCorrectorProducersSequence = cms.Sequence( process.hltAK4CaloFastJetCorrector + process.hltAK4CaloRelativeCorrector + process.hltAK4CaloAbsoluteCorrector + process.hltAK4CaloResidualCorrector + process.hltAK4CaloCorrector )
process.HLTAK4CaloJetsCorrectionSequence = cms.Sequence( process.hltFixedGridRhoFastjetAllCalo + process.HLTAK4CaloCorrectorProducersSequence + process.hltAK4CaloJetsCorrected + process.hltAK4CaloJetsCorrectedIDPassed )
process.HLTAK4CaloJetsSequence = cms.Sequence( process.HLTAK4CaloJetsReconstructionSequence + process.HLTAK4CaloJetsCorrectionSequence )
process.HLTMuonLocalRecoMeanTimerSequence = cms.Sequence( process.hltMuonDTDigis + process.hltDt1DRecHits + process.hltDt4DSegmentsMeanTimer + process.hltMuonCSCDigis + process.hltCsc2DRecHits + process.hltCscSegments + process.hltMuonRPCDigisCPPF + process.hltOmtfDigis + process.hltMuonRPCDigisTwinMux + process.hltMuonRPCDigis + process.hltRpcRecHits + process.hltMuonGEMDigis + process.hltGemRecHits + process.hltGemSegments )
process.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence = cms.Sequence( process.HLTMuonLocalRecoMeanTimerSequence + process.hltL2CosmicOfflineMuonSeeds + process.hltL2CosmicMuonSeeds + process.hltL2CosmicMuons )
process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer = cms.Sequence( process.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence + process.hltL2MuonCandidatesNoVtxMeanTimerCosmicSeed )
process.HLTPPSPixelRecoSequence = cms.Sequence( process.hltCTPPSPixelDigis + process.hltCTPPSPixelClusters + process.hltCTPPSPixelRecHits + process.hltCTPPSPixelLocalTracks )
process.HLTDoLocalStripFullSequence = cms.Sequence( process.hltSiStripExcludedFEDListProducer + process.hltFullSiStripRawToClustersFacility + process.hltFullMeasurementTrackerEvent + process.hltGlobalSiStripMatchedRecHitsFull )
process.HLTCTFCosmicsSequence = cms.Sequence( process.hltSimpleCosmicBONSeedingLayers + process.hltSimpleCosmicBONSeeds + process.hltCombinatorialcosmicseedingtripletsP5 + process.hltCombinatorialcosmicseedingpairsTOBP5 + process.hltCombinatorialcosmicseedingpairsTECposP5 + process.hltCombinatorialcosmicseedingpairsTECnegP5 + process.hltCombinatorialcosmicseedfinderP5 + process.hltCombinedP5SeedsForCTF + process.hltCkfTrackCandidatesP5 + process.hltCtfWithMaterialTracksCosmics + process.hltCtfWithMaterialTracksP5 )
process.HLTDatasetPathBeginSequence = cms.Sequence( process.hltGtStage2Digis )

process.HLTriggerFirstPath = cms.Path( process.hltGetRaw + process.hltPSetMap + process.hltBoolFalse )
process.Status_OnCPU = cms.Path( process.hltBackend + ~process.hltStatusOnGPUFilter )
process.Status_OnGPU = cms.Path( process.hltBackend + process.hltStatusOnGPUFilter )
process.AlCa_EcalPhiSym_v20 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasIorAlwaysTrueIorIsolatedBunch + process.hltPreAlCaEcalPhiSym + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltEcalPhiSymFilter + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.AlCa_EcalEtaEBonly_v25 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaEcalPi0Eta + process.hltPreAlCaEcalEtaEBonly + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltSimple3x3Clusters + process.hltAlCaEtaRecHitsFilterEBonlyRegional + process.hltAlCaEtaEBUncalibrator + process.hltAlCaEtaEBRechitsToDigis + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.AlCa_EcalEtaEEonly_v25 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaEcalPi0Eta + process.hltPreAlCaEcalEtaEEonly + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltSimple3x3Clusters + process.hltAlCaEtaRecHitsFilterEEonlyRegional + process.hltAlCaEtaEEUncalibrator + process.hltAlCaEtaEERechitsToDigis + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.AlCa_EcalPi0EBonly_v25 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaEcalPi0Eta + process.hltPreAlCaEcalPi0EBonly + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltSimple3x3Clusters + process.hltAlCaPi0RecHitsFilterEBonlyRegional + process.hltAlCaPi0EBUncalibrator + process.hltAlCaPi0EBRechitsToDigis + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.AlCa_EcalPi0EEonly_v25 = cms.Path( process.HLTBeginSequence + process.hltL1sAlCaEcalPi0Eta + process.hltPreAlCaEcalPi0EEonly + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltSimple3x3Clusters + process.hltAlCaPi0RecHitsFilterEEonlyRegional + process.hltAlCaPi0EEUncalibrator + process.hltAlCaPi0EERechitsToDigis + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.AlCa_RPCMuonNormalisation_v23 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMu5IorSingleMu14erIorSingleMu16er + process.hltPreAlCaRPCMuonNormalisation + process.hltRPCMuonNormaL1Filtered0 + process.HLTFEDSelectorsForRPCMonitor + process.HLTEndSequence )
process.AlCa_LumiPixelsCounts_Random_v10 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreAlCaLumiPixelsCountsRandom + process.HLTBeamSpot + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltAlcaPixelClusterCounts + process.HLTEndSequence )
process.AlCa_LumiPixelsCounts_ZeroBias_v12 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreAlCaLumiPixelsCountsZeroBias + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltAlcaPixelClusterCounts + process.HLTEndSequence )
process.DQM_PixelReconstruction_v12 = cms.Path( process.HLTBeginSequence + process.hltL1sDQMPixelReconstruction + process.hltPreDQMPixelReconstruction + process.hltBackend + process.hltStatusOnGPUFilter + process.HLTDoLocalPixelSequence + process.HLTDoLocalPixelSequenceSerialSync + process.HLTRecopixelvertexingSequence + process.HLTRecopixelvertexingSequenceSerialSync + process.HLTDQMPixelReconstruction + process.HLTEndSequence )
process.DQM_EcalReconstruction_v12 = cms.Path( process.HLTBeginSequence + process.hltL1sDQMEcalReconstruction + process.hltPreDQMEcalReconstruction + process.hltBackend + process.hltStatusOnGPUFilter + process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequence + process.HLTDoFullUnpackingEgammaEcalWithoutPreshowerSequenceSerialSync + process.HLTEndSequence )
process.DQM_HcalReconstruction_v10 = cms.Path( process.HLTBeginSequence + process.hltL1sDQMHcalReconstruction + process.hltPreDQMHcalReconstruction + process.hltBackend + process.hltStatusOnGPUFilter + process.HLTDoLocalHcalSequence + process.HLTDoLocalHcalSequenceSerialSync + process.HLTPFHcalClustering + process.HLTPFHcalClusteringSerialSync + process.HLTEndSequence )
process.DQM_Random_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreDQMRandom + process.HLTEndSequence )
process.DQM_ZeroBias_v3 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreDQMZeroBias + process.HLTEndSequence )
process.DST_ZeroBias_v11 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreDSTZeroBias + process.hltFEDSelectorL1 + process.hltFEDSelectorL1uGTTest + process.hltFEDSelectorTCDS + process.HLTEndSequence )
process.DST_Physics_v16 = cms.Path( process.HLTBeginSequence + process.hltPreDSTPhysics + process.hltFEDSelectorL1 + process.hltFEDSelectorL1uGTTest + process.hltFEDSelectorTCDS + process.HLTEndSequence )
process.HLT_EcalCalibration_v4 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreEcalCalibration + process.hltEcalCalibrationRaw + process.HLTEndSequence )
process.HLT_HcalCalibration_v6 = cms.Path( process.HLTBeginSequenceCalibration + process.hltPreHcalCalibration + process.hltHcalCalibrationRaw + process.HLTEndSequence )
process.HLT_HcalNZS_v21 = cms.Path( process.HLTBeginSequenceNZS + process.hltL1sHcalNZS + process.hltPreHcalNZS + process.HLTEndSequence )
process.HLT_HcalPhiSym_v23 = cms.Path( process.HLTBeginSequenceNZS + process.hltL1sSingleEGorSingleorDoubleMu + process.hltPreHcalPhiSym + process.HLTEndSequence )
process.HLT_Random_v3 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreRandom + process.HLTEndSequence )
process.HLT_Physics_v14 = cms.Path( process.HLTBeginSequenceL1Fat + process.hltPrePhysics + process.HLTEndSequence )
process.HLT_ZeroBias_v13 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreZeroBias + process.HLTEndSequence )
process.HLT_ZeroBias_Alignment_v8 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreZeroBiasAlignment + process.HLTEndSequence )
process.HLT_ZeroBias_Beamspot_v16 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPreZeroBiasBeamspot + process.HLTTrackingForBeamSpot + process.hltVerticesPF + process.hltVerticesPFSelector + process.hltVerticesPFFilter + process.hltFEDSelectorOnlineMetaData + process.hltFEDSelectorTCDS + process.HLTEndSequence )
process.HLT_ZeroBias_IsolatedBunches_v12 = cms.Path( process.HLTBeginSequence + process.hltL1sIsolatedBunch + process.hltPreZeroBiasIsolatedBunches + process.HLTEndSequence )
process.HLT_ZeroBias_FirstBXAfterTrain_v10 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBiasFirstBunchAfterTrain + process.hltPreZeroBiasFirstBXAfterTrain + process.HLTEndSequence )
process.HLT_ZeroBias_FirstCollisionAfterAbortGap_v12 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBiasFirstCollisionAfterAbortGap + process.hltPreZeroBiasFirstCollisionAfterAbortGap + process.HLTEndSequence )
process.HLT_ZeroBias_FirstCollisionInTrain_v11 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBiasFirstCollisionInTrainNOTFirstCollisionInOrbit + process.hltPreZeroBiasFirstCollisionInTrain + process.HLTEndSequence )
process.HLT_ZeroBias_LastCollisionInTrain_v10 = cms.Path( process.HLTBeginSequence + process.hltL1sL1ZeroBiasLastBunchInTrain + process.hltPreZeroBiasLastCollisionInTrain + process.HLTEndSequence )
process.HLT_HT300_Beamspot_v23 = cms.Path( process.HLTBeginSequence + process.hltL1sHTTForBeamSpot + process.hltPreHT300Beamspot + process.HLTAK4CaloJetsSequence + process.hltHtMht + process.hltHT300 + process.HLTTrackingForBeamSpot + process.hltVerticesPF + process.hltVerticesPFSelector + process.hltVerticesPFFilter + process.hltFEDSelectorOnlineMetaData + process.hltFEDSelectorTCDS + process.HLTEndSequence )
process.HLT_IsoTrackHB_v14 = cms.Path( process.HLTBeginSequence + process.hltL1sV0SingleJet3OR + process.hltPreIsoTrackHB + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingSequence + process.hltPixelTracksQuadruplets + process.hltIsolPixelTrackProdHB + process.hltIsolPixelTrackL2FilterHB + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltIsolEcalPixelTrackProdHB + process.hltEcalIsolPixelTrackL2FilterHB + process.HLTDoLocalStripSequence + process.hltIter0PFLowPixelSeedsFromPixelTracks + process.hltIter0PFlowCkfTrackCandidates + process.hltIter0PFlowCtfWithMaterialTracks + process.hltHcalITIPTCorrectorHB + process.hltIsolPixelTrackL3FilterHB + process.HLTEndSequence )
process.HLT_IsoTrackHE_v14 = cms.Path( process.HLTBeginSequence + process.hltL1sV0SingleJet3OR + process.hltPreIsoTrackHE + process.HLTDoLocalPixelSequence + process.HLTRecopixelvertexingSequence + process.hltPixelTracksQuadruplets + process.hltIsolPixelTrackProdHE + process.hltIsolPixelTrackL2FilterHE + process.HLTDoFullUnpackingEgammaEcalSequence + process.hltIsolEcalPixelTrackProdHE + process.hltEcalIsolPixelTrackL2FilterHE + process.HLTDoLocalStripSequence + process.hltIter0PFLowPixelSeedsFromPixelTracks + process.hltIter0PFlowCkfTrackCandidates + process.hltIter0PFlowCtfWithMaterialTracks + process.hltHcalITIPTCorrectorHE + process.hltIsolPixelTrackL3FilterHE + process.HLTEndSequence )
process.HLT_L1SingleMuCosmics_v8 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuCosmics + process.hltPreL1SingleMuCosmics + process.hltL1MuCosmicsL1Filtered0 + process.HLTEndSequence )
process.HLT_L2Mu10_NoVertex_NoBPTX3BX_v14 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + process.hltPreL2Mu10NoVertexNoBPTX3BX + process.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 + process.HLTEndSequence )
process.HLT_L2Mu10_NoVertex_NoBPTX_v15 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpenNotBptxOR + process.hltPreL2Mu10NoVertexNoBPTX + process.hltL1fL1sMuOpenNotBptxORL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sMuOpenNotBptxORL1f0NoVtxCosmicSeedMeanTimerL2Filtered10 + process.HLTEndSequence )
process.HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + process.hltPreL2Mu45NoVertex3StaNoBPTX3BX + process.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered45Sta3 + process.HLTEndSequence )
process.HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpenEr1p4NotBptxOR3BXORL1sSingleMuOpenEr1p1NotBptxOR3BX + process.hltPreL2Mu40NoVertex3StaNoBPTX3BX + process.hltL1fL1sMuOpenNotBptxORNoHaloMu3BXL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sMuOpenNotBptxORNoHaloMu3BXL1f0NoVtxCosmicSeedMeanTimerL2Filtered40Sta3 + process.HLTEndSequence )
process.HLT_CDC_L2cosmic_10_er1p0_v10 = cms.Path( process.HLTBeginSequence + process.hltL1sCDC + process.hltPreCDCL2cosmic10er1p0 + process.hltL1fL1sCDCL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations10er1p0 + process.HLTEndSequence )
process.HLT_CDC_L2cosmic_5p5_er1p0_v10 = cms.Path( process.HLTBeginSequence + process.hltL1sCDC + process.hltPreCDCL2cosmic5p5er1p0 + process.hltL1fL1sCDCL1Filtered0 + process.HLTL2muonrecoSequenceNoVtxCosmicSeedMeanTimer + process.hltL2fL1sCDCL2CosmicMuL2Filtered3er2stations5p5er1p0 + process.HLTEndSequence )
process.HLT_PPSMaxTracksPerArm1_v9 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPrePPSMaxTracksPerArm1 + process.HLTPPSPixelRecoSequence + process.hltPPSExpCalFilter + process.hltPPSCalibrationRaw + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.HLT_PPSMaxTracksPerRP4_v9 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBias + process.hltPrePPSMaxTracksPerRP4 + process.HLTPPSPixelRecoSequence + process.hltPPSPrCalFilter + process.hltPPSCalibrationRaw + process.hltFEDSelectorL1 + process.HLTEndSequence )
process.HLT_PPSRandom_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPrePPSRandom + process.hltPPSCalibrationRaw + process.HLTEndSequence )
process.HLT_SpecialHLTPhysics_v7 = cms.Path( process.HLTBeginSequence + process.hltPreSpecialHLTPhysics + process.HLTEndSequence )
process.AlCa_LumiPixelsCounts_RandomHighRate_v4 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreAlCaLumiPixelsCountsRandomHighRate + process.HLTBeamSpot + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltAlcaPixelClusterCounts + process.HLTEndSequence )
process.AlCa_LumiPixelsCounts_ZeroBiasVdM_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrZeroBiasCopy + process.hltPreAlCaLumiPixelsCountsZeroBiasVdM + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltAlcaPixelClusterCounts + process.HLTEndSequence )
process.AlCa_LumiPixelsCounts_ZeroBiasGated_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR + process.hltPreAlCaLumiPixelsCountsZeroBiasGated + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltAlcaPixelClusterCountsGated + process.HLTEndSequence )
process.HLT_L1SingleMuOpen_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpen + process.hltPreL1SingleMuOpen + process.hltL1MuOpenL1Filtered0 + process.HLTEndSequence )
process.HLT_L1SingleMuOpen_DT_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuOpen + process.hltPreL1SingleMuOpenDT + process.hltL1MuOpenL1FilteredDT + process.HLTEndSequence )
process.HLT_L1SingleMu3_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMu3 + process.hltPreL1SingleMu3 + process.hltL1fL1sMu3L1Filtered0 + process.HLTEndSequence )
process.HLT_L1SingleMu5_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMu5 + process.hltPreL1SingleMu5 + process.hltL1fL1sMu5L1Filtered0 + process.HLTEndSequence )
process.HLT_L1SingleMu7_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMu7 + process.hltPreL1SingleMu7 + process.hltL1fL1sMu7L1Filtered0 + process.HLTEndSequence )
process.HLT_L1DoubleMu0_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sDoubleMu0 + process.hltPreL1DoubleMu0 + process.hltDoubleMu0L1Filtered + process.HLTEndSequence )
process.HLT_L1SingleJet8erHE_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet8erHE + process.hltPreL1SingleJet8erHE + process.HLTEndSequence )
process.HLT_L1SingleJet10erHE_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet10erHE + process.hltPreL1SingleJet10erHE + process.HLTEndSequence )
process.HLT_L1SingleJet12erHE_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet12erHE + process.hltPreL1SingleJet12erHE + process.HLTEndSequence )
process.HLT_L1SingleJet35_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet35 + process.hltPreL1SingleJet35 + process.HLTEndSequence )
process.HLT_L1SingleJet200_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet200 + process.hltPreL1SingleJet200 + process.HLTEndSequence )
process.HLT_L1SingleEG8er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG8er2p5 + process.hltPreL1SingleEG8er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG10er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG10er2p5 + process.hltPreL1SingleEG10er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG15er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG15er2p5 + process.hltPreL1SingleEG15er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG26er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG26er2p5 + process.hltPreL1SingleEG26er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG28er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG28er2p5 + process.hltPreL1SingleEG28er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG28er2p1_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG28er2p1 + process.hltPreL1SingleEG28er2p1 + process.HLTEndSequence )
process.HLT_L1SingleEG28er1p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG28er1p5 + process.hltPreL1SingleEG28er1p5 + process.HLTEndSequence )
process.HLT_L1SingleEG34er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG34er2p5 + process.hltPreL1SingleEG34er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG36er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG36er2p5 + process.hltPreL1SingleEG36er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG38er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG38er2p5 + process.hltPreL1SingleEG38er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG40er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG40er2p5 + process.hltPreL1SingleEG40er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG42er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG42er2p5 + process.hltPreL1SingleEG42er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG45er2p5_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleEG45er2p5 + process.hltPreL1SingleEG45er2p5 + process.HLTEndSequence )
process.HLT_L1SingleEG50_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sL1SingleEG50 + process.hltPreL1SingleEG50 + process.HLTEndSequence )
process.HLT_L1SingleJet60_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet60 + process.hltPreL1SingleJet60 + process.HLTEndSequence )
process.HLT_L1SingleJet90_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet90 + process.hltPreL1SingleJet90 + process.HLTEndSequence )
process.HLT_L1SingleJet120_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet120 + process.hltPreL1SingleJet120 + process.HLTEndSequence )
process.HLT_L1SingleJet180_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleJet180 + process.hltPreL1SingleJet180 + process.HLTEndSequence )
process.HLT_L1HTT120er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT120er + process.hltPreL1HTT120er + process.HLTEndSequence )
process.HLT_L1HTT160er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT160er + process.hltPreL1HTT160er + process.HLTEndSequence )
process.HLT_L1HTT200er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT200er + process.hltPreL1HTT200er + process.HLTEndSequence )
process.HLT_L1HTT255er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT255er + process.hltPreL1HTT255er + process.HLTEndSequence )
process.HLT_L1HTT280er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT280er + process.hltPreL1HTT280er + process.HLTEndSequence )
process.HLT_L1HTT320er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT320er + process.hltPreL1HTT320er + process.HLTEndSequence )
process.HLT_L1HTT360er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT360er + process.hltPreL1HTT360er + process.HLTEndSequence )
process.HLT_L1HTT400er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT400er + process.hltPreL1HTT400er + process.HLTEndSequence )
process.HLT_L1HTT450er_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sHTT450er + process.hltPreL1HTT450er + process.HLTEndSequence )
process.HLT_L1ETM120_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sETM120 + process.hltPreL1ETM120 + process.HLTEndSequence )
process.HLT_L1ETM150_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sETM150 + process.hltPreL1ETM150 + process.HLTEndSequence )
process.HLT_L1EXT_HCAL_LaserMon1_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sEXTHCALLaserMon1 + process.hltPreL1EXTHCALLaserMon1 + process.HLTEndSequence )
process.HLT_L1EXT_HCAL_LaserMon4_v5 = cms.Path( process.HLTBeginSequence + process.hltL1sEXTHCALLaserMon4 + process.hltPreL1EXTHCALLaserMon4 + process.HLTEndSequence )
process.HLT_L1MinimumBiasHF0ANDBptxAND_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sL1MinimumBiasHF0ANDBptxAND + process.hltPreL1MinimumBiasHF0ANDBptxAND + process.HLTEndSequence )
process.HLT_CscCluster_Cosmic_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sMuShowerOneNominal + process.hltPreCscClusterCosmic + process.HLTMuonLocalRecoSequence + process.hltCSCrechitClusters + process.hltCscClusterCosmic + process.HLTEndSequence )
process.HLT_HT60_Beamspot_v22 = cms.Path( process.HLTBeginSequence + process.hltL1sHTTForBeamSpotHT60 + process.hltPreHT60Beamspot + process.HLTAK4CaloJetsSequence + process.hltHtMht + process.hltHT60 + process.HLTTrackingForBeamSpot + process.hltVerticesPF + process.hltVerticesPFSelector + process.hltVerticesPFFilter + process.hltFEDSelectorOnlineMetaData + process.hltFEDSelectorTCDS + process.HLTEndSequence )
process.HLT_HT300_Beamspot_PixelClusters_WP2_v7 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrMinBias + process.hltPreHT300BeamspotPixelClustersWP2 + process.hltPixelTrackerHVOn + process.HLTAK4CaloJetsSequence + process.hltHtMht + process.hltHT300 + process.HLTDoLocalPixelSequence + process.hltPixelActivityFilterWP2 + process.HLTTrackingForBeamSpot + process.hltVerticesPF + process.hltVerticesPFSelector + process.hltVerticesPFFilter + process.HLTEndSequence )
process.HLT_PixelClusters_WP2_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrMinBias + process.hltPrePixelClustersWP2 + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltPixelActivityFilterWP2 + process.HLTEndSequence )
process.HLT_PixelClusters_WP2_HighRate_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrMinBias + process.hltPrePixelClustersWP2HighRate + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltPixelActivityFilterWP2 + process.HLTEndSequence )
process.HLT_PixelClusters_WP1_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrMinBias + process.hltPrePixelClustersWP1 + process.hltPixelTrackerHVOn + process.HLTDoLocalPixelSequence + process.hltPixelActivityFilterWP1 + process.HLTEndSequence )
process.HLT_BptxOR_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sBptxOR + process.hltPreBptxOR + process.HLTEndSequence )
process.HLT_L1SingleMuCosmics_EMTF_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuCosmicsEMTF + process.hltPreL1SingleMuCosmicsEMTF + process.HLTEndSequence )
process.HLT_L1SingleMuCosmics_CosmicTracking_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuCosmics + process.hltPreL1SingleMuCosmicsCosmicTracking + process.hltL1MuCosmicsL1Filtered0 + process.HLTDoLocalPixelSequence + process.HLTDoLocalStripFullSequence + process.HLTCTFCosmicsSequence + process.hltCtfWithMaterialTracksP5TrackCountFilter + process.HLTEndSequence )
process.HLT_L1SingleMuCosmics_PointingCosmicTracking_v1 = cms.Path( process.HLTBeginSequence + process.hltL1sSingleMuCosmics + process.hltPreL1SingleMuCosmicsPointingCosmicTracking + process.hltL1MuCosmicsL1Filtered0 + process.HLTL2muonrecoNocandCosmicSeedMeanTimerSequence + process.hltL2CosmicsMuonTrackerPointingFilter + process.HLTDoLocalPixelSequence + process.HLTDoLocalStripFullSequence + process.HLTCTFCosmicsSequence + process.hltCtfWithMaterialTracksP5TrackCountFilter + process.HLTEndSequence )
process.HLT_L1FatEvents_v5 = cms.Path( process.HLTBeginSequenceL1Fat + process.hltPreL1FatEvents + process.HLTEndSequence )
process.HLT_Random_HighRate_v1 = cms.Path( process.HLTBeginSequenceRandom + process.hltPreRandomHighRate + process.HLTEndSequence )
process.HLT_ZeroBias_HighRate_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrZeroBiasCopy + process.hltPreZeroBiasHighRate + process.HLTEndSequence )
process.HLT_ZeroBias_Gated_v4 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasOrZeroBiasCopyOrAlwaysTrueOrBptxOR + process.hltPreZeroBiasGated + process.hltBXGateFilter + process.HLTEndSequence )
process.HLT_SpecialZeroBias_v6 = cms.Path( process.HLTBeginSequence + process.hltL1sZeroBiasCopyOrAlwaysTrue + process.hltPreSpecialZeroBias + process.HLTEndSequence )
process.HLTriggerFinalPath = cms.Path( process.hltGtStage2Digis + process.hltTriggerSummaryAOD + process.hltTriggerSummaryRAW + process.hltBoolFalse )
process.HLTAnalyzerEndpath = cms.EndPath( process.hltGtStage2Digis + process.hltL1TGlobalSummary + process.hltTrigReport )
process.Dataset_AlCaLumiPixelsCountsExpress = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsExpress + process.hltPreDatasetAlCaLumiPixelsCountsExpress )
process.Dataset_AlCaLumiPixelsCountsPrompt = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPrompt + process.hltPreDatasetAlCaLumiPixelsCountsPrompt )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate0 )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate1 )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate2 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate2 )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate3 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate3 )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate4 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate4 )
process.Dataset_AlCaLumiPixelsCountsPromptHighRate5 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsPromptHighRate + process.hltPreDatasetAlCaLumiPixelsCountsPromptHighRate5 )
process.Dataset_AlCaLumiPixelsCountsGated = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaLumiPixelsCountsGated + process.hltPreDatasetAlCaLumiPixelsCountsGated )
process.Dataset_AlCaP0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaP0 + process.hltPreDatasetAlCaP0 )
process.Dataset_AlCaPPSExpress = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaPPSExpress + process.hltPreDatasetAlCaPPSExpress )
process.Dataset_AlCaPPSPrompt = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaPPSPrompt + process.hltPreDatasetAlCaPPSPrompt )
process.Dataset_AlCaPhiSym = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetAlCaPhiSym + process.hltPreDatasetAlCaPhiSym )
process.Dataset_Commissioning = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetCommissioning + process.hltPreDatasetCommissioning )
process.Dataset_Cosmics = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetCosmics + process.hltPreDatasetCosmics )
process.Dataset_DQMGPUvsCPU = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetDQMGPUvsCPU + process.hltPreDatasetDQMGPUvsCPU )
process.Dataset_DQMOnlineBeamspot = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetDQMOnlineBeamspot + process.hltPreDatasetDQMOnlineBeamspot )
process.Dataset_DQMPPSRandom = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetDQMPPSRandom + process.hltPreDatasetDQMPPSRandom )
process.Dataset_EcalLaser = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetEcalLaser + process.hltPreDatasetEcalLaser )
process.Dataset_EventDisplay = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetEventDisplay + process.hltPreDatasetEventDisplay )
process.Dataset_ExpressAlignment = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetExpressAlignment + process.hltPreDatasetExpressAlignment )
process.Dataset_ExpressCosmics = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetExpressCosmics + process.hltPreDatasetExpressCosmics )
process.Dataset_ExpressPhysics = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetExpressPhysics + process.hltPreDatasetExpressPhysics )
process.Dataset_HLTMonitor = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetHLTMonitor + process.hltPreDatasetHLTMonitor )
process.Dataset_HLTPhysics = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetHLTPhysics + process.hltPreDatasetHLTPhysics )
process.Dataset_HcalNZS = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetHcalNZS + process.hltPreDatasetHcalNZS )
process.Dataset_L1Accept = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetL1Accept + process.hltPreDatasetL1Accept )
process.Dataset_MinimumBias = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetMinimumBias + process.hltPreDatasetMinimumBias )
process.Dataset_MuonShower = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetMuonShower + process.hltPreDatasetMuonShower )
process.Dataset_NoBPTX = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetNoBPTX + process.hltPreDatasetNoBPTX )
process.Dataset_OnlineMonitor = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetOnlineMonitor + process.hltPreDatasetOnlineMonitor )
process.Dataset_RPCMonitor = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetRPCMonitor + process.hltPreDatasetRPCMonitor )
process.Dataset_TestEnablesEcalHcal = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetTestEnablesEcalHcal + process.hltPreDatasetTestEnablesEcalHcal )
process.Dataset_TestEnablesEcalHcalDQM = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetTestEnablesEcalHcalDQM + process.hltPreDatasetTestEnablesEcalHcalDQM )
process.Dataset_VRRandom0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom0 )
process.Dataset_VRRandom1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom1 )
process.Dataset_VRRandom2 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom2 )
process.Dataset_VRRandom3 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom3 )
process.Dataset_VRRandom4 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom4 )
process.Dataset_VRRandom5 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom5 )
process.Dataset_VRRandom6 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom6 )
process.Dataset_VRRandom7 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom7 )
process.Dataset_VRRandom8 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom8 )
process.Dataset_VRRandom9 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom9 )
process.Dataset_VRRandom10 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom10 )
process.Dataset_VRRandom11 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom11 )
process.Dataset_VRRandom12 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom12 )
process.Dataset_VRRandom13 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom13 )
process.Dataset_VRRandom14 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom14 )
process.Dataset_VRRandom15 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetVRRandom + process.hltPreDatasetVRRandom15 )
process.Dataset_ZeroBias = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetZeroBias + process.hltPreDatasetZeroBias )
process.Dataset_SpecialRandom0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom0 )
process.Dataset_SpecialRandom1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom1 )
process.Dataset_SpecialRandom2 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom2 )
process.Dataset_SpecialRandom3 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom3 )
process.Dataset_SpecialRandom4 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom4 )
process.Dataset_SpecialRandom5 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom5 )
process.Dataset_SpecialRandom6 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom6 )
process.Dataset_SpecialRandom7 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom7 )
process.Dataset_SpecialRandom8 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom8 )
process.Dataset_SpecialRandom9 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom9 )
process.Dataset_SpecialRandom10 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom10 )
process.Dataset_SpecialRandom11 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom11 )
process.Dataset_SpecialRandom12 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom12 )
process.Dataset_SpecialRandom13 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom13 )
process.Dataset_SpecialRandom14 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom14 )
process.Dataset_SpecialRandom15 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom15 )
process.Dataset_SpecialRandom16 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom16 )
process.Dataset_SpecialRandom17 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom17 )
process.Dataset_SpecialRandom18 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom18 )
process.Dataset_SpecialRandom19 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialRandom + process.hltPreDatasetSpecialRandom19 )
process.Dataset_SpecialZeroBias0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias0 )
process.Dataset_SpecialZeroBias1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias1 )
process.Dataset_SpecialZeroBias2 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias2 )
process.Dataset_SpecialZeroBias3 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias3 )
process.Dataset_SpecialZeroBias4 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias4 )
process.Dataset_SpecialZeroBias5 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias5 )
process.Dataset_SpecialZeroBias6 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias6 )
process.Dataset_SpecialZeroBias7 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias7 )
process.Dataset_SpecialZeroBias8 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias8 )
process.Dataset_SpecialZeroBias9 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias9 )
process.Dataset_SpecialZeroBias10 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias10 )
process.Dataset_SpecialZeroBias11 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias11 )
process.Dataset_SpecialZeroBias12 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias12 )
process.Dataset_SpecialZeroBias13 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias13 )
process.Dataset_SpecialZeroBias14 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias14 )
process.Dataset_SpecialZeroBias15 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias15 )
process.Dataset_SpecialZeroBias16 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias16 )
process.Dataset_SpecialZeroBias17 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias17 )
process.Dataset_SpecialZeroBias18 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias18 )
process.Dataset_SpecialZeroBias19 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias19 )
process.Dataset_SpecialZeroBias20 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias20 )
process.Dataset_SpecialZeroBias21 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias21 )
process.Dataset_SpecialZeroBias22 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias22 )
process.Dataset_SpecialZeroBias23 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias23 )
process.Dataset_SpecialZeroBias24 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias24 )
process.Dataset_SpecialZeroBias25 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias25 )
process.Dataset_SpecialZeroBias26 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias26 )
process.Dataset_SpecialZeroBias27 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias27 )
process.Dataset_SpecialZeroBias28 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias28 )
process.Dataset_SpecialZeroBias29 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias29 )
process.Dataset_SpecialZeroBias30 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias30 )
process.Dataset_SpecialZeroBias31 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialZeroBias + process.hltPreDatasetSpecialZeroBias31 )
process.Dataset_SpecialHLTPhysics0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics0 )
process.Dataset_SpecialHLTPhysics1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics1 )
process.Dataset_SpecialHLTPhysics2 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics2 )
process.Dataset_SpecialHLTPhysics3 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics3 )
process.Dataset_SpecialHLTPhysics4 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics4 )
process.Dataset_SpecialHLTPhysics5 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics5 )
process.Dataset_SpecialHLTPhysics6 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics6 )
process.Dataset_SpecialHLTPhysics7 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics7 )
process.Dataset_SpecialHLTPhysics8 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics8 )
process.Dataset_SpecialHLTPhysics9 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics9 )
process.Dataset_SpecialHLTPhysics10 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics10 )
process.Dataset_SpecialHLTPhysics11 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics11 )
process.Dataset_SpecialHLTPhysics12 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics12 )
process.Dataset_SpecialHLTPhysics13 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics13 )
process.Dataset_SpecialHLTPhysics14 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics14 )
process.Dataset_SpecialHLTPhysics15 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics15 )
process.Dataset_SpecialHLTPhysics16 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics16 )
process.Dataset_SpecialHLTPhysics17 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics17 )
process.Dataset_SpecialHLTPhysics18 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics18 )
process.Dataset_SpecialHLTPhysics19 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialHLTPhysics + process.hltPreDatasetSpecialHLTPhysics19 )
process.Dataset_SpecialMinimumBias0 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialMinimumBias + process.hltPreDatasetSpecialMinimumBias0 )
process.Dataset_SpecialMinimumBias1 = cms.Path( process.HLTDatasetPathBeginSequence + process.hltDatasetSpecialMinimumBias + process.hltPreDatasetSpecialMinimumBias1 )
process.ALCALumiPixelsCountsExpressOutput = cms.EndPath( process.hltOutputALCALumiPixelsCountsExpress )
process.ALCALumiPixelsCountsGatedOutput = cms.EndPath( process.hltOutputALCALumiPixelsCountsGated )
process.ALCALumiPixelsCountsPromptOutput = cms.EndPath( process.hltOutputALCALumiPixelsCountsPrompt )
process.ALCALumiPixelsCountsPromptHighRate0Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate0 )
process.ALCALumiPixelsCountsPromptHighRate1Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate1 )
process.ALCALumiPixelsCountsPromptHighRate2Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate2 )
process.ALCALumiPixelsCountsPromptHighRate3Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate3 )
process.ALCALumiPixelsCountsPromptHighRate4Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate4 )
process.ALCALumiPixelsCountsPromptHighRate5Output = cms.EndPath( process.hltOutputALCALumiPixelsCountsPromptHighRate5 )
process.ALCAP0Output = cms.EndPath( process.hltOutputALCAP0 )
process.ALCAPHISYMOutput = cms.EndPath( process.hltOutputALCAPHISYM )
process.ALCAPPSExpressOutput = cms.EndPath( process.hltOutputALCAPPSExpress )
process.ALCAPPSPromptOutput = cms.EndPath( process.hltOutputALCAPPSPrompt )
process.CalibrationOutput = cms.EndPath( process.hltOutputCalibration )

# load the DQMStore and DQMRootOutputModule
process.load( "DQMServices.Core.DQMStore_cfi" )

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)
process.DQMOutput = cms.EndPath( process.dqmOutput + process.hltOutputDQM )
process.DQMCalibrationOutput = cms.EndPath( process.hltOutputDQMCalibration )
process.DQMEventDisplayOutput = cms.EndPath( process.hltOutputDQMEventDisplay )
process.DQMGPUvsCPUOutput = cms.EndPath( process.hltOutputDQMGPUvsCPU )
process.DQMOnlineBeamspotOutput = cms.EndPath( process.hltOutputDQMOnlineBeamspot )
process.DQMPPSRandomOutput = cms.EndPath( process.hltOutputDQMPPSRandom )
process.EcalCalibrationOutput = cms.EndPath( process.hltOutputEcalCalibration )
process.ExpressOutput = cms.EndPath( process.hltOutputExpress )
process.ExpressAlignmentOutput = cms.EndPath( process.hltOutputExpressAlignment )
process.ExpressCosmicsOutput = cms.EndPath( process.hltOutputExpressCosmics )
process.HLTMonitorOutput = cms.EndPath( process.hltOutputHLTMonitor )
process.NanoDSTOutput = cms.EndPath( process.hltOutputNanoDST )
process.PhysicsCommissioningOutput = cms.EndPath( process.hltOutputPhysicsCommissioning )
process.PhysicsSpecialHLTPhysics0Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics0 )
process.PhysicsSpecialHLTPhysics1Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics1 )
process.PhysicsSpecialHLTPhysics10Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics10 )
process.PhysicsSpecialHLTPhysics11Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics11 )
process.PhysicsSpecialHLTPhysics12Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics12 )
process.PhysicsSpecialHLTPhysics13Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics13 )
process.PhysicsSpecialHLTPhysics14Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics14 )
process.PhysicsSpecialHLTPhysics15Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics15 )
process.PhysicsSpecialHLTPhysics16Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics16 )
process.PhysicsSpecialHLTPhysics17Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics17 )
process.PhysicsSpecialHLTPhysics18Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics18 )
process.PhysicsSpecialHLTPhysics19Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics19 )
process.PhysicsSpecialHLTPhysics2Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics2 )
process.PhysicsSpecialHLTPhysics3Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics3 )
process.PhysicsSpecialHLTPhysics4Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics4 )
process.PhysicsSpecialHLTPhysics5Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics5 )
process.PhysicsSpecialHLTPhysics6Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics6 )
process.PhysicsSpecialHLTPhysics7Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics7 )
process.PhysicsSpecialHLTPhysics8Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics8 )
process.PhysicsSpecialHLTPhysics9Output = cms.EndPath( process.hltOutputPhysicsSpecialHLTPhysics9 )
process.PhysicsSpecialMinimumBias0Output = cms.EndPath( process.hltOutputPhysicsSpecialMinimumBias0 )
process.PhysicsSpecialMinimumBias1Output = cms.EndPath( process.hltOutputPhysicsSpecialMinimumBias1 )
process.PhysicsSpecialRandom0Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom0 )
process.PhysicsSpecialRandom1Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom1 )
process.PhysicsSpecialRandom2Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom2 )
process.PhysicsSpecialRandom3Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom3 )
process.PhysicsSpecialRandom4Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom4 )
process.PhysicsSpecialRandom5Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom5 )
process.PhysicsSpecialRandom6Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom6 )
process.PhysicsSpecialRandom7Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom7 )
process.PhysicsSpecialRandom8Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom8 )
process.PhysicsSpecialRandom9Output = cms.EndPath( process.hltOutputPhysicsSpecialRandom9 )
process.PhysicsSpecialZeroBias0Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias0 )
process.PhysicsSpecialZeroBias1Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias1 )
process.PhysicsSpecialZeroBias10Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias10 )
process.PhysicsSpecialZeroBias11Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias11 )
process.PhysicsSpecialZeroBias12Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias12 )
process.PhysicsSpecialZeroBias13Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias13 )
process.PhysicsSpecialZeroBias14Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias14 )
process.PhysicsSpecialZeroBias15Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias15 )
process.PhysicsSpecialZeroBias2Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias2 )
process.PhysicsSpecialZeroBias3Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias3 )
process.PhysicsSpecialZeroBias4Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias4 )
process.PhysicsSpecialZeroBias5Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias5 )
process.PhysicsSpecialZeroBias6Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias6 )
process.PhysicsSpecialZeroBias7Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias7 )
process.PhysicsSpecialZeroBias8Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias8 )
process.PhysicsSpecialZeroBias9Output = cms.EndPath( process.hltOutputPhysicsSpecialZeroBias9 )
process.PhysicsVRRandom0Output = cms.EndPath( process.hltOutputPhysicsVRRandom0 )
process.PhysicsVRRandom1Output = cms.EndPath( process.hltOutputPhysicsVRRandom1 )
process.PhysicsVRRandom2Output = cms.EndPath( process.hltOutputPhysicsVRRandom2 )
process.PhysicsVRRandom3Output = cms.EndPath( process.hltOutputPhysicsVRRandom3 )
process.PhysicsVRRandom4Output = cms.EndPath( process.hltOutputPhysicsVRRandom4 )
process.PhysicsVRRandom5Output = cms.EndPath( process.hltOutputPhysicsVRRandom5 )
process.PhysicsVRRandom6Output = cms.EndPath( process.hltOutputPhysicsVRRandom6 )
process.PhysicsVRRandom7Output = cms.EndPath( process.hltOutputPhysicsVRRandom7 )
process.RPCMONOutput = cms.EndPath( process.hltOutputRPCMON )


process.schedule = cms.Schedule( *(process.HLTriggerFirstPath, process.Status_OnCPU, process.Status_OnGPU, process.AlCa_EcalPhiSym_v20, process.AlCa_EcalEtaEBonly_v25, process.AlCa_EcalEtaEEonly_v25, process.AlCa_EcalPi0EBonly_v25, process.AlCa_EcalPi0EEonly_v25, process.AlCa_RPCMuonNormalisation_v23, process.AlCa_LumiPixelsCounts_Random_v10, process.AlCa_LumiPixelsCounts_ZeroBias_v12, process.DQM_PixelReconstruction_v12, process.DQM_EcalReconstruction_v12, process.DQM_HcalReconstruction_v10, process.DQM_Random_v1, process.DQM_ZeroBias_v3, process.DST_ZeroBias_v11, process.DST_Physics_v16, process.HLT_EcalCalibration_v4, process.HLT_HcalCalibration_v6, process.HLT_HcalNZS_v21, process.HLT_HcalPhiSym_v23, process.HLT_Random_v3, process.HLT_Physics_v14, process.HLT_ZeroBias_v13, process.HLT_ZeroBias_Alignment_v8, process.HLT_ZeroBias_Beamspot_v16, process.HLT_ZeroBias_IsolatedBunches_v12, process.HLT_ZeroBias_FirstBXAfterTrain_v10, process.HLT_ZeroBias_FirstCollisionAfterAbortGap_v12, process.HLT_ZeroBias_FirstCollisionInTrain_v11, process.HLT_ZeroBias_LastCollisionInTrain_v10, process.HLT_HT300_Beamspot_v23, process.HLT_IsoTrackHB_v14, process.HLT_IsoTrackHE_v14, process.HLT_L1SingleMuCosmics_v8, process.HLT_L2Mu10_NoVertex_NoBPTX3BX_v14, process.HLT_L2Mu10_NoVertex_NoBPTX_v15, process.HLT_L2Mu45_NoVertex_3Sta_NoBPTX3BX_v13, process.HLT_L2Mu40_NoVertex_3Sta_NoBPTX3BX_v14, process.HLT_CDC_L2cosmic_10_er1p0_v10, process.HLT_CDC_L2cosmic_5p5_er1p0_v10, process.HLT_PPSMaxTracksPerArm1_v9, process.HLT_PPSMaxTracksPerRP4_v9, process.HLT_PPSRandom_v1, process.HLT_SpecialHLTPhysics_v7, process.AlCa_LumiPixelsCounts_RandomHighRate_v4, process.AlCa_LumiPixelsCounts_ZeroBiasVdM_v4, process.AlCa_LumiPixelsCounts_ZeroBiasGated_v5, process.HLT_L1SingleMuOpen_v6, process.HLT_L1SingleMuOpen_DT_v6, process.HLT_L1SingleMu3_v5, process.HLT_L1SingleMu5_v5, process.HLT_L1SingleMu7_v5, process.HLT_L1DoubleMu0_v5, process.HLT_L1SingleJet8erHE_v5, process.HLT_L1SingleJet10erHE_v5, process.HLT_L1SingleJet12erHE_v5, process.HLT_L1SingleJet35_v5, process.HLT_L1SingleJet200_v5, process.HLT_L1SingleEG8er2p5_v4, process.HLT_L1SingleEG10er2p5_v4, process.HLT_L1SingleEG15er2p5_v4, process.HLT_L1SingleEG26er2p5_v4, process.HLT_L1SingleEG28er2p5_v4, process.HLT_L1SingleEG28er2p1_v4, process.HLT_L1SingleEG28er1p5_v4, process.HLT_L1SingleEG34er2p5_v4, process.HLT_L1SingleEG36er2p5_v4, process.HLT_L1SingleEG38er2p5_v4, process.HLT_L1SingleEG40er2p5_v4, process.HLT_L1SingleEG42er2p5_v4, process.HLT_L1SingleEG45er2p5_v4, process.HLT_L1SingleEG50_v4, process.HLT_L1SingleJet60_v4, process.HLT_L1SingleJet90_v4, process.HLT_L1SingleJet120_v4, process.HLT_L1SingleJet180_v4, process.HLT_L1HTT120er_v4, process.HLT_L1HTT160er_v4, process.HLT_L1HTT200er_v4, process.HLT_L1HTT255er_v4, process.HLT_L1HTT280er_v4, process.HLT_L1HTT320er_v4, process.HLT_L1HTT360er_v4, process.HLT_L1HTT400er_v4, process.HLT_L1HTT450er_v4, process.HLT_L1ETM120_v4, process.HLT_L1ETM150_v4, process.HLT_L1EXT_HCAL_LaserMon1_v5, process.HLT_L1EXT_HCAL_LaserMon4_v5, process.HLT_L1MinimumBiasHF0ANDBptxAND_v1, process.HLT_CscCluster_Cosmic_v4, process.HLT_HT60_Beamspot_v22, process.HLT_HT300_Beamspot_PixelClusters_WP2_v7, process.HLT_PixelClusters_WP2_v4, process.HLT_PixelClusters_WP2_HighRate_v1, process.HLT_PixelClusters_WP1_v4, process.HLT_BptxOR_v6, process.HLT_L1SingleMuCosmics_EMTF_v4, process.HLT_L1SingleMuCosmics_CosmicTracking_v1, process.HLT_L1SingleMuCosmics_PointingCosmicTracking_v1, process.HLT_L1FatEvents_v5, process.HLT_Random_HighRate_v1, process.HLT_ZeroBias_HighRate_v4, process.HLT_ZeroBias_Gated_v4, process.HLT_SpecialZeroBias_v6, process.HLTriggerFinalPath, process.HLTAnalyzerEndpath, process.Dataset_AlCaLumiPixelsCountsExpress, process.Dataset_AlCaLumiPixelsCountsPrompt, process.Dataset_AlCaLumiPixelsCountsPromptHighRate0, process.Dataset_AlCaLumiPixelsCountsPromptHighRate1, process.Dataset_AlCaLumiPixelsCountsPromptHighRate2, process.Dataset_AlCaLumiPixelsCountsPromptHighRate3, process.Dataset_AlCaLumiPixelsCountsPromptHighRate4, process.Dataset_AlCaLumiPixelsCountsPromptHighRate5, process.Dataset_AlCaLumiPixelsCountsGated, process.Dataset_AlCaP0, process.Dataset_AlCaPPSExpress, process.Dataset_AlCaPPSPrompt, process.Dataset_AlCaPhiSym, process.Dataset_Commissioning, process.Dataset_Cosmics, process.Dataset_DQMGPUvsCPU, process.Dataset_DQMOnlineBeamspot, process.Dataset_DQMPPSRandom, process.Dataset_EcalLaser, process.Dataset_EventDisplay, process.Dataset_ExpressAlignment, process.Dataset_ExpressCosmics, process.Dataset_ExpressPhysics, process.Dataset_HLTMonitor, process.Dataset_HLTPhysics, process.Dataset_HcalNZS, process.Dataset_L1Accept, process.Dataset_MinimumBias, process.Dataset_MuonShower, process.Dataset_NoBPTX, process.Dataset_OnlineMonitor, process.Dataset_RPCMonitor, process.Dataset_TestEnablesEcalHcal, process.Dataset_TestEnablesEcalHcalDQM, process.Dataset_VRRandom0, process.Dataset_VRRandom1, process.Dataset_VRRandom2, process.Dataset_VRRandom3, process.Dataset_VRRandom4, process.Dataset_VRRandom5, process.Dataset_VRRandom6, process.Dataset_VRRandom7, process.Dataset_VRRandom8, process.Dataset_VRRandom9, process.Dataset_VRRandom10, process.Dataset_VRRandom11, process.Dataset_VRRandom12, process.Dataset_VRRandom13, process.Dataset_VRRandom14, process.Dataset_VRRandom15, process.Dataset_ZeroBias, process.Dataset_SpecialRandom0, process.Dataset_SpecialRandom1, process.Dataset_SpecialRandom2, process.Dataset_SpecialRandom3, process.Dataset_SpecialRandom4, process.Dataset_SpecialRandom5, process.Dataset_SpecialRandom6, process.Dataset_SpecialRandom7, process.Dataset_SpecialRandom8, process.Dataset_SpecialRandom9, process.Dataset_SpecialRandom10, process.Dataset_SpecialRandom11, process.Dataset_SpecialRandom12, process.Dataset_SpecialRandom13, process.Dataset_SpecialRandom14, process.Dataset_SpecialRandom15, process.Dataset_SpecialRandom16, process.Dataset_SpecialRandom17, process.Dataset_SpecialRandom18, process.Dataset_SpecialRandom19, process.Dataset_SpecialZeroBias0, process.Dataset_SpecialZeroBias1, process.Dataset_SpecialZeroBias2, process.Dataset_SpecialZeroBias3, process.Dataset_SpecialZeroBias4, process.Dataset_SpecialZeroBias5, process.Dataset_SpecialZeroBias6, process.Dataset_SpecialZeroBias7, process.Dataset_SpecialZeroBias8, process.Dataset_SpecialZeroBias9, process.Dataset_SpecialZeroBias10, process.Dataset_SpecialZeroBias11, process.Dataset_SpecialZeroBias12, process.Dataset_SpecialZeroBias13, process.Dataset_SpecialZeroBias14, process.Dataset_SpecialZeroBias15, process.Dataset_SpecialZeroBias16, process.Dataset_SpecialZeroBias17, process.Dataset_SpecialZeroBias18, process.Dataset_SpecialZeroBias19, process.Dataset_SpecialZeroBias20, process.Dataset_SpecialZeroBias21, process.Dataset_SpecialZeroBias22, process.Dataset_SpecialZeroBias23, process.Dataset_SpecialZeroBias24, process.Dataset_SpecialZeroBias25, process.Dataset_SpecialZeroBias26, process.Dataset_SpecialZeroBias27, process.Dataset_SpecialZeroBias28, process.Dataset_SpecialZeroBias29, process.Dataset_SpecialZeroBias30, process.Dataset_SpecialZeroBias31, process.Dataset_SpecialHLTPhysics0, process.Dataset_SpecialHLTPhysics1, process.Dataset_SpecialHLTPhysics2, process.Dataset_SpecialHLTPhysics3, process.Dataset_SpecialHLTPhysics4, process.Dataset_SpecialHLTPhysics5, process.Dataset_SpecialHLTPhysics6, process.Dataset_SpecialHLTPhysics7, process.Dataset_SpecialHLTPhysics8, process.Dataset_SpecialHLTPhysics9, process.Dataset_SpecialHLTPhysics10, process.Dataset_SpecialHLTPhysics11, process.Dataset_SpecialHLTPhysics12, process.Dataset_SpecialHLTPhysics13, process.Dataset_SpecialHLTPhysics14, process.Dataset_SpecialHLTPhysics15, process.Dataset_SpecialHLTPhysics16, process.Dataset_SpecialHLTPhysics17, process.Dataset_SpecialHLTPhysics18, process.Dataset_SpecialHLTPhysics19, process.Dataset_SpecialMinimumBias0, process.Dataset_SpecialMinimumBias1, process.ALCALumiPixelsCountsExpressOutput, process.ALCALumiPixelsCountsGatedOutput, process.ALCALumiPixelsCountsPromptOutput, process.ALCALumiPixelsCountsPromptHighRate0Output, process.ALCALumiPixelsCountsPromptHighRate1Output, process.ALCALumiPixelsCountsPromptHighRate2Output, process.ALCALumiPixelsCountsPromptHighRate3Output, process.ALCALumiPixelsCountsPromptHighRate4Output, process.ALCALumiPixelsCountsPromptHighRate5Output, process.ALCAP0Output, process.ALCAPHISYMOutput, process.ALCAPPSExpressOutput, process.ALCAPPSPromptOutput, process.CalibrationOutput, process.DQMOutput, process.DQMCalibrationOutput, process.DQMEventDisplayOutput, process.DQMGPUvsCPUOutput, process.DQMOnlineBeamspotOutput, process.DQMPPSRandomOutput, process.EcalCalibrationOutput, process.ExpressOutput, process.ExpressAlignmentOutput, process.ExpressCosmicsOutput, process.HLTMonitorOutput, process.NanoDSTOutput, process.PhysicsCommissioningOutput, process.PhysicsSpecialHLTPhysics0Output, process.PhysicsSpecialHLTPhysics1Output, process.PhysicsSpecialHLTPhysics10Output, process.PhysicsSpecialHLTPhysics11Output, process.PhysicsSpecialHLTPhysics12Output, process.PhysicsSpecialHLTPhysics13Output, process.PhysicsSpecialHLTPhysics14Output, process.PhysicsSpecialHLTPhysics15Output, process.PhysicsSpecialHLTPhysics16Output, process.PhysicsSpecialHLTPhysics17Output, process.PhysicsSpecialHLTPhysics18Output, process.PhysicsSpecialHLTPhysics19Output, process.PhysicsSpecialHLTPhysics2Output, process.PhysicsSpecialHLTPhysics3Output, process.PhysicsSpecialHLTPhysics4Output, process.PhysicsSpecialHLTPhysics5Output, process.PhysicsSpecialHLTPhysics6Output, process.PhysicsSpecialHLTPhysics7Output, process.PhysicsSpecialHLTPhysics8Output, process.PhysicsSpecialHLTPhysics9Output, process.PhysicsSpecialMinimumBias0Output, process.PhysicsSpecialMinimumBias1Output, process.PhysicsSpecialRandom0Output, process.PhysicsSpecialRandom1Output, process.PhysicsSpecialRandom2Output, process.PhysicsSpecialRandom3Output, process.PhysicsSpecialRandom4Output, process.PhysicsSpecialRandom5Output, process.PhysicsSpecialRandom6Output, process.PhysicsSpecialRandom7Output, process.PhysicsSpecialRandom8Output, process.PhysicsSpecialRandom9Output, process.PhysicsSpecialZeroBias0Output, process.PhysicsSpecialZeroBias1Output, process.PhysicsSpecialZeroBias10Output, process.PhysicsSpecialZeroBias11Output, process.PhysicsSpecialZeroBias12Output, process.PhysicsSpecialZeroBias13Output, process.PhysicsSpecialZeroBias14Output, process.PhysicsSpecialZeroBias15Output, process.PhysicsSpecialZeroBias2Output, process.PhysicsSpecialZeroBias3Output, process.PhysicsSpecialZeroBias4Output, process.PhysicsSpecialZeroBias5Output, process.PhysicsSpecialZeroBias6Output, process.PhysicsSpecialZeroBias7Output, process.PhysicsSpecialZeroBias8Output, process.PhysicsSpecialZeroBias9Output, process.PhysicsVRRandom0Output, process.PhysicsVRRandom1Output, process.PhysicsVRRandom2Output, process.PhysicsVRRandom3Output, process.PhysicsVRRandom4Output, process.PhysicsVRRandom5Output, process.PhysicsVRRandom6Output, process.PhysicsVRRandom7Output, process.RPCMONOutput, ))


# source module (EDM inputs)
process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:RelVal_Raw_Special_DATA.root',
    ),
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

# limit the number of events to be processed
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( 100 )
)

# enable TrigReport, TimeReport and MultiThreading
process.options.wantSummary = True
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0

# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    from Configuration.AlCa.GlobalTag import GlobalTag as customiseGlobalTag
    process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = 'auto:run3_hlt_Special')

# show summaries from trigger analysers used at HLT
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.TriggerSummaryProducerAOD = cms.untracked.PSet()
    process.MessageLogger.L1GtTrigReport = cms.untracked.PSet()
    process.MessageLogger.L1TGlobalSummary = cms.untracked.PSet()
    process.MessageLogger.HLTrigReport = cms.untracked.PSet()
    process.MessageLogger.FastReport = cms.untracked.PSet()
    process.MessageLogger.ThroughputService = cms.untracked.PSet()

# add specific customizations
_customInfo = {}
_customInfo['menuType'  ]= "Special"
_customInfo['globalTags']= {}
_customInfo['globalTags'][True ] = "auto:run3_hlt_Special"
_customInfo['globalTags'][False] = "auto:run3_mc_Special"
_customInfo['inputFiles']={}
_customInfo['inputFiles'][True]  = "file:RelVal_Raw_Special_DATA.root"
_customInfo['inputFiles'][False] = "file:RelVal_Raw_Special_MC.root"
_customInfo['maxEvents' ]=  100
_customInfo['globalTag' ]= "auto:run3_hlt_Special"
_customInfo['inputFile' ]=  ['file:RelVal_Raw_Special_DATA.root']
_customInfo['realData'  ]=  True

from HLTrigger.Configuration.customizeHLTforALL import customizeHLTforAll
process = customizeHLTforAll(process,"Special",_customInfo)

from HLTrigger.Configuration.customizeHLTforCMSSW import customizeHLTforCMSSW
process = customizeHLTforCMSSW(process,"Special")

# Eras-based customisations
from HLTrigger.Configuration.Eras import modifyHLTforEras
modifyHLTforEras(process)

