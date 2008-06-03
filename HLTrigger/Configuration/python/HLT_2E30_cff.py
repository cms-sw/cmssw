# /dev/CMSSW_2_1_0_pre5/HLT/V3 (CMSSW_2_1_0_pre5)

import FWCore.ParameterSet.Config as cms

BTagRecord = cms.ESSource( "EmptyESSource",
  recordName = cms.string( "JetTagComputerRecord" ),
  iovIsRunNotTime = cms.bool( True ),
  firstValid = cms.vuint32( 1 )
)
MCJetCorrectorIcone5 = cms.ESSource( "MCJetCorrectionService",
  tagName = cms.string( "CMSSW_152_iterativeCone5" ),
  label = cms.string( "MCJetCorrectorIcone5" )
)

CaloTopologyBuilder = cms.ESProducer( "CaloTopologyBuilder" )
Chi2EstimatorForL2Refit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForL2Refit" ),
  MaxChi2 = cms.double( 1000.0 ),
  nSigma = cms.double( 3.0 )
)
KFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  minHits = cms.int32( 3 )
)
KFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 )
)
KFFitterSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "KFFitterSmootherForL2Muon" ),
  Fitter = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "KFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True )
)
CaloTowerConstituentsMapBuilder = cms.ESProducer( "CaloTowerConstituentsMapBuilder",
  MapFile = cms.untracked.string( "Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz" )
)
Chi2EstimatorForL3Refit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForL3Refit" ),
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 )
)
Chi2EstimatorForMuonTrackLoader = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForMuonTrackLoader" ),
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 )
)
Chi2EstimatorForRefit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForRefit" ),
  MaxChi2 = cms.double( 100000.0 ),
  nSigma = cms.double( 3.0 )
)
Chi2MeasurementEstimator = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2" ),
  MaxChi2 = cms.double( 30.0 ),
  nSigma = cms.double( 3.0 )
)
CkfTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "CkfTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "ckfBaseTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True )
)
FitterRK = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "FitterRK" ),
  Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  minHits = cms.int32( 3 )
)
FittingSmootherRK = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "FittingSmootherRK" ),
  Fitter = cms.string( "FitterRK" ),
  Smoother = cms.string( "SmootherRK" ),
  EstimateCut = cms.double( -1.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True )
)
GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" )
GroupedCkfTrajectoryBuilder = cms.ESProducer( "GroupedCkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "GroupedCkfTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "ckfBaseTrajectoryFilter" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  foundHitBonus = cms.double( 5.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( True ),
  lockHits = cms.bool( True ),
  bestHitOnly = cms.bool( True ),
  requireSeedHitsInRebuild = cms.bool( True ),
  minNrOfHitsForRebuild = cms.int32( 5 )
)
KFFitterForRefitInsideOut = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "KFFitterForRefitInsideOut" ),
  Propagator = cms.string( "SmartPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForRefit" ),
  minHits = cms.int32( 3 )
)
KFSmootherForMuonTrackLoader = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFSmootherForMuonTrackLoader" ),
  Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForMuonTrackLoader" ),
  errorRescaling = cms.double( 10.0 ),
  minHits = cms.int32( 3 )
)
KFSmootherForRefitInsideOut = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFSmootherForRefitInsideOut" ),
  Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForRefit" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 )
)
KFUpdatorESProducer = cms.ESProducer( "KFUpdatorESProducer",
  ComponentName = cms.string( "KFUpdator" )
)
L3MuKFFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "L3MuKFFitter" ),
  Propagator = cms.string( "SmartPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL3Refit" ),
  minHits = cms.int32( 3 )
)
MaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterial" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
MeasurementTracker = cms.ESProducer( "MeasurementTrackerESProducer",
  ComponentName = cms.string( "" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  HitMatcher = cms.string( "StandardMatcher" ),
  Regional = cms.bool( True ),
  OnDemand = cms.bool( True ),
  UseStripModuleQualityDB = cms.bool( False ),
  DebugStripModuleQualityDB = cms.untracked.bool( False ),
  UseStripAPVFiberQualityDB = cms.bool( False ),
  DebugStripAPVFiberQualityDB = cms.untracked.bool( False ),
  UseStripStripQualityDB = cms.bool( False ),
  DebugStripStripQualityDB = cms.untracked.bool( False ),
  pixelClusterProducer = cms.string( "hltSiPixelClusters" ),
  stripClusterProducer = cms.string( "hltSiStripClusters" ),
  stripLazyGetterProducer = cms.string( "hltSiStripRawToClustersFacility" )
)
MuonCkfTrajectoryBuilder = cms.ESProducer( "MuonCkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "muonCkfTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  propagatorProximity = cms.string( "SteppingHelixPropagatorAny" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "muonCkfTrajectoryFilter" ),
  useSeedLayer = cms.bool( False ),
  rescaleErrorIfFail = cms.double( 1.0 ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( False ),
  alwaysUseInvalidHits = cms.bool( True )
)
MuonDetLayerGeometryESProducer = cms.ESProducer( "MuonDetLayerGeometryESProducer" )
MuonTransientTrackingRecHitBuilderESProducer = cms.ESProducer( "MuonTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "MuonRecHitBuilder" )
)
OppositeMaterialPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "PropagatorWithMaterialOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( False )
)
PixelCPEGenericESProducer = cms.ESProducer( "PixelCPEGenericESProducer",
  ComponentName = cms.string( "PixelCPEGeneric" ),
  eff_charge_cut_lowX = cms.untracked.double( 0.0 ),
  eff_charge_cut_lowY = cms.untracked.double( 0.0 ),
  eff_charge_cut_highX = cms.untracked.double( 1.0 ),
  eff_charge_cut_highY = cms.untracked.double( 1.0 ),
  size_cutX = cms.untracked.double( 3.0 ),
  size_cutY = cms.untracked.double( 3.0 ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True )
)
RKTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "RKTrackerPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
RungeKuttaTrackerPropagator = cms.ESProducer( "PropagatorWithMaterialESProducer",
  ComponentName = cms.string( "RungeKuttaTrackerPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Mass = cms.double( 0.105 ),
  MaxDPhi = cms.double( 1.6 ),
  useRungeKutta = cms.bool( True )
)
SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
)
SmartPropagator = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagator" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAlong" )
)
SmartPropagatorAny = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorAny" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterial" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" )
)
SmartPropagatorAnyOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorAnyOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" )
)
SmartPropagatorAnyRK = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorAnyRK" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "RKTrackerPropagator" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAny" )
)
SmartPropagatorOpposite = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "PropagatorWithMaterialOpposite" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorOpposite" )
)
SmartPropagatorRK = cms.ESProducer( "SmartPropagatorESProducer",
  ComponentName = cms.string( "SmartPropagatorRK" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  Epsilon = cms.double( 5.0 ),
  TrackerPropagator = cms.string( "RKTrackerPropagator" ),
  MuonPropagator = cms.string( "SteppingHelixPropagatorAlong" )
)
SmootherRK = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "SmootherRK" ),
  Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 )
)
SteppingHelixPropagatorAlong = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorAlong" ),
  PropagationDirection = cms.string( "alongMomentum" ),
  SetVBFPointer = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useInTeslaFromMagField = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False )
)
SteppingHelixPropagatorAny = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorAny" ),
  PropagationDirection = cms.string( "anyDirection" ),
  SetVBFPointer = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useInTeslaFromMagField = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False )
)
SteppingHelixPropagatorOpposite = cms.ESProducer( "SteppingHelixPropagatorESProducer",
  ComponentName = cms.string( "SteppingHelixPropagatorOpposite" ),
  PropagationDirection = cms.string( "oppositeToMomentum" ),
  SetVBFPointer = cms.bool( False ),
  VBFName = cms.string( "VolumeBasedMagneticField" ),
  useInTeslaFromMagField = cms.bool( False ),
  ApplyRadX0Correction = cms.bool( True ),
  AssumeNoMaterial = cms.bool( False ),
  NoErrorPropagation = cms.bool( False ),
  debug = cms.bool( False ),
  useMagVolumes = cms.bool( True ),
  useMatVolumes = cms.bool( True ),
  useIsYokeFlag = cms.bool( True ),
  returnTangentPlane = cms.bool( True ),
  sendLogWarning = cms.bool( False ),
  useTuningForL2Speed = cms.bool( False )
)
TrackerRecoGeometryESProducer = cms.ESProducer( "TrackerRecoGeometryESProducer"
)
TransientTrackBuilderESProducer = cms.ESProducer( "TransientTrackBuilderESProducer",
  ComponentName = cms.string( "TransientTrackBuilder" )
)
bJetRegionalTrajectoryBuilder = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryBuilder" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "bJetRegionalTrajectoryFilter" ),
  maxCand = cms.int32( 1 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
bJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryFilter" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 1.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
ckfBaseTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "ckfBaseTrajectoryFilter" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( -1 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
hltCkfTrajectoryBuilderMumu = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilderMumu" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  maxCand = cms.int32( 3 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
hltCkfTrajectoryBuilderMumuk = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryBuilderMumuk" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "hltCkfTrajectoryFilterMumuk" ),
  maxCand = cms.int32( 3 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
hltCkfTrajectoryFilterMumu = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 3.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
hltCkfTrajectoryFilterMumuk = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumuk" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 3.0 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
muonCkfTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "muonCkfTrajectoryFilter" ),
  filterPset = cms.PSet( 
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxConsecLostHits = cms.int32( 1 ),
    minimumNumberOfHits = cms.int32( 5 ),
    minPt = cms.double( 0.9 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minHitsMinPt = cms.int32( 3 ),
    maxNumberOfHits = cms.int32( -1 ),
    chargeSignificance = cms.double( -1.0 )
  )
)
navigationSchoolESProducer = cms.ESProducer( "NavigationSchoolESProducer",
  ComponentName = cms.string( "SimpleNavigationSchool" )
)
pixellayerpairs = cms.ESProducer( "PixelLayerPairsESProducer",
  ComponentName = cms.string( "PixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    hitErrorRZ = cms.double( 0.0036 )
  )
)
TTRHBuilderPixelOnly = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "TTRHBuilderPixelOnly" ),
  StripCPE = cms.string( "Fake" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" )
)
WithTrackAngle = cms.ESProducer( "TkTransientTrackingRecHitBuilderESProducer",
  ComponentName = cms.string( "WithTrackAngle" ),
  StripCPE = cms.string( "StripCPEfromTrackAngle" ),
  PixelCPE = cms.string( "PixelCPEGeneric" ),
  Matcher = cms.string( "StandardMatcher" )
)
pixellayertriplets = cms.ESProducer( "PixelLayerTripletsESProducer",
  ComponentName = cms.string( "PixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    useErrorsFromParam = cms.untracked.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    hitErrorRZ = cms.double( 0.0036 )
  )
)
softLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  distance = cms.double( 0.5 )
)
softLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer" )
trackCounting3D2nd = cms.ESProducer( "TrackCountingESProducer",
  nthTrack = cms.int32( 2 ),
  impactParameterType = cms.int32( 0 ),
  deltaR = cms.double( -1.0 ),
  maximumDecayLength = cms.double( 5.0 ),
  maximumDistanceToJetAxis = cms.double( 0.07 ),
  trackQualityClass = cms.string( "any" )
)
trajBuilderL3 = cms.ESProducer( "CkfTrajectoryBuilderESProducer",
  ComponentName = cms.string( "trajBuilderL3" ),
  updator = cms.string( "KFUpdator" ),
  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
  estimator = cms.string( "Chi2" ),
  TTRHBuilder = cms.string( "WithTrackAngle" ),
  MeasurementTrackerName = cms.string( "" ),
  trajectoryFilterName = cms.string( "trajFilterL3" ),
  maxCand = cms.int32( 5 ),
  lostHitPenalty = cms.double( 30.0 ),
  intermediateCleaning = cms.bool( True ),
  alwaysUseInvalidHits = cms.bool( False )
)
trajFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "trajFilterL3" ),
  filterPset = cms.PSet( 
    minHitsMinPt = cms.int32( 3 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minPt = cms.double( 0.9 ),
    minimumNumberOfHits = cms.int32( 5 ),
    maxNumberOfHits = cms.int32( 7 ),
    maxConsecLostHits = cms.int32( 1 ),
    maxLostHits = cms.int32( 1 ),
    chargeSignificance = cms.double( -1.0 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" )
  )
)
trajectoryCleanerBySharedHits = cms.ESProducer( "TrajectoryCleanerESProducer",
  ComponentName = cms.string( "TrajectoryCleanerBySharedHits" )
)

UpdaterService = cms.Service( "UpdaterService",
)

hlt2GetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
hltGtDigis = cms.EDProducer( "L1GlobalTriggerRawToDigi",
    DaqGtInputTag = cms.InputTag( "rawDataCollector" ),
    DaqGtFedId = cms.untracked.int32( 813 ),
    ActiveBoardsMask = cms.uint32( 0x101 ),
    UnpackBxInEvent = cms.int32( 1 )
)
hltGctDigis = cms.EDProducer( "GctRawToDigi",
    inputLabel = cms.InputTag( "rawDataCollector" ),
    gctFedId = cms.int32( 745 ),
    hltMode = cms.bool( False ),
    grenCompatibilityMode = cms.bool( False ),
    unpackInternEm = cms.untracked.bool( False ),
    unpackFibres = cms.untracked.bool( False )
)
hltL1GtObjectMap = cms.EDProducer( "L1GlobalTrigger",
    GmtInputTag = cms.InputTag( "hltGtDigis" ),
    GctInputTag = cms.InputTag( "hltGctDigis" ),
    TechnicalTriggersInputTag = cms.InputTag( "techTrigDigis" ),
    ProduceL1GtDaqRecord = cms.bool( False ),
    ProduceL1GtEvmRecord = cms.bool( False ),
    ProduceL1GtObjectMapRecord = cms.bool( True ),
    WritePsbL1GtDaqRecord = cms.bool( False ),
    ReadTechnicalTriggerRecords = cms.bool( True ),
    EmulateBxInEvent = cms.int32( 1 )
)
hltL1extraParticles = cms.EDProducer( "L1ExtraParticlesProd",
    produceMuonParticles = cms.bool( True ),
    muonSource = cms.InputTag( "hltGtDigis" ),
    produceCaloParticles = cms.bool( True ),
    isolatedEmSource = cms.InputTag( 'hltGctDigis','isoEm' ),
    nonIsolatedEmSource = cms.InputTag( 'hltGctDigis','nonIsoEm' ),
    centralJetSource = cms.InputTag( 'hltGctDigis','cenJets' ),
    forwardJetSource = cms.InputTag( 'hltGctDigis','forJets' ),
    tauJetSource = cms.InputTag( 'hltGctDigis','tauJets' ),
    etTotalSource = cms.InputTag( "hltGctDigis" ),
    etHadSource = cms.InputTag( "hltGctDigis" ),
    etMissSource = cms.InputTag( "hltGctDigis" ),
    centralBxOnly = cms.bool( True )
)
hltOfflineBeamSpot = cms.EDProducer( "BeamSpotProducer" )
hltBoolFirst = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltBoolEnd = cms.EDFilter( "HLTBool",
    result = cms.bool( True )
)
hltL1s2jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre2jet = cms.EDFilter( "HLTPrescaler" )
hltEcalPreshowerDigis = cms.EDProducer( "ESRawToDigi",
    Label = cms.string( "rawDataCollector" ),
    InstanceES = cms.string( "" ),
    ESdigiCollection = cms.string( "" )
)
hltEcalRegionalJetsFEDs = cms.EDProducer( "EcalListOfFEDSProducer",
    debug = cms.untracked.bool( False ),
    Jets = cms.untracked.bool( True ),
    Ptmin_jets = cms.untracked.double( 50.0 ),
    CentralSource = cms.untracked.InputTag( 'hltL1extraParticles','Central' ),
    ForwardSource = cms.untracked.InputTag( 'hltL1extraParticles','Forward' ),
    TauSource = cms.untracked.InputTag( 'hltL1extraParticles','Tau' ),
    OutputLabel = cms.untracked.string( "" )
)
hltEcalRegionalJetsDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    DoRegional = cms.untracked.bool( True ),
    FedLabel = cms.untracked.InputTag( "hltEcalRegionalJetsFEDs" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalRegionalJetsWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalRegionalJetsDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalRegionalJetsDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRegionalJetsRecHitTmp = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalJetsWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalJetsWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltEcalRegionalJetsRecHit = cms.EDProducer( "EcalRecHitsMerger",
    debug = cms.untracked.bool( False ),
    EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
    MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
    TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
    JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
    RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
    EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
    MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
    TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
    JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
    RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
    OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
    OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
    EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
    EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
)
hltEcalPreshowerRecHit = cms.EDProducer( "ESRecHitProducer",
    ESdigiCollection = cms.InputTag( "hltEcalPreshowerDigis" ),
    ESrechitCollection = cms.string( "EcalRecHitsES" )
)
hltHcalDigis = cms.EDProducer( "HcalRawToDigi",
    InputLabel = cms.InputTag( "rawDataCollector" ),
    UnpackCalib = cms.untracked.bool( True ),
    UnpackZDC = cms.untracked.bool( True ),
    firstSample = cms.int32( 0 ),
    lastSample = cms.int32( 9 ),
    FilterDataQuality = cms.bool( True )
)
hltHbhereco = cms.EDProducer( "HcalSimpleReconstructor",
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    Subdetector = cms.string( "HBHE" ),
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 )
)
hltHfreco = cms.EDProducer( "HcalSimpleReconstructor",
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    Subdetector = cms.string( "HF" ),
    firstSample = cms.int32( 3 ),
    samplesToAdd = cms.int32( 1 ),
    correctForTimeslew = cms.bool( False ),
    correctForPhaseContainment = cms.bool( False ),
    correctionPhaseNS = cms.double( 0.0 )
)
hltHoreco = cms.EDProducer( "HcalSimpleReconstructor",
    digiLabel = cms.InputTag( "hltHcalDigis" ),
    Subdetector = cms.string( "HO" ),
    firstSample = cms.int32( 4 ),
    samplesToAdd = cms.int32( 4 ),
    correctForTimeslew = cms.bool( True ),
    correctForPhaseContainment = cms.bool( True ),
    correctionPhaseNS = cms.double( 13.0 )
)
hltTowerMakerForJets = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( ('hltEcalRegionalJetsRecHit','EcalRecHitsEB'),('hltEcalRegionalJetsRecHit','EcalRecHitsEE') )
)
hltIterativeCone5CaloJetsRegional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltTowerMakerForJets" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltMCJetCorJetIcone5Regional = cms.EDProducer( "JetCorrectionProducer",
    src = cms.InputTag( "hltIterativeCone5CaloJetsRegional" ),
    alias = cms.untracked.string( "corJetIcone5" ),
    correctors = cms.vstring( 'MCJetCorrectorIcone5' )
)
hlt2jet150 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 150.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1s3jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre3jet = cms.EDFilter( "HLTPrescaler" )
hlt3jet85 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 85.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltL1s4jet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre4jet = cms.EDFilter( "HLTPrescaler" )
hltEcalRegionalRestFEDs = cms.EDProducer( "EcalListOfFEDSProducer",
    debug = cms.untracked.bool( False ),
    OutputLabel = cms.untracked.string( "" )
)
hltEcalRegionalRestDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    DoRegional = cms.untracked.bool( True ),
    FedLabel = cms.untracked.InputTag( "hltEcalRegionalRestFEDs" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalRegionalRestWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalRegionalRestDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalRegionalRestDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRegionalRestRecHitTmp = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalRestWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalRestWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltEcalRecHitAll = cms.EDProducer( "EcalRecHitsMerger",
    debug = cms.untracked.bool( False ),
    EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
    MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
    TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
    JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
    RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
    EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
    MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
    TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
    JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
    RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
    OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
    OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
    EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
    EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
)
hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( ('hltEcalRecHitAll','EcalRecHitsEB'),('hltEcalRecHitAll','EcalRecHitsEE') )
)
hltIterativeCone5CaloJets = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltTowerMakerForAll" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltMCJetCorJetIcone5 = cms.EDProducer( "JetCorrectionProducer",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    alias = cms.untracked.string( "MCJetCorJetIcone5" ),
    correctors = cms.vstring( 'MCJetCorrectorIcone5' )
)
hltMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltTowerMakerForAll" ),
    InputType = cms.string( "CandidateCollection" ),
    METType = cms.string( "CaloMET" ),
    alias = cms.string( "RawCaloMET" ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False )
)
hltHtMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltMCJetCorJetIcone5" ),
    InputType = cms.string( "CaloJetCollection" ),
    METType = cms.string( "MET" ),
    alias = cms.string( "HTMET" ),
    globalThreshold = cms.double( 5.0 ),
    noHF = cms.bool( False )
)
hlt4jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1s2jetAco = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre2jetAco = cms.EDFilter( "HLTPrescaler" )
hlt2jet125 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 125.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hlt2jetAco = cms.EDFilter( "HLT2JetJet",
    inputTag1 = cms.InputTag( "hlt2jet125" ),
    inputTag2 = cms.InputTag( "hlt2jet125" ),
    MinDphi = cms.double( 0.0 ),
    MaxDphi = cms.double( 2.1 ),
    MinDeta = cms.double( 0.0 ),
    MaxDeta = cms.double( -1.0 ),
    MinMinv = cms.double( 0.0 ),
    MaxMinv = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet1METAco = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet1METAco = cms.EDFilter( "HLTPrescaler" )
hlt1MET60 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hlt1jet100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hlt1jet1METAco = cms.EDFilter( "HLT2JetMET",
    inputTag1 = cms.InputTag( "hlt1jet100" ),
    inputTag2 = cms.InputTag( "hlt1MET60" ),
    MinDphi = cms.double( 0.0 ),
    MaxDphi = cms.double( 2.1 ),
    MinDeta = cms.double( 0.0 ),
    MaxDeta = cms.double( -1.0 ),
    MinMinv = cms.double( 0.0 ),
    MaxMinv = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt1jet180 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s2jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre2jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt2jet125New = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 125.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1s3jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre3jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt3jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltL1s4jet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre4jet1MET = cms.EDFilter( "HLTPrescaler" )
hlt4jet35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1s1MET1HT = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET1HT = cms.EDFilter( "HLTPrescaler" )
hlt1MET65 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 65.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hlt1HT350 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 350.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s2jetvbfMET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM40" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre2jetvbfMET = cms.EDFilter( "HLTPrescaler" )
hlt2jetvbf = cms.EDFilter( "HLTJetVBFFilter",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    minEt = cms.double( 40.0 ),
    minDeltaEta = cms.double( 2.5 )
)
hltL1snvMET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrenv = cms.EDFilter( "HLTPrescaler" )
hltnv = cms.EDFilter( "HLTNVFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET60" ),
    minNV = cms.double( 0.1 ),
    minEtJet1 = cms.double( 80.0 ),
    minEtJet2 = cms.double( 20.0 )
)
hltL1sPhi2MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrephi2met = cms.EDFilter( "HLTPrescaler" )
hltPhi2metAco = cms.EDFilter( "HLTPhi2METFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET60" ),
    minDeltaPhi = cms.double( 0.377 ),
    maxDeltaPhi = cms.double( 3.1514 ),
    minEtJet1 = cms.double( 60.0 ),
    minEtJet2 = cms.double( 60.0 )
)
hltL1sPhiJet1MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrephijet1met = cms.EDFilter( "HLTPrescaler" )
hlt1MET70 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPhiJet1metAco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    minDeltaPhi = cms.double( 0.0 ),
    maxDeltaPhi = cms.double( 2.89 ),
    minEtJet1 = cms.double( 60.0 ),
    minEtJet2 = cms.double( -1.0 ),
    Acoplanar = cms.string( "Jet1Met" )
)
hltL1sPhiJet2MET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrephijet2met = cms.EDFilter( "HLTPrescaler" )
hltPhiJet2metAco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    minDeltaPhi = cms.double( 0.377 ),
    maxDeltaPhi = cms.double( 3.141593 ),
    minEtJet1 = cms.double( 50.0 ),
    minEtJet2 = cms.double( 50.0 ),
    Acoplanar = cms.string( "Jet2Met" )
)
hltL1sPhiJet1Jet2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrephijet1jet2 = cms.EDFilter( "HLTPrescaler" )
hltPhiJet1Jet2Aco = cms.EDFilter( "HLTAcoFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    inputMETTag = cms.InputTag( "hlt1MET70" ),
    minDeltaPhi = cms.double( 0.0 ),
    maxDeltaPhi = cms.double( 2.7646 ),
    minEtJet1 = cms.double( 40.0 ),
    minEtJet2 = cms.double( 40.0 ),
    Acoplanar = cms.string( "Jet1Jet2" )
)
hltL1RapGap = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet15_ForJet10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrerapgap = cms.EDFilter( "HLTPrescaler" )
hltRapGap = cms.EDFilter( "HLTRapGapFilter",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEta = cms.double( 3.0 ),
    maxEta = cms.double( 5.0 ),
    caloThresh = cms.double( 20.0 )
)
hltL1seedSingle = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltEcalRegionalEgammaFEDs = cms.EDProducer( "EcalListOfFEDSProducer",
    debug = cms.untracked.bool( False ),
    EGamma = cms.untracked.bool( True ),
    EM_l1TagIsolated = cms.untracked.InputTag( 'hltL1extraParticles','Isolated' ),
    EM_l1TagNonIsolated = cms.untracked.InputTag( 'hltL1extraParticles','NonIsolated' ),
    Ptmin_iso = cms.untracked.double( 5.0 ),
    Ptmin_noniso = cms.untracked.double( 5.0 ),
    OutputLabel = cms.untracked.string( "" )
)
hltEcalRegionalEgammaDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    DoRegional = cms.untracked.bool( True ),
    FedLabel = cms.untracked.InputTag( "hltEcalRegionalEgammaFEDs" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalRegionalEgammaWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalRegionalEgammaDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalRegionalEgammaDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRegionalEgammaRecHitTmp = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalEgammaWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalEgammaWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltEcalRegionalEgammaRecHit = cms.EDProducer( "EcalRecHitsMerger",
    debug = cms.untracked.bool( False ),
    EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
    MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
    TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
    JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
    RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
    EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
    MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
    TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
    JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
    RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
    OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
    OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
    EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
    EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
)
hltIslandBasicClustersEndcapL1Isolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "ecalRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltIslandBasicClustersBarrelL1Isolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( False ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "ecalRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( True ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 10 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 1.0 )
)
hltIslandSuperClustersL1Isolated = cms.EDProducer( "SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersEndcapL1Isolated" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersBarrelL1Isolated" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClusters" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.2 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.4 ),
    seedTransverseEnergyThreshold = cms.double( 1.5 )
)
hltCorrectedIslandEndcapSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1Isolated','islandEndcapSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedIslandBarrelSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1Isolated','islandBarrelSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1Isolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( -0.01217, 0.031, 0.9887, -3.776E-4, 1.598 ),
      fEtEtaVec = cms.vdouble( 1.001, -0.8654, 3.131, 0.0, 0.735, 20.72, 1.169, 8.0, 1.023, -0.00181, 0.0 ),
      corrF = cms.vint32( 1, 1, 0 ),
      brLinearLowThr = cms.double( 0.7 ),
      brLinearHighThr = cms.double( 8.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( "hltCorrectedIslandEndcapSuperClustersL1Isolated" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
hltL1IsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1IsoSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSingleL1MatchFilter" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedElectronHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.15 )
)
hltL1IsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSiPixelDigis = cms.EDProducer( "SiPixelRawToDigi",
    InputLabel = cms.untracked.string( "rawDataCollector" )
)
hltSiPixelClusters = cms.EDProducer( "SiPixelClusterProducer",
    src = cms.InputTag( "hltSiPixelDigis" ),
    payloadType = cms.string( "HLT" ),
    ChannelThreshold = cms.int32( 2500 ),
    SeedThreshold = cms.int32( 3000 ),
    ClusterThreshold = cms.double( 5050.0 ),
    VCaltoElectronGain = cms.int32( 65 ),
    VCaltoElectronOffset = cms.int32( 0 ),
    MissCalibrate = cms.untracked.bool( True )
)
hltSiPixelRecHits = cms.EDProducer( "SiPixelRecHitConverter",
    src = cms.InputTag( "hltSiPixelClusters" ),
    CPE = cms.string( "PixelCPEGeneric" )
)
hltSiStripRawToClustersFacility = cms.EDProducer( "SiStripRawToClusters",
    ProductLabel = cms.untracked.string( "rawDataCollector" ),
    MaxHolesInCluster = cms.untracked.uint32( 0 ),
    ClusterThreshold = cms.untracked.double( 5.0 ),
    SeedThreshold = cms.untracked.double( 3.0 ),
    ChannelThreshold = cms.untracked.double( 2.0 ),
    ClusterizerAlgorithm = cms.untracked.string( "ThreeThreshold" )
)
hltSiStripClusters = cms.EDProducer( "MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag( "hltSiStripRawToClustersFacility" ),
    measurementTrackerName = cms.string( "" )
)
hltL1IsoElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.025 ),
      pPhiMin1 = cms.double( -0.015 ),
      ePhiMax1 = cms.double( 0.015 ),
      ePhiMin1 = cms.double( -0.025 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.11 ),
      rMinI = cms.double( -0.11 ),
      r2MaxF = cms.double( 0.08 ),
      r2MinF = cms.double( -0.08 ),
      z2MaxB = cms.double( 0.05 ),
      z2MinB = cms.double( -0.05 ),
      PhiMax2 = cms.double( 0.0010 ),
      PhiMin2 = cms.double( -0.0010 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1IsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltCkfL1IsoTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1IsoWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1IsoTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1Iso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltL1IsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1IsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1IsoElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1IsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1IsoElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1IsoElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    trackProducer = cms.InputTag( "hltL1IsoElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1IsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltSingleElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingle = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltIslandBasicClustersEndcapL1NonIsolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "ecalRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltIslandBasicClustersBarrelL1NonIsolated = cms.EDProducer( "EgammaHLTIslandClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( False ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "ecalRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    IslandBarrelSeedThr = cms.double( 0.5 ),
    IslandEndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.3 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0_barl = cms.double( 7.4 ),
    posCalc_t0_endc = cms.double( 3.1 ),
    posCalc_t0_endcPresh = cms.double( 1.2 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 )
)
hltHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    doIsolated = cms.bool( False ),
    l1LowerThr = cms.double( 5.0 ),
    l1UpperThr = cms.double( 999.0 ),
    l1LowerThrIgnoreIsolation = cms.double( 999.0 ),
    regionEtaMargin = cms.double( 0.14 ),
    regionPhiMargin = cms.double( 0.4 ),
    posCalc_logweight = cms.bool( True ),
    posCalc_t0 = cms.double( 7.4 ),
    posCalc_w0 = cms.double( 4.2 ),
    posCalc_x0 = cms.double( 0.89 ),
    HybridBarrelSeedThr = cms.double( 1.5 ),
    step = cms.int32( 10 ),
    ethresh = cms.double( 0.1 ),
    eseed = cms.double( 0.35 ),
    ewing = cms.double( 1.0 )
)
hltIslandSuperClustersL1NonIsolated = cms.EDProducer( "SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltIslandBasicClustersEndcapL1NonIsolated" ),
    barrelClusterProducer = cms.string( "hltIslandBasicClustersBarrelL1NonIsolated" ),
    endcapClusterCollection = cms.string( "islandEndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "islandBarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "islandEndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "islandBarrelSuperClusters" ),
    doBarrel = cms.bool( True ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.2 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.4 ),
    seedTransverseEnergyThreshold = cms.double( 1.5 )
)
hltCorrectedIslandEndcapSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1NonIsolated','islandEndcapSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedIslandBarrelSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( 'hltIslandSuperClustersL1NonIsolated','islandBarrelSuperClusters' ),
    superClusterAlgo = cms.string( "Island" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 0.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( 0.0 ),
      fEtEtaVec = cms.vdouble( 0.0 ),
      corrF = cms.vint32( 0 ),
      brLinearLowThr = cms.double( 0.0 ),
      brLinearHighThr = cms.double( 0.0 )
    ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1NonIsolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      fBremVec = cms.vdouble( -0.01217, 0.031, 0.9887, -3.776E-4, 1.598 ),
      fEtEtaVec = cms.vdouble( 1.001, -0.8654, 3.131, 0.0, 0.735, 20.72, 1.169, 8.0, 1.023, -0.00181, 0.0 ),
      corrF = cms.vint32( 1, 1, 0 ),
      brLinearLowThr = cms.double( 0.7 ),
      brLinearHighThr = cms.double( 8.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( "hltCorrectedIslandEndcapSuperClustersL1NonIsolated" ),
    preshClusterCollectionX = cms.string( "preshowerXClusters" ),
    preshClusterCollectionY = cms.string( "preshowerYClusters" ),
    preshNclust = cms.int32( 4 ),
    etThresh = cms.double( 5.0 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    assocSClusterCollection = cms.string( "" ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    debugLevel = cms.string( "" )
)
hltL1NonIsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1NonIsoSingleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsolatedElectronHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.15 )
)
hltL1NonIsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.025 ),
      pPhiMin1 = cms.double( -0.015 ),
      ePhiMax1 = cms.double( 0.015 ),
      ePhiMin1 = cms.double( -0.025 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.11 ),
      rMinI = cms.double( -0.11 ),
      r2MaxF = cms.double( 0.08 ),
      r2MinF = cms.double( -0.08 ),
      z2MaxB = cms.double( 0.05 ),
      z2MinB = cms.double( -0.05 ),
      PhiMax2 = cms.double( 0.0010 ),
      PhiMin2 = cms.double( -0.0010 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1NonIsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltCkfL1NonIsoTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1NonIsoWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1NonIsoTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1NonIso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1NonIsoElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1NonIsoElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1NonIsoElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    trackProducer = cms.InputTag( "hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltSingleElectronL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedDouble = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleIsoEG8" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1IsoDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedDouble = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1NonIsoDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleElectronL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoSinglePhotonL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSinglePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSinglePhotonL1MatchFilter" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalIsolationProducersRegional",
    bcBarrelProducer = cms.InputTag( 'hltIslandBasicClustersBarrelL1Isolated','islandBarrelBasicClusters' ),
    bcEndcapProducer = cms.InputTag( 'hltIslandBasicClustersEndcapL1Isolated','islandEndcapBasicClusters' ),
    scIslandBarrelProducer = cms.InputTag( "hltCorrectedIslandBarrelSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    egEcalIsoEtMin = cms.double( 0.0 ),
    egEcalIsoConeSize = cms.double( 0.3 ),
    SCAlgoType = cms.int32( 0 )
)
hltL1IsoSinglePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1IsoSinglePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEgammaRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 15.0 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "pixelMatchElectrons" ),
    UseZInVertex = cms.bool( False ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1IsoEgammaRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoEgammaRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1IsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1IsoEgammaRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1IsoPhotonTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1IsoEgammaRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.0 )
)
hltL1IsoSinglePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSinglePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalIsolationProducersRegional",
    bcBarrelProducer = cms.InputTag( 'hltIslandBasicClustersBarrelL1NonIsolated','islandBarrelBasicClusters' ),
    bcEndcapProducer = cms.InputTag( 'hltIslandBasicClustersEndcapL1NonIsolated','islandEndcapBasicClusters' ),
    scIslandBarrelProducer = cms.InputTag( "hltCorrectedIslandBarrelSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    egEcalIsoEtMin = cms.double( 0.0 ),
    egEcalIsoConeSize = cms.double( 0.3 ),
    SCAlgoType = cms.int32( 0 )
)
hltL1NonIsoSinglePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1NonIsoSinglePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoEgammaRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 15.0 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "pixelMatchElectrons" ),
    UseZInVertex = cms.bool( False ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1NonIsoEgammaRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoEgammaRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1NonIsoEgammaRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1NonIsoPhotonTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.0 )
)
hltL1NonIsoSinglePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoDoublePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoublePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoublePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoublePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoublePhotonDoubleEtFilter = cms.EDFilter( "HLTEgammaDoubleEtFilter",
    candTag = cms.InputTag( "hltL1IsoDoublePhotonTrackIsolFilter" ),
    etcut1 = cms.double( 20.0 ),
    etcut2 = cms.double( 20.0 ),
    npaircut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoublePhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoDoublePhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoDoublePhotonEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoDoublePhotonL1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoDoublePhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoDoublePhotonDoubleEtFilter = cms.EDFilter( "HLTEgammaDoubleEtFilter",
    candTag = cms.InputTag( "hltL1NonIsoDoublePhotonTrackIsolFilter" ),
    etcut1 = cms.double( 20.0 ),
    etcut2 = cms.double( 20.0 ),
    npaircut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoublePhotonL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSingleEMHighEtL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEMHighEtEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtL1MatchFilterRegional" ),
    etcut = cms.double( 80.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSingleEMHighEtEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEMHighEtEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 5.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoSingleEMHighEtHOEFilter = cms.EDFilter( "HLTEgammaHOEFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 0.05 ),
    hcalisolendcapcut = cms.double( 0.05 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltHcalDoubleCone = cms.EDProducer( "EgammaHLTHcalIsolationDoubleConeProducers",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 ),
    egHcalExclusion = cms.double( 0.15 )
)
hltL1NonIsoEMHcalDoubleCone = cms.EDProducer( "EgammaHLTHcalIsolationDoubleConeProducers",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 ),
    egHcalExclusion = cms.double( 0.15 )
)
hltL1NonIsoSingleEMHighEtHcalDBCFilter = cms.EDFilter( "HLTEgammaHcalDBCFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtHOEFilter" ),
    isoTag = cms.InputTag( "hltHcalDoubleCone" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoEMHcalDoubleCone" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoSingleEMHighEtTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSingleEMHighEtHcalDBCFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 4.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSingleEMVHighEtL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional" ),
    etcut = cms.double( 200.0 ),
    ncandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSingleEMVHEL1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoDoubleElectronZeeL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleElectronZeeEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoDoubleElectronZeeL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeeHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeePixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleElectronZeeEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeePixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleElectronZeeTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltL1IsoDoubleElectronZeePMMassFilter = cms.EDFilter( "HLTPMMassFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleElectronZeeTrackIsolFilter" ),
    lowerMassCut = cms.double( 54.22 ),
    upperMassCut = cms.double( 99999.9 ),
    nZcandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    relaxed = cms.untracked.bool( False ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltZeeCounterPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedExclusiveDouble = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ExclusiveDoubleIsoEG6" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1IsoDoubleExclElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedExclusiveDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleExclElectronEtPhiFilter = cms.EDFilter( "HLTEgammaDoubleEtPhiFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronL1MatchFilterRegional" ),
    etcut1 = cms.double( 6.0 ),
    etcut2 = cms.double( 6.0 ),
    MinAcop = cms.double( -0.1 ),
    MaxAcop = cms.double( 0.6 ),
    MinEtBalance = cms.double( -1.0 ),
    MaxEtBalance = cms.double( 10.0 ),
    npaircut = cms.int32( 1 )
)
hltL1IsoDoubleExclElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronEtPhiFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleExclElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoDoubleExclElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltDoubleExclElectronL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoDoubleExclPhotonL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedExclusiveDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoDoubleExclPhotonEtPhiFilter = cms.EDFilter( "HLTEgammaDoubleEtPhiFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonL1MatchFilterRegional" ),
    etcut1 = cms.double( 10.0 ),
    etcut2 = cms.double( 10.0 ),
    MinAcop = cms.double( -0.1 ),
    MaxAcop = cms.double( 0.6 ),
    MinEtBalance = cms.double( -1.0 ),
    MaxEtBalance = cms.double( 10.0 ),
    npaircut = cms.int32( 1 )
)
hltL1IsoDoubleExclPhotonEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonEtPhiFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 2.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoDoubleExclPhotonHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 8.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoDoubleExclPhotonTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoDoubleExclPhotonHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltDoubleExclPhotonL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedSinglePrescaled = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1IsoSinglePhotonPrescaledL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSinglePrescaled" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoSinglePhotonPrescaledEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledL1MatchFilter" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonPrescaledEcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonEcalNonIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoSinglePhotonPrescaledHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoSinglePhotonPrescaledTrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1IsoSinglePhotonPrescaledHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltPhotonNonIsoTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltSinglePhotonPrescaledL1IsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoLargeWindowSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoLargeWindowSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoLargeWindowSingleL1MatchFilter" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.045 ),
      pPhiMin1 = cms.double( -0.03 ),
      ePhiMax1 = cms.double( 0.03 ),
      ePhiMin1 = cms.double( -0.045 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.2 ),
      rMinI = cms.double( -0.2 ),
      r2MaxF = cms.double( 0.3 ),
      r2MinF = cms.double( -0.3 ),
      z2MaxB = cms.double( 0.2 ),
      z2MinB = cms.double( -0.2 ),
      PhiMax2 = cms.double( 0.01 ),
      PhiMin2 = cms.double( -0.01 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1IsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltCkfL1IsoLargeWindowTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1IsoLargeWindowWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1IsoLargeWindowTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1IsoLargeWindow = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoLargeWindowWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsL1NonIsoLargeWindowForHLT" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1IsoLargeWindowElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    trackProducer = cms.InputTag( "hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1IsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltSingleElectronL1IsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingle" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoLargeWindowSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.045 ),
      pPhiMin1 = cms.double( -0.03 ),
      ePhiMax1 = cms.double( 0.03 ),
      ePhiMin1 = cms.double( -0.045 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.2 ),
      rMinI = cms.double( -0.2 ),
      r2MaxF = cms.double( 0.3 ),
      r2MinF = cms.double( -0.3 ),
      z2MaxB = cms.double( 0.2 ),
      z2MinB = cms.double( -0.2 ),
      PhiMax2 = cms.double( 0.01 ),
      PhiMin2 = cms.double( -0.01 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltCkfL1NonIsoLargeWindowTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1NonIsoLargeWindowWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1NonIsoLargeWindowTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1NonIsoLargeWindow = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoLargeWindowWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1NonIsoLargeWindowElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    trackProducer = cms.InputTag( "hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltSingleElectronL1NonIsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoLargeWindowDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True )
)
hltL1IsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoLargeWindowDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltDoubleElectronL1IsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDouble" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoLargeWindowDoubleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9.0 ),
    hcalisolendcapcut = cms.double( 9.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    eoverpbarrelcut = cms.double( 15000.0 ),
    eoverpendcapcut = cms.double( 24500.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoLargeWindowDoubleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.4 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltDoubleElectronL1NonIsoLargeWindowPresc = cms.EDFilter( "HLTPrescaler" )
hltPrescaleSingleMuIso = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuonDTDigis = cms.EDProducer( "DTUnpackingModule",
    dataType = cms.string( "DDU" ),
    fedbyType = cms.untracked.bool( False ),
    fedColl = cms.untracked.string( "rawDataCollector" ),
    readOutParameters = cms.PSet( 
      localDAQ = cms.untracked.bool( False ),
      performDataIntegrityMonitor = cms.untracked.bool( False ),
      debug = cms.untracked.bool( False ),
      rosParameters = cms.PSet( 
        localDAQ = cms.untracked.bool( False ),
        readingDDU = cms.untracked.bool( True ),
        writeSC = cms.untracked.bool( True ),
        readDDUIDfromDDU = cms.untracked.bool( True ),
        performDataIntegrityMonitor = cms.untracked.bool( False ),
        debug = cms.untracked.bool( False )
      )
    ),
    rosParameters = cms.PSet( 
      localDAQ = cms.untracked.bool( False ),
      readingDDU = cms.untracked.bool( True ),
    )
)
hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
    dtDigiLabel = cms.InputTag( "hltMuonDTDigis" ),
    recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
    recAlgoConfig = cms.PSet( 
      tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
      tTrigModeConfig = cms.PSet( 
        debug = cms.untracked.bool( False ),
        kFactor = cms.double( -2.0 ),
        vPropWire = cms.double( 24.4 ),
        doT0Correction = cms.bool( True ),
        doTOFCorrection = cms.bool( True ),
        tofCorrType = cms.int32( 1 ),
        doWirePropCorrection = cms.bool( True ),
        wirePropCorrType = cms.int32( 1 )
      ),
      minTime = cms.double( -3.0 ),
      maxTime = cms.double( 415.0 ),
      interpolate = cms.bool( True ),
      debug = cms.untracked.bool( False )
    )
)
hltDt4DSegments = cms.EDProducer( "DTRecSegment4DProducer",
    debug = cms.untracked.bool( False ),
    recHits1DLabel = cms.InputTag( "hltDt1DRecHits" ),
    recHits2DLabel = cms.InputTag( "dt2DSegments" ),
    Reco4DAlgoName = cms.string( "DTCombinatorialPatternReco4D" ),
    Reco4DAlgoConfig = cms.PSet( 
      Reco2DAlgoConfig = cms.PSet( 
        recAlgoConfig = cms.PSet( 
          tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
          tTrigModeConfig = cms.PSet( 
            debug = cms.untracked.bool( False ),
            kFactor = cms.double( -2.0 ),
            vPropWire = cms.double( 24.4 ),
            doT0Correction = cms.bool( True ),
            doTOFCorrection = cms.bool( True ),
            tofCorrType = cms.int32( 1 ),
            doWirePropCorrection = cms.bool( True ),
            wirePropCorrType = cms.int32( 1 )
          ),
          minTime = cms.double( -3.0 ),
          maxTime = cms.double( 415.0 ),
          interpolate = cms.bool( True ),
          debug = cms.untracked.bool( False )
        ),
        recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
        MaxAllowedHits = cms.uint32( 50 ),
        AlphaMaxTheta = cms.double( 0.1 ),
        AlphaMaxPhi = cms.double( 1.0 ),
        debug = cms.untracked.bool( False ),
        nSharedHitsMax = cms.int32( 2 ),
        nUnSharedHitsMin = cms.int32( 2 ),
        segmCleanerMode = cms.int32( 1 )
      ),
      Reco2DAlgoName = cms.string( "DTCombinatorialPatternReco" ),
      recAlgoConfig = cms.PSet( 
        tTrigMode = cms.string( "DTTTrigSyncFromDB" ),
        tTrigModeConfig = cms.PSet( 
          debug = cms.untracked.bool( False ),
          kFactor = cms.double( -2.0 ),
          vPropWire = cms.double( 24.4 ),
          doT0Correction = cms.bool( True ),
          doTOFCorrection = cms.bool( True ),
          tofCorrType = cms.int32( 1 ),
          doWirePropCorrection = cms.bool( True ),
          wirePropCorrType = cms.int32( 1 )
        ),
        minTime = cms.double( -3.0 ),
        maxTime = cms.double( 415.0 ),
        interpolate = cms.bool( True ),
        debug = cms.untracked.bool( False )
      ),
      recAlgo = cms.string( "DTParametrizedDriftAlgo" ),
      AllDTRecHits = cms.bool( True ),
      debug = cms.untracked.bool( False ),
      nSharedHitsMax = cms.int32( 2 ),
      nUnSharedHitsMin = cms.int32( 2 ),
      segmCleanerMode = cms.int32( 1 )
    )
)
hltMuonCSCDigis = cms.EDProducer( "CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool( False ),
    UseExaminer = cms.untracked.bool( False ),
    ExaminerMask = cms.untracked.uint32( 0x1febf3f6 ),
    ErrorMask = cms.untracked.uint32( 0x0 ),
    InputObjects = cms.InputTag( "rawDataCollector" )
)
hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    CSCUseCalibrations = cms.untracked.bool( True ),
    stripDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCStripDigi' ),
    wireDigiTag = cms.InputTag( 'hltMuonCSCDigis','MuonCSCWireDigi' ),
    CSCstripWireDeltaTime = cms.untracked.int32( 8 ),
    CSCStripPeakThreshold = cms.untracked.double( 10.0 ),
    CSCStripClusterChargeCut = cms.untracked.double( 25.0 ),
    CSCWireClusterDeltaT = cms.untracked.int32( 1 ),
    CSCStripxtalksOffset = cms.untracked.double( 0.03 ),
    NoiseLevel_ME1a = cms.untracked.double( 7.0 ),
    XTasymmetry_ME1a = cms.untracked.double( 0.0 ),
    ConstSyst_ME1a = cms.untracked.double( 0.022 ),
    NoiseLevel_ME1b = cms.untracked.double( 7.0 ),
    XTasymmetry_ME1b = cms.untracked.double( 0.0 ),
    ConstSyst_ME1b = cms.untracked.double( 0.02 ),
    NoiseLevel_ME12 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME12 = cms.untracked.double( 0.025 ),
    ConstSyst_ME12 = cms.untracked.double( 0.045 ),
    NoiseLevel_ME13 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME13 = cms.untracked.double( 0.025 ),
    ConstSyst_ME13 = cms.untracked.double( 0.065 ),
    NoiseLevel_ME21 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME21 = cms.untracked.double( 0.025 ),
    ConstSyst_ME21 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME22 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME22 = cms.untracked.double( 0.025 ),
    ConstSyst_ME22 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME31 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME31 = cms.untracked.double( 0.025 ),
    ConstSyst_ME31 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME32 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME32 = cms.untracked.double( 0.025 ),
    ConstSyst_ME32 = cms.untracked.double( 0.06 ),
    NoiseLevel_ME41 = cms.untracked.double( 7.0 ),
    XTasymmetry_ME41 = cms.untracked.double( 0.025 ),
    ConstSyst_ME41 = cms.untracked.double( 0.06 ),
    readBadChannels = cms.bool( False )
)
hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_type = cms.int32( 4 ),
    algo_psets = cms.VPSet( 
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoSK" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  minLayersApart = cms.int32( 2 ),
            wideSeg = cms.double( 3.0 ),
            chi2Max = cms.double( 99999.0 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            dPhiMax = cms.double( 0.0030 ),
            dRPhiMax = cms.double( 8.0 ),
            verboseInfo = cms.untracked.bool( True )
          ),
          cms.PSet(  minLayersApart = cms.int32( 2 ),
            wideSeg = cms.double( 3.0 ),
            chi2Max = cms.double( 99999.0 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 3.0 ),
            dPhiMax = cms.double( 0.025 ),
            dRPhiMax = cms.double( 8.0 ),
            verboseInfo = cms.untracked.bool( True )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoTC" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  SegmentSorting = cms.int32( 1 ),
            minLayersApart = cms.int32( 2 ),
            chi2ndfProbMin = cms.double( 1.0E-4 ),
            chi2Max = cms.double( 6000.0 ),
            dPhiFineMax = cms.double( 0.02 ),
            dRPhiFineMax = cms.double( 6.0 ),
            dPhiMax = cms.double( 0.0030 ),
            dRPhiMax = cms.double( 1.2 ),
            verboseInfo = cms.untracked.bool( True )
          ),
          cms.PSet(  SegmentSorting = cms.int32( 1 ),
            minLayersApart = cms.int32( 2 ),
            chi2ndfProbMin = cms.double( 1.0E-4 ),
            chi2Max = cms.double( 6000.0 ),
            dPhiFineMax = cms.double( 0.013 ),
            dRPhiFineMax = cms.double( 3.0 ),
            dPhiMax = cms.double( 0.00198 ),
            dRPhiMax = cms.double( 0.6 ),
            verboseInfo = cms.untracked.bool( True )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoDF" ),
        parameters_per_chamber_type = cms.vint32( 3, 1, 2, 2, 1, 2, 1, 2, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 10 ),
            dYclusBoxMax = cms.double( 8.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.5 ),
            tanThetaMax = cms.double( 1.2 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 10 ),
            dYclusBoxMax = cms.double( 12.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.8 ),
            tanThetaMax = cms.double( 2.0 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 12.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            Pruning = cms.untracked.bool( False ),
            nHitsPerClusterIsShower = cms.int32( 20 ),
            minHitsForPreClustering = cms.int32( 30 ),
            dYclusBoxMax = cms.double( 8.0 ),
            dXclusBoxMax = cms.double( 8.0 ),
            preClustering = cms.untracked.bool( False ),
            chi2Max = cms.double( 5000.0 ),
            tanPhiMax = cms.double( 0.5 ),
            tanThetaMax = cms.double( 1.2 ),
            minLayersApart = cms.int32( 2 ),
            dPhiFineMax = cms.double( 0.025 ),
            dRPhiFineMax = cms.double( 8.0 ),
            minHitsPerSegment = cms.int32( 3 ),
            CSCSegmentDebug = cms.untracked.bool( False )
          )
        )
      ),
      cms.PSet(  algo_name = cms.string( "CSCSegAlgoST" ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1 ),
        chamber_types = cms.vstring( 'ME1/a', 'ME1/b', 'ME1/2', 'ME1/3', 'ME2/1', 'ME2/2', 'ME3/1', 'ME3/2', 'ME4/1' ),
        algo_psets = cms.VPSet( 
          cms.PSet(  curvePenalty = cms.untracked.double( 2.0 ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            onlyBestSegment = cms.untracked.bool( False ),
            BrutePruning = cms.untracked.bool( False ),
            Pruning = cms.untracked.bool( False ),
            preClustering = cms.untracked.bool( True ),
            maxRecHitsInCluster = cms.untracked.int32( 20 ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            CSCDebug = cms.untracked.bool( False )
          ),
          cms.PSet(  curvePenalty = cms.untracked.double( 2.0 ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            onlyBestSegment = cms.untracked.bool( False ),
            BrutePruning = cms.untracked.bool( False ),
            Pruning = cms.untracked.bool( False ),
            preClustering = cms.untracked.bool( True ),
            maxRecHitsInCluster = cms.untracked.int32( 24 ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            CSCDebug = cms.untracked.bool( False )
          )
        )
      )
    )
)
hltMuonRPCDigis = cms.EDProducer( "RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag( "rawDataCollector" )
)
hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    rpcDigiLabel = cms.InputTag( "hltMuonRPCDigis" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    recAlgoConfig = cms.PSet(  )
)
hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    InputObjects = cms.InputTag( "hltL1extraParticles" ),
    GMTReadoutCollection = cms.InputTag( "hltGtDigis" ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 1 ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    )
)
hltL2Muons = cms.EDProducer( "L2MuonProducer",
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    ),
    TrackLoaderParameters = cms.PSet( 
      MuonUpdatorAtVertexParameters = cms.PSet( 
        Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        MaxChi2 = cms.double( 1000000.0 )
      ),
      VertexConstraint = cms.bool( True ),
      DoSmoothing = cms.bool( False ),
      Smoother = cms.string( "KFSmootherForMuonTrackLoader" )
    ),
    L2TrajBuilderParameters = cms.PSet( 
      SeedPosition = cms.string( "in" ),
      SeedPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      NavigationType = cms.string( "Standard" ),
      RefitterParameters = cms.PSet( 
        FitterName = cms.string( "KFFitterSmootherForL2Muon" ),
        Option = cms.int32( 1 )
      ),
      BWFilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableDTMeasurement = cms.bool( True ),
        EnableCSCMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        FitDirection = cms.string( "outsideIn" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        MaxChi2 = cms.double( 25.0 ),
        NumberOfSigma = cms.double( 3.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          Granularity = cms.int32( 2 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 )
        )
      ),
      DoRefit = cms.bool( False ),
      FilterParameters = cms.PSet( 
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        EnableDTMeasurement = cms.bool( True ),
        EnableCSCMeasurement = cms.bool( True ),
        EnableRPCMeasurement = cms.bool( True ),
        FitDirection = cms.string( "insideOut" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        MaxChi2 = cms.double( 1000.0 ),
        NumberOfSigma = cms.double( 3.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 1000.0 ),
          Granularity = cms.int32( 0 ),
          RescaleError = cms.bool( False ),
          RescaleErrorFactor = cms.double( 100.0 )
        )
      ),
      DoBackwardFilter = cms.bool( True )
    )
)
hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
hltSingleMuIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltEcalRegionalMuonsFEDs = cms.EDProducer( "EcalListOfFEDSProducer",
    debug = cms.untracked.bool( False ),
    Muon = cms.untracked.bool( True ),
    MuonSource = cms.untracked.InputTag( "hltL1extraParticles" ),
    OutputLabel = cms.untracked.string( "" )
)
hltEcalRegionalMuonsDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    DoRegional = cms.untracked.bool( True ),
    FedLabel = cms.untracked.InputTag( "hltEcalRegionalMuonsFEDs" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalRegionalMuonsWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalRegionalMuonsDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalRegionalMuonsDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRegionalMuonsRecHitTmp = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalMuonsWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalMuonsWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltEcalRegionalMuonsRecHit = cms.EDProducer( "EcalRecHitsMerger",
    debug = cms.untracked.bool( False ),
    EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
    MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
    TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
    JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
    RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
    EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
    MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
    TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
    JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
    RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
    OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
    OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
    EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
    EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
)
hltTowerMakerForMuons = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( ('hltEcalRegionalMuonsRecHit','EcalRecHitsEB'),('hltEcalRegionalMuonsRecHit','EcalRecHitsEE') )
)
hltL2MuonIsolations = cms.EDProducer( "L2MuonIsolationProducer",
    StandAloneCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    OutputMuIsoDeposits = cms.bool( True ),
    EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
    ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
    Thresholds = cms.vdouble( 4.0, 3.7, 4.0, 3.5, 3.4, 3.4, 3.2, 3.4, 3.1, 2.9, 2.9, 2.7, 3.1, 3.0, 2.4, 2.1, 2.0, 2.3, 2.2, 2.4, 2.5, 2.5, 2.6, 2.9, 3.1, 2.9 ),
    ExtractorPSet = cms.PSet( 
      ComponentName = cms.string( "CaloExtractor" ),
      CaloTowerCollectionLabel = cms.InputTag( "hltTowerMakerForMuons" ),
      DepositLabel = cms.untracked.string( "EcalPlusHcal" ),
      Weight_E = cms.double( 1.5 ),
      Weight_H = cms.double( 1.0 ),
      Threshold_E = cms.double( 0.2 ),
      Threshold_H = cms.double( 0.5 ),
      DR_Veto_E = cms.double( 0.07 ),
      DR_Veto_H = cms.double( 0.1 ),
      DR_Max = cms.double( 0.24 ),
      Vertex_Constraint_XY = cms.bool( False ),
      Vertex_Constraint_Z = cms.bool( False )
    )
)
hltSingleMuIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltL3TrajectorySeed = cms.EDProducer( "TSGFromL2Muon",
    PtCut = cms.double( 1.0 ),
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    tkSeedGenerator = cms.string( "TSGFromCombinedHits" ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    ),
    MuonTrackingRegionBuilder = cms.PSet( 
      beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
      UseVertex = cms.bool( False ),
      Rescale_eta = cms.double( 3.0 ),
      Rescale_phi = cms.double( 3.0 ),
      Rescale_Dz = cms.double( 3.0 ),
      DeltaZ_Region = cms.double( 15.9 ),
      DeltaR = cms.double( 0.2 ),
      EscapePt = cms.double( 1.5 ),
      Phi_min = cms.double( 0.1 ),
      Eta_min = cms.double( 0.1 ),
      UseFixedRegion = cms.bool( False ),
      EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
      EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
      PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
      PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
      vertexCollection = cms.InputTag( "pixelVertices" ),
      Eta_fixed = cms.double( 0.2 ),
      Phi_fixed = cms.double( 0.2 )
    ),
    TrackerSeedCleaner = cms.PSet( 
      TTRHBuilder = cms.string( "WithTrackAngle" ),
      beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
      directionCleaner = cms.bool( False ),
      cleanerFromSharedHits = cms.bool( True ),
      ptCleaner = cms.bool( False )
    ),
    TSGFromMixedPairs = cms.PSet( 
      ComponentName = cms.string( "TSGFromOrderedHits" ),
      TTRHBuilder = cms.string( "WithTrackAngle" ),
      OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitPairGenerator" ),
        SeedingLayers = cms.string( "MixedLayerPairs" )
      )
    ),
    TSGFromPixelTriplets = cms.PSet( 
      ComponentName = cms.string( "TSGFromOrderedHits" ),
      TTRHBuilder = cms.string( "WithTrackAngle" ),
      OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitTripletGenerator" ),
        SeedingLayers = cms.string( "PixelLayerTriplets" ),
        GeneratorPSet = cms.PSet( 
          ComponentName = cms.string( "PixelTripletHLTGenerator" ),
          useFixedPreFiltering = cms.bool( False ),
          phiPreFiltering = cms.double( 0.3 ),
          extraHitRZtolerance = cms.double( 0.06 ),
          extraHitRPhitolerance = cms.double( 0.06 ),
          useBending = cms.bool( True ),
          useMultScattering = cms.bool( True )
        )
      )
    ),
    TSGFromPixelPairs = cms.PSet( 
      ComponentName = cms.string( "TSGFromOrderedHits" ),
      TTRHBuilder = cms.string( "WithTrackAngle" ),
      OrderedHitsFactoryPSet = cms.PSet( 
        ComponentName = cms.string( "StandardHitPairGenerator" ),
        SeedingLayers = cms.string( "PixelLayerPairs" )
      )
    ),
    TSGForRoadSearchOI = cms.PSet( 
      ComponentName = cms.string( "TSGForRoadSearch" ),
      option = cms.uint32( 3 ),
      copyMuonRecHit = cms.bool( False ),
      manySeeds = cms.bool( False ),
      maxChi2 = cms.double( 40.0 ),
      propagatorName = cms.string( "SteppingHelixPropagatorAlong" ),
      propagatorCompatibleName = cms.string( "SteppingHelixPropagatorAny" ),
      errorMatrixPset = cms.PSet( 
        errorMatrixValuesPSet = cms.PSet( 
          xAxis = cms.vdouble( 0.0, 3.16, 6.69, 10.695, 15.319, 20.787, 27.479, 36.106, 48.26, 69.03, 200.0 ),
          yAxis = cms.vdouble( 0.0, 0.2, 0.3, 0.7, 0.9, 1.15, 1.35, 1.55, 1.75, 2.2, 2.5 ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V11 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 4.593, 5.253, 1.895, 1.985, 2.344, 5.37, 2.059, 2.423, 1.985, 2.054, 2.071, 2.232, 2.159, 2.1, 2.355, 3.862, 1.855, 2.311, 1.784, 1.766, 1.999, 2.18, 2.071, 2.03, 2.212, 2.266, 1.693, 1.984, 1.664, 1.602, 1.761, 2.007, 1.985, 1.982, 2.118, 1.734, 1.647, 1.705, 1.56, 1.542, 1.699, 2.058, 2.037, 1.934, 2.067, 1.555, 1.566, 1.638, 1.51, 1.486, 1.635, 1.977, 1.944, 1.865, 1.925, 1.415, 1.542, 1.571, 1.499, 1.468, 1.608, 1.899, 1.893, 1.788, 1.851, 1.22, 1.49, 1.54, 1.493, 1.457, 1.572, 1.876, 1.848, 1.751, 1.827, 1.223, 1.51, 1.583, 1.486, 1.431, 1.534, 1.79, 1.802, 1.65, 1.755, 1.256, 1.489, 1.641, 1.464, 1.438, 1.48, 1.888, 1.839, 1.657, 1.903, 1.899 ) ),
          pf3_V12 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V13 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V14 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V15 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V22 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 66.152, 3.219, 66.051, 1.298, 1.186, 1.197, 1.529, 2.807, 1.056, 1.092, 1.15, 1.158, 1.163, 1.05, 1.191, 1.287, 1.371, 2.039, 1.02, 1.059, 1.048, 1.087, 1.087, 1.041, 1.072, 1.118, 1.097, 1.229, 1.042, 1.07, 1.071, 1.063, 1.039, 1.038, 1.061, 1.052, 1.058, 1.188, 1.099, 1.075, 1.082, 1.055, 1.084, 1.024, 1.058, 1.069, 1.022, 1.184, 1.117, 1.105, 1.093, 1.082, 1.086, 1.053, 1.097, 1.07, 1.044, 1.125, 1.141, 1.167, 1.136, 1.133, 1.146, 1.089, 1.081, 1.117, 1.085, 1.075, 1.212, 1.199, 1.186, 1.212, 1.168, 1.125, 1.127, 1.119, 1.114, 1.062, 1.273, 1.229, 1.272, 1.293, 1.172, 1.124, 1.141, 1.123, 1.158, 1.115, 1.419, 1.398, 1.425, 1.394, 1.278, 1.132, 1.132, 1.115, 1.26, 1.096 ) ),
          pf3_V23 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V24 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V25 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V33 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 6.174, 56.89, 1.019, 2.206, 1.694, 1.698, 1.776, 3.563, 2.141, 2.432, 1.898, 1.834, 1.763, 1.797, 1.944, 1.857, 2.068, 2.894, 1.76, 2.185, 1.664, 1.656, 1.761, 1.964, 1.925, 1.89, 2.012, 2.014, 1.651, 1.825, 1.573, 1.534, 1.634, 1.856, 1.962, 1.879, 1.95, 1.657, 1.556, 1.639, 1.481, 1.433, 1.605, 1.943, 1.99, 1.885, 1.916, 1.511, 1.493, 1.556, 1.445, 1.457, 1.543, 1.897, 1.919, 1.884, 1.797, 1.394, 1.489, 1.571, 1.436, 1.425, 1.534, 1.796, 1.845, 1.795, 1.763, 1.272, 1.472, 1.484, 1.452, 1.412, 1.508, 1.795, 1.795, 1.773, 1.741, 1.207, 1.458, 1.522, 1.437, 1.399, 1.485, 1.747, 1.739, 1.741, 1.716, 1.187, 1.463, 1.589, 1.411, 1.404, 1.471, 1.92, 1.86, 1.798, 1.867, 1.436 ) ),
          pf3_V34 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V35 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V44 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.622, 2.139, 2.08, 1.178, 1.044, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.002, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.011, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.002, 1.0, 1.002, 1.013, 1.0, 1.0, 1.0, 1.0, 1.0, 1.005, 1.0, 1.004, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.009, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V45 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V55 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 27.275, 15.167, 13.818, 1.0, 1.0, 1.0, 1.037, 1.129, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.023, 1.028, 1.063, 1.08, 1.077, 1.054, 1.068, 1.065, 1.047, 1.025, 1.046, 1.064, 1.082, 1.078, 1.137, 1.12, 1.163, 1.158, 1.112, 1.072, 1.054, 1.095, 1.101, 1.092, 1.219, 1.167, 1.186, 1.203, 1.144, 1.096, 1.095, 1.109, 1.111, 1.105, 1.236, 1.187, 1.203, 1.262, 1.2, 1.086, 1.106, 1.112, 1.138, 1.076, 1.287, 1.255, 1.241, 1.334, 1.244, 1.112, 1.083, 1.111, 1.127, 1.025, 1.309, 1.257, 1.263, 1.393, 1.23, 1.091, 1.075, 1.078, 1.135, 1.042, 1.313, 1.303, 1.295, 1.436, 1.237, 1.064, 1.078, 1.075, 1.149, 1.037, 1.329, 1.509, 1.369, 1.546, 1.269, 1.079, 1.084, 1.047, 1.183, 1.008 ) )
        ),
        action = cms.string( "use" ),
        atIP = cms.bool( True )
      )
    ),
    TSGForRoadSearchIOpxl = cms.PSet( 
      ComponentName = cms.string( "TSGForRoadSearch" ),
      option = cms.uint32( 4 ),
      copyMuonRecHit = cms.bool( False ),
      manySeeds = cms.bool( False ),
      maxChi2 = cms.double( 40.0 ),
      propagatorName = cms.string( "SteppingHelixPropagatorAlong" ),
      propagatorCompatibleName = cms.string( "SteppingHelixPropagatorAny" ),
      errorMatrixPset = cms.PSet( 
        errorMatrixValuesPSet = cms.PSet( 
          xAxis = cms.vdouble( 0.0, 3.16, 6.69, 10.695, 15.319, 20.787, 27.479, 36.106, 48.26, 69.03, 200.0 ),
          yAxis = cms.vdouble( 0.0, 0.2, 0.3, 0.7, 0.9, 1.15, 1.35, 1.55, 1.75, 2.2, 2.5 ),
          zAxis = cms.vdouble( -3.14159, 3.14159 ),
          pf3_V11 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 4.593, 5.253, 1.895, 1.985, 2.344, 5.37, 2.059, 2.423, 1.985, 2.054, 2.071, 2.232, 2.159, 2.1, 2.355, 3.862, 1.855, 2.311, 1.784, 1.766, 1.999, 2.18, 2.071, 2.03, 2.212, 2.266, 1.693, 1.984, 1.664, 1.602, 1.761, 2.007, 1.985, 1.982, 2.118, 1.734, 1.647, 1.705, 1.56, 1.542, 1.699, 2.058, 2.037, 1.934, 2.067, 1.555, 1.566, 1.638, 1.51, 1.486, 1.635, 1.977, 1.944, 1.865, 1.925, 1.415, 1.542, 1.571, 1.499, 1.468, 1.608, 1.899, 1.893, 1.788, 1.851, 1.22, 1.49, 1.54, 1.493, 1.457, 1.572, 1.876, 1.848, 1.751, 1.827, 1.223, 1.51, 1.583, 1.486, 1.431, 1.534, 1.79, 1.802, 1.65, 1.755, 1.256, 1.489, 1.641, 1.464, 1.438, 1.48, 1.888, 1.839, 1.657, 1.903, 1.899 ) ),
          pf3_V12 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V13 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V14 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V15 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V22 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 66.152, 3.219, 66.051, 1.298, 1.186, 1.197, 1.529, 2.807, 1.056, 1.092, 1.15, 1.158, 1.163, 1.05, 1.191, 1.287, 1.371, 2.039, 1.02, 1.059, 1.048, 1.087, 1.087, 1.041, 1.072, 1.118, 1.097, 1.229, 1.042, 1.07, 1.071, 1.063, 1.039, 1.038, 1.061, 1.052, 1.058, 1.188, 1.099, 1.075, 1.082, 1.055, 1.084, 1.024, 1.058, 1.069, 1.022, 1.184, 1.117, 1.105, 1.093, 1.082, 1.086, 1.053, 1.097, 1.07, 1.044, 1.125, 1.141, 1.167, 1.136, 1.133, 1.146, 1.089, 1.081, 1.117, 1.085, 1.075, 1.212, 1.199, 1.186, 1.212, 1.168, 1.125, 1.127, 1.119, 1.114, 1.062, 1.273, 1.229, 1.272, 1.293, 1.172, 1.124, 1.141, 1.123, 1.158, 1.115, 1.419, 1.398, 1.425, 1.394, 1.278, 1.132, 1.132, 1.115, 1.26, 1.096 ) ),
          pf3_V23 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V24 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V25 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V33 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 6.174, 56.89, 1.019, 2.206, 1.694, 1.698, 1.776, 3.563, 2.141, 2.432, 1.898, 1.834, 1.763, 1.797, 1.944, 1.857, 2.068, 2.894, 1.76, 2.185, 1.664, 1.656, 1.761, 1.964, 1.925, 1.89, 2.012, 2.014, 1.651, 1.825, 1.573, 1.534, 1.634, 1.856, 1.962, 1.879, 1.95, 1.657, 1.556, 1.639, 1.481, 1.433, 1.605, 1.943, 1.99, 1.885, 1.916, 1.511, 1.493, 1.556, 1.445, 1.457, 1.543, 1.897, 1.919, 1.884, 1.797, 1.394, 1.489, 1.571, 1.436, 1.425, 1.534, 1.796, 1.845, 1.795, 1.763, 1.272, 1.472, 1.484, 1.452, 1.412, 1.508, 1.795, 1.795, 1.773, 1.741, 1.207, 1.458, 1.522, 1.437, 1.399, 1.485, 1.747, 1.739, 1.741, 1.716, 1.187, 1.463, 1.589, 1.411, 1.404, 1.471, 1.92, 1.86, 1.798, 1.867, 1.436 ) ),
          pf3_V34 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V35 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V44 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.622, 2.139, 2.08, 1.178, 1.044, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.002, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.001, 1.0, 1.0, 1.011, 1.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.002, 1.0, 1.002, 1.013, 1.0, 1.0, 1.0, 1.0, 1.0, 1.005, 1.0, 1.004, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.009, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V45 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ) ),
          pf3_V55 = cms.PSet(  values = cms.vdouble( 1.0, 1.0, 27.275, 15.167, 13.818, 1.0, 1.0, 1.0, 1.037, 1.129, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.023, 1.028, 1.063, 1.08, 1.077, 1.054, 1.068, 1.065, 1.047, 1.025, 1.046, 1.064, 1.082, 1.078, 1.137, 1.12, 1.163, 1.158, 1.112, 1.072, 1.054, 1.095, 1.101, 1.092, 1.219, 1.167, 1.186, 1.203, 1.144, 1.096, 1.095, 1.109, 1.111, 1.105, 1.236, 1.187, 1.203, 1.262, 1.2, 1.086, 1.106, 1.112, 1.138, 1.076, 1.287, 1.255, 1.241, 1.334, 1.244, 1.112, 1.083, 1.111, 1.127, 1.025, 1.309, 1.257, 1.263, 1.393, 1.23, 1.091, 1.075, 1.078, 1.135, 1.042, 1.313, 1.303, 1.295, 1.436, 1.237, 1.064, 1.078, 1.075, 1.149, 1.037, 1.329, 1.509, 1.369, 1.546, 1.269, 1.079, 1.084, 1.047, 1.183, 1.008 ) )
        ),
        action = cms.string( "use" ),
        atIP = cms.bool( True )
      )
    ),
    TSGFromPropagation = cms.PSet( 
      ComponentName = cms.string( "TSGFromPropagation" ),
      Propagator = cms.string( "SmartPropagatorAnyOpposite" ),
      MaxChi2 = cms.double( 30.0 ),
      ErrorRescaling = cms.double( 10.0 ),
      UseVertexState = cms.bool( True ),
      UpdateState = cms.bool( False ),
      UseSecondMeasurements = cms.bool( False )
    ),
    TSGFromCombinedHits = cms.PSet( 
      ComponentName = cms.string( "CombinedTSG" ),
      PSetNames = cms.vstring( 'firstTSG', 'secondTSG' ),
      firstTSG = cms.PSet( 
        ComponentName = cms.string( "TSGFromOrderedHits" ),
        TTRHBuilder = cms.string( "WithTrackAngle" ),
        OrderedHitsFactoryPSet = cms.PSet( 
          ComponentName = cms.string( "StandardHitTripletGenerator" ),
          SeedingLayers = cms.string( "PixelLayerTriplets" ),
          GeneratorPSet = cms.PSet( 
            ComponentName = cms.string( "PixelTripletHLTGenerator" ),
            useFixedPreFiltering = cms.bool( False ),
            phiPreFiltering = cms.double( 0.3 ),
            extraHitRZtolerance = cms.double( 0.06 ),
            extraHitRPhitolerance = cms.double( 0.06 ),
            useBending = cms.bool( True ),
            useMultScattering = cms.bool( True )
          )
        )
      ),
      secondTSG = cms.PSet( 
        ComponentName = cms.string( "TSGFromOrderedHits" ),
        TTRHBuilder = cms.string( "WithTrackAngle" ),
        OrderedHitsFactoryPSet = cms.PSet( 
          ComponentName = cms.string( "StandardHitPairGenerator" ),
          SeedingLayers = cms.string( "PixelLayerPairs" )
        )
      ),
      thirdTSG = cms.PSet( 
        ComponentName = cms.string( "DualByEtaTSG" ),
        etaSeparation = cms.double( 2.0 ),
        PSetNames = cms.vstring( 'endcapTSG', 'barrelTSG' ),
        barrelTSG = cms.PSet(  ),
        endcapTSG = cms.PSet( 
          ComponentName = cms.string( "TSGFromOrderedHits" ),
          TTRHBuilder = cms.string( "WithTrackAngle" ),
          OrderedHitsFactoryPSet = cms.PSet( 
            ComponentName = cms.string( "StandardHitPairGenerator" ),
            SeedingLayers = cms.string( "MixedLayerPairs" )
          )
        )
      )
    )
)
hltL3TrackCandidateFromL2 = cms.EDProducer( "CkfTrajectoryMaker",
    trackCandidateAlso = cms.bool( True ),
    SeedProducer = cms.string( "hltL3TrajectorySeed" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "muonCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    doSeedingRegionRebuilding = cms.bool( False )
)
hltL3Muons = cms.EDProducer( "L3MuonProducer",
    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
    ServiceParameters = cms.PSet( 
      UseMuonNavigation = cms.untracked.bool( True ),
      RPCLayers = cms.bool( True ),
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny', 'SteppingHelixPropagatorAlong', 'SteppingHelixPropagatorOpposite', 'PropagatorWithMaterial', 'PropagatorWithMaterialOpposite', 'SmartPropagator', 'SmartPropagatorOpposite', 'SmartPropagatorAnyOpposite', 'SmartPropagatorAny', 'SmartPropagatorRK', 'SmartPropagatorAnyRK' )
    ),
    TrackLoaderParameters = cms.PSet( 
      MuonUpdatorAtVertexParameters = cms.PSet( 
        Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 ),
        MaxChi2 = cms.double( 1000000.0 )
      ),
      VertexConstraint = cms.bool( False ),
      PutTkTrackIntoEvent = cms.untracked.bool( True ),
      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
      DoSmoothing = cms.bool( True ),
      SmoothTkTrack = cms.untracked.bool( False ),
      Smoother = cms.string( "KFSmootherForMuonTrackLoader" )
    ),
    L3TrajBuilderParameters = cms.PSet( 
      TrackTransformer = cms.PSet( 
        Fitter = cms.string( "KFFitterForRefitInsideOut" ),
        Smoother = cms.string( "KFSmootherForRefitInsideOut" ),
        TrackerRecHitBuilder = cms.string( "WithTrackAngle" ),
        MuonRecHitBuilder = cms.string( "MuonRecHitBuilder" ),
        RefitDirection = cms.string( "insideOut" ),
        RefitRPCHits = cms.bool( True )
      ),
      GlobalMuonTrackMatcher = cms.PSet( 
        Chi2Cut = cms.double( 50.0 ),
        MinP = cms.double( 2.5 ),
        MinPt = cms.double( 1.0 ),
        DeltaRCut = cms.double( 0.2 ),
        DeltaDCut = cms.double( 10.0 )
      ),
      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      Chi2CutRPC = cms.double( 1.0 ),
      Chi2CutCSC = cms.double( 150.0 ),
      Chi2CutDT = cms.double( 10.0 ),
      HitThreshold = cms.int32( 1 ),
      Chi2ProbabilityCut = cms.double( 30.0 ),
      PtCut = cms.double( 1.0 ),
      Direction = cms.int32( 0 ),
      MuonHitsOption = cms.int32( 1 ),
      TrackRecHitBuilder = cms.string( "WithTrackAngle" ),
      RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
      CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
      DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
      MuonTrackingRegionBuilder = cms.PSet( 
        beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
        UseVertex = cms.bool( False ),
        Rescale_eta = cms.double( 3.0 ),
        Rescale_phi = cms.double( 3.0 ),
        Rescale_Dz = cms.double( 3.0 ),
        DeltaZ_Region = cms.double( 15.9 ),
        DeltaR = cms.double( 0.2 ),
        EscapePt = cms.double( 1.5 ),
        Phi_min = cms.double( 0.05 ),
        Eta_min = cms.double( 0.05 ),
        UseFixedRegion = cms.bool( False ),
        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),
        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),
        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),
        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),
        vertexCollection = cms.InputTag( "pixelVertices" ),
        Eta_fixed = cms.double( 0.2 ),
        Phi_fixed = cms.double( 0.2 )
      ),
      StateOnTrackerBoundOutPropagator = cms.string( "SmartPropagatorAny" ),
      l3SeedLabel = cms.InputTag( "" ),
      tkTrajLabel = cms.InputTag( "hltL3TrackCandidateFromL2" ),
      TkTrackBuilder = cms.string( "muonCkfTrajectoryBuilder" ),
      SeedGeneratorParameters = cms.PSet( 
        ComponentName = cms.string( "TSGFromOrderedHits" ),
        TTRHBuilder = cms.string( "WithTrackAngle" ),
        OrderedHitsFactoryPSet = cms.PSet( 
          ComponentName = cms.string( "StandardHitPairGenerator" ),
          SeedingLayers = cms.string( "PixelLayerPairs" )
        )
      ),
      KFFitter = cms.string( "L3MuKFFitter" ),
      TransformerOutPropagator = cms.string( "SmartPropagatorAny" ),
      MatcherOutPropagator = cms.string( "SmartPropagator" )
    )
)
hltL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL3Muons" )
)
hltSingleMuIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltPixelTracks = cms.EDProducer( "PixelTrackProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducer" ),
      RegionPSet = cms.PSet( 
        ptMin = cms.double( 0.9 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 22.7 ),
        originXPos = cms.double( 0.0 ),
        originYPos = cms.double( 0.0 ),
        originZPos = cms.double( 0.0 ),
        precise = cms.bool( True )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "PixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRZtolerance = cms.double( 0.06 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useBending = cms.bool( True ),
        useMultScattering = cms.bool( True )
      )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" )
    ),
    FilterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      ptMin = cms.double( 0.0 ),
      tipMax = cms.double( 1.0 ),
      chi2 = cms.double( 1000.0 ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) )
)
hltL3MuonIsolations = cms.EDProducer( "L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag( "hltL3Muons" ),
    OutputMuIsoDeposits = cms.bool( True ),
    TrackPt_Min = cms.double( -1.0 ),
    ExtractorPSet = cms.PSet( 
      ComponentName = cms.string( "TrackExtractor" ),
      inputTrackCollection = cms.InputTag( "hltPixelTracks" ),
      DepositLabel = cms.untracked.string( "PXLS" ),
      Diff_r = cms.double( 0.1 ),
      Diff_z = cms.double( 0.2 ),
      DR_Veto = cms.double( 0.01 ),
      DR_Max = cms.double( 0.24 ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" ),
      BeamSpotLabel = cms.InputTag( "hltOfflineBeamSpot" ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      Pt_Min = cms.double( -1.0 )
    ),
    CutsPSet = cms.PSet( 
      ComponentName = cms.string( "SimpleCuts" ),
      EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
      ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
      Thresholds = cms.vdouble( 1.1, 1.1, 1.1, 1.1, 1.2, 1.1, 1.2, 1.1, 1.2, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 1.1, 0.9, 1.1, 1.0, 1.0, 0.9, 0.8, 0.1 )
    )
)
hltSingleMuIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltPrescaleSingleMuNoIso = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonIso = cms.EDFilter( "HLTPrescaler" )
hltDiMuonIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltDiMuonIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltDiMuonIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDiMuonIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonIsoL1Filtered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltDiMuonIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltDiMuonIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonIsoL2IsoFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltDiMuonIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIso = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltDiMuonNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltDiMuonNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDiMuonNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL1Filtered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleJPsiMM = cms.EDFilter( "HLTPrescaler" )
hltJpsiMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltJpsiMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsiMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltJpsiMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 5.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltJpsiMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 2.8 ),
    MaxInvMass = cms.double( 3.4 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleUpsilonMM = cms.EDFilter( "HLTPrescaler" )
hltUpsilonMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltUpsilonMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltUpsilonMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltUpsilonMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltUpsilonMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 6.0 ),
    MaxInvMass = cms.double( 13.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltUpsilonMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltUpsilonMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 8.0 ),
    MaxInvMass = cms.double( 11.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleZMM = cms.EDFilter( "HLTPrescaler" )
hltZMMLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltZMML1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltZMMLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltZMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltZMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 7.0 ),
    MinPtMin = cms.double( 7.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltZMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltZMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 7.0 ),
    MinPtMin = cms.double( 7.0 ),
    MinInvMass = cms.double( 70.0 ),
    MaxInvMass = cms.double( 1.0E30 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleMultiMuonNoIso = cms.EDFilter( "HLTPrescaler" )
hltMultiMuonNoIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_TripleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMultiMuonNoIsoL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMultiMuonNoIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 3 )
)
hltMultiMuonNoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMultiMuonNoIsoL1Filtered" ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMultiMuonNoIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMultiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 3 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSameSignMu = cms.EDFilter( "HLTPrescaler" )
hltSameSignMuLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSameSignMuL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSameSignMuLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltSameSignMuL2PreFiltered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSameSignMuL1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSameSignMuL3PreFiltered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSameSignMuL2PreFiltered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 1 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 0.0 ),
    MaxInvMass = cms.double( 9999.0 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuPrescale3 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale3Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuPrescale3L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale3Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale3L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale3L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale3L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale3L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuPrescale5 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale5Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuPrescale5L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale5Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale5L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale5L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale5L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale5L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreSingleMuPrescale77 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale77Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuPrescale77L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale77Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale77L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale77L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale77L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale77L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreSingleMuPrescale710 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuPrescale710Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuPrescale710L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuPrescale710Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuPrescale710L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale710L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuPrescale710L3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuPrescale710L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleMuLevel1Path = cms.EDFilter( "HLTPrescaler" )
hltMuLevel1PathLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7 OR L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuLevel1PathL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuLevel1PathLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIsoRelaxedVtx2cm = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIsoRelaxedVtx2mm = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 0.2 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIsoRelaxedVtx2cm = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleDiMuonNoIsoRelaxedVtx2mm = cms.EDFilter( "HLTPrescaler" )
hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonNoIsoL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 0.2 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescalerBLifetime1jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltBLifetime1jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBLifetimeHighestEtJets = cms.EDProducer( "LargestEtCaloJetSelector",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 4 )
)
hltBLifetimeL25Jets = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBLifetimeHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 35.0 )
)
hltPixelVertices = cms.EDProducer( "PixelVertexProducer",
    Verbosity = cms.int32( 0 ),
    Finder = cms.string( "DivisiveVertexFinder" ),
    UseError = cms.bool( True ),
    WtAverage = cms.bool( True ),
    ZOffset = cms.double( 5.0 ),
    ZSeparation = cms.double( 0.05 ),
    NTrkMin = cms.int32( 2 ),
    PtMin = cms.double( 1.0 ),
    TrackCollection = cms.InputTag( "hltPixelTracks" ),
    Method2 = cms.bool( True ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" )
)
hltBLifetimeL25Associator = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL25Jets" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL25TagInfos = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL25Associator" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL25BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( ("hltBLifetimeL25TagInfos") ),
    tagInfo = cms.InputTag( "hltBLifetimeL25TagInfos" )
)
hltBLifetimeL25filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL25BJetTags" ),
    MinTag = cms.double( 3.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBLifetimeL3Jets = cms.EDProducer( "GetJetsFromHLTobject",
    jets = cms.InputTag( "hltBLifetimeL25filter" )
)
hltBLifetimeRegionalPixelSeedGenerator = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        JetSrc = cms.InputTag( "hltBLifetimeL3Jets" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        precise = cms.bool( True ),
        ptMin = cms.double( 1.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        originZPos = cms.double( 0.0 ),
        deltaEtaRegion = cms.double( 0.25 ),
        deltaPhiRegion = cms.double( 0.25 )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltBLifetimeRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltBLifetimeRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "bJetRegionalTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltBLifetimeRegionalCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltBLifetimeRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltBLifetimeL3Associator = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL3Jets" ),
    tracks = cms.InputTag( "hltBLifetimeRegionalCtfWithMaterialTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL3TagInfos = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL3Associator" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL3BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( ("hltBLifetimeL3TagInfos") ),
    tagInfo = cms.InputTag( "hltBLifetimeL3TagInfos" )
)
hltBLifetimeL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL3BJetTags" ),
    MinTag = cms.double( 6.0 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBLifetime2jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime3jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime4jet = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetimeHT = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 470.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuon1jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonNjetL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltBSoftmuon1jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBSoftmuonHighestEtJets = cms.EDProducer( "LargestEtCaloJetSelector",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 2 )
)
hltBSoftmuonL25Jets = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBSoftmuonHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 20.0 )
)
hltBSoftmuonL25TagInfos = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftmuonL25Jets" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL2Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 )
)
hltBSoftmuonL25BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( ("hltBSoftmuonL25TagInfos") ),
    tagInfo = cms.InputTag( "hltBSoftmuonL25TagInfos" )
)
hltBSoftmuonL25filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL25BJetTags" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBSoftmuonL3TagInfos = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftmuonL25Jets" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL3Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 )
)
hltBSoftmuonL3BJetTags = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByPt" ),
    tagInfos = cms.VInputTag( ("hltBSoftmuonL3TagInfos") ),
    tagInfo = cms.InputTag( "hltBSoftmuonL3TagInfos" )
)
hltBSoftmuonL3BJetTagsByDR = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( ("hltBSoftmuonL3TagInfos") ),
    tagInfo = cms.InputTag( "hltBSoftmuonL3TagInfos" )
)
hltBSoftmuonByDRL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTagsByDR" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBSoftmuon2jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon2jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltBSoftmuonL3filter = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTags" ),
    MinTag = cms.double( 0.7 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBSoftmuon3jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 70.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon4jet = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuonHT = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltBSoftmuonHTL2filter = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 370.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltJpsitoMumuL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltJpsitoMumuL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsitoMumuL1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltMumuPixelSeedFromL2Candidate = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "L3MumuTrackingRegion" ),
      RegionPSet = cms.PSet( 
        ptMin = cms.double( 3.0 ),
        vertexZDefault = cms.double( 0.0 ),
        vertexSrc = cms.string( "hltPixelVertices" ),
        originRadius = cms.double( 1.0 ),
        originHalfLength = cms.double( 1.0 ),
        deltaEtaRegion = cms.double( 0.15 ),
        deltaPhiRegion = cms.double( 0.15 ),
        TrkSrc = cms.InputTag( "hltL2Muons" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesMumu = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltMumuPixelSeedFromL2Candidate" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "hltCkfTrajectoryBuilderMumu" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksMumu = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesMumu" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltMuTracks = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltCtfWithMaterialTracksMumu" ),
    particleType = cms.string( "mu-" )
)
hltDisplacedJpsitoMumuFilter = cms.EDFilter( "HLTDisplacedmumuFilter",
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 4.0 ),
    MinPtPair = cms.double( 4.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 6.0 ),
    ChargeOpt = cms.int32( -1 ),
    FastAccept = cms.bool( False )
)
hltMuMukL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuMukL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuMukL1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDisplacedMuMukFilter = cms.EDFilter( "HLTDisplacedmumuFilter",
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinPtPair = cms.double( 0.0 ),
    MinInvMass = cms.double( 0.2 ),
    MaxInvMass = cms.double( 3.0 ),
    ChargeOpt = cms.int32( 0 ),
    FastAccept = cms.bool( False )
)
hltMumukPixelSeedFromL2Candidate = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "L3MumuTrackingRegion" ),
      RegionPSet = cms.PSet( 
        ptMin = cms.double( 3.0 ),
        vertexZDefault = cms.double( 0.0 ),
        vertexSrc = cms.string( "hltPixelVertices" ),
        originRadius = cms.double( 1.0 ),
        originHalfLength = cms.double( 1.0 ),
        deltaEtaRegion = cms.double( 0.15 ),
        deltaPhiRegion = cms.double( 0.15 ),
        TrkSrc = cms.InputTag( "hltL2Muons" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesMumuk = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltMumukPixelSeedFromL2Candidate" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "hltCkfTrajectoryBuilderMumuk" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksMumuk = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesMumuk" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltMumukAllConeTracks = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltCtfWithMaterialTracksMumuk" ),
    particleType = cms.string( "mu-" )
)
hltmmkFilter = cms.EDFilter( "HLTmmkFilter",
    ThirdTrackMass = cms.double( 0.106 ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.2 ),
    MaxInvMass = cms.double( 2.2 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinLxySignificance = cms.double( 3.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    FastAccept = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    MuCand = cms.InputTag( "hltMuTracks" ),
    TrackCand = cms.InputTag( "hltMumukAllConeTracks" )
)
hltElectronBPrescale = cms.EDFilter( "HLTPrescaler" )
hltElectronBL1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltElBElectronL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltElectronBL1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltElBElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltElBElectronL1MatchFilter" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltElBElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltElBElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElBElectronEoverpFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltElBElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltElBElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltElBElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltMuBPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuBLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuBLifetimeL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuBLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 7.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuBLifetimeIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuBLifetimeL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBLifetimeIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBLifetimeIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuBLifetimeIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuBLifetimeIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBLifetimeIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBLifetimeIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuBsoftMuPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuBSoftL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuBLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltMuBSoftIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuBSoftL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBSoftIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBSoftIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuBSoftIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuBSoftIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuBSoftIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuBSoftIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltL1seedEJet = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1IsoEJetSingleEEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSingleL1MatchFilter" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 2.0 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoEJetSingleETrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltej1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltej2jet80 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 80.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltej3jet60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltej4jet35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltMuJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 7.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuJetsL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuJetsL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltMuJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuJetsL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuJetsL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuJetsL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuJetsHLT1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuNoL2IsoJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuNoL2IsoJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuNoL2IsoJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoL2IsoJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 8.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoL2IsoJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoL2IsoJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 6.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoL2IsoJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoL2IsoJetsL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoL2IsoJetsL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuNoL2IsoJetsL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoL2IsoJetsHLT1jet40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsPrescale = cms.EDFilter( "HLTPrescaler" )
hltMuNoIsoJetsLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuNoIsoJetsL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoIsoJetsLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 14.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoIsoJetsL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 14.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoIsoJetsHLT1jet50 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltemuPrescale = cms.EDFilter( "HLTPrescaler" )
hltEMuonLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_IsoEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltEMuL1MuonFilter = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltEMuonLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 4.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltemuL1IsoSingleL1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltEMuonLevel1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltemuL1IsoSingleElectronEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltemuL1IsoSingleL1MatchFilter" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuL1IsoSingleElectronHcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltEMuL2MuonPreFilter = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltEMuL1MuonFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltEMuL2MuonIsoFilter = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltEMuL2MuonPreFilter" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltemuL1IsoSingleElectronPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronHcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuL1IsoSingleElectronEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 1.5 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltEMuL3MuonPreFilter = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltEMuL2MuonIsoFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltEMuL3MuonIsoFilter = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltEMuL3MuonPreFilter" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltemuL1IsoSingleElectronTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltemuL1IsoSingleElectronEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltemuNonIsoPrescale = cms.EDFilter( "HLTPrescaler" )
hltemuNonIsoLevel1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_EG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltNonIsoEMuL1MuonFilter = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltemuNonIsoLevel1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltemuNonIsoL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltemuNonIsoLevel1Seed" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltemuNonIsoL1IsoEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltemuNonIsoL1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuNonIsoL1HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltNonIsoEMuL2MuonPreFilter = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltNonIsoEMuL1MuonFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 8.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltemuNonIsoL1IsoPixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltemuNonIsoL1HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltemuNonIsoL1IsoEoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoPixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIso" ),
    eoverpbarrelcut = cms.double( 1.5 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltNonIsoEMuL3MuonPreFilter = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltNonIsoEMuL2MuonPreFilter" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 10.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltemuNonIsoL1IsoTrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltemuNonIsoL1IsoEoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltLevel1seedHLTBackwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "38 OR 39" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTBackwardBSC = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTForwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "36 OR 37" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTForwardBSC = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTCSCBeamHalo = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTCSCBeamHalo = cms.EDFilter( "HLTPrescaler" )
hltLevel1seedHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( True ),
    ring2 = cms.bool( False ),
    fillHists = cms.bool( False )
)
hltLevel1seedHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( False ),
    ring2 = cms.bool( True ),
    fillHists = cms.bool( False )
)
hltLevel1seedHLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTPrescaler" )
hltFilter23HLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTCSCRing2or3Filter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 )
)
hltLevel1seedHLTTrackerCosmics = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "24 OR 25 OR 26 OR 27 OR 28" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPrescaleHLTTrackerCosmics = cms.EDFilter( "HLTPrescaler" )
hltPreMinBiasPixel = cms.EDFilter( "HLTPrescaler" )
hltL1seedMinBiasPixel = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPixelTracksForMinBias = cms.EDProducer( "PixelTrackProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "GlobalRegionProducer" ),
      RegionPSet = cms.PSet( 
        ptMin = cms.double( 0.2 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 22.7 ),
        originXPos = cms.double( 0.0 ),
        originYPos = cms.double( 0.0 ),
        originZPos = cms.double( 0.0 ),
        precise = cms.bool( True )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitTripletGenerator" ),
      SeedingLayers = cms.string( "PixelLayerTriplets" ),
      GeneratorPSet = cms.PSet( 
        ComponentName = cms.string( "PixelTripletHLTGenerator" ),
        useFixedPreFiltering = cms.bool( False ),
        phiPreFiltering = cms.double( 0.3 ),
        extraHitRZtolerance = cms.double( 0.06 ),
        extraHitRPhitolerance = cms.double( 0.06 ),
        useBending = cms.bool( True ),
        useMultScattering = cms.bool( True )
      )
    ),
    FitterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
      TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" )
    ),
    FilterPSet = cms.PSet( 
      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
      ptMin = cms.double( 0.0 ),
      tipMax = cms.double( 1.0 ),
      chi2 = cms.double( 1000.0 ),
      nSigmaInvPtTolerance = cms.double( 0.0 ),
      nSigmaTipMaxTolerance = cms.double( 0.0 )
    ),
    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) )
)
hltPixelCands = cms.EDProducer( "ConcreteChargedCandidateProducer",
    src = cms.InputTag( "hltPixelTracksForMinBias" ),
    particleType = cms.string( "pi+" )
)
hltMinBiasPixelFilter = cms.EDFilter( "HLTPixlMBFilt",
    pixlTag = cms.InputTag( "hltPixelCands" ),
    MinPt = cms.double( 0.0 ),
    MinTrks = cms.uint32( 2 ),
    MinSep = cms.double( 1.0 )
)
hltPreMBForAlignment = cms.EDFilter( "HLTPrescaler" )
hltPixelMBForAlignment = cms.EDFilter( "HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag( "hltPixelCands" ),
    MinPt = cms.double( 5.0 ),
    MinTrks = cms.uint32( 2 ),
    MinSep = cms.double( 1.0 ),
    MinIsol = cms.double( 0.05 )
)
hltl1sMin = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_MinBias_HTT10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltpreMin = cms.EDFilter( "HLTPrescaler" )
hltl1sZero = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltpreZero = cms.EDFilter( "HLTPrescaler" )
hltPrescaleTriggerType = cms.EDFilter( "HLTPrescaler" )
hltFilterTriggerType = cms.EDFilter( "TriggerTypeFilter",
    InputLabel = cms.string( "rawDataCollector" ),
    TriggerFedId = cms.int32( 812 ),
    SelectedTriggerType = cms.int32( 2 )
)
hltL1gtTrigReport = cms.EDAnalyzer( "L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool( False ),
    L1GtRecordInputTag = cms.InputTag( "hltGtDigis" )
)
hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','HLT' )
)
hltPrescalerElectronTau = cms.EDFilter( "HLTPrescaler" )
hltLevel1GTSeedElectronTau = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_TauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltEgammaL1MatchFilterRegionalElectronTau = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltLevel1GTSeedElectronTau" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltEgammaEtFilterElectronTau = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltEgammaL1MatchFilterRegionalElectronTau" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltEgammaHcalIsolFilterElectronTau = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltEgammaEtFilterElectronTau" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElectronPixelMatchFilterElectronTau = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltEgammaHcalIsolFilterElectronTau" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltElectronOneOEMinusOneOPFilterElectronTau = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltElectronPixelMatchFilterElectronTau" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsL1NonIsoForHLT" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltElectronOneOEMinusOneOPFilterElectronTau" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltEcalRegionalTausFEDs = cms.EDProducer( "EcalListOfFEDSProducer",
    debug = cms.untracked.bool( False ),
    Jets = cms.untracked.bool( True ),
    Ptmin_jets = cms.untracked.double( 20.0 ),
    CentralSource = cms.untracked.InputTag( 'hltL1extraParticles','Central' ),
    ForwardSource = cms.untracked.InputTag( 'hltL1extraParticles','Forward' ),
    TauSource = cms.untracked.InputTag( 'hltL1extraParticles','Tau' ),
    JETS_doCentral = cms.untracked.bool( False ),
    JETS_doForward = cms.untracked.bool( False ),
    OutputLabel = cms.untracked.string( "" )
)
hltEcalRegionalTausDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    DoRegional = cms.untracked.bool( True ),
    FedLabel = cms.untracked.InputTag( "hltEcalRegionalTausFEDs" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalRegionalTausWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalRegionalTausDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalRegionalTausDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRegionalTausRecHitTmp = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalTausWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalRegionalTausWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltEcalRegionalTausRecHit = cms.EDProducer( "EcalRecHitsMerger",
    debug = cms.untracked.bool( False ),
    EgammaSource_EB = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEB' ),
    MuonsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEB' ),
    TausSource_EB = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEB' ),
    JetsSource_EB = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEB' ),
    RestSource_EB = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEB' ),
    EgammaSource_EE = cms.untracked.InputTag( 'hltEcalRegionalEgammaRecHitTmp','EcalRecHitsEE' ),
    MuonsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalMuonsRecHitTmp','EcalRecHitsEE' ),
    TausSource_EE = cms.untracked.InputTag( 'hltEcalRegionalTausRecHitTmp','EcalRecHitsEE' ),
    JetsSource_EE = cms.untracked.InputTag( 'hltEcalRegionalJetsRecHitTmp','EcalRecHitsEE' ),
    RestSource_EE = cms.untracked.InputTag( 'hltEcalRegionalRestRecHitTmp','EcalRecHitsEE' ),
    OutputLabel_EB = cms.untracked.string( "EcalRecHitsEB" ),
    OutputLabel_EE = cms.untracked.string( "EcalRecHitsEE" ),
    EcalRecHitCollectionEB = cms.untracked.string( "EcalRecHitsEB" ),
    EcalRecHitCollectionEE = cms.untracked.string( "EcalRecHitsEE" )
)
hltTowerMakerForTaus = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( True ),
    MomConstrMethod = cms.int32( 0 ),
    MomEmDepth = cms.double( 0.0 ),
    MomHadDepth = cms.double( 0.0 ),
    MomTotDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    ecalInputs = cms.VInputTag( ('hltEcalRegionalTausRecHit','EcalRecHitsEB'),('hltEcalRegionalTausRecHit','EcalRecHitsEE') )
)
hltCaloTowersTau1Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIcone5Tau1Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau1Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau2Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIcone5Tau2Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau2Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau3Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIcone5Tau3Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau3Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau4Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForTaus" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIcone5Tau4Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau4Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltL2TauJetsProviderElectronTau = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltLevel1GTSeedElectronTau" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1Regional"),("hltIcone5Tau2Regional"),("hltIcone5Tau3Regional"),("hltIcone5Tau4Regional") )
)
hltL2ElectronTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJetsProviderElectronTau" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2ElectronTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2ElectronTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterEcalIsolatedTauJetsElectronTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2ElectronTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltJetTracksAssociatorAtVertexL25ElectronTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2ElectronTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25ElectronTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltJetTracksAssociatorAtVertexL25ElectronTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedTauJetsSelectorL25ElectronTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 5.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25ElectronTau") )
)
hltFilterIsolatedTauJetsL25ElectronTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedTauJetsSelectorL25ElectronTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerMuonTau = cms.EDFilter( "HLTPrescaler" )
hltLevel1GTSeedMuonTau = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu5_TauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuonTauL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltLevel1GTSeedMuonTau" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuonTauIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuonTauL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuonTauIsoL2IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuonTauIsoL2PreFiltered" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltL2TauJetsProviderMuonTau = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltLevel1GTSeedMuonTau" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1Regional"),("hltIcone5Tau2Regional"),("hltIcone5Tau3Regional"),("hltIcone5Tau4Regional") )
)
hltL2MuonTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJetsProviderMuonTau" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2MuonTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2MuonTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterEcalIsolatedTauJetsMuonTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2MuonTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltJetsPixelTracksAssociatorMuonTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2MuonTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltPixelTrackConeIsolationMuonTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( " hltJetsPixelTracksAssociatorMuonTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltPixelTrackIsolatedTauJetsSelectorMuonTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 5.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltPixelTrackConeIsolationMuonTau") )
)
hltFilterPixelTrackIsolatedTauJetsMuonTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltPixelTrackIsolatedTauJetsSelectorMuonTau" ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltMuonTauIsoL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuonTauIsoL2IsoFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 14.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuonTauIsoL3IsoFiltered = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltMuonTauIsoL3PreFiltered" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltSingleTauMETPrescaler = cms.EDFilter( "HLTPrescaler" )
hltSingleTauMETL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_TauJet30_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltCaloTowersTau1 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIcone5Tau1 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau1" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau2 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIcone5Tau2 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau2" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau3 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIcone5Tau3 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau3" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hltCaloTowersTau4 = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForAll" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'hltL1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIcone5Tau4 = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.5 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau4" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" )
)
hlt1METSingleTauMET = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauMETJets = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltSingleTauMETL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1"),("hltIcone5Tau2"),("hltIcone5Tau3"),("hltIcone5Tau4") )
)
hltL2SingleTauMETIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( " hltL2SingleTauMETJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2SingleTauMETIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauMETIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauMETEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauMETIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTauMET = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauMETIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTauMET = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTauMET" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTauMET = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 5.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25SingleTauMET") )
)
hltFilterL25SingleTauMET = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTauMET" ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL3SingleTauMETPixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        deltaPhiRegion = cms.double( 0.1 ),
        deltaEtaRegion = cms.double( 0.1 ),
        ptMin = cms.double( 10.0 ),
        originZPos = cms.double( 0.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        precise = cms.bool( True ),
        JetSrc = cms.InputTag( "hltIsolatedL25SingleTauMET" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesL3SingleTauMET = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL3SingleTauMETPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "trajBuilderL3" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksL3SingleTauMET = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesL3SingleTauMET" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltAssociatorL3SingleTauMET = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTauMET" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTauMET" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTauMET = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTauMET" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTauMET = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 15.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL3SingleTauMET") )
)
hltFilterL3SingleTauMET = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTauMET" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltSingleTauPrescaler = cms.EDFilter( "HLTPrescaler" )
hltSingleTauL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleTauJet80" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hlt1METSingleTau = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 65.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauJets = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltSingleTauL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1"),("hltIcone5Tau2"),("hltIcone5Tau3"),("hltIcone5Tau4") )
)
hltL2SingleTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2SingleTauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2SingleTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.5 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 5.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25SingleTau") )
)
hltFilterL25SingleTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTau" ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL3SingleTauPixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        deltaEtaRegion = cms.double( 0.1 ),
        deltaPhiRegion = cms.double( 0.1 ),
        ptMin = cms.double( 10.0 ),
        originZPos = cms.double( 0.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        precise = cms.bool( True ),
        JetSrc = cms.InputTag( "hltIsolatedL25SingleTau" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesL3SingleTau = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL3SingleTauPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "trajBuilderL3" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksL3SingleTau = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesL3SingleTau" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltAssociatorL3SingleTau = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTau" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTau" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTau = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTau" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 20.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL3SingleTau") )
)
hltFilterL3SingleTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltSingleElectronEt10L1NonIsoHLTNonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEt8 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG8" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoStartUpElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.025 ),
      pPhiMin1 = cms.double( -0.015 ),
      ePhiMax1 = cms.double( 0.015 ),
      ePhiMin1 = cms.double( -0.025 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.11 ),
      rMinI = cms.double( -0.11 ),
      r2MaxF = cms.double( 0.096 ),
      r2MinF = cms.double( -0.096 ),
      z2MaxB = cms.double( 0.06 ),
      z2MinB = cms.double( -0.06 ),
      PhiMax2 = cms.double( 0.0050 ),
      PhiMin2 = cms.double( -0.0050 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1NonIsoStartUpElectronPixelSeeds = cms.EDProducer( "ElectronPixelSeedProducer",
    barrelSuperClusters = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    endcapSuperClusters = cms.InputTag( "hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated" ),
    SeedConfiguration = cms.PSet( 
      SCEtCut = cms.double( 5.0 ),
      maxHOverE = cms.double( 0.2 ),
      hOverEConeSize = cms.double( 0.1 ),
      hbheInstance = cms.string( "" ),
      hbheModule = cms.string( "hbhereco" ),
      pPhiMax1 = cms.double( 0.025 ),
      pPhiMin1 = cms.double( -0.015 ),
      ePhiMax1 = cms.double( 0.015 ),
      ePhiMin1 = cms.double( -0.025 ),
      DeltaPhi2 = cms.double( 0.0040 ),
      DeltaPhi1High = cms.double( 0.08 ),
      DeltaPhi1Low = cms.double( 0.23 ),
      SizeWindowENeg = cms.double( 0.675 ),
      HighPtThreshold = cms.double( 35.0 ),
      LowPtThreshold = cms.double( 5.0 ),
      searchInTIDTEC = cms.bool( True ),
      dynamicPhiRoad = cms.bool( False ),
      rMaxI = cms.double( 0.11 ),
      rMinI = cms.double( -0.11 ),
      r2MaxF = cms.double( 0.096 ),
      r2MinF = cms.double( -0.096 ),
      z2MaxB = cms.double( 0.06 ),
      z2MinB = cms.double( -0.06 ),
      PhiMax2 = cms.double( 0.0050 ),
      PhiMin2 = cms.double( -0.0050 ),
      hcalRecHits = cms.InputTag( "hltHbhereco" ),
      fromTrackerSeeds = cms.bool( False ),
      preFilteredSeeds = cms.bool( False ),
      initialSeeds = cms.InputTag( "globalMixedSeeds" )
    )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltCkfL1IsoStartUpTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoStartUpElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1IsoStartUpWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1IsoStartUpTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchStartUpElectronsL1Iso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoStartUpWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltCkfL1NonIsoStartUpTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "GroupedCkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfL1NonIsoStartUpWithMaterialTracks = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfL1NonIsoStartUpTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchStartUpElectronsL1NonIso = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoStartUpWithMaterialTracks" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)
hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1IsoStartUpElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1IsoStartUpElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1IsoStartUpElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1IsoStartUpElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    )
)
hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltEgammaRegionalCTFFinalFitWithMaterial" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL1IsoStartUpElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    trackProducer = cms.InputTag( "hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoStartupElectronTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    trackProducer = cms.InputTag( "hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltSingleElectronEt8L1NonIsoHLTnoIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedSingleEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt5" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1seedRelaxedDoubleEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional" ),
    etcut = cms.double( 5.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltSinglePhotonEt10L1NonIsoPresc = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt10EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoSinglePhotonEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt10TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt10HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1sIsolTrack = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80 " ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPreIsolTrackNoEcalIso = cms.EDFilter( "HLTPrescaler" )
hltIsolPixelTrackProd = cms.EDProducer( "IsolatedPixelTrackCandidateProducer",
    L1eTauJetsSource = cms.InputTag( 'hltL1extraParticles','Tau' ),
    PixelTracksSource = cms.InputTag( "hltPixelTracks" ),
    L1GTSeedLabel = cms.InputTag( "hltL1sIsolTrack" ),
    ecalFilterLabel = cms.InputTag( "aaa" )
)
hltIsolPixelTrackFilter = cms.EDFilter( "HLTPixelIsolTrackFilter",
    candTag = cms.InputTag( "hltIsolPixelTrackProd" ),
    MinPtTrack = cms.double( 20.0 ),
    MaxPtNearby = cms.double( 2.0 ),
    MaxEtaTrack = cms.double( 2.1 ),
    filterTrackEnergy = cms.bool( False ),
    MinEnergyTrack = cms.double( 15.0 )
)
hltSiStripRegFED = cms.EDProducer( "SiStripRegFEDSelector",
    regSeedLabel = cms.InputTag( "hltIsolPixelTrackFilter" ),
    delta = cms.double( 1.0 ),
    rawInputLabel = cms.InputTag( "rawDataCollector" )
)
hltEcalRegFED = cms.EDProducer( "ECALRegFEDSelector",
    regSeedLabel = cms.InputTag( "hltIsolPixelTrackFilter" ),
    delta = cms.double( 1.0 ),
    rawInputLabel = cms.InputTag( "rawDataCollector" )
)
hltSubdetFED = cms.EDProducer( "SubdetFEDSelector",
    getECAL = cms.bool( False ),
    getSiStrip = cms.bool( False ),
    getSiPixel = cms.bool( True ),
    getHCAL = cms.bool( True ),
    getMuon = cms.bool( False ),
    getTrigger = cms.bool( True ),
    rawInputLabel = cms.InputTag( "rawDataCollector" )
)
hltL1sHcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltHcalPhiSymPresc = cms.EDFilter( "HLTPrescaler" )
hltAlCaHcalPhiSymStream = cms.EDFilter( "HLTHcalPhiSymFilter",
    HBHEHitCollection = cms.InputTag( "hltHbhereco" ),
    HOHitCollection = cms.InputTag( "hltHoreco" ),
    HFHitCollection = cms.InputTag( "hltHfreco" ),
    phiSymHBHEHitCollection = cms.string( "phiSymHcalRecHitsHBHE" ),
    phiSymHOHitCollection = cms.string( "phiSymHcalRecHitsHO" ),
    phiSymHFHitCollection = cms.string( "phiSymHcalRecHitsHF" ),
    eCut_HB = cms.double( -10.0 ),
    eCut_HE = cms.double( -10.0 ),
    eCut_HO = cms.double( -10.0 ),
    eCut_HF = cms.double( -10.0 )
)
hltL1sEcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltEcalPhiSymPresc = cms.EDFilter( "HLTPrescaler" )
hltEcalDigis = cms.EDProducer( "EcalRawToDigiDev",
    syncCheck = cms.untracked.bool( False ),
    eventPut = cms.untracked.bool( True ),
    InputLabel = cms.untracked.string( "rawDataCollector" ),
    orderedFedList = cms.untracked.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    orderedDCCIdList = cms.untracked.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 )
)
hltEcalWeightUncalibRecHit = cms.EDProducer( "EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag( 'hltEcalDigis','ebDigis' ),
    EEdigiCollection = cms.InputTag( 'hltEcalDigis','eeDigis' ),
    EBhitCollection = cms.string( "EcalUncalibRecHitsEB" ),
    EEhitCollection = cms.string( "EcalUncalibRecHitsEE" )
)
hltEcalRecHit = cms.EDProducer( "EcalRecHitProducer",
    EBuncalibRecHitCollection = cms.InputTag( 'hltEcalWeightUncalibRecHit','EcalUncalibRecHitsEB' ),
    EEuncalibRecHitCollection = cms.InputTag( 'hltEcalWeightUncalibRecHit','EcalUncalibRecHitsEE' ),
    EBrechitCollection = cms.string( "EcalRecHitsEB" ),
    EErechitCollection = cms.string( "EcalRecHitsEE" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
)
hltAlCaPhiSymStream = cms.EDFilter( "HLTEcalPhiSymFilter",
    barrelHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEB' ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHit','EcalRecHitsEE' ),
    phiSymBarrelHitCollection = cms.string( "phiSymEcalRecHitsEB" ),
    phiSymEndcapHitCollection = cms.string( "phiSymEcalRecHitsEE" ),
    eCut_barrel = cms.double( 0.15 ),
    eCut_endcap = cms.double( 0.75 )
)
hltPrePi0Ecal = cms.EDFilter( "HLTPrescaler" )
hltL1sEcalPi0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltAlCaPi0RegRecHits = cms.EDFilter( "HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    gammaCandEtaSize = cms.int32( 9 ),
    gammaCandPhiSize = cms.int32( 21 ),
    clusSeedThr = cms.double( 0.5 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    selePtGammaOne = cms.double( 0.9 ),
    selePtGammaTwo = cms.double( 0.9 ),
    selePtPi0 = cms.double( 2.5 ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleXtalMinEnergy = cms.double( 0.0 ),
    seleNRHMax = cms.int32( 1000 ),
    seleS4S9GammaOne = cms.double( 0.85 ),
    seleS4S9GammaTwo = cms.double( 0.85 ),
    selePi0Iso = cms.double( 0.5 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 5.7 ),
    ParameterW0 = cms.double( 4.2 )
)
hltPrescaleSingleMuLevel2 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuLevel2NoIsoL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltJpsitoMumuL1SeedRelaxed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltJpsitoMumuL1FilteredRelaxed = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltJpsitoMumuL1SeedRelaxed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 2 )
)
hltDisplacedJpsitoMumuFilterRelaxed = cms.EDFilter( "HLTDisplacedmumuFilter",
    MinLxySignificance = cms.double( 3.0 ),
    MaxNormalisedChi2 = cms.double( 10.0 ),
    MinCosinePointingAngle = cms.double( 0.9 ),
    Src = cms.InputTag( "hltMuTracks" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 3.0 ),
    MinPtPair = cms.double( 4.0 ),
    MinInvMass = cms.double( 1.0 ),
    MaxInvMass = cms.double( 6.0 ),
    ChargeOpt = cms.int32( -1 ),
    FastAccept = cms.bool( False )
)
PreHLT2Photon10L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon8L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1ElectronLWEt12L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltL1seedRelaxedSingleEt10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
PreHLT1ElectronLWEt15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
hltL1seedRelaxedSingleEt15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
PreHLT1Photon20L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon25L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional" ),
    etcut = cms.double( 25.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Electron18L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
hltL1seedRelaxedSingleEt12 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG12" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
PreHLT1Electron15L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.0 ),
    hcalisolendcapcut = cms.double( 9999999.0 ),
    HoverEcut = cms.double( 999999.0 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1ElectronLW12L1IHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSingleElectronEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1ElectronLWEt18L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional" ),
    etcut = cms.double( 18.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
PreHLT1ElectronLWEt15L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoLargeWindowElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1IsoLargeWindow" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLargeWindow" )
)
PreHLT1Photon40L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon30L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon45L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional" ),
    etcut = cms.double( 45.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon30L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional" ),
    etcut = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon25L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional" ),
    etcut = cms.double( 25.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon20L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT1Photon15L1RHIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 1.5 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 4.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2ElectronLWonlyPMEt8L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional" ),
    etcut = cms.double( 8.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2ElectronLWonlyPMEt10L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt5" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional" ),
    etcut = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1seedRelaxedDoubleEt10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
PreHLT2ElectronLWonlyPMEt12L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional" ),
    etcut = cms.double( 12.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon20L1RHNonIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 9999999.9 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 9999999.9 ),
    hcalisolendcapcut = cms.double( 9999999.9 ),
    HoverEcut = cms.double( 9999999.9 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 9999999.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltl1sMinHcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltpreMinHcal = cms.EDFilter( "HLTPrescaler" )
hltl1sMinEcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG2 OR L1_DoubleEG1" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltpreMinEcal = cms.EDFilter( "HLTPrescaler" )
PreHLT1Electron15L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt12" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional" ),
    etcut = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    hcalisolbarrelcut = cms.double( 6.0 ),
    hcalisolendcapcut = cms.double( 6.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter( "HLTElectronOneOEMinusOneOPFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" ),
    barrelcut = cms.double( 999.03 ),
    endcapcut = cms.double( 999.03 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter" ),
    isoTag = cms.InputTag( "hltL1IsoStartUpElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoStartupElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.12 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchStartUpElectronsL1NonIso" )
)
PreHLT1Photon40L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedSingleEt15" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional" ),
    etcut = cms.double( 40.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
PreHLT2Photon20L1RHLooseIso = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedRelaxedDoubleEt10" ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional" ),
    etcut = cms.double( 20.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter( "HLTEgammaEcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    ecalisolcut = cms.double( 3.0 ),
    ecalIsoOverEtCut = cms.double( 0.05 ),
    ecalIsoOverEt2Cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    hcalisolbarrelcut = cms.double( 12.0 ),
    hcalisolendcapcut = cms.double( 8.0 ),
    HoverEcut = cms.double( 0.1 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter( "HLTPhotonTrackIsolFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonTrackIsol" ),
    numtrackisolcut = cms.double( 3.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1s4jet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_QuadJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre4jet30 = cms.EDFilter( "HLTPrescaler" )
hlt4jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hlt1METSingleTauRelaxed = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 10000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTauRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauIsolationSelectorRelaxed','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTauRelaxed = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTauRelaxed" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTauRelaxed = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25SingleTauRelaxed") )
)
hltFilterL25SingleTauRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTauRelaxed" ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL3SingleTauPixelSeedsRelaxed = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        deltaEtaRegion = cms.double( 0.1 ),
        deltaPhiRegion = cms.double( 0.1 ),
        ptMin = cms.double( 10.0 ),
        originZPos = cms.double( 0.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        precise = cms.bool( True ),
        JetSrc = cms.InputTag( "hltIsolatedL25SingleTauRelaxed" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesL3SingleTauRelaxed = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL3SingleTauPixelSeedsRelaxed" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "trajBuilderL3" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksL3SingleTauRelaxed = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesL3SingleTauRelaxed" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltAssociatorL3SingleTauRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTauRelaxed" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTauRelaxed" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTauRelaxed = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTauRelaxed" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTauRelaxed = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL3SingleTauRelaxed") )
)
hltFilterL3SingleTauRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTauRelaxed" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hlt1METSingleTauMETRelaxed = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL2SingleTauMETIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2SingleTauMETIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterSingleTauMETEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2SingleTauMETIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltAssociatorL25SingleTauMETRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2SingleTauMETIsolationSelectorRelaxed','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25SingleTauMETRelaxed = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25SingleTauMETRelaxed" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25SingleTauMETRelaxed = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25SingleTauMETRelaxed") )
)
hltFilterL25SingleTauMETRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25SingleTauMETRelaxed" ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL3SingleTauMETPixelSeedsRelaxed = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        deltaPhiRegion = cms.double( 0.1 ),
        deltaEtaRegion = cms.double( 0.1 ),
        ptMin = cms.double( 10.0 ),
        originZPos = cms.double( 0.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        precise = cms.bool( True ),
        JetSrc = cms.InputTag( "hltIsolatedL25SingleTauMETRelaxed" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltCkfTrackCandidatesL3SingleTauMETRelaxed = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltL3SingleTauMETPixelSeedsRelaxed" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "trajBuilderL3" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltCtfWithMaterialTracksL3SingleTauMETRelaxed = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltCkfTrackCandidatesL3SingleTauMETRelaxed" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltAssociatorL3SingleTauMETRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltIsolatedL25SingleTauMETRelaxed" ),
    tracks = cms.InputTag( "hltCtfWithMaterialTracksL3SingleTauMETRelaxed" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL3SingleTauMETRelaxed = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL3SingleTauMETRelaxed" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 5 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL3SingleTauMETRelaxed = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL3SingleTauMETRelaxed") )
)
hltFilterL3SingleTauMETRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL3SingleTauMETRelaxed" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltDoubleTauPrescaler = cms.EDFilter( "HLTPrescaler" )
hltDoubleTauL1SeedFilterRelaxed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleTauJet20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL2DoubleTauJetsRelaxed = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltDoubleTauL1SeedFilterRelaxed" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1Regional"),("hltIcone5Tau2Regional"),("hltIcone5Tau3Regional"),("hltIcone5Tau4Regional") )
)
hltL2DoubleTauIsolationProducerRelaxed = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2DoubleTauJetsRelaxed" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2DoubleTauIsolationSelectorRelaxed = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2DoubleTauIsolationProducerRelaxed','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterDoubleTauEcalIsolationRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2DoubleTauIsolationSelectorRelaxed','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltAssociatorL25PixelTauIsolatedRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2DoubleTauIsolationSelectorRelaxed','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25PixelTauIsolatedRelaxed = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25PixelTauIsolatedRelaxed" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25PixelTauRelaxed = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.2 ),
    IsolationCone = cms.double( 0.1 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25PixelTauIsolatedRelaxed") )
)
hltFilterL25PixelTauRelaxed = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25PixelTauRelaxed" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltDoubleTauL1SeedFilter = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleTauJet40" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL2DoubleTauJets = cms.EDProducer( "L2TauJetsProvider",
    L1Particles = cms.InputTag( 'hltL1extraParticles','Tau' ),
    L1TauTrigger = cms.InputTag( "hltDoubleTauL1SeedFilter" ),
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( ("hltIcone5Tau1Regional"),("hltIcone5Tau2Regional"),("hltIcone5Tau3Regional"),("hltIcone5Tau4Regional") )
)
hltL2DoubleTauIsolationProducer = cms.EDProducer( "L2TauIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2DoubleTauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalTausRecHit','EcalRecHitsEE' ),
    crystalThreshold = cms.double( 0.1 ),
    towerThreshold = cms.double( 0.2 ),
    ECALIsolation = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      runAlgorithm = cms.bool( False ),
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 )
    )
)
hltL2DoubleTauIsolationSelector = cms.EDProducer( "L2TauIsolationSelector",
    L2InfoAssociation = cms.InputTag( 'hltL2DoubleTauIsolationProducer','L2TauIsolationInfoAssociator' ),
    ECALIsolEt = cms.double( 5.0 ),
    TowerIsolEt = cms.double( 1000.0 ),
    ClusterEtaRMS = cms.double( 1000.0 ),
    ClusterPhiRMS = cms.double( 1000.0 ),
    ClusterDRRMS = cms.double( 1000.0 ),
    ClusterNClusters = cms.int32( 1000 ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 )
)
hltFilterDoubleTauEcalIsolation = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( 'hltL2DoubleTauIsolationSelector','Isolated' ),
    MinPt = cms.double( 1.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltAssociatorL25PixelTauIsolated = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( 'hltL2DoubleTauIsolationSelector','Isolated' ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltConeIsolationL25PixelTauIsolated = cms.EDProducer( "ConeIsolation",
    JetTrackSrc = cms.InputTag( "hltAssociatorL25PixelTauIsolated" ),
    vertexSrc = cms.InputTag( "hltPixelVertices" ),
    useVertex = cms.bool( True ),
    useBeamSpot = cms.bool( True ),
    BeamSpotProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    MinimumNumberOfPixelHits = cms.int32( 2 ),
    MinimumNumberOfHits = cms.int32( 2 ),
    MaximumTransverseImpactParameter = cms.double( 300.0 ),
    MinimumTransverseMomentum = cms.double( 1.0 ),
    MaximumChiSquared = cms.double( 100.0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.45 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 6.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    UseFixedSizeCone = cms.bool( True ),
    VariableConeParameter = cms.double( 3.5 ),
    VariableMaxCone = cms.double( 0.17 ),
    VariableMinCone = cms.double( 0.05 )
)
hltIsolatedL25PixelTau = cms.EDProducer( "IsolatedTauJetsSelector",
    MatchingCone = cms.double( 0.1 ),
    SignalCone = cms.double( 0.07 ),
    IsolationCone = cms.double( 0.3 ),
    MinimumTransverseMomentumInIsolationRing = cms.double( 1.0 ),
    MinimumTransverseMomentumLeadingTrack = cms.double( 3.0 ),
    MaximumNumberOfTracksIsolationRing = cms.int32( 0 ),
    DeltaZetTrackVertex = cms.double( 0.2 ),
    UseVertex = cms.bool( False ),
    VertexSrc = cms.InputTag( "hltPixelVertices" ),
    UseInHLTOpen = cms.bool( False ),
    JetSrc = cms.VInputTag( ("hltConeIsolationL25PixelTauIsolated") )
)
hltFilterL25PixelTau = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltIsolatedL25PixelTau" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 0.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPre1Level1jet15 = cms.EDFilter( "HLTPrescaler" )
hltL1s1Level1jet15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1s1jet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet30 = cms.EDFilter( "HLTPrescaler" )
hlt1jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet50 = cms.EDFilter( "HLTPrescaler" )
hlt1jet50 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet80 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet80 = cms.EDFilter( "HLTPrescaler" )
hlt1jet80 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 80.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet110 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet110 = cms.EDFilter( "HLTPrescaler" )
hlt1jet110 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 110.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet250 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet250 = cms.EDFilter( "HLTPrescaler" )
hlt1jet250 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 250.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1SumET = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETT60" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET1SumET = cms.EDFilter( "HLTPrescaler" )
hlt1SumET120 = cms.EDFilter( "HLTGlobalSumMET",
    inputTag = cms.InputTag( "hltMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 120.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1jet180 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1jet180 = cms.EDFilter( "HLTPrescaler" )
hlt1jet180regional = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5Regional" ),
    MinPt = cms.double( 180.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPreLevel1MET20 = cms.EDFilter( "HLTPrescaler" )
hltL1sLevel1MET20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1s1MET25 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM20" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET25 = cms.EDFilter( "HLTPrescaler" )
hlt1MET25 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 25.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET35 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET35 = cms.EDFilter( "HLTPrescaler" )
hlt1MET35 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM40" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET50 = cms.EDFilter( "HLTPrescaler" )
hlt1MET50 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1s1MET65 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET65 = cms.EDFilter( "HLTPrescaler" )
hltL1s1MET75 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM50" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPre1MET75 = cms.EDFilter( "HLTPrescaler" )
hlt1MET75 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    MinPt = cms.double( 75.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1sdijetave15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave15 = cms.EDFilter( "HLTPrescaler" )
hltdijetave15 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 15.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave30 = cms.EDFilter( "HLTPrescaler" )
hltdijetave30 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 30.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave50 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet50" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave50 = cms.EDFilter( "HLTPrescaler" )
hltdijetave50 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 50.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave70 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave70 = cms.EDFilter( "HLTPrescaler" )
hltdijetave70 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 70.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave130 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave130 = cms.EDFilter( "HLTPrescaler" )
hltdijetave130 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 130.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sdijetave220 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet70" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltPredijetave220 = cms.EDFilter( "HLTPrescaler" )
hltdijetave220 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    minEtAve = cms.double( 220.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltPrescalerBLifetime1jet120 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeL1seedsLowEnergy = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltBLifetime1jetL2filter120 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 120.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltBLifetimeL25JetsRelaxed = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltBLifetimeHighestEtJets" ),
    filter = cms.bool( False ),
    etMin = cms.double( 30.0 )
)
hltBLifetimeL25AssociatorRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL25JetsRelaxed" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL25TagInfosRelaxed = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL25AssociatorRelaxed" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL25BJetTagsRelaxed = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( ("hltBLifetimeL25TagInfosRelaxed") ),
    tagInfo = cms.InputTag( "hltBLifetimeL25TagInfosRelaxed" )
)
hltBLifetimeL25filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL25BJetTagsRelaxed" ),
    MinTag = cms.double( 2.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBLifetimeL3JetsRelaxed = cms.EDProducer( "GetJetsFromHLTobject",
    jets = cms.InputTag( "hltBLifetimeL25filterRelaxed" )
)
hltBLifetimeRegionalPixelSeedGeneratorRelaxed = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
    RegionFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "TauRegionalPixelSeedGenerator" ),
      RegionPSet = cms.PSet( 
        JetSrc = cms.InputTag( "hltBLifetimeL3JetsRelaxed" ),
        vertexSrc = cms.InputTag( "hltPixelVertices" ),
        precise = cms.bool( True ),
        ptMin = cms.double( 1.0 ),
        originRadius = cms.double( 0.2 ),
        originHalfLength = cms.double( 0.2 ),
        originZPos = cms.double( 0.0 ),
        deltaEtaRegion = cms.double( 0.5 ),
        deltaPhiRegion = cms.double( 0.5 )
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "PixelLayerPairs" )
    ),
    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),
    TTRHBuilder = cms.string( "WithTrackAngle" )
)
hltBLifetimeRegionalCkfTrackCandidatesRelaxed = cms.EDProducer( "CkfTrackCandidateMaker",
    SeedProducer = cms.string( "hltBLifetimeRegionalPixelSeedGeneratorRelaxed" ),
    SeedLabel = cms.string( "" ),
    TrajectoryBuilder = cms.string( "bJetRegionalTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    )
)
hltBLifetimeRegionalCtfWithMaterialTracksRelaxed = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltBLifetimeRegionalCkfTrackCandidatesRelaxed" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltBLifetimeL3AssociatorRelaxed = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL3JetsRelaxed" ),
    tracks = cms.InputTag( "hltBLifetimeRegionalCtfWithMaterialTracksRelaxed" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL3TagInfosRelaxed = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL3AssociatorRelaxed" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumDecayLength = cms.double( 5.0 ),
    maximumChiSquared = cms.double( 20.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    maximumDistanceToJetAxis = cms.double( 0.07 ),
    jetDirectionUsingTracks = cms.bool( False )
)
hltBLifetimeL3BJetTagsRelaxed = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( ("hltBLifetimeL3TagInfosRelaxed") ),
    tagInfo = cms.InputTag( "hltBLifetimeL3TagInfosRelaxed" )
)
hltBLifetimeL3filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL3BJetTagsRelaxed" ),
    MinTag = cms.double( 3.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBLifetime1jet160 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime1jetL2filter160 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 160.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBLifetime2jet100 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime2jet60 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime2jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBSoftmuon2jet100 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon2jetL2filter100 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltBSoftmuonL3filterRelaxed = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftmuonL3BJetTags" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltPrescalerBSoftmuon2jet60 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon2jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltPrescalerBLifetime3jet40 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime3jet60 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime3jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon3jet40 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter40 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 40.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBSoftmuon3jet60 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon3jetL2filter60 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 60.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescalerBLifetime4jet30 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetime4jet35 = cms.EDFilter( "HLTPrescaler" )
hltBLifetime4jetL2filter35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuon4jet30 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBSoftmuon4jet35 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuon4jetL2filter35 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltPrescalerBLifetimeHT320 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter320 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 320.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBLifetimeHT420 = cms.EDFilter( "HLTPrescaler" )
hltBLifetimeHTL2filter420 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 420.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuonHT250 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL1seedsLowEnergy = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_HTT200" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltBSoftmuonHTL2filter250 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 250.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltPrescalerBSoftmuonHT330 = cms.EDFilter( "HLTPrescaler" )
hltBSoftmuonHTL2filter330 = cms.EDFilter( "HLTGlobalSumHT",
    inputTag = cms.InputTag( "hltHtMet" ),
    observable = cms.string( "sumEt" ),
    Min = cms.double( 330.0 ),
    Max = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1seedEJet30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_EG5_TripleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltL1IsoSingleEJet30L1MatchFilter = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'hltL1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltRecoNonIsolatedEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'hltL1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1seedEJet30" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1IsoEJetSingleEEt5Filter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1IsoSingleEJet30L1MatchFilter" ),
    etcut = cms.double( 5.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5HcalIsolFilter = cms.EDFilter( "HLTEgammaHcalIsolFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5Filter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltSingleEgammaHcalNonIsol" ),
    hcalisolbarrelcut = cms.double( 3.0 ),
    hcalisolendcapcut = cms.double( 3.0 ),
    HoverEcut = cms.double( 0.05 ),
    HoverEt2cut = cms.double( 0.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "electronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoEJetSingleEEt5EoverpFilter = cms.EDFilter( "HLTElectronEoverpFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5PixelMatchFilter" ),
    electronIsolatedProducer = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    electronNonIsolatedProducer = cms.InputTag( "pixelMatchElectronsForHLT" ),
    eoverpbarrelcut = cms.double( 2.0 ),
    eoverpendcapcut = cms.double( 2.45 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True )
)
hltL1IsoEJetSingleEEt5TrackIsolFilter = cms.EDFilter( "HLTElectronTrackIsolFilterRegional",
    candTag = cms.InputTag( "hltL1IsoEJetSingleEEt5EoverpFilter" ),
    isoTag = cms.InputTag( "hltL1IsoElectronTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoElectronTrackIsol" ),
    pttrackisolcut = cms.double( 0.06 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( True ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltPixelMatchElectronsL1Iso" ),
    L1NonIsoCand = cms.InputTag( "hltPixelMatchElectronsL1NonIso" )
)
hltej3jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltMuNoIsoJets30Level1Seed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_TripleJet15" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuNoIsoJetsMinPt4L1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuNoIsoJets30Level1Seed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 5.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltMuNoIsoJetsMinPt4L2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsMinPt4L1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltMuNoIsoJetsMinPtL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltMuNoIsoJetsMinPt4L2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltMuNoIsoHLTJets3jet30 = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltMCJetCorJetIcone5" ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 3 )
)
hltPrescaleMuLevel1Open = cms.EDFilter( "HLTPrescaler" )
hltMuLevel1PathLevel1OpenSeed = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltMuLevel1PathLevel1OpenSeed" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso9 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered9 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered7" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso11 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered9 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered11 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered9" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso13 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoLevel1Seed10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuNoIsoL1Filtered10 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuNoIsoLevel1Seed10" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuNoIsoL2PreFiltered11 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered13 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered11" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 13.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuNoIso15 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuNoIsoL2PreFiltered12 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuNoIsoL3PreFiltered15 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered12" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso9 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoL2PreFiltered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered7 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered7" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered9 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered7" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered9 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered9" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso13 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoLevel1Seed10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu10" ),
    L1GtReadoutRecordTag = cms.InputTag( "hltGtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "hltL1GtObjectMap" ),
    L1CollectionsTag = cms.InputTag( "hltL1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "hltL1extraParticles" )
)
hltSingleMuIsoL1Filtered10 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "hltSingleMuIsoLevel1Seed10" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinQuality = cms.int32( -1 ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL2PreFiltered11 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered11 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered11" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered13 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered11" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 13.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered13 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered13" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescaleSingleMuIso15 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoL2PreFiltered12 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered10" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 12.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL2IsoFiltered12 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered12" ),
    IsoTag = cms.InputTag( "hltL2MuonIsolations" ),
    MinN = cms.int32( 1 )
)
hltSingleMuIsoL3PreFiltered15 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered12" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMuIsoL3IsoFiltered15 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered15" ),
    IsoTag = cms.InputTag( "hltL3MuonIsolations" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)
hltPrescalePsi2SMM = cms.EDFilter( "HLTPrescaler" )
hltPsi2SMML2Filtered = cms.EDFilter( "HLTMuonDimuonL2Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltJpsiMML1Filtered" ),
    SeedTag = cms.InputTag( "hltL2MuonSeeds" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 100.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 1.6 ),
    MaxInvMass = cms.double( 5.6 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltPsi2SMML3Filtered = cms.EDFilter( "HLTMuonDimuonL3Filter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    LinksTag = cms.InputTag( "hltL3Muons" ),
    PreviousCandTag = cms.InputTag( "hltPsi2SMML2Filtered" ),
    FastAccept = cms.bool( False ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    ChargeOpt = cms.int32( 0 ),
    MinPtPair = cms.double( 0.0 ),
    MinPtMax = cms.double( 3.0 ),
    MinPtMin = cms.double( 3.0 ),
    MinInvMass = cms.double( 3.5 ),
    MaxInvMass = cms.double( 3.9 ),
    MinAcop = cms.double( -1.0 ),
    MaxAcop = cms.double( 3.15 ),
    MinPtBalance = cms.double( -1.0 ),
    MaxPtBalance = cms.double( 999999.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltL3TkMuons = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "" ),
    Fitter = cms.string( "FittingSmootherRK" ),
    Propagator = cms.string( "RungeKuttaTrackerPropagator" ),
    src = cms.InputTag( "hltL3TrackCandidateFromL2" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltL3TkMuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL3TkMuons" )
)
hltSingleMuNoIsoL3TkPreFilter = cms.EDFilter( "HLTMuonL3TkPreFilter",
    BeamSpotTag = cms.InputTag( "hltOfflineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3TkMuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuNoIsoL2PreFiltered" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 99999.0 ),
    MinPt = cms.double( 15.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
hltTriggerSummaryRAWprescaler = cms.EDFilter( "HLTPrescaler" )
hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
hltBoolFinal = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)

HLTBeginSequence = cms.Sequence( hlt2GetRaw + hltGtDigis + hltGctDigis + hltL1GtObjectMap + hltL1extraParticles + hltOfflineBeamSpot )
HLTEndSequence = cms.Sequence( hltBoolEnd )
HLTDoLocalHcalSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco + hltHoreco )
HLTRecoJetRegionalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalJetsFEDs + hltEcalRegionalJetsDigis + hltEcalRegionalJetsWeightUncalibRecHit + hltEcalRegionalJetsRecHitTmp + hltEcalRegionalJetsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForJets + hltIterativeCone5CaloJetsRegional + hltMCJetCorJetIcone5Regional )
HLTDoCaloSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalRestFEDs + hltEcalRegionalRestDigis + hltEcalRegionalRestWeightUncalibRecHit + hltEcalRegionalRestRecHitTmp + hltEcalRecHitAll + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForAll )
HLTDoJetRecoSequence = cms.Sequence( hltIterativeCone5CaloJets + hltMCJetCorJetIcone5 )
HLTDoHTRecoSequence = cms.Sequence( hltHtMet )
HLTRecoJetMETSequence = cms.Sequence( HLTDoCaloSequence + HLTDoJetRecoSequence + hltMet + HLTDoHTRecoSequence )
HLTDoRegionalEgammaEcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalEgammaFEDs + hltEcalRegionalEgammaDigis + hltEcalRegionalEgammaWeightUncalibRecHit + hltEcalRegionalEgammaRecHitTmp + hltEcalRegionalEgammaRecHit + hltEcalPreshowerRecHit )
HLTL1IsolatedEcalClustersSequence = cms.Sequence( hltIslandBasicClustersEndcapL1Isolated + hltIslandBasicClustersBarrelL1Isolated + hltHybridSuperClustersL1Isolated + hltIslandSuperClustersL1Isolated + hltCorrectedIslandEndcapSuperClustersL1Isolated + hltCorrectedIslandBarrelSuperClustersL1Isolated + hltCorrectedHybridSuperClustersL1Isolated + hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated )
HLTDoLocalHcalWithoutHOSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco )
HLTDoLocalPixelSequence = cms.Sequence( hltSiPixelDigis + hltSiPixelClusters + hltSiPixelRecHits )
HLTDoLocalStripSequence = cms.Sequence( hltSiStripRawToClustersFacility + hltSiStripClusters )
HLTPixelMatchElectronL1IsoSequence = cms.Sequence( hltL1IsoElectronPixelSeeds )
HLTPixelMatchElectronL1IsoTrackingSequence = cms.Sequence( hltCkfL1IsoTrackCandidates + hltCtfL1IsoWithMaterialTracks + hltPixelMatchElectronsL1Iso )
HLTL1IsoElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1IsoElectronsRegionalPixelSeedGenerator + hltL1IsoElectronsRegionalCkfTrackCandidates + hltL1IsoElectronsRegionalCTFFinalFitWithMaterial )
HLTL1NonIsolatedEcalClustersSequence = cms.Sequence( hltIslandBasicClustersEndcapL1NonIsolated + hltIslandBasicClustersBarrelL1NonIsolated + hltHybridSuperClustersL1NonIsolated + hltIslandSuperClustersL1NonIsolated + hltCorrectedIslandEndcapSuperClustersL1NonIsolated + hltCorrectedIslandBarrelSuperClustersL1NonIsolated + hltCorrectedHybridSuperClustersL1NonIsolated + hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated )
HLTPixelMatchElectronL1NonIsoSequence = cms.Sequence( hltL1NonIsoElectronPixelSeeds )
HLTPixelMatchElectronL1NonIsoTrackingSequence = cms.Sequence( hltCkfL1NonIsoTrackCandidates + hltCtfL1NonIsoWithMaterialTracks + hltPixelMatchElectronsL1NonIso )
HLTL1NonIsoElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1NonIsoElectronsRegionalPixelSeedGenerator + hltL1NonIsoElectronsRegionalCkfTrackCandidates + hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial )
HLTDoLocalTrackerSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence )
HLTL1IsoEgammaRegionalRecoTrackerSequence = cms.Sequence( hltL1IsoEgammaRegionalPixelSeedGenerator + hltL1IsoEgammaRegionalCkfTrackCandidates + hltL1IsoEgammaRegionalCTFFinalFitWithMaterial )
HLTL1NonIsoEgammaRegionalRecoTrackerSequence = cms.Sequence( hltL1NonIsoEgammaRegionalPixelSeedGenerator + hltL1NonIsoEgammaRegionalCkfTrackCandidates + hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial )
HLTPixelMatchElectronL1IsoLargeWindowSequence = cms.Sequence( hltL1IsoLargeWindowElectronPixelSeeds )
HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence = cms.Sequence( hltCkfL1IsoLargeWindowTrackCandidates + hltCtfL1IsoLargeWindowWithMaterialTracks + hltPixelMatchElectronsL1IsoLargeWindow )
HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator + hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates + hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial )
HLTPixelMatchElectronL1NonIsoLargeWindowSequence = cms.Sequence( hltL1NonIsoLargeWindowElectronPixelSeeds )
HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence = cms.Sequence( hltCkfL1NonIsoLargeWindowTrackCandidates + hltCtfL1NonIsoLargeWindowWithMaterialTracks + hltPixelMatchElectronsL1NonIsoLargeWindow )
HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator + hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates + hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial )
HLTL1muonrecoSequence = cms.Sequence( HLTBeginSequence )
HLTL2muonrecoNocandSequence = cms.Sequence( hltMuonDTDigis + hltDt1DRecHits + hltDt4DSegments + hltMuonCSCDigis + hltCsc2DRecHits + hltCscSegments + hltMuonRPCDigis + hltRpcRecHits + hltL2MuonSeeds + hltL2Muons )
HLTL2muonrecoSequence = cms.Sequence( HLTL2muonrecoNocandSequence + hltL2MuonCandidates )
HLTL2muonisorecoSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalMuonsFEDs + hltEcalRegionalMuonsDigis + hltEcalRegionalMuonsWeightUncalibRecHit + hltEcalRegionalMuonsRecHitTmp + hltEcalRegionalMuonsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForMuons + hltL2MuonIsolations )
HLTL3muonTkCandidateSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL3TrajectorySeed + hltL3TrackCandidateFromL2 )
HLTL3muonrecoNocandSequence = cms.Sequence( HLTL3muonTkCandidateSequence + hltL3Muons )
HLTL3muonrecoSequence = cms.Sequence( HLTL3muonrecoNocandSequence + hltL3MuonCandidates )
HLTL3muonisorecoSequence = cms.Sequence( hltPixelTracks + hltL3MuonIsolations )
HLTBCommonL2recoSequence = cms.Sequence( HLTDoCaloSequence + HLTDoJetRecoSequence + HLTDoHTRecoSequence )
HLTRecopixelvertexingSequence = cms.Sequence( hltPixelTracks + hltPixelVertices )
HLTBLifetimeL25recoSequence = cms.Sequence( hltBLifetimeHighestEtJets + hltBLifetimeL25Jets + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltBLifetimeL25Associator + hltBLifetimeL25TagInfos + hltBLifetimeL25BJetTags )
HLTBLifetimeL3recoSequence = cms.Sequence( hltBLifetimeL3Jets + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltBLifetimeRegionalPixelSeedGenerator + hltBLifetimeRegionalCkfTrackCandidates + hltBLifetimeRegionalCtfWithMaterialTracks + hltBLifetimeL3Associator + hltBLifetimeL3TagInfos + hltBLifetimeL3BJetTags )
HLTBSoftmuonL25recoSequence = cms.Sequence( hltBSoftmuonHighestEtJets + hltBSoftmuonL25Jets + HLTL2muonrecoNocandSequence + hltBSoftmuonL25TagInfos + hltBSoftmuonL25BJetTags )
HLTBSoftmuonL3recoSequence = cms.Sequence( HLTL3muonrecoNocandSequence + hltBSoftmuonL3TagInfos + hltBSoftmuonL3BJetTags + hltBSoftmuonL3BJetTagsByDR )
HLTL3displacedMumurecoSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTRecopixelvertexingSequence + hltMumuPixelSeedFromL2Candidate + hltCkfTrackCandidatesMumu + hltCtfWithMaterialTracksMumu + hltMuTracks )
HLTL1EplusJetSequence = cms.Sequence( HLTBeginSequence + hltL1seedEJet )
HLTEJetElectronSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSingleL1MatchFilter + hltL1IsoEJetSingleEEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoEJetSingleEHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoEJetSingleEPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoEJetSingleEEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoEJetSingleETrackIsolFilter + hltSingleElectronL1IsoPresc )
HLTPixelTrackingForMinBiasSequence = cms.Sequence( hltPixelTracksForMinBias )
HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltEgammaL1MatchFilterRegionalElectronTau + hltEgammaEtFilterElectronTau + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltEgammaHcalIsolFilterElectronTau + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltElectronPixelMatchFilterElectronTau + HLTPixelMatchElectronL1IsoTrackingSequence + hltElectronOneOEMinusOneOPFilterElectronTau + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau )
HLTCaloTausCreatorRegionalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRegionalTausFEDs + hltEcalRegionalTausDigis + hltEcalRegionalTausWeightUncalibRecHit + hltEcalRegionalTausRecHitTmp + hltEcalRegionalTausRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForTaus + hltCaloTowersTau1Regional + hltIcone5Tau1Regional + hltCaloTowersTau2Regional + hltIcone5Tau2Regional + hltCaloTowersTau3Regional + hltIcone5Tau3Regional + hltCaloTowersTau4Regional + hltIcone5Tau4Regional )
HLTL2TauJetsElectronTauSequnce = cms.Sequence( HLTCaloTausCreatorRegionalSequence + hltL2TauJetsProviderElectronTau )
HLTCaloTausCreatorSequence = cms.Sequence( HLTDoCaloSequence + hltCaloTowersTau1 + hltIcone5Tau1 + hltCaloTowersTau2 + hltIcone5Tau2 + hltCaloTowersTau3 + hltIcone5Tau3 + hltCaloTowersTau4 + hltIcone5Tau4 )
HLTPixelMatchStartUpElectronL1IsoTrackingSequence = cms.Sequence( hltCkfL1IsoStartUpTrackCandidates + hltCtfL1IsoStartUpWithMaterialTracks + hltPixelMatchStartUpElectronsL1Iso )
HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence = cms.Sequence( hltCkfL1NonIsoStartUpTrackCandidates + hltCtfL1NonIsoStartUpWithMaterialTracks + hltPixelMatchStartUpElectronsL1NonIso )
HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1IsoStartUpElectronsRegionalPixelSeedGenerator + hltL1IsoStartUpElectronsRegionalCkfTrackCandidates + hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial )
HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence( hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator + hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates + hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial )
HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence = cms.Sequence( HLTBeginSequence + hltSingleElectronEt10L1NonIsoHLTNonIsoPresc + hltL1seedRelaxedSingleEt8 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter )
HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltSingleElectronEt8L1NonIsoHLTnoIsoPresc + hltL1seedRelaxedSingleEt5 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional + hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter )
HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence = cms.Sequence( HLTBeginSequence + hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc + hltL1seedRelaxedDoubleEt5 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter )
HLTSinglePhotonEt10L1NonIsolatedSequence = cms.Sequence( HLTBeginSequence + hltSinglePhotonEt10L1NonIsoPresc + hltL1seedRelaxedSingleEt8 + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional + hltL1NonIsoSinglePhotonEt10EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSinglePhotonEt10EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonEt10HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSinglePhotonEt10TrackIsolFilter )
HLTL1SeedFilterSequence = cms.Sequence( hltL1sIsolTrack )
HLTL3PixelIsolFilterSequence = cms.Sequence( HLTDoLocalPixelSequence + hltPixelTracks + hltIsolPixelTrackProd + hltIsolPixelTrackFilter )
HLTIsoTrRegFEDSelection = cms.Sequence( hltSiStripRegFED + hltEcalRegFED + hltSubdetFED )
HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt5 + PreHLT2Photon10L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter )
HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt5 + PreHLT2Photon8L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter )
HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt8 + PreHLT1ElectronLWEt12L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter )
HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt10 + PreHLT1ElectronLWEt15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter )
HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon20L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt10 + PreHLT1Photon15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter )
HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon25L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter )
HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Electron18L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter )
HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt12 + PreHLT1Electron15L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter )
HLTSingleElectronEt12L1NonIsoHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt8 + PreHLT1ElectronLW12L1IHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional + hltL1NonIsoHLTIsoSingleElectronEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter )
HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1ElectronLWEt18L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter )
HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt12 + PreHLT1ElectronLWEt15L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter )
HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon40L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter )
HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon30L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter )
HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon45L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter )
HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon30L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter )
HLTSinglePhoton25L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon25L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter )
HLTSinglePhoton20L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon20L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt12 + PreHLT1Photon15L1RHIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional + hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter )
HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt5 + PreHLT2ElectronLWonlyPMEt8L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter )
HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt5 + PreHLT2ElectronLWonlyPMEt10L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter )
HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt10 + PreHLT2ElectronLWonlyPMEt12L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter )
HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt10 + PreHLT2Photon20L1RHNonIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter )
HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt12 + PreHLT1Electron15L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter + HLTPixelMatchStartUpElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter + HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + hltL1IsoStartUpElectronTrackIsol + hltL1NonIsoStartupElectronTrackIsol + hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter )
HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedSingleEt15 + PreHLT1Photon40L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter )
HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence( HLTBeginSequence + hltL1seedRelaxedDoubleEt10 + PreHLT2Photon20L1RHLooseIso + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter )
HLTBLifetimeL25recoSequenceRelaxed = cms.Sequence( hltBLifetimeHighestEtJets + hltBLifetimeL25JetsRelaxed + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltBLifetimeL25AssociatorRelaxed + hltBLifetimeL25TagInfosRelaxed + hltBLifetimeL25BJetTagsRelaxed )
HLTBLifetimeL3recoSequenceRelaxed = cms.Sequence( hltBLifetimeL3JetsRelaxed + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltBLifetimeRegionalPixelSeedGeneratorRelaxed + hltBLifetimeRegionalCkfTrackCandidatesRelaxed + hltBLifetimeRegionalCtfWithMaterialTracksRelaxed + hltBLifetimeL3AssociatorRelaxed + hltBLifetimeL3TagInfosRelaxed + hltBLifetimeL3BJetTagsRelaxed )
HLTL1EplusJet30Sequence = cms.Sequence( HLTBeginSequence + hltL1seedEJet30 )
HLTE3Jet30ElectronSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSingleEJet30L1MatchFilter + hltL1IsoEJetSingleEEt5Filter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoEJetSingleEEt5HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoEJetSingleEEt5PixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoEJetSingleEEt5EoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoEJetSingleEEt5TrackIsolFilter + hltSingleElectronL1IsoPresc )
HLTL3TkmuonrecoNocandSequence = cms.Sequence( HLTL3muonTkCandidateSequence + hltL3TkMuons )

HLTriggerFirstPath = cms.Path( HLTBeginSequence + hltBoolFirst + HLTEndSequence )
HLT2jet = cms.Path( HLTBeginSequence + hltL1s2jet + hltPre2jet + HLTRecoJetRegionalSequence + hlt2jet150 + HLTEndSequence )
HLT3jet = cms.Path( HLTBeginSequence + hltL1s3jet + hltPre3jet + HLTRecoJetRegionalSequence + hlt3jet85 + HLTEndSequence )
HLT4jet = cms.Path( HLTBeginSequence + hltL1s4jet + hltPre4jet + HLTRecoJetMETSequence + hlt4jet60 + HLTEndSequence )
HLT2jetAco = cms.Path( HLTBeginSequence + hltL1s2jetAco + hltPre2jetAco + HLTRecoJetRegionalSequence + hlt2jet125 + hlt2jetAco + HLTEndSequence )
HLT1jet1METAco = cms.Path( HLTBeginSequence + hltL1s1jet1METAco + hltPre1jet1METAco + HLTRecoJetMETSequence + hlt1MET60 + hlt1jet100 + hlt1jet1METAco + HLTEndSequence )
HLT1jet1MET = cms.Path( HLTBeginSequence + hltL1s1jet1MET + hltPre1jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt1jet180 + HLTEndSequence )
HLT2jet1MET = cms.Path( HLTBeginSequence + hltL1s2jet1MET + hltPre2jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt2jet125New + HLTEndSequence )
HLT3jet1MET = cms.Path( HLTBeginSequence + hltL1s3jet1MET + hltPre3jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt3jet60 + HLTEndSequence )
HLT4jet1MET = cms.Path( HLTBeginSequence + hltL1s4jet1MET + hltPre4jet1MET + HLTRecoJetMETSequence + hlt1MET60 + hlt4jet35 + HLTEndSequence )
HLT1MET1HT = cms.Path( HLTBeginSequence + hltL1s1MET1HT + hltPre1MET1HT + HLTRecoJetMETSequence + hlt1MET65 + hlt1HT350 + HLTEndSequence )
HLT2jetvbfMET = cms.Path( HLTBeginSequence + hltL1s2jetvbfMET + hltPre2jetvbfMET + HLTRecoJetMETSequence + hlt1MET60 + hlt2jetvbf + HLTEndSequence )
HLTS2jet1METNV = cms.Path( HLTBeginSequence + hltL1snvMET + hltPrenv + HLTRecoJetMETSequence + hlt1MET60 + hltnv + HLTEndSequence )
HLTS2jet1METAco = cms.Path( HLTBeginSequence + hltL1sPhi2MET + hltPrephi2met + HLTRecoJetMETSequence + hlt1MET60 + hltPhi2metAco + HLTEndSequence )
HLTSjet1MET1Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet1MET + hltPrephijet1met + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet1metAco + HLTEndSequence )
HLTSjet2MET1Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet2MET + hltPrephijet2met + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet2metAco + HLTEndSequence )
HLTS2jetMET1Aco = cms.Path( HLTBeginSequence + hltL1sPhiJet1Jet2 + hltPrephijet1jet2 + HLTRecoJetMETSequence + hlt1MET70 + hltPhiJet1Jet2Aco + HLTEndSequence )
HLTJetMETRapidityGap = cms.Path( HLTBeginSequence + hltL1RapGap + hltPrerapgap + HLTRecoJetMETSequence + hltRapGap + HLTEndSequence )
HLT1Electron = cms.Path( HLTBeginSequence + hltL1seedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSingleL1MatchFilter + hltL1IsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoSingleElectronTrackIsolFilter + hltSingleElectronL1IsoPresc + HLTEndSequence )
HLT1ElectronRelaxed = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleElectronL1MatchFilterRegional + hltL1NonIsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltL1NonIsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltL1NonIsoSingleElectronTrackIsolFilter + hltSingleElectronL1NonIsoPresc + HLTEndSequence )
HLT2Electron = cms.Path( HLTBeginSequence + hltL1seedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleElectronL1MatchFilterRegional + hltL1IsoDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleElectronTrackIsolFilter + hltDoubleElectronL1IsoPresc + HLTEndSequence )
HLT2ElectronRelaxed = cms.Path( HLTBeginSequence + hltL1seedRelaxedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoDoubleElectronL1MatchFilterRegional + hltL1NonIsoDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltL1NonIsoDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltL1NonIsoDoubleElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltL1NonIsoDoubleElectronTrackIsolFilter + hltDoubleElectronL1NonIsoPresc + HLTEndSequence )
HLT1Photon = cms.Path( HLTBeginSequence + hltL1seedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSinglePhotonL1MatchFilter + hltL1IsoSinglePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoSinglePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoSinglePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoSinglePhotonTrackIsolFilter + hltSinglePhotonL1IsoPresc + HLTEndSequence )
HLT1PhotonRelaxed = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonL1MatchFilterRegional + hltL1NonIsoSinglePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSinglePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSinglePhotonTrackIsolFilter + hltSinglePhotonL1NonIsoPresc + HLTEndSequence )
HLT2Photon = cms.Path( HLTBeginSequence + hltL1seedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoublePhotonL1MatchFilterRegional + hltL1IsoDoublePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoDoublePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoDoublePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoDoublePhotonTrackIsolFilter + hltL1IsoDoublePhotonDoubleEtFilter + hltDoublePhotonL1IsoPresc + HLTEndSequence )
HLT2PhotonRelaxed = cms.Path( HLTBeginSequence + hltL1seedRelaxedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoDoublePhotonL1MatchFilterRegional + hltL1NonIsoDoublePhotonEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoDoublePhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoDoublePhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoDoublePhotonTrackIsolFilter + hltL1NonIsoDoublePhotonDoubleEtFilter + hltDoublePhotonL1NonIsoPresc + HLTEndSequence )
HLT1EMHighEt = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleEMHighEtL1MatchFilterRegional + hltL1NonIsoSinglePhotonEMHighEtEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSingleEMHighEtEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1NonIsolatedElectronHcalIsol + hltL1IsolatedElectronHcalIsol + hltL1NonIsoSingleEMHighEtHOEFilter + hltHcalDoubleCone + hltL1NonIsoEMHcalDoubleCone + hltL1NonIsoSingleEMHighEtHcalDBCFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1NonIsoPhotonTrackIsol + hltL1NonIsoSingleEMHighEtTrackIsolFilter + hltSingleEMVHighEtL1NonIsoPresc + HLTEndSequence )
HLT1EMVeryHighEt = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional + hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter + hltSingleEMVHEL1NonIsoPresc + HLTEndSequence )
HLT2ElectronZCounter = cms.Path( HLTBeginSequence + hltL1seedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleElectronZeeL1MatchFilterRegional + hltL1IsoDoubleElectronZeeEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleElectronZeeHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleElectronZeePixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleElectronZeeEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleElectronZeeTrackIsolFilter + hltL1IsoDoubleElectronZeePMMassFilter + hltZeeCounterPresc + HLTEndSequence )
HLT2ElectronExclusive = cms.Path( HLTBeginSequence + hltL1seedExclusiveDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleExclElectronL1MatchFilterRegional + hltL1IsoDoubleExclElectronEtPhiFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoDoubleExclElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltL1IsoDoubleExclElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltL1IsoDoubleExclElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1IsoDoubleExclElectronTrackIsolFilter + hltDoubleExclElectronL1IsoPresc + HLTEndSequence )
HLT2PhotonExclusive = cms.Path( HLTBeginSequence + hltL1seedExclusiveDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoDoubleExclPhotonL1MatchFilterRegional + hltL1IsoDoubleExclPhotonEtPhiFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoDoubleExclPhotonEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoDoubleExclPhotonHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoDoubleExclPhotonTrackIsolFilter + hltDoubleExclPhotonL1IsoPresc + HLTEndSequence )
HLT1PhotonL1Isolated = cms.Path( HLTBeginSequence + hltL1seedSinglePrescaled + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoSinglePhotonPrescaledL1MatchFilter + hltL1IsoSinglePhotonPrescaledEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1IsoSinglePhotonPrescaledEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1IsoSinglePhotonPrescaledHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonTrackIsol + hltL1IsoSinglePhotonPrescaledTrackIsolFilter + hltSinglePhotonPrescaledL1IsoPresc + HLTEndSequence )
CandHLT1ElectronStartup = cms.Path( HLTBeginSequence + hltL1seedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoLargeWindowSingleL1MatchFilter + hltL1IsoLargeWindowSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoLargeWindowSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + hltL1IsoLargeWindowSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1IsoLargeWindowSingleElectronTrackIsolFilter + hltSingleElectronL1IsoLargeWindowPresc + HLTEndSequence )
CandHLT1ElectronRelaxedStartup = cms.Path( HLTBeginSequence + hltL1seedRelaxedSingle + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional + hltL1NonIsoLargeWindowSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + HLTPixelMatchElectronL1NonIsoLargeWindowSequence + hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter + hltSingleElectronL1NonIsoLargeWindowPresc + HLTEndSequence )
CandHLT2ElectronStartup = cms.Path( HLTBeginSequence + hltL1seedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional + hltL1IsoLargeWindowDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1IsoLargeWindowDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + hltL1IsoLargeWindowDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + hltL1IsoLargeWindowDoubleElectronEoverpFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1IsoLargeWindowDoubleElectronTrackIsolFilter + hltDoubleElectronL1IsoLargeWindowPresc + HLTEndSequence )
CandHLT2ElectronRelaxedStartup = cms.Path( HLTBeginSequence + hltL1seedRelaxedDouble + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional + hltL1NonIsoLargeWindowDoubleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoLargeWindowSequence + HLTPixelMatchElectronL1NonIsoLargeWindowSequence + hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence + HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence + hltL1NonIsoLargeWindowDoubleElectronEoverpFilter + HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence + HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence + hltL1IsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowElectronTrackIsol + hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter + hltDoubleElectronL1NonIsoLargeWindowPresc + HLTEndSequence )
HLT1MuonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso + hltSingleMuIsoLevel1Seed + hltSingleMuIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered + HLTEndSequence )
HLT1MuonNonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered + HLTEndSequence )
HLT2MuonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonIso + hltDiMuonIsoLevel1Seed + hltDiMuonIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonIsoL2PreFiltered + HLTL2muonisorecoSequence + hltDiMuonIsoL2IsoFiltered + HLTL3muonrecoSequence + hltDiMuonIsoL3PreFiltered + HLTL3muonisorecoSequence + hltDiMuonIsoL3IsoFiltered + HLTEndSequence )
HLT2MuonNonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIso + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFiltered + HLTEndSequence )
HLT2MuonJPsi = cms.Path( HLTL1muonrecoSequence + hltPrescaleJPsiMM + hltJpsiMMLevel1Seed + hltJpsiMML1Filtered + HLTL2muonrecoSequence + hltJpsiMML2Filtered + HLTL3muonrecoSequence + hltJpsiMML3Filtered + HLTEndSequence )
HLT2MuonUpsilon = cms.Path( HLTL1muonrecoSequence + hltPrescaleUpsilonMM + hltUpsilonMMLevel1Seed + hltUpsilonMML1Filtered + HLTL2muonrecoSequence + hltUpsilonMML2Filtered + HLTL3muonrecoSequence + hltUpsilonMML3Filtered + HLTEndSequence )
HLT2MuonZ = cms.Path( HLTL1muonrecoSequence + hltPrescaleZMM + hltZMMLevel1Seed + hltZMML1Filtered + HLTL2muonrecoSequence + hltZMML2Filtered + HLTL3muonrecoSequence + hltZMML3Filtered + HLTEndSequence )
HLTNMuonNonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleMultiMuonNoIso + hltMultiMuonNoIsoLevel1Seed + hltMultiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltMultiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltMultiMuonNoIsoL3PreFiltered + HLTEndSequence )
HLT2MuonSameSign = cms.Path( HLTL1muonrecoSequence + hltPrescaleSameSignMu + hltSameSignMuLevel1Seed + hltSameSignMuL1Filtered + HLTL2muonrecoSequence + hltSameSignMuL2PreFiltered + HLTL3muonrecoSequence + hltSameSignMuL3PreFiltered + HLTEndSequence )
HLT1MuonPrescalePt3 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuPrescale3 + hltSingleMuPrescale3Level1Seed + hltSingleMuPrescale3L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale3L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale3L3PreFiltered + HLTEndSequence )
HLT1MuonPrescalePt5 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuPrescale5 + hltSingleMuPrescale5Level1Seed + hltSingleMuPrescale5L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale5L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale5L3PreFiltered + HLTEndSequence )
HLT1MuonPrescalePt7x7 = cms.Path( HLTL1muonrecoSequence + hltPreSingleMuPrescale77 + hltSingleMuPrescale77Level1Seed + hltSingleMuPrescale77L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale77L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale77L3PreFiltered + HLTEndSequence )
HLT1MuonPrescalePt7x10 = cms.Path( HLTL1muonrecoSequence + hltPreSingleMuPrescale710 + hltSingleMuPrescale710Level1Seed + hltSingleMuPrescale710L1Filtered + HLTL2muonrecoSequence + hltSingleMuPrescale710L2PreFiltered + HLTL3muonrecoSequence + hltSingleMuPrescale710L3PreFiltered + HLTEndSequence )
HLT1MuonLevel1 = cms.Path( HLTL1muonrecoSequence + hltPrescaleMuLevel1Path + hltMuLevel1PathLevel1Seed + hltMuLevel1PathL1Filtered + HLTEndSequence )
CandHLT1MuonPrescaleVtx2cm = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIsoRelaxedVtx2cm + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm + HLTEndSequence )
CandHLT1MuonPrescaleVtx2mm = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIsoRelaxedVtx2mm + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm + HLTEndSequence )
CandHLT2MuonPrescaleVtx2cm = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIsoRelaxedVtx2cm + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm + HLTEndSequence )
CandHLT2MuonPrescaleVtx2mm = cms.Path( HLTL1muonrecoSequence + hltPrescaleDiMuonNoIsoRelaxedVtx2mm + hltDiMuonNoIsoLevel1Seed + hltDiMuonNoIsoL1Filtered + HLTL2muonrecoSequence + hltDiMuonNoIsoL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm + HLTEndSequence )
HLTB1Jet = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTB2Jet = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTB3Jet = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTB4Jet = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTBHT = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTB1JetMu = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon1jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon1jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonByDRL3filter + HLTEndSequence )
HLTB2JetMu = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + HLTEndSequence )
HLTB3JetMu = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + HLTEndSequence )
HLTB4JetMu = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + HLTEndSequence )
HLTBHTMu = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT + hltBSoftmuonHTL1seeds + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + HLTEndSequence )
HLTBJPsiMuMu = cms.Path( HLTBeginSequence + hltJpsitoMumuL1Seed + hltJpsitoMumuL1Filtered + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedJpsitoMumuFilter + HLTEndSequence )
HLTTauTo3Mu = cms.Path( HLTBeginSequence + hltMuMukL1Seed + hltMuMukL1Filtered + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedMuMukFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTRecopixelvertexingSequence + hltMumukPixelSeedFromL2Candidate + hltCkfTrackCandidatesMumuk + hltCtfWithMaterialTracksMumuk + hltMumukAllConeTracks + hltmmkFilter + HLTEndSequence )
HLTXElectronBJet = cms.Path( HLTBeginSequence + hltElectronBPrescale + hltElectronBL1Seed + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltElBElectronL1MatchFilter + hltElBElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltElBElectronHcalIsolFilter + HLTBCommonL2recoSequence + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1NonIsoSequence + HLTPixelMatchElectronL1IsoSequence + hltElBElectronPixelMatchFilter + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltElBElectronEoverpFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltElBElectronTrackIsolFilter + HLTEndSequence )
HLTXMuonBJet = cms.Path( HLTBeginSequence + hltMuBPrescale + hltMuBLevel1Seed + hltMuBLifetimeL1Filtered + HLTL2muonrecoSequence + hltMuBLifetimeIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuBLifetimeIsoL2IsoFiltered + HLTBCommonL2recoSequence + HLTBLifetimeL25recoSequence + hltBLifetimeL25filter + HLTL3muonrecoSequence + hltMuBLifetimeIsoL3PreFiltered + HLTL3muonisorecoSequence + hltMuBLifetimeIsoL3IsoFiltered + HLTBLifetimeL3recoSequence + hltBLifetimeL3filter + HLTEndSequence )
HLTXMuonBJetSoftMuon = cms.Path( HLTBeginSequence + hltMuBsoftMuPrescale + hltMuBLevel1Seed + hltMuBSoftL1Filtered + HLTL2muonrecoSequence + hltMuBSoftIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuBSoftIsoL2IsoFiltered + HLTBCommonL2recoSequence + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTL3muonrecoSequence + hltMuBSoftIsoL3PreFiltered + HLTL3muonisorecoSequence + hltMuBSoftIsoL3IsoFiltered + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filter + HLTEndSequence )
HLTXElectron1Jet = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej1jet40 + HLTEndSequence )
HLTXElectron2Jet = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej2jet80 + HLTEndSequence )
HLTXElectron3Jet = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej3jet60 + HLTEndSequence )
HLTXElectron4Jet = cms.Path( HLTL1EplusJetSequence + HLTEJetElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej4jet35 + HLTEndSequence )
HLTXMuonJets = cms.Path( HLTL1muonrecoSequence + hltMuJetsPrescale + hltMuJetsLevel1Seed + hltMuJetsL1Filtered + HLTL2muonrecoSequence + hltMuJetsL2PreFiltered + HLTL2muonisorecoSequence + hltMuJetsL2IsoFiltered + HLTL3muonrecoSequence + hltMuJetsL3PreFiltered + HLTL3muonisorecoSequence + hltMuJetsL3IsoFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuJetsHLT1jet40 + HLTEndSequence )
CandHLTXMuonNoL2IsoJets = cms.Path( HLTL1muonrecoSequence + hltMuNoL2IsoJetsPrescale + hltMuNoL2IsoJetsLevel1Seed + hltMuNoL2IsoJetsL1Filtered + HLTL2muonrecoSequence + hltMuNoL2IsoJetsL2PreFiltered + HLTL3muonrecoSequence + hltMuNoL2IsoJetsL3PreFiltered + HLTL3muonisorecoSequence + hltMuNoL2IsoJetsL3IsoFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoL2IsoJetsHLT1jet40 + HLTEndSequence )
CandHLTXMuonNoIsoJets = cms.Path( HLTL1muonrecoSequence + hltMuNoIsoJetsPrescale + hltMuNoIsoJetsLevel1Seed + hltMuNoIsoJetsL1Filtered + HLTL2muonrecoSequence + hltMuNoIsoJetsL2PreFiltered + HLTL3muonrecoSequence + hltMuNoIsoJetsL3PreFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoIsoJetsHLT1jet50 + HLTEndSequence )
HLTXElectronMuon = cms.Path( HLTBeginSequence + hltemuPrescale + hltEMuonLevel1Seed + hltEMuL1MuonFilter + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltemuL1IsoSingleL1MatchFilter + hltemuL1IsoSingleElectronEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltemuL1IsoSingleElectronHcalIsolFilter + HLTL2muonrecoSequence + hltEMuL2MuonPreFilter + HLTL2muonisorecoSequence + hltEMuL2MuonIsoFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + hltemuL1IsoSingleElectronPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + hltemuL1IsoSingleElectronEoverpFilter + HLTL3muonrecoSequence + hltEMuL3MuonPreFilter + HLTL3muonisorecoSequence + hltEMuL3MuonIsoFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltemuL1IsoSingleElectronTrackIsolFilter + HLTEndSequence )
HLTXElectronMuonRelaxed = cms.Path( HLTBeginSequence + hltemuNonIsoPrescale + hltemuNonIsoLevel1Seed + hltNonIsoEMuL1MuonFilter + HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltemuNonIsoL1MatchFilterRegional + hltemuNonIsoL1IsoEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltemuNonIsoL1HcalIsolFilter + HLTL2muonrecoSequence + hltNonIsoEMuL2MuonPreFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + HLTPixelMatchElectronL1IsoSequence + HLTPixelMatchElectronL1NonIsoSequence + hltemuNonIsoL1IsoPixelMatchFilter + HLTPixelMatchElectronL1IsoTrackingSequence + HLTPixelMatchElectronL1NonIsoTrackingSequence + hltemuNonIsoL1IsoEoverpFilter + HLTL3muonrecoSequence + hltNonIsoEMuL3MuonPreFilter + HLTL1IsoElectronsRegionalRecoTrackerSequence + HLTL1NonIsoElectronsRegionalRecoTrackerSequence + hltL1IsoElectronTrackIsol + hltL1NonIsoElectronTrackIsol + hltemuNonIsoL1IsoTrackIsolFilter + HLTEndSequence )
CandHLTBackwardBSC = cms.Path( HLTBeginSequence + hltLevel1seedHLTBackwardBSC + hltPrescaleHLTBackwardBSC + HLTEndSequence )
CandHLTForwardBSC = cms.Path( HLTBeginSequence + hltLevel1seedHLTForwardBSC + hltPrescaleHLTForwardBSC + HLTEndSequence )
CandHLTCSCBeamHalo = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHalo + hltPrescaleHLTCSCBeamHalo + HLTEndSequence )
CandHLTCSCBeamHaloOverlapRing1 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloOverlapRing1 + hltPrescaleHLTCSCBeamHaloOverlapRing1 + hltMuonCSCDigis + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing1 + HLTEndSequence )
CandHLTCSCBeamHaloOverlapRing2 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloOverlapRing2 + hltPrescaleHLTCSCBeamHaloOverlapRing2 + hltMuonCSCDigis + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing2 + HLTEndSequence )
CandHLTCSCBeamHaloRing2or3 = cms.Path( HLTBeginSequence + hltLevel1seedHLTCSCBeamHaloRing2or3 + hltPrescaleHLTCSCBeamHaloRing2or3 + hltMuonCSCDigis + hltCsc2DRecHits + hltFilter23HLTCSCBeamHaloRing2or3 + HLTEndSequence )
CandHLTTrackerCosmics = cms.Path( HLTBeginSequence + hltLevel1seedHLTTrackerCosmics + hltPrescaleHLTTrackerCosmics + HLTEndSequence )
HLTMinBiasPixel = cms.Path( HLTBeginSequence + hltPreMinBiasPixel + hltL1seedMinBiasPixel + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltMinBiasPixelFilter + HLTEndSequence )
CandHLTMinBiasForAlignment = cms.Path( HLTBeginSequence + hltPreMBForAlignment + hltL1seedMinBiasPixel + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltPixelMBForAlignment + HLTEndSequence )
HLTMinBias = cms.Path( HLTBeginSequence + hltl1sMin + hltpreMin + HLTEndSequence )
HLTZeroBias = cms.Path( HLTBeginSequence + hltl1sZero + hltpreZero + HLTEndSequence )
HLTriggerType = cms.Path( HLTBeginSequence + hltPrescaleTriggerType + hltFilterTriggerType + HLTEndSequence )
HLTEndpath1 = cms.EndPath( hltL1gtTrigReport + hltTrigReport )
HLTXElectronTau = cms.Path( HLTBeginSequence + hltPrescalerElectronTau + hltLevel1GTSeedElectronTau + HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence + HLTL2TauJetsElectronTauSequnce + hltL2ElectronTauIsolationProducer + hltL2ElectronTauIsolationSelector + hltFilterEcalIsolatedTauJetsElectronTau + HLTRecopixelvertexingSequence + hltJetTracksAssociatorAtVertexL25ElectronTau + hltConeIsolationL25ElectronTau + hltIsolatedTauJetsSelectorL25ElectronTau + hltFilterIsolatedTauJetsL25ElectronTau + HLTEndSequence )
HLTXMuonTau = cms.Path( HLTBeginSequence + hltPrescalerMuonTau + hltLevel1GTSeedMuonTau + hltMuonTauL1Filtered + HLTL2muonrecoSequence + hltMuonTauIsoL2PreFiltered + HLTL2muonisorecoSequence + hltMuonTauIsoL2IsoFiltered + HLTCaloTausCreatorRegionalSequence + hltL2TauJetsProviderMuonTau + hltL2MuonTauIsolationProducer + hltL2MuonTauIsolationSelector + hltFilterEcalIsolatedTauJetsMuonTau + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltJetsPixelTracksAssociatorMuonTau + hltPixelTrackConeIsolationMuonTau + hltPixelTrackIsolatedTauJetsSelectorMuonTau + hltFilterPixelTrackIsolatedTauJetsMuonTau + HLTDoLocalStripSequence + HLTL3muonrecoSequence + hltMuonTauIsoL3PreFiltered + HLTL3muonisorecoSequence + hltMuonTauIsoL3IsoFiltered + HLTEndSequence )
HLT1Tau1MET = cms.Path( HLTBeginSequence + hltSingleTauMETPrescaler + hltSingleTauMETL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauMET + hltL2SingleTauMETJets + hltL2SingleTauMETIsolationProducer + hltL2SingleTauMETIsolationSelector + hltFilterSingleTauMETEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTauMET + hltConeIsolationL25SingleTauMET + hltIsolatedL25SingleTauMET + hltFilterL25SingleTauMET + HLTDoLocalStripSequence + hltL3SingleTauMETPixelSeeds + hltCkfTrackCandidatesL3SingleTauMET + hltCtfWithMaterialTracksL3SingleTauMET + hltAssociatorL3SingleTauMET + hltConeIsolationL3SingleTauMET + hltIsolatedL3SingleTauMET + hltFilterL3SingleTauMET + HLTEndSequence )
HLT1Tau = cms.Path( HLTBeginSequence + hltSingleTauPrescaler + hltSingleTauL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTau + hltL2SingleTauJets + hltL2SingleTauIsolationProducer + hltL2SingleTauIsolationSelector + hltFilterSingleTauEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTau + hltConeIsolationL25SingleTau + hltIsolatedL25SingleTau + hltFilterL25SingleTau + HLTDoLocalStripSequence + hltL3SingleTauPixelSeeds + hltCkfTrackCandidatesL3SingleTau + hltCtfWithMaterialTracksL3SingleTau + hltAssociatorL3SingleTau + hltConeIsolationL3SingleTau + hltIsolatedL3SingleTau + hltFilterL3SingleTau + HLTEndSequence )
HLT1Electron10_L1R_NI = cms.Path( HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence + HLTEndSequence )
HLT1Electron8_L1R_NI = cms.Path( HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT2Electron5_L1R_NI = cms.Path( HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence + HLTEndSequence )
HLT1Photon10_L1R = cms.Path( HLTSinglePhotonEt10L1NonIsolatedSequence + HLTEndSequence )
AlCaIsoTrack = cms.Path( HLTBeginSequence + HLTL1SeedFilterSequence + hltPreIsolTrackNoEcalIso + HLTL3PixelIsolFilterSequence + HLTIsoTrRegFEDSelection + HLTEndSequence )
AlCaHcalPhiSym = cms.Path( HLTBeginSequence + hltL1sHcalPhiSym + hltHcalPhiSymPresc + HLTDoLocalHcalSequence + hltAlCaHcalPhiSymStream + HLTEndSequence )
AlCaEcalPhiSym = cms.Path( HLTBeginSequence + hltL1sEcalPhiSym + hltEcalPhiSymPresc + hltEcalDigis + hltEcalWeightUncalibRecHit + hltEcalRecHit + hltAlCaPhiSymStream + HLTEndSequence )
AlCaEcalPi0 = cms.Path( HLTBeginSequence + hltPrePi0Ecal + hltL1sEcalPi0 + HLTDoRegionalEgammaEcalSequence + hltAlCaPi0RegRecHits + HLTEndSequence )
HLT1MuonLevel2 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuLevel2 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuLevel2NoIsoL2PreFiltered + HLTEndSequence )
HLTBJPsiMuMuRelaxed = cms.Path( HLTBeginSequence + hltJpsitoMumuL1SeedRelaxed + hltJpsitoMumuL1FilteredRelaxed + HLTL2muonrecoSequence + HLTL3displacedMumurecoSequence + hltDisplacedJpsitoMumuFilterRelaxed + HLTEndSequence )
HLT2PhotonEt10_L1R_NI = cms.Path( HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT2PhotonEt8_L1R_NI = cms.Path( HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT1ElectronLWEt12_L1R_NI = cms.Path( HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT1ElectronLWEt15_L1R_NI = cms.Path( HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT1PhotonEt20_L1R_LI = cms.Path( HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence + HLTEndSequence )
HLT1PhotonEt15_L1R_NI = cms.Path( HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT1PhotonEt25_L1R_NI = cms.Path( HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT1ElectronEt18_L1R_NI = cms.Path( HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT1ElectronEt15_L1R_NI = cms.Path( HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT1ElectronEt12_L1R_HI = cms.Path( HLTSingleElectronEt12L1NonIsoHLTIsoSequence + HLTEndSequence )
HLT1ElectronLWEt18_L1R_LI = cms.Path( HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence + HLTEndSequence )
HLT1ElectronLWEt15_L1R_LI = cms.Path( HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence + HLTEndSequence )
HLT1PhotonEt40_L1R_NI = cms.Path( HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT1PhotonEt30_L1R_NI = cms.Path( HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLT1PhotonEt45_L1R_LI = cms.Path( HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence + HLTEndSequence )
HLT1PhotonEt30_L1R_LI = cms.Path( HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence + HLTEndSequence )
HLT1PhotonEt25_L1R_HI = cms.Path( HLTSinglePhoton25L1NonIsolatedHLTIsoSequence + HLTEndSequence )
HLT1PhotonEt20_L1R_HI = cms.Path( HLTSinglePhoton20L1NonIsolatedHLTIsoSequence + HLTEndSequence )
HLT1PhotonEt15_L1R_HI = cms.Path( HLTSinglePhoton15L1NonIsolatedHLTIsoSequence + HLTEndSequence )
HLT2ElectronLWonlyPMEt8_L1R_NI = cms.Path( HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT2ElectronLWonlyPMEt10_L1R_NI = cms.Path( HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT2ElectronLWonlyPMEt12_L1R_NI = cms.Path( HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence + HLTEndSequence )
HLT2PhotonEt20_L1R_NI = cms.Path( HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence + HLTEndSequence )
HLTMinBiasHcal = cms.Path( HLTBeginSequence + hltl1sMinHcal + hltpreMinHcal + HLTEndSequence )
HLTMinBiasEcal = cms.Path( HLTBeginSequence + hltl1sMinEcal + hltpreMinEcal + HLTEndSequence )
HLT1ElectronEt15_L1R_LI = cms.Path( HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence + HLTEndSequence )
HLT1PhotonEt40_L1R_LI = cms.Path( HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence + HLTEndSequence )
HLT2PhotonEt20_L1R_LI = cms.Path( HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence + HLTEndSequence )
HLT4jet30 = cms.Path( HLTBeginSequence + hltL1s4jet30 + hltPre4jet30 + HLTRecoJetMETSequence + hlt4jet30 + HLTEndSequence )
HLT1TauRelaxed = cms.Path( HLTBeginSequence + hltSingleTauPrescaler + hltSingleTauL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauRelaxed + hltL2SingleTauJets + hltL2SingleTauIsolationProducer + hltL2SingleTauIsolationSelectorRelaxed + hltFilterSingleTauEcalIsolationRelaxed + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTauRelaxed + hltConeIsolationL25SingleTauRelaxed + hltIsolatedL25SingleTauRelaxed + hltFilterL25SingleTauRelaxed + HLTDoLocalStripSequence + hltL3SingleTauPixelSeedsRelaxed + hltCkfTrackCandidatesL3SingleTauRelaxed + hltCtfWithMaterialTracksL3SingleTauRelaxed + hltAssociatorL3SingleTauRelaxed + hltConeIsolationL3SingleTauRelaxed + hltIsolatedL3SingleTauRelaxed + hltFilterL3SingleTauRelaxed + HLTEndSequence )
HLT1Tau1METRelaxed = cms.Path( HLTBeginSequence + hltSingleTauMETPrescaler + hltSingleTauMETL1SeedFilter + HLTCaloTausCreatorSequence + hltMet + hlt1METSingleTauMETRelaxed + hltL2SingleTauMETJets + hltL2SingleTauMETIsolationProducer + hltL2SingleTauMETIsolationSelectorRelaxed + hltFilterSingleTauMETEcalIsolationRelaxed + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25SingleTauMETRelaxed + hltConeIsolationL25SingleTauMETRelaxed + hltIsolatedL25SingleTauMETRelaxed + hltFilterL25SingleTauMETRelaxed + HLTDoLocalStripSequence + hltL3SingleTauMETPixelSeedsRelaxed + hltCkfTrackCandidatesL3SingleTauMETRelaxed + hltCtfWithMaterialTracksL3SingleTauMETRelaxed + hltAssociatorL3SingleTauMETRelaxed + hltConeIsolationL3SingleTauMETRelaxed + hltIsolatedL3SingleTauMETRelaxed + hltFilterL3SingleTauMETRelaxed + HLTEndSequence )
HLT2TauPixelRelaxed = cms.Path( HLTBeginSequence + hltDoubleTauPrescaler + hltDoubleTauL1SeedFilterRelaxed + HLTCaloTausCreatorRegionalSequence + hltL2DoubleTauJetsRelaxed + hltL2DoubleTauIsolationProducerRelaxed + hltL2DoubleTauIsolationSelectorRelaxed + hltFilterDoubleTauEcalIsolationRelaxed + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25PixelTauIsolatedRelaxed + hltConeIsolationL25PixelTauIsolatedRelaxed + hltIsolatedL25PixelTauRelaxed + hltFilterL25PixelTauRelaxed + HLTEndSequence )
HLT2TauPixel = cms.Path( HLTBeginSequence + hltDoubleTauPrescaler + hltDoubleTauL1SeedFilter + HLTCaloTausCreatorRegionalSequence + hltL2DoubleTauJets + hltL2DoubleTauIsolationProducer + hltL2DoubleTauIsolationSelector + hltFilterDoubleTauEcalIsolation + HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltAssociatorL25PixelTauIsolated + hltConeIsolationL25PixelTauIsolated + hltIsolatedL25PixelTau + hltFilterL25PixelTau + HLTEndSequence )
HLT1Level1jet15 = cms.Path( HLTBeginSequence + hltPre1Level1jet15 + hltL1s1Level1jet15 + HLTEndSequence )
HLT1jet30 = cms.Path( HLTBeginSequence + hltL1s1jet30 + hltPre1jet30 + HLTRecoJetMETSequence + hlt1jet30 + HLTEndSequence )
HLT1jet50 = cms.Path( HLTBeginSequence + hltL1s1jet50 + hltPre1jet50 + HLTRecoJetMETSequence + hlt1jet50 + HLTEndSequence )
HLT1jet80 = cms.Path( HLTBeginSequence + hltL1s1jet80 + hltPre1jet80 + HLTRecoJetRegionalSequence + hlt1jet80 + HLTEndSequence )
HLT1jet110 = cms.Path( HLTBeginSequence + hltL1s1jet110 + hltPre1jet110 + HLTRecoJetRegionalSequence + hlt1jet110 + HLTEndSequence )
HLT1jet250 = cms.Path( HLTBeginSequence + hltL1s1jet250 + hltPre1jet250 + HLTRecoJetRegionalSequence + hlt1jet250 + HLTEndSequence )
HLT1SumET = cms.Path( HLTBeginSequence + hltL1s1SumET + hltPre1MET1SumET + HLTRecoJetMETSequence + hlt1SumET120 + HLTEndSequence )
HLT1jet180 = cms.Path( HLTBeginSequence + hltL1s1jet180 + hltPre1jet180 + HLTRecoJetRegionalSequence + hlt1jet180regional + HLTEndSequence )
HLT1Level1MET20 = cms.Path( HLTBeginSequence + hltPreLevel1MET20 + hltL1sLevel1MET20 + HLTEndSequence )
HLT1MET25 = cms.Path( HLTBeginSequence + hltL1s1MET25 + hltPre1MET25 + HLTRecoJetMETSequence + hlt1MET25 + HLTEndSequence )
HLT1MET35 = cms.Path( HLTBeginSequence + hltL1s1MET35 + hltPre1MET35 + HLTRecoJetMETSequence + hlt1MET35 + HLTEndSequence )
HLT1MET50 = cms.Path( HLTBeginSequence + hltL1s1MET50 + hltPre1MET50 + HLTRecoJetMETSequence + hlt1MET50 + HLTEndSequence )
HLT1MET65 = cms.Path( HLTBeginSequence + hltL1s1MET65 + hltPre1MET65 + HLTRecoJetMETSequence + hlt1MET65 + HLTEndSequence )
HLT1MET75 = cms.Path( HLTBeginSequence + hltL1s1MET75 + hltPre1MET75 + HLTRecoJetMETSequence + hlt1MET75 + HLTEndSequence )
HLT2jetAve15 = cms.Path( HLTBeginSequence + hltL1sdijetave15 + hltPredijetave15 + HLTRecoJetMETSequence + hltdijetave15 + HLTEndSequence )
HLT2jetAve30 = cms.Path( HLTBeginSequence + hltL1sdijetave30 + hltPredijetave30 + HLTRecoJetMETSequence + hltdijetave30 + HLTEndSequence )
HLT2jetAve50 = cms.Path( HLTBeginSequence + hltL1sdijetave50 + hltPredijetave50 + HLTRecoJetMETSequence + hltdijetave50 + HLTEndSequence )
HLT2jetAve70 = cms.Path( HLTBeginSequence + hltL1sdijetave70 + hltPredijetave70 + HLTRecoJetMETSequence + hltdijetave70 + HLTEndSequence )
HLT2jetAve130 = cms.Path( HLTBeginSequence + hltL1sdijetave130 + hltPredijetave130 + HLTRecoJetMETSequence + hltdijetave130 + HLTEndSequence )
HLT2jetAve220 = cms.Path( HLTBeginSequence + hltL1sdijetave220 + hltPredijetave220 + HLTRecoJetMETSequence + hltdijetave220 + HLTEndSequence )
HLTB1Jet120 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet120 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter120 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB1Jet160 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime1jet160 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime1jetL2filter160 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB2Jet100 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet100 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter100 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB2Jet60 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime2jet60 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime2jetL2filter60 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB2JetMu100 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet100 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter100 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTB2JetMu60 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon2jet60 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon2jetL2filter60 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTB3Jet40 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet40 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter40 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB3Jet60 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime3jet60 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime3jetL2filter60 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB3JetMu40 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet40 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter40 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTB3JetMu60 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon3jet60 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon3jetL2filter60 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTB4Jet30 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet30 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter30 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB4Jet35 = cms.Path( HLTBeginSequence + hltPrescalerBLifetime4jet35 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetime4jetL2filter35 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTB4JetMu30 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet30 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter30 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTB4JetMu35 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuon4jet35 + hltBSoftmuonNjetL1seeds + HLTBCommonL2recoSequence + hltBSoftmuon4jetL2filter35 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTBHT320 = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT320 + hltBLifetimeL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter320 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTBHT420 = cms.Path( HLTBeginSequence + hltPrescalerBLifetimeHT420 + hltBLifetimeL1seeds + HLTBCommonL2recoSequence + hltBLifetimeHTL2filter420 + HLTBLifetimeL25recoSequenceRelaxed + hltBLifetimeL25filterRelaxed + HLTBLifetimeL3recoSequenceRelaxed + hltBLifetimeL3filterRelaxed + HLTEndSequence )
HLTBHTMu250 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT250 + hltBSoftmuonHTL1seedsLowEnergy + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter250 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTBHTMu330 = cms.Path( HLTBeginSequence + hltPrescalerBSoftmuonHT330 + hltBSoftmuonHTL1seeds + HLTBCommonL2recoSequence + hltBSoftmuonHTL2filter330 + HLTBSoftmuonL25recoSequence + hltBSoftmuonL25filter + HLTBSoftmuonL3recoSequence + hltBSoftmuonL3filterRelaxed + HLTEndSequence )
HLTXElectron3Jet30 = cms.Path( HLTL1EplusJet30Sequence + HLTE3Jet30ElectronSequence + HLTDoCaloSequence + HLTDoJetRecoSequence + hltej3jet30 + HLTEndSequence )
HLTXMuonNoIso3Jets30 = cms.Path( HLTL1muonrecoSequence + hltMuNoIsoJetsPrescale + hltMuNoIsoJets30Level1Seed + hltMuNoIsoJetsMinPt4L1Filtered + HLTL2muonrecoSequence + hltMuNoIsoJetsMinPt4L2PreFiltered + HLTL3muonrecoSequence + hltMuNoIsoJetsMinPtL3PreFiltered + HLTDoCaloSequence + HLTDoJetRecoSequence + hltMuNoIsoHLTJets3jet30 + HLTEndSequence )
HLT1MuonL1Open = cms.Path( HLTL1muonrecoSequence + hltPrescaleMuLevel1Open + hltMuLevel1PathLevel1OpenSeed + hltMuLevel1PathL1OpenFiltered + HLTEndSequence )
HLT1MuonNonIso9 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso9 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered7 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered9 + HLTEndSequence )
HLT1MuonNonIso11 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso11 + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered9 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered11 + HLTEndSequence )
HLT1MuonNonIso13 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso13 + hltSingleMuNoIsoLevel1Seed10 + hltSingleMuNoIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered11 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered13 + HLTEndSequence )
HLT1MuonNonIso15 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso15 + hltSingleMuNoIsoLevel1Seed10 + hltSingleMuNoIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered12 + HLTL3muonrecoSequence + hltSingleMuNoIsoL3PreFiltered15 + HLTEndSequence )
HLT1MuonIso9 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso9 + hltSingleMuIsoLevel1Seed + hltSingleMuIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered7 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered7 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered9 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered9 + HLTEndSequence )
HLT1MuonIso13 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso13 + hltSingleMuIsoLevel1Seed10 + hltSingleMuIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered11 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered11 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered13 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered13 + HLTEndSequence )
HLT1MuonIso15 = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuIso15 + hltSingleMuIsoLevel1Seed10 + hltSingleMuIsoL1Filtered10 + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered12 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered12 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered15 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered15 + HLTEndSequence )
CandHLT2MuonPsi2S = cms.Path( HLTBeginSequence + hltPrescalePsi2SMM + hltJpsiMMLevel1Seed + hltJpsiMML1Filtered + HLTL2muonrecoSequence + hltPsi2SMML2Filtered + HLTL3muonrecoSequence + hltPsi2SMML3Filtered + HLTEndSequence )
CandHLT1MuonTrackerNonIso = cms.Path( HLTL1muonrecoSequence + hltPrescaleSingleMuNoIso + hltSingleMuNoIsoLevel1Seed + hltSingleMuNoIsoL1Filtered + HLTL2muonrecoSequence + hltSingleMuNoIsoL2PreFiltered + HLTL3TkmuonrecoNocandSequence + hltL3TkMuonCandidates + hltSingleMuNoIsoL3TkPreFilter + HLTEndSequence )
HLTriggerFinalPath = cms.Path( hltTriggerSummaryAOD + hltTriggerSummaryRAWprescaler + hltTriggerSummaryRAW + hltBoolFinal )




