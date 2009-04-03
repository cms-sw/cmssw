# /dev/CMSSW_3_1_0/pre2/8E29_V258/V2 (CMSSW_3_1_X_2009-04-02-0600_HLT2)
# Begin replace statements specific to the FastSim HLT
# For all HLTLevel1GTSeed objects, make the following replacements:
#   - L1GtReadoutRecordTag changed from hltGtDigis to gtDigis
#   - L1CollectionsTag changed from l1extraParticles to l1extraParticles
#   - L1MuonCollectionTag changed from l1extraParticles to l1ParamMuons
# For hltL2MuonSeeds: InputObjects and GMTReadoutCollection set to l1ParamMuons
# All other occurances of l1extraParticles recast as l1extraParticles
# L1GtObjectMapTag: hltL1GtObjectMap recast as gtDigis
# L1GtReadoutRecordTag: hltGtDigis recast as gtDigis
# hltMuon[CSC/DT/RPC]Digis changed to muon[CSC/DT/RPC]Digis
# Replace offlineBeamSpot with offlineBeamSpot
# AlCaIsoTrack needs HLTpixelTracking instead of pixelTracks
# Some HLT modules were recast as FastSim sequences: 
#   - hltL3TrackCandidateFromL2, see FastSimulation/HighLevelTrigger/data/Muon/HLTFastRecoForMuon.cff
#   - hltCkfTrackCandidatesL3SingleTau[MET][Relaxed], see FastSimulation/HighLevelTrigger/data/btau/HLTFastRecoForTau.cff
#   - hltCkfTrackCandidatesMumu, see FastSimulation/HighLevelTrigger/data/btau/L3ForDisplacedMumuTrigger.cff
#   - hltCkfTrackCandidatesMumuk, see FastSimulation/HighLevelTrigger/data/btau/L3ForMuMuk.cff
#   - hltBLifetimeRegionalCkfTrackCandidates[Relaxed], see FastSimulation/HighLevelTrigger/data/btau/lifetimeRegionalTracking.cff
# See FastSimulation/Configuration/test/getFastSimHLTcff.py for other documentation
# (L1Menu2007 only) Replace L1_QuadJet30 with L1_QuadJet40
# (Temporary) Remove PSet begin and end from block
# End replace statements specific to the FastSim HLT
# Additional import to make this file self contained
from FastSimulation.HighLevelTrigger.HLTSetup_cff import *

import FWCore.ParameterSet.Config as cms


HLTConfigVersion = cms.PSet(
  tableName = cms.string('/dev/CMSSW_3_1_0/pre2/8E29_V258/V2')
)

essourceSev = cms.ESSource( "EmptyESSource",
  recordName = cms.string( "HcalSeverityLevelComputerRcd" ),
  iovIsRunNotTime = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  firstValid = cms.vuint32( 1 )
)
L2RelativeCorrectionService = cms.ESSource( "L2RelativeCorrectionService",
  appendToDataLabel = cms.string( "" ),
  tagName = cms.string( "Summer08_L2Relative_IC5Calo" ),
  label = cms.string( "L2RelativeJetCorrector" )
)
L3AbsoluteCorrectionService = cms.ESSource( "L3AbsoluteCorrectionService",
  appendToDataLabel = cms.string( "" ),
  tagName = cms.string( "Summer08_L3Absolute_IC5Calo" ),
  label = cms.string( "L3AbsoluteJetCorrector" )
)
MCJetCorrectorIcone5 = cms.ESSource( "JetCorrectionServiceChain",
  label = cms.string( "MCJetCorrectorIcone5" ),
  appendToDataLabel = cms.string( "" ),
  correctors = cms.vstring( 'L2RelativeJetCorrector',
    'L3AbsoluteJetCorrector' )
)
SiStripQualityFakeESSource = cms.ESSource( "SiStripQualityFakeESSource" )

hcalRecAlgos = cms.ESProducer( "HcalRecAlgoESProducer",
  RecoveredRecHitBits = cms.vstring( '' ),
  appendToDataLabel = cms.string( "" ),
  DropChannelStatusBits = cms.vstring( '' ),
  SeverityLevels = cms.VPSet( 
    cms.PSet(  Level = cms.int32( 0 ),
      RecHitFlags = cms.vstring( '' ),
      ChannelStatus = cms.vstring( '' )
    )
  )
)
AnyDirectionAnalyticalPropagator = cms.ESProducer( "AnalyticalPropagatorESProducer",
  ComponentName = cms.string( "AnyDirectionAnalyticalPropagator" ),
  PropagationDirection = cms.string( "anyDirection" ),
  MaxDPhi = cms.double( 1.6 ),
  appendToDataLabel = cms.string( "" )
)
Chi2EstimatorForL2Refit = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
  ComponentName = cms.string( "Chi2EstimatorForL2Refit" ),
  MaxChi2 = cms.double( 1000.0 ),
  nSigma = cms.double( 3.0 ),
  appendToDataLabel = cms.string( "" )
)
EcalRegionCablingESProducer = cms.ESProducer( "EcalRegionCablingESProducer",
  appendToDataLabel = cms.string( "" )
)
EcalUnpackerWorkerESProducer = cms.ESProducer( "EcalUnpackerWorkerESProducer",
  appendToDataLabel = cms.string( "" ),
  DCCDataUnpacker = cms.PSet( 
    tccUnpacking = cms.bool( True ),
    orderedDCCIdList = cms.vint32( 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54 ),
    srpUnpacking = cms.bool( False ),
    syncCheck = cms.bool( False ),
    headerUnpacking = cms.bool( False ),
    orderedFedList = cms.vint32( 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654 ),
    feUnpacking = cms.bool( True ),
    feIdCheck = cms.bool( True ),
    memUnpacking = cms.bool( False )
  ),
  ElectronicsMapper = cms.PSet( 
    numbXtalTSamples = cms.uint32( 10 ),
    numbTriggerTSamples = cms.uint32( 1 )
  ),
  UncalibRHAlgo = cms.PSet(  Type = cms.string( "EcalUncalibRecHitWorkerWeights" ) ),
  CalibRHAlgo = cms.PSet( 
    Type = cms.string( "EcalRecHitWorkerSimple" ),
    ChannelStatusToBeExcluded = cms.vint32(  )
  )
)
KFFitterSmootherForL2Muon = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "KFFitterSmootherForL2Muon" ),
  Fitter = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Smoother = cms.string( "KFTrajectorySmootherForL2Muon" ),
  EstimateCut = cms.double( -1.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
KFTrajectoryFitterForL2Muon = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "KFTrajectoryFitterForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorAny" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
KFTrajectorySmootherForL2Muon = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "KFTrajectorySmootherForL2Muon" ),
  Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2EstimatorForL2Refit" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
ParametrizedMagneticFieldProducer = cms.ESProducer( "ParametrizedMagneticFieldProducer",
  label = cms.untracked.string( "parametrizedField" ),
  version = cms.string( "OAE_1103l_071212" ),
  appendToDataLabel = cms.string( "" ),
  parameters = cms.PSet(  BValue = cms.string( "3_8T" ) )
)
PixelCPEGenericESProducer = cms.ESProducer( "PixelCPEGenericESProducer",
  ComponentName = cms.string( "PixelCPEGeneric" ),
  eff_charge_cut_lowX = cms.double( 0.0 ),
  eff_charge_cut_lowY = cms.double( 0.0 ),
  eff_charge_cut_highX = cms.double( 1.0 ),
  eff_charge_cut_highY = cms.double( 1.0 ),
  size_cutX = cms.double( 3.0 ),
  size_cutY = cms.double( 3.0 ),
  EdgeClusterErrorX = cms.double( 50.0 ),
  EdgeClusterErrorY = cms.double( 85.0 ),
  inflate_errors = cms.bool( False ),
  inflate_all_errors_no_trk_angle = cms.bool( False ),
  UseErrorsFromTemplates = cms.bool( True ),
  TruncatePixelCharge = cms.bool( True ),
  IrradiationBiasCorrection = cms.bool( False ),
  DoCosmics = cms.bool( False ),
  LoadTemplatesFromDB = cms.bool( True ),
  appendToDataLabel = cms.string( "" ),
  TanLorentzAnglePerTesla = cms.double( 0.106 ),
  PixelErrorParametrization = cms.string( "NOTcmsim" ),
  Alpha2Order = cms.bool( True ),
  ClusterProbComputationFlag = cms.int32( 0 )
)
SiStripRegionConnectivity = cms.ESProducer( "SiStripRegionConnectivity",
  EtaDivisions = cms.untracked.uint32( 20 ),
  PhiDivisions = cms.untracked.uint32( 20 ),
  EtaMax = cms.untracked.double( 2.5 )
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
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
bJetRegionalTrajectoryFilter = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "bJetRegionalTrajectoryFilter" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    chargeSignificance = cms.double( -1.0 ),
    minPt = cms.double( 1.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 8 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
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
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
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
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltCkfTrajectoryFilterMumu = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumu" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    chargeSignificance = cms.double( -1.0 ),
    minPt = cms.double( 3.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
  )
)
hltCkfTrajectoryFilterMumuk = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "hltCkfTrajectoryFilterMumuk" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    chargeSignificance = cms.double( -1.0 ),
    minPt = cms.double( 3.0 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 5 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
  )
)
hltKFFitter = cms.ESProducer( "KFTrajectoryFitterESProducer",
  ComponentName = cms.string( "hltKFFitter" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
hltKFFittingSmoother = cms.ESProducer( "KFFittingSmootherESProducer",
  ComponentName = cms.string( "hltKFFittingSmoother" ),
  Fitter = cms.string( "hltKFFitter" ),
  Smoother = cms.string( "hltKFSmoother" ),
  EstimateCut = cms.double( -1.0 ),
  MinNumberOfHits = cms.int32( 5 ),
  RejectTracks = cms.bool( True ),
  BreakTrajWith2ConsecutiveMissing = cms.bool( False ),
  NoInvalidHitsBeginEnd = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
hltKFSmoother = cms.ESProducer( "KFTrajectorySmootherESProducer",
  ComponentName = cms.string( "hltKFSmoother" ),
  Propagator = cms.string( "PropagatorWithMaterial" ),
  Updator = cms.string( "KFUpdator" ),
  Estimator = cms.string( "Chi2" ),
  errorRescaling = cms.double( 100.0 ),
  minHits = cms.int32( 3 ),
  appendToDataLabel = cms.string( "" )
)
mixedlayerpairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "MixedLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg',
    'FPix2_pos+TEC1_pos',
    'FPix2_pos+TEC2_pos',
    'TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'FPix2_neg+TEC1_neg',
    'FPix2_neg+TEC2_neg',
    'TEC1_neg+TEC2_neg',
    'TEC2_neg+TEC3_neg' ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  TEC = cms.PSet( 
    useRingSlector = cms.untracked.bool( True ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    minRing = cms.int32( 1 ),
    maxRing = cms.int32( 1 )
  )
)
pixellayerpairs = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "PixelLayerPairs" ),
  layerList = cms.vstring( 'BPix1+BPix2',
    'BPix1+BPix3',
    'BPix2+BPix3',
    'BPix1+FPix1_pos',
    'BPix1+FPix1_neg',
    'BPix1+FPix2_pos',
    'BPix1+FPix2_neg',
    'BPix2+FPix1_pos',
    'BPix2+FPix1_neg',
    'BPix2+FPix2_pos',
    'BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos',
    'FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  TEC = cms.PSet(  )
)
pixellayertriplets = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "PixelLayerTriplets" ),
  layerList = cms.vstring( 'BPix1+BPix2+BPix3',
    'BPix1+BPix2+FPix1_pos',
    'BPix1+BPix2+FPix1_neg',
    'BPix1+FPix1_pos+FPix2_pos',
    'BPix1+FPix1_neg+FPix2_neg' ),
  BPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0060 ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  FPix = cms.PSet( 
    hitErrorRZ = cms.double( 0.0036 ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    useErrorsFromParam = cms.untracked.bool( True )
  ),
  TEC = cms.PSet(  )
)
sistripconn = cms.ESProducer( "SiStripConnectivity" )
softLeptonByDistance = cms.ESProducer( "LeptonTaggerByDistanceESProducer",
  appendToDataLabel = cms.string( "" ),
  distance = cms.double( 0.5 )
)
softLeptonByPt = cms.ESProducer( "LeptonTaggerByPtESProducer",
  appendToDataLabel = cms.string( "" ),
  ipSign = cms.string( "any" )
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
  alwaysUseInvalidHits = cms.bool( False ),
  appendToDataLabel = cms.string( "" )
)
trajFilterL3 = cms.ESProducer( "TrajectoryFilterESProducer",
  ComponentName = cms.string( "trajFilterL3" ),
  appendToDataLabel = cms.string( "" ),
  filterPset = cms.PSet( 
    chargeSignificance = cms.double( -1.0 ),
    minPt = cms.double( 0.9 ),
    minHitsMinPt = cms.int32( 3 ),
    ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
    maxLostHits = cms.int32( 1 ),
    maxNumberOfHits = cms.int32( 7 ),
    maxConsecLostHits = cms.int32( 1 ),
    nSigmaMinPt = cms.double( 5.0 ),
    minimumNumberOfHits = cms.int32( 5 )
  )
)

hltGetRaw = cms.EDAnalyzer( "HLTGetRaw",
    RawDataCollection = cms.InputTag( "rawDataCollector" )
)
hltPreFirstPath = cms.EDFilter( "HLTPrescaler" )
hltBoolFirstPath = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltL1sL1Jet6U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Jet6U = cms.EDFilter( "HLTPrescaler" )
hltL1sJet15U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreJet15U = cms.EDFilter( "HLTPrescaler" )
hltTowerMakerForAll = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold0 = cms.double( 1.1 ),
    HOThresholdPlus1 = cms.double( 1.1 ),
    HOThresholdMinus1 = cms.double( 1.1 ),
    HOThresholdPlus2 = cms.double( 1.1 ),
    HOThresholdMinus2 = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    HcalAcceptSeverityLevel = cms.uint32( 999 ),
    EcalAcceptSeverityLevel = cms.uint32( 1 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
    ecalInputs = cms.VInputTag( 'hltEcalRecHitAll:EcalRecHitsEB','hltEcalRecHitAll:EcalRecHitsEE' )
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
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltMCJetCorJetIcone5 = cms.EDProducer( "CaloJetCorrectionProducer",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    verbose = cms.untracked.bool( False ),
    alias = cms.untracked.string( "MCJetCorJetIcone5" ),
    correctors = cms.vstring( 'MCJetCorrectorIcone5' )
)
hltMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltTowerMakerForAll" ),
    InputType = cms.string( "CandidateCollection" ),
    METType = cms.string( "CaloMET" ),
    alias = cms.string( "RawCaloMET" ),
    globalThreshold = cms.double( 0.5 ),
    noHF = cms.bool( False ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    HO_PhiResPar = cms.vdouble( 0.02511 )
)
hltHtMet = cms.EDProducer( "METProducer",
    src = cms.InputTag( "hltMCJetCorJetIcone5" ),
    InputType = cms.string( "CaloJetCollection" ),
    METType = cms.string( "MET" ),
    alias = cms.string( "HTMET" ),
    globalThreshold = cms.double( 5.0 ),
    noHF = cms.bool( False ),
    HO_EtResPar = cms.vdouble( 0.0, 1.3, 0.0050 ),
    HF_EtResPar = cms.vdouble( 0.0, 1.82, 0.09 ),
    HB_PhiResPar = cms.vdouble( 0.02511 ),
    HE_PhiResPar = cms.vdouble( 0.02511 ),
    EE_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    EB_PhiResPar = cms.vdouble( 0.00502 ),
    EE_PhiResPar = cms.vdouble( 0.02511 ),
    HB_EtResPar = cms.vdouble( 0.0, 1.22, 0.05 ),
    EB_EtResPar = cms.vdouble( 0.2, 0.03, 0.0050 ),
    HF_PhiResPar = cms.vdouble( 0.05022 ),
    HE_EtResPar = cms.vdouble( 0.0, 1.3, 0.05 ),
    HO_PhiResPar = cms.vdouble( 0.02511 )
)
hlt1jet15U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sJet30U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet20U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreJet30U = cms.EDFilter( "HLTPrescaler" )
hlt1jet30U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sJet50U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreJet50U = cms.EDFilter( "HLTPrescaler" )
hlt1jet50U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sFwdJet20U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_IsoEG10_Jet6U_ForJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreFwdJet20U = cms.EDFilter( "HLTPrescaler" )
hltRapGap20U = cms.EDFilter( "HLTRapGapFilter",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEta = cms.double( 3.0 ),
    maxEta = cms.double( 5.0 ),
    caloThresh = cms.double( 20.0 )
)
hltL1sDiJetAve15U8E29 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreDiJetAve15U8E29 = cms.EDFilter( "HLTPrescaler" )
hltDiJetAve15U8E29 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 15.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sDiJetAve30U8E29 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet20U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreDiJetAve30U8E29 = cms.EDFilter( "HLTPrescaler" )
hltDiJetAve30U8E29 = cms.EDFilter( "HLTDiJetAveFilter",
    inputJetTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    minEtAve = cms.double( 30.0 ),
    minEtJet3 = cms.double( 3000.0 ),
    minDphi = cms.double( 0.0 )
)
hltL1sQuadJet15U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_QuadJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreQuadJet15U = cms.EDFilter( "HLTPrescaler" )
hlt4jet15U = cms.EDFilter( "HLT1CaloJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 30.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 4 )
)
hltL1sL1MET20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1MET20 = cms.EDFilter( "HLTPrescaler" )
hltL1sMET35 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreMET35 = cms.EDFilter( "HLTPrescaler" )
hlt1MET35 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 35.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1sMET100 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_ETM80" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreMET100 = cms.EDFilter( "HLTPrescaler" )
hlt1MET100 = cms.EDFilter( "HLT1CaloMET",
    inputTag = cms.InputTag( "hltMet" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 100.0 ),
    MaxEta = cms.double( -1.0 ),
    MinN = cms.int32( 1 )
)
hltL1sL1MuOpen = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1MuOpen = cms.EDFilter( "HLTPrescaler" )
hltL1MuOpenL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1MuOpen" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    SelectQualities = cms.vint32(  )
)
hltL1sL1Mu = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7 OR L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Mu = cms.EDFilter( "HLTPrescaler" )
hltL1MuL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1Mu" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    SelectQualities = cms.vint32(  )
)
hltL1sL1SingleMu20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Mu20 = cms.EDFilter( "HLTPrescaler" )
hltL1Mu20L1Filtered20 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu20" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 20.0 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    SelectQualities = cms.vint32(  )
)
hltL1sL1SingleMu7 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu7" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL2Mu9 = cms.EDFilter( "HLTPrescaler" )
hltL1SingleMu7L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu7" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SelectQualities = cms.vint32(  )
)
hltDt1DRecHits = cms.EDProducer( "DTRecHitProducer",
    debug = cms.untracked.bool( False ),
    dtDigiLabel = cms.InputTag( "simMuonDTDigis" ),
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
        segmCleanerMode = cms.int32( 1 ),
        performT0SegCorrection = cms.bool( False ),
        performT0_vdriftSegCorrection = cms.bool( False ),
        hit_afterT0_resolution = cms.double( 0.03 ),
        T0SegCorrectionDebug = cms.untracked.bool( False )
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
      segmCleanerMode = cms.int32( 1 ),
      performT0SegCorrection = cms.bool( False ),
      performT0_vdriftSegCorrection = cms.bool( False ),
      hit_afterT0_resolution = cms.double( 0.03 ),
      T0SegCorrectionDebug = cms.untracked.bool( False )
    )
)
hltCsc2DRecHits = cms.EDProducer( "CSCRecHitDProducer",
    CSCUseCalibrations = cms.untracked.bool( True ),
    stripDigiTag = cms.InputTag( 'simMuonCSCDigis','MuonCSCStripDigi' ),
    wireDigiTag = cms.InputTag( 'simMuonCSCDigis','MuonCSCWireDigi' ),
    CSCstripWireDeltaTime = cms.untracked.int32( 8 ),
    CSCUseStaticPedestals = cms.bool( False ),
    CSCNoOfTimeBinsForDynamicPedestal = cms.int32( 2 ),
    CSCStripPeakThreshold = cms.untracked.double( 10.0 ),
    CSCStripClusterChargeCut = cms.untracked.double( 25.0 ),
    CSCWireClusterDeltaT = cms.untracked.int32( 1 ),
    CSCStripxtalksOffset = cms.untracked.double( 0.03 ),
    NoiseLevel_ME1a = cms.untracked.double( 7.0 ),
    XTasymmetry_ME1a = cms.untracked.double( 0.0 ),
    ConstSyst_ME1a = cms.untracked.double( 0.022 ),
    NoiseLevel_ME1b = cms.untracked.double( 8.0 ),
    XTasymmetry_ME1b = cms.untracked.double( 0.0 ),
    ConstSyst_ME1b = cms.untracked.double( 0.0070 ),
    NoiseLevel_ME12 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME12 = cms.untracked.double( 0.0 ),
    ConstSyst_ME12 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME13 = cms.untracked.double( 8.0 ),
    XTasymmetry_ME13 = cms.untracked.double( 0.0 ),
    ConstSyst_ME13 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME21 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME21 = cms.untracked.double( 0.0 ),
    ConstSyst_ME21 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME22 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME22 = cms.untracked.double( 0.0 ),
    ConstSyst_ME22 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME31 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME31 = cms.untracked.double( 0.0 ),
    ConstSyst_ME31 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME32 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME32 = cms.untracked.double( 0.0 ),
    ConstSyst_ME32 = cms.untracked.double( 0.0 ),
    NoiseLevel_ME41 = cms.untracked.double( 9.0 ),
    XTasymmetry_ME41 = cms.untracked.double( 0.0 ),
    ConstSyst_ME41 = cms.untracked.double( 0.0 ),
    readBadChannels = cms.bool( False ),
    readBadChambers = cms.bool( False )
)
hltCscSegments = cms.EDProducer( "CSCSegmentProducer",
    inputObjects = cms.InputTag( "hltCsc2DRecHits" ),
    algo_type = cms.int32( 1 ),
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
        algo_psets = cms.VPSet( 
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            maxRecHitsInCluster = cms.untracked.int32( 20 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            tanPhiMax = cms.double( 0.5 ),
            onlyBestSegment = cms.untracked.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            curvePenalty = cms.untracked.double( 2.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            BrutePruning = cms.untracked.bool( True ),
            tanThetaMax = cms.double( 1.2 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            useShowering = cms.untracked.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            dPhiFineMax = cms.double( 0.025 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            preClustering = cms.untracked.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.untracked.bool( True ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            BPMinImprovement = cms.untracked.double( 10000.0 )
          ),
          cms.PSet(  maxRatioResidualPrune = cms.double( 3.0 ),
            yweightPenalty = cms.untracked.double( 1.5 ),
            maxRecHitsInCluster = cms.untracked.int32( 24 ),
            hitDropLimit6Hits = cms.untracked.double( 0.3333 ),
            tanPhiMax = cms.double( 0.5 ),
            onlyBestSegment = cms.untracked.bool( False ),
            dRPhiFineMax = cms.double( 8.0 ),
            curvePenalty = cms.untracked.double( 2.0 ),
            dXclusBoxMax = cms.untracked.double( 4.0 ),
            BrutePruning = cms.untracked.bool( True ),
            tanThetaMax = cms.double( 1.2 ),
            hitDropLimit4Hits = cms.untracked.double( 0.6 ),
            useShowering = cms.untracked.bool( False ),
            CSCDebug = cms.untracked.bool( False ),
            curvePenaltyThreshold = cms.untracked.double( 0.85 ),
            minHitsPerSegment = cms.untracked.int32( 3 ),
            dPhiFineMax = cms.double( 0.025 ),
            yweightPenaltyThreshold = cms.untracked.double( 1.0 ),
            hitDropLimit5Hits = cms.untracked.double( 0.8 ),
            preClustering = cms.untracked.bool( True ),
            maxDPhi = cms.double( 999.0 ),
            maxDTheta = cms.double( 999.0 ),
            Pruning = cms.untracked.bool( True ),
            dYclusBoxMax = cms.untracked.double( 8.0 ),
            BPMinImprovement = cms.untracked.double( 10000.0 )
          )
        ),
        parameters_per_chamber_type = cms.vint32( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
      )
    )
)
hltRpcRecHits = cms.EDProducer( "RPCRecHitProducer",
    rpcDigiLabel = cms.InputTag( "simMuonRPCDigis" ),
    recAlgo = cms.string( "RPCRecHitStandardAlgo" ),
    maskSource = cms.string( "File" ),
    maskvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCMaskVec.dat" ),
    deadSource = cms.string( "File" ),
    deadvecfile = cms.FileInPath( "RecoLocalMuon/RPCRecHit/data/RPCDeadVec.dat" ),
    recAlgoConfig = cms.PSet(  )
)
hltL2MuonSeeds = cms.EDProducer( "L2MuonSeedGenerator",
    InputObjects = cms.InputTag( "l1ParamMuons" ),
    GMTReadoutCollection = cms.InputTag( "l1ParamMuons" ),
    Propagator = cms.string( "SteppingHelixPropagatorAny" ),
    L1MinPt = cms.double( 0.0 ),
    L1MaxEta = cms.double( 2.5 ),
    L1MinQuality = cms.uint32( 1 ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny',
        'SteppingHelixPropagatorAlong',
        'SteppingHelixPropagatorOpposite',
        'PropagatorWithMaterial',
        'PropagatorWithMaterialOpposite',
        'SmartPropagator',
        'SmartPropagatorOpposite',
        'SmartPropagatorAnyOpposite',
        'SmartPropagatorAny',
        'SmartPropagatorRK',
        'SmartPropagatorAnyRK' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    )
)
hltL2Muons = cms.EDProducer( "L2MuonProducer",
    InputObjects = cms.InputTag( "hltL2MuonSeeds" ),
    L2TrajBuilderParameters = cms.PSet( 
      RefitterParameters = cms.PSet( 
        FitterName = cms.string( "KFFitterSmootherForL2Muon" ),
        Option = cms.int32( 1 )
      ),
      DoRefit = cms.bool( False ),
      SeedPropagator = cms.string( "SteppingHelixPropagatorAny" ),
      NavigationType = cms.string( "Standard" ),
      SeedTransformerParameters = cms.PSet( 
        Fitter = cms.string( "KFFitterSmootherForL2Muon" ),
        RescaleError = cms.double( 100.0 ),
        MuonRecHitBuilder = cms.string( "MuonRecHitBuilder" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        NMinRecHits = cms.uint32( 2 )
      ),
      DoBackwardFilter = cms.bool( True ),
      SeedPosition = cms.string( "in" ),
      BWFilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        FitDirection = cms.string( "outsideIn" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 25.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 25.0 ),
          Granularity = cms.int32( 2 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        BWSeedType = cms.string( "fromGenerator" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      ),
      DoSeedRefit = cms.bool( False ),
      FilterParameters = cms.PSet( 
        NumberOfSigma = cms.double( 3.0 ),
        FitDirection = cms.string( "insideOut" ),
        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
        MaxChi2 = cms.double( 1000.0 ),
        MuonTrajectoryUpdatorParameters = cms.PSet( 
          MaxChi2 = cms.double( 1000.0 ),
          Granularity = cms.int32( 0 ),
          RescaleErrorFactor = cms.double( 100.0 ),
          RescaleError = cms.bool( False )
        ),
        EnableRPCMeasurement = cms.bool( True ),
        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
        EnableDTMeasurement = cms.bool( True ),
        RPCRecSegmentLabel = cms.InputTag( "hltRpcRecHits" ),
        Propagator = cms.string( "SteppingHelixPropagatorAny" ),
        EnableCSCMeasurement = cms.bool( True )
      )
    ),
    ServiceParameters = cms.PSet( 
      Propagators = cms.untracked.vstring( 'SteppingHelixPropagatorAny',
        'SteppingHelixPropagatorAlong',
        'SteppingHelixPropagatorOpposite',
        'PropagatorWithMaterial',
        'PropagatorWithMaterialOpposite',
        'SmartPropagator',
        'SmartPropagatorOpposite',
        'SmartPropagatorAnyOpposite',
        'SmartPropagatorAny',
        'SmartPropagatorRK',
        'SmartPropagatorAnyRK' ),
      RPCLayers = cms.bool( True ),
      UseMuonNavigation = cms.untracked.bool( True )
    ),
    TrackLoaderParameters = cms.PSet( 
      Smoother = cms.string( "KFSmootherForMuonTrackLoader" ),
      DoSmoothing = cms.bool( False ),
      MuonUpdatorAtVertexParameters = cms.PSet( 
        MaxChi2 = cms.double( 1000000.0 ),
        BeamSpotPosition = cms.vdouble( 0.0, 0.0, 0.0 ),
        Propagator = cms.string( "SteppingHelixPropagatorOpposite" ),
        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
      ),
      VertexConstraint = cms.bool( True )
    )
)
hltL2MuonCandidates = cms.EDProducer( "L2MuonCandidateProducer",
    InputObjects = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
)
hltL2Mu9L2Filtered9 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu7L1Filtered0" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreL2Mu11 = cms.EDFilter( "HLTPrescaler" )
hltL2Mu11L2Filtered11 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu7L1Filtered0" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 11.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltL1sL1SingleMu3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreIsoMu3 = cms.EDFilter( "HLTPrescaler" )
hltSingleMuIsoL1Filtered3 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SelectQualities = cms.vint32(  )
)
hltSingleMuIsoL2PreFiltered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL1Filtered3" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltTowerMakerForMuons = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold0 = cms.double( 1.1 ),
    HOThresholdPlus1 = cms.double( 1.1 ),
    HOThresholdMinus1 = cms.double( 1.1 ),
    HOThresholdPlus2 = cms.double( 1.1 ),
    HOThresholdMinus2 = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    HcalAcceptSeverityLevel = cms.uint32( 999 ),
    EcalAcceptSeverityLevel = cms.uint32( 1 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalMuonsRecHit:EcalRecHitsEB','hltEcalRegionalMuonsRecHit:EcalRecHitsEE' )
)
hltL2MuonIsolations = cms.EDProducer( "L2MuonIsolationProducer",
    StandAloneCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' ),
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
    ),
    IsolatorPSet = cms.PSet( 
      ComponentName = cms.string( "SimpleCutsIsolator" ),
      ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
      EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
      Thresholds = cms.vdouble( 4.0, 3.7, 4.0, 3.5, 3.4, 3.4, 3.2, 3.4, 3.1, 2.9, 2.9, 2.7, 3.1, 3.0, 2.4, 2.1, 2.0, 2.3, 2.2, 2.4, 2.5, 2.5, 2.6, 2.9, 3.1, 2.9 )
    )
)
hltSingleMuIsoL2IsoFiltered3 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2PreFiltered3" ),
    MinN = cms.int32( 1 ),
    DepTag = cms.VInputTag( 'hltL2MuonIsolations' ),
    IsolatorPSet = cms.PSet(  )
)
hltL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
    InputObjects = cms.InputTag( "hltL3Muons" )
)
hltSingleMuIsoL3PreFiltered3 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL2IsoFiltered3" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltL3MuonIsolations = cms.EDProducer( "L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag( "hltL3Muons" ),
    OutputMuIsoDeposits = cms.bool( True ),
    TrackPt_Min = cms.double( -1.0 ),
    CutsPSet = cms.PSet( 
      ComponentName = cms.string( "SimpleCuts" ),
      EtaBounds = cms.vdouble( 0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 1.785, 1.88, 1.9865, 2.1075, 2.247, 2.411 ),
      ConeSizes = cms.vdouble( 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24 ),
      Thresholds = cms.vdouble( 1.1, 1.1, 1.1, 1.1, 1.2, 1.1, 1.2, 1.1, 1.2, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 0.9, 1.1, 0.9, 1.1, 1.0, 1.0, 0.9, 0.8, 0.1 ),
      applyCutsORmaxNTracks = cms.bool( False ),
      maxNTracks = cms.int32( -1 )
    ),
    ExtractorPSet = cms.PSet( 
      ComponentName = cms.string( "PixelTrackExtractor" ),
      inputTrackCollection = cms.InputTag( "hltPixelTracks" ),
      DepositLabel = cms.untracked.string( "PXLS" ),
      Diff_r = cms.double( 0.1 ),
      Diff_z = cms.double( 0.2 ),
      DR_Veto = cms.double( 0.01 ),
      DR_Max = cms.double( 0.24 ),
      BeamlineOption = cms.string( "BeamSpotFromEvent" ),
      BeamSpotLabel = cms.InputTag( "offlineBeamSpot" ),
      NHits_Min = cms.uint32( 0 ),
      Chi2Ndof_Max = cms.double( 1.0E64 ),
      Chi2Prob_Min = cms.double( -1.0 ),
      Pt_Min = cms.double( -1.0 ),
      PropagateTracksToRadius = cms.bool( True ),
      ReferenceRadius = cms.double( 6.0 ),
      VetoLeadingTrack = cms.bool( True ),
      PtVeto_Min = cms.double( 2.0 ),
      DR_VetoPt = cms.double( 0.025 )
    )
)
hltSingleMuIsoL3IsoFiltered3 = cms.EDFilter( "HLTMuonIsoFilter",
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMuIsoL3PreFiltered3" ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    DepTag = cms.VInputTag( 'hltL3MuonIsolations' ),
    IsolatorPSet = cms.PSet(  )
)
hltPreMu3 = cms.EDFilter( "HLTPrescaler" )
hltL1SingleMu3L1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu3" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 1 ),
    SelectQualities = cms.vint32(  )
)
hltSingleMu3L2Filtered3 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu3L1Filtered0" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMu3L3Filtered3 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMu3L2Filtered3" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreMu5 = cms.EDFilter( "HLTPrescaler" )
hltSingleMu5L2Filtered4 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu3L1Filtered0" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 4.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMu5L3Filtered5 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMu5L2Filtered4" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 5.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltPreMu9 = cms.EDFilter( "HLTPrescaler" )
hltSingleMu9L2Filtered7 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltL1SingleMu7L1Filtered0" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 7.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltSingleMu9L3Filtered9 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltSingleMu9L2Filtered7" ),
    MinN = cms.int32( 1 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 9.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltL1sL1DoubleMuOpen = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMuOpen" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1DoubleMuOpen = cms.EDFilter( "HLTPrescaler" )
hltDoubleMuLevel1PathL1OpenFiltered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpen" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SaveTag = cms.untracked.bool( True ),
    SelectQualities = cms.vint32(  )
)
hltPreDoubleMu0 = cms.EDFilter( "HLTPrescaler" )
hltDiMuonL1Filtered0 = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMuOpen" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SelectQualities = cms.vint32(  )
)
hltDiMuonL2PreFiltered0 = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonL1Filtered0" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 0.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonL3PreFiltered0 = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonL2PreFiltered0" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 0.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltL1sL1DoubleMu3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleMu3" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreDoubleMu3 = cms.EDFilter( "HLTPrescaler" )
hltDiMuonL1Filtered = cms.EDFilter( "HLTMuonL1Filter",
    CandTag = cms.InputTag( "l1ParamMuons" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1DoubleMu3" ),
    MaxEta = cms.double( 2.5 ),
    MinPt = cms.double( 0.0 ),
    MinN = cms.int32( 2 ),
    SelectQualities = cms.vint32(  )
)
hltDiMuonL2PreFiltered = cms.EDFilter( "HLTMuonL2PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL2MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonL1Filtered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 9999.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 )
)
hltDiMuonL3PreFiltered = cms.EDFilter( "HLTMuonL3PreFilter",
    BeamSpotTag = cms.InputTag( "offlineBeamSpot" ),
    CandTag = cms.InputTag( "hltL3MuonCandidates" ),
    PreviousCandTag = cms.InputTag( "hltDiMuonL2PreFiltered" ),
    MinN = cms.int32( 2 ),
    MaxEta = cms.double( 2.5 ),
    MinNhits = cms.int32( 0 ),
    MaxDr = cms.double( 2.0 ),
    MaxDz = cms.double( 9999.0 ),
    MinPt = cms.double( 3.0 ),
    NSigmaPt = cms.double( 0.0 ),
    SaveTag = cms.untracked.bool( True )
)
hltL1sRelaxedSingleEgammaEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1SingleEG5 = cms.EDFilter( "HLTPrescaler" )
hltL1sRelaxedSingleEgammaEt8 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG8" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1SingleEG8 = cms.EDFilter( "HLTPrescaler" )
hltPreEle10LWL1R = cms.EDFilter( "HLTPrescaler" )
hltHybridSuperClustersL1Isolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
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
      brLinearLowThr = cms.double( 1.1 ),
      fBremVec = cms.vdouble( -0.05208, 0.1331, 0.9196, -5.735E-4, 1.343 ),
      brLinearHighThr = cms.double( 8.0 ),
      fEtEtaVec = cms.vdouble( 1.0012, -0.5714, 0.0, 0.0, 0.0, 0.5549, 12.74, 1.0448, 0.0, 0.0, 0.0, 0.0, 8.0, 1.023, -0.00181, 0.0, 0.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltMulti5x5BasicClustersL1Isolated = cms.EDProducer( "EgammaHLTMulti5x5ClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( True ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    Multi5x5BarrelSeedThr = cms.double( 0.5 ),
    Multi5x5EndcapSeedThr = cms.double( 0.18 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
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
hltMulti5x5SuperClustersL1Isolated = cms.EDProducer( "Multi5x5SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersL1Isolated" ),
    barrelClusterProducer = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    dynamicPhiRoad = cms.bool( False ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet(  ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      ),
      doEndcaps = cms.bool( True ),
      doBarrel = cms.bool( False )
    )
)
hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersL1Isolated','multi5x5EndcapSuperClusters' ),
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
hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.6 ),
      fBremVec = cms.vdouble( -0.04163, 0.08552, 0.95048, -0.002308, 1.077 ),
      brLinearHighThr = cms.double( 6.0 ),
      fEtEtaVec = cms.vdouble( 0.9746, -6.512, 0.0, 0.0, 0.02771, 4.983, 0.0, 0.0, -0.007288, -0.9446, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 )
    )
)
hltHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTHybridClusterProducer",
    debugLevel = cms.string( "INFO" ),
    basicclusterCollection = cms.string( "" ),
    superclusterCollection = cms.string( "" ),
    ecalhitproducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalhitcollection = cms.string( "EcalRecHitsEB" ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
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
hltCorrectedHybridSuperClustersL1NonIsolatedTemp = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEB' ),
    rawSuperClusterProducer = cms.InputTag( "hltHybridSuperClustersL1NonIsolated" ),
    superClusterAlgo = cms.string( "Hybrid" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.03 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 1.1 ),
      fBremVec = cms.vdouble( -0.05208, 0.1331, 0.9196, -5.735E-4, 1.343 ),
      brLinearHighThr = cms.double( 8.0 ),
      fEtEtaVec = cms.vdouble( 1.0012, -0.5714, 0.0, 0.0, 0.0, 0.5549, 12.74, 1.0448, 0.0, 0.0, 0.0, 0.0, 8.0, 1.023, -0.00181, 0.0, 0.0 )
    ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet(  )
)
hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDProducer( "EgammaHLTRemoveDuplicatedSC",
    L1NonIsoUskimmedSC = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolatedTemp" ),
    L1IsoSC = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    L1NonIsoSkimmedCollection = cms.string( "" )
)
hltMulti5x5BasicClustersL1NonIsolated = cms.EDProducer( "EgammaHLTMulti5x5ClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    doIsolated = cms.bool( False ),
    barrelHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    endcapHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    barrelHitCollection = cms.string( "EcalRecHitsEB" ),
    endcapHitCollection = cms.string( "EcalRecHitsEE" ),
    barrelClusterCollection = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    Multi5x5BarrelSeedThr = cms.double( 0.5 ),
    Multi5x5EndcapSeedThr = cms.double( 0.5 ),
    l1TagIsolated = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1TagNonIsolated = cms.InputTag( 'l1extraParticles','NonIsolated' ),
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
hltMulti5x5SuperClustersL1NonIsolated = cms.EDProducer( "Multi5x5SuperClusterProducer",
    VerbosityLevel = cms.string( "ERROR" ),
    endcapClusterProducer = cms.string( "hltMulti5x5BasicClustersL1NonIsolated" ),
    barrelClusterProducer = cms.string( "notused" ),
    endcapClusterCollection = cms.string( "multi5x5EndcapBasicClusters" ),
    barrelClusterCollection = cms.string( "multi5x5BarrelBasicClusters" ),
    endcapSuperclusterCollection = cms.string( "multi5x5EndcapSuperClusters" ),
    barrelSuperclusterCollection = cms.string( "multi5x5BarrelSuperClusters" ),
    doBarrel = cms.bool( False ),
    doEndcaps = cms.bool( True ),
    barrelEtaSearchRoad = cms.double( 0.06 ),
    barrelPhiSearchRoad = cms.double( 0.8 ),
    endcapEtaSearchRoad = cms.double( 0.14 ),
    endcapPhiSearchRoad = cms.double( 0.6 ),
    seedTransverseEnergyThreshold = cms.double( 1.0 ),
    dynamicPhiRoad = cms.bool( False ),
    bremRecoveryPset = cms.PSet( 
      barrel = cms.PSet(  ),
      endcap = cms.PSet( 
        a = cms.double( 47.85 ),
        c = cms.double( 0.1201 ),
        b = cms.double( 108.8 )
      ),
      doEndcaps = cms.bool( True ),
      doBarrel = cms.bool( False )
    )
)
hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "PreshowerClusterProducer",
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    endcapSClusterProducer = cms.InputTag( 'hltMulti5x5SuperClustersL1NonIsolated','multi5x5EndcapSuperClusters' ),
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
hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp = cms.EDProducer( "EgammaSCCorrectionMaker",
    VerbosityLevel = cms.string( "ERROR" ),
    recHitProducer = cms.InputTag( 'hltEcalRegionalEgammaRecHit','EcalRecHitsEE' ),
    rawSuperClusterProducer = cms.InputTag( "hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated" ),
    superClusterAlgo = cms.string( "Multi5x5" ),
    applyEnergyCorrection = cms.bool( True ),
    sigmaElectronicNoise = cms.double( 0.15 ),
    etThresh = cms.double( 5.0 ),
    corectedSuperClusterCollection = cms.string( "" ),
    hyb_fCorrPset = cms.PSet(  ),
    isl_fCorrPset = cms.PSet(  ),
    dyn_fCorrPset = cms.PSet(  ),
    fix_fCorrPset = cms.PSet( 
      brLinearLowThr = cms.double( 0.6 ),
      fBremVec = cms.vdouble( -0.04163, 0.08552, 0.95048, -0.002308, 1.077 ),
      brLinearHighThr = cms.double( 6.0 ),
      fEtEtaVec = cms.vdouble( 0.9746, -6.512, 0.0, 0.0, 0.02771, 4.983, 0.0, 0.0, -0.007288, -0.9446, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 )
    )
)
hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer( "EgammaHLTRemoveDuplicatedSC",
    L1NonIsoUskimmedSC = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp" ),
    L1IsoSC = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    L1NonIsoSkimmedCollection = cms.string( "" )
)
hltL1IsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1Isolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1NonIsoRecoEcalCandidate = cms.EDProducer( "EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag( "hltCorrectedHybridSuperClustersL1NonIsolated" ),
    scIslandEndcapProducer = cms.InputTag( "hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated" ),
    recoEcalCandidateCollection = cms.string( "" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional" ),
    etcutEB = cms.double( 10.0 ),
    etcutEE = cms.double( 10.0 ),
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
hltL1NonIsolatedElectronHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.15 )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1sRelaxedSingleEgammaEt10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreEle15LWL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt10" ),
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
    etcutEB = cms.double( 15.0 ),
    etcutEE = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
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
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1sRelaxedDoubleEgammaEt5 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleEG5" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1DoubleEG5 = cms.EDFilter( "HLTPrescaler" )
hltPreDoubleEle5SWL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedDoubleEgammaEt5" ),
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
    etcutEB = cms.double( 5.0 ),
    etcutEE = cms.double( 5.0 ),
    ncandcut = cms.int32( 2 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedElectronHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedElectronHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter = cms.EDFilter( "HLTElectronPixelMatchFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoDoubleElectronEt5HcalIsolFilter" ),
    L1IsoPixelSeedsTag = cms.InputTag( "hltL1IsoStartUpElectronPixelSeeds" ),
    L1NonIsoPixelSeedsTag = cms.InputTag( "hltL1NonIsoStartUpElectronPixelSeeds" ),
    npixelmatchcut = cms.double( 1.0 ),
    ncandcut = cms.int32( 2 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPreEle15SC10LWL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESscWrapper = cms.EDFilter( "HLTEgammaTriggerFilterObjectWrapper",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    doIsolated = cms.bool( False )
)
hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10 = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESscWrapper" ),
    etcutEB = cms.double( 10.0 ),
    etcutEE = cms.double( 10.0 ),
    ncandcut = cms.int32( 2 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPreEle25LWL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet25 = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter" ),
    etcutEB = cms.double( 25.0 ),
    etcutEE = cms.double( 25.0 ),
    ncandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton10L1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt10L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt8" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt10EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt10L1MatchFilterRegional" ),
    etcutEB = cms.double( 10.0 ),
    etcutEE = cms.double( 10.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1NonIsolatedPhotonHcalIsol = cms.EDProducer( "EgammaHLTHcalIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    hbRecHitProducer = cms.InputTag( "hltHbhereco" ),
    hfRecHitProducer = cms.InputTag( "hltHfreco" ),
    egHcalIsoPtMin = cms.double( 0.0 ),
    egHcalIsoConeSize = cms.double( 0.3 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt10EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton15L1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt10" ),
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
    etcutEB = cms.double( 15.0 ),
    etcutEE = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton15HTIL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSinglePhotonEt15HTIL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEt15HTIEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15HTIL1MatchFilterRegional" ),
    etcutEB = cms.double( 15.0 ),
    etcutEE = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt15HTIHcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15HTIEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsoPhotonHollowTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1IsoEgammaRegionalCTFFinalFitWithMaterial" ),
    countTracks = cms.bool( False ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.06 )
)
hltL1NonIsoPhotonHollowTrackIsol = cms.EDProducer( "EgammaHLTPhotonTrackIsolationProducersRegional",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    trackProducer = cms.InputTag( "hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial" ),
    countTracks = cms.bool( False ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.3 ),
    egTrkIsoZSpan = cms.double( 999999.0 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.06 )
)
hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15HTIHcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsoPhotonHollowTrackIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsoPhotonHollowTrackIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 0.0 ),
    thrRegularEE = cms.double( 0.0 ),
    thrOverEEB = cms.untracked.double( 0.05 ),
    thrOverEEE = cms.untracked.double( 0.05 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton15LEIL1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoSinglePhotonEt15LEIL1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoSinglePhotonEt15LEIEtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15LEIL1MatchFilterRegional" ),
    etcutEB = cms.double( 15.0 ),
    etcutEE = cms.double( 15.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1IsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
    ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
    etMinBarrel = cms.double( -9999.0 ),
    eMinBarrel = cms.double( 0.08 ),
    etMinEndcap = cms.double( -9999.0 ),
    eMinEndcap = cms.double( 0.3 ),
    intRadiusBarrel = cms.double( 0.045 ),
    intRadiusEndcap = cms.double( 0.07 ),
    extRadius = cms.double( 0.4 ),
    jurassicWidth = cms.double( 0.02 ),
    useIsolEt = cms.bool( True ),
    tryBoth = cms.bool( True ),
    subtract = cms.bool( False )
)
hltL1NonIsolatedPhotonEcalIsol = cms.EDProducer( "EgammaHLTEcalRecIsolationProducer",
    recoEcalCandidateProducer = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    ecalBarrelRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalBarrelRecHitCollection = cms.InputTag( "EcalRecHitsEB" ),
    ecalEndcapRecHitProducer = cms.InputTag( "hltEcalRegionalEgammaRecHit" ),
    ecalEndcapRecHitCollection = cms.InputTag( "EcalRecHitsEE" ),
    etMinBarrel = cms.double( -9999.0 ),
    eMinBarrel = cms.double( 0.08 ),
    etMinEndcap = cms.double( -9999.0 ),
    eMinEndcap = cms.double( 0.3 ),
    intRadiusBarrel = cms.double( 0.045 ),
    intRadiusEndcap = cms.double( 0.07 ),
    extRadius = cms.double( 0.4 ),
    jurassicWidth = cms.double( 0.02 ),
    useIsolEt = cms.bool( True ),
    tryBoth = cms.bool( True ),
    subtract = cms.bool( False )
)
hltL1NonIsoSinglePhotonEt15LEIEcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15LEIEtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonEcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonEcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 3.0 ),
    thrRegularEE = cms.double( 3.0 ),
    thrOverEEB = cms.untracked.double( 0.2 ),
    thrOverEEE = cms.untracked.double( 0.2 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoSinglePhotonEt15LEIEcalIsolFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton20L1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter( "HLTEgammaL1MatchFilterRegional",
    candIsolatedTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    candNonIsolatedTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    L1SeedFilterTag = cms.InputTag( "hltL1sRelaxedSingleEgammaEt10" ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    region_eta_size = cms.double( 0.522 ),
    region_eta_size_ecap = cms.double( 1.0 ),
    region_phi_size = cms.double( 1.044 ),
    barrel_end = cms.double( 1.4791 ),
    endcap_end = cms.double( 2.65 )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt20EtFilter = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt20L1MatchFilterRegional" ),
    etcutEB = cms.double( 20.0 ),
    etcutEE = cms.double( 20.0 ),
    ncandcut = cms.int32( 1 ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter( "HLTEgammaGenericFilter",
    candTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt20EtFilter" ),
    isoTag = cms.InputTag( "hltL1IsolatedPhotonHcalIsol" ),
    nonIsoTag = cms.InputTag( "hltL1NonIsolatedPhotonHcalIsol" ),
    lessThan = cms.bool( True ),
    thrRegularEB = cms.double( 999999.9 ),
    thrRegularEE = cms.double( 999999.9 ),
    ncandcut = cms.int32( 1 ),
    doIsolated = cms.bool( False ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltPrePhoton30L1R = cms.EDFilter( "HLTPrescaler" )
hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilterESet30 = cms.EDFilter( "HLTEgammaEtFilter",
    inputTag = cms.InputTag( "hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter" ),
    etcutEB = cms.double( 30.0 ),
    etcutEE = cms.double( 30.0 ),
    ncandcut = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True ),
    L1IsoCand = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    L1NonIsoCand = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" )
)
hltL1sSingleLooseIsoTau20 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleTauJet20U OR L1_SingleJet30U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreSingleLooseIsoTau20 = cms.EDFilter( "HLTPrescaler" )
hltTowerMakerForJets = cms.EDProducer( "CaloTowersCreator",
    EBThreshold = cms.double( 0.09 ),
    EEThreshold = cms.double( 0.45 ),
    HcalThreshold = cms.double( -1000.0 ),
    HBThreshold = cms.double( 0.9 ),
    HESThreshold = cms.double( 1.4 ),
    HEDThreshold = cms.double( 1.4 ),
    HOThreshold0 = cms.double( 1.1 ),
    HOThresholdPlus1 = cms.double( 1.1 ),
    HOThresholdMinus1 = cms.double( 1.1 ),
    HOThresholdPlus2 = cms.double( 1.1 ),
    HOThresholdMinus2 = cms.double( 1.1 ),
    HF1Threshold = cms.double( 1.2 ),
    HF2Threshold = cms.double( 1.8 ),
    EBWeight = cms.double( 1.0 ),
    EEWeight = cms.double( 1.0 ),
    HBWeight = cms.double( 1.0 ),
    HESWeight = cms.double( 1.0 ),
    HEDWeight = cms.double( 1.0 ),
    HOWeight = cms.double( 1.0E-99 ),
    HF1Weight = cms.double( 1.0 ),
    HF2Weight = cms.double( 1.0 ),
    EcutTower = cms.double( -1000.0 ),
    EBSumThreshold = cms.double( 0.2 ),
    EESumThreshold = cms.double( 0.45 ),
    UseHO = cms.bool( False ),
    MomConstrMethod = cms.int32( 1 ),
    MomHBDepth = cms.double( 0.2 ),
    MomHEDepth = cms.double( 0.4 ),
    MomEBDepth = cms.double( 0.3 ),
    MomEEDepth = cms.double( 0.0 ),
    hbheInput = cms.InputTag( "hltHbhereco" ),
    hoInput = cms.InputTag( "hltHoreco" ),
    hfInput = cms.InputTag( "hltHfreco" ),
    HcalAcceptSeverityLevel = cms.uint32( 999 ),
    EcalAcceptSeverityLevel = cms.uint32( 1 ),
    UseHcalRecoveredHits = cms.bool( True ),
    UseEcalRecoveredHits = cms.bool( True ),
    ecalInputs = cms.VInputTag( 'hltEcalRegionalJetsRecHit:EcalRecHitsEB','hltEcalRegionalJetsRecHit:EcalRecHitsEE' )
)
hltCaloTowersTau1Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIconeTau1Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau1Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersTau2Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIconeTau2Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau2Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersTau3Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIconeTau3Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau3Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersTau4Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Tau' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIconeTau4Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersTau4Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersCentral1Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Central' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 0 )
)
hltIconeCentral1Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersCentral1Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersCentral2Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Central' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 1 )
)
hltIconeCentral2Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersCentral2Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersCentral3Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Central' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 2 )
)
hltIconeCentral3Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersCentral3Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltCaloTowersCentral4Regional = cms.EDProducer( "CaloTowerCreatorForTauHLT",
    towers = cms.InputTag( "hltTowerMakerForJets" ),
    UseTowersInCone = cms.double( 0.8 ),
    TauTrigger = cms.InputTag( 'l1extraParticles','Central' ),
    minimumEt = cms.double( 0.5 ),
    minimumE = cms.double( 0.8 ),
    TauId = cms.int32( 3 )
)
hltIconeCentral4Regional = cms.EDProducer( "IterativeConeJetProducer",
    seedThreshold = cms.double( 1.0 ),
    coneRadius = cms.double( 0.2 ),
    verbose = cms.untracked.bool( False ),
    jetType = cms.untracked.string( "CaloJet" ),
    src = cms.InputTag( "hltCaloTowersCentral4Regional" ),
    jetPtMin = cms.double( 0.0 ),
    inputEMin = cms.double( 0.0 ),
    inputEtMin = cms.double( 0.5 ),
    debugLevel = cms.untracked.int32( 0 ),
    alias = cms.untracked.string( "IC5CaloJet" ),
    correctInputToSignalVertex = cms.bool( False ),
    pvCollection = cms.InputTag( "offlinePrimaryVertices" ),
    maxBadEcalCells = cms.uint32( 9999999 ),
    maxRecoveredEcalCells = cms.uint32( 9999999 ),
    maxProblematicEcalCells = cms.uint32( 9999999 ),
    maxBadHcalCells = cms.uint32( 9999999 ),
    maxRecoveredHcalCells = cms.uint32( 9999999 ),
    maxProblematicHcalCells = cms.uint32( 9999999 )
)
hltL2TauJets = cms.EDProducer( "L2TauJetsMerger",
    EtMin = cms.double( 15.0 ),
    JetSrc = cms.VInputTag( 'hltIconeTau1Regional','hltIconeTau2Regional','hltIconeTau3Regional','hltIconeTau4Regional','hltIconeCentral1Regional','hltIconeCentral2Regional','hltIconeCentral3Regional','hltIconeCentral4Regional' )
)
hltFilterL2EtCutSingleLooseIsoTau20 = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltL2TauJets" ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL2TauNarrowConeIsolationProducer = cms.EDProducer( "L2TauNarrowConeIsolationProducer",
    L2TauJetCollection = cms.InputTag( "hltL2TauJets" ),
    EBRecHits = cms.InputTag( 'hltEcalRegionalJetsRecHit','EcalRecHitsEB' ),
    EERecHits = cms.InputTag( 'hltEcalRegionalJetsRecHit','EcalRecHitsEE' ),
    CaloTowers = cms.InputTag( "hltTowerMakerForJets" ),
    associationRadius = cms.double( 0.5 ),
    crystalThresholdEE = cms.double( 0.45 ),
    crystalThresholdEB = cms.double( 0.15 ),
    towerThreshold = cms.double( 1.0 ),
    ECALIsolation = cms.PSet( 
      innerCone = cms.double( 0.15 ),
      outerCone = cms.double( 0.5 ),
      runAlgorithm = cms.bool( True )
    ),
    ECALClustering = cms.PSet( 
      runAlgorithm = cms.bool( True ),
      clusterRadius = cms.double( 0.08 )
    ),
    TowerIsolation = cms.PSet( 
      innerCone = cms.double( 0.2 ),
      outerCone = cms.double( 0.5 ),
      runAlgorithm = cms.bool( True )
    )
)
hltL2TauRelaxingIsolationSelector = cms.EDProducer( "L2TauRelaxingIsolationSelector",
    L2InfoAssociation = cms.InputTag( "hltL2TauNarrowConeIsolationProducer" ),
    MinJetEt = cms.double( 15.0 ),
    SeedTowerEt = cms.double( -10.0 ),
    EcalIsolationEt = cms.vdouble( 5.0, 0.025, 7.5E-4 ),
    TowerIsolationEt = cms.vdouble( 1000.0, 0.0, 0.0 ),
    NumberOfClusters = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ClusterPhiRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ClusterEtaRMS = cms.vdouble( 1000.0, 0.0, 0.0 ),
    ClusterDRRMS = cms.vdouble( 1000.0, 0.0, 0.0 )
)
hltL1HLTSingleLooseIsoTau20JetsMatch = cms.EDProducer( "L1HLTJetsMatching",
    JetSrc = cms.InputTag( 'hltL2TauRelaxingIsolationSelector','Isolated' ),
    L1TauTrigger = cms.InputTag( "hltL1sSingleLooseIsoTau20" ),
    EtMin = cms.double( 20.0 )
)
hltFilterL2EcalIsolationSingleLooseIsoTau20 = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltL1HLTSingleLooseIsoTau20JetsMatch" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 20.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 1 )
)
hltL1sDoubleLooseIsoTau15 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_DoubleTauJet14U OR L1_DoubleJet30U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreDoubleLooseIsoTau15 = cms.EDFilter( "HLTPrescaler" )
hltFilterL2EtCutDoubleLooseIsoTau15 = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltL2TauJets" ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1HLTDoubleLooseIsoTau15JetsMatch = cms.EDProducer( "L1HLTJetsMatching",
    JetSrc = cms.InputTag( 'hltL2TauRelaxingIsolationSelector','Isolated' ),
    L1TauTrigger = cms.InputTag( "hltL1sDoubleLooseIsoTau15" ),
    EtMin = cms.double( 15.0 )
)
hltFilterL2EcalIsolationDoubleLooseIsoTau15 = cms.EDFilter( "HLT1Tau",
    inputTag = cms.InputTag( "hltL1HLTDoubleLooseIsoTau15JetsMatch" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 15.0 ),
    MaxEta = cms.double( 5.0 ),
    MinN = cms.int32( 2 )
)
hltL1sBTagMuJet10U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_Mu3_Jet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreBTagMuJet10U = cms.EDFilter( "HLTPrescaler" )
hltBJet10U = cms.EDFilter( "HLT1CaloBJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 10.0 ),
    MaxEta = cms.double( 3.0 ),
    MinN = cms.int32( 1 )
)
hltSelector4JetsU = cms.EDProducer( "LargestEtCaloJetSelector",
    src = cms.InputTag( "hltIterativeCone5CaloJets" ),
    filter = cms.bool( False ),
    maxNumber = cms.uint32( 4 )
)
hltBSoftMuonL25JetsU = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltSelector4JetsU" ),
    filter = cms.bool( False ),
    etMin = cms.double( 10.0 )
)
hltBSoftMuonL25TagInfosU = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftMuonL25JetsU" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL2Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 ),
    muonSelection = cms.uint32( 0 )
)
hltBSoftMuonL25BJetTagsUByDR = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftMuonL25TagInfosU' )
)
hltBSoftMuonL25FilterUByDR = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftMuonL25BJetTagsUByDR" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBSoftMuonL3TagInfosU = cms.EDProducer( "SoftLepton",
    jets = cms.InputTag( "hltBSoftMuonL25JetsU" ),
    primaryVertex = cms.InputTag( "nominal" ),
    leptons = cms.InputTag( "hltL3Muons" ),
    refineJetAxis = cms.uint32( 0 ),
    leptonDeltaRCut = cms.double( 0.4 ),
    leptonChi2Cut = cms.double( 0.0 ),
    leptonQualityCut = cms.double( 0.0 ),
    muonSelection = cms.uint32( 0 )
)
hltBSoftMuonL3BJetTagsUByPt = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByPt" ),
    tagInfos = cms.VInputTag( 'hltBSoftMuonL3TagInfosU' )
)
hltBSoftMuonL3BJetTagsUByDR = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "softLeptonByDistance" ),
    tagInfos = cms.VInputTag( 'hltBSoftMuonL3TagInfosU' )
)
hltBSoftMuonL3FilterUByDR = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBSoftMuonL3BJetTagsUByDR" ),
    MinTag = cms.double( 0.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltL1sBTagIPJet50U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet30U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreBTagIPJet50U = cms.EDFilter( "HLTPrescaler" )
hltBJet50U = cms.EDFilter( "HLT1CaloBJet",
    inputTag = cms.InputTag( "hltIterativeCone5CaloJets" ),
    saveTag = cms.untracked.bool( True ),
    MinPt = cms.double( 50.0 ),
    MaxEta = cms.double( 3.0 ),
    MinN = cms.int32( 1 )
)
hltBLifetimeL25JetsStartupU = cms.EDProducer( "EtMinCaloJetSelector",
    src = cms.InputTag( "hltSelector4JetsU" ),
    filter = cms.bool( False ),
    etMin = cms.double( 15.0 )
)
hltBLifetimeL25AssociatorStartupU = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL25JetsStartupU" ),
    tracks = cms.InputTag( "hltPixelTracks" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL25TagInfosStartupU = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL25AssociatorStartupU" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    computeGhostTrack = cms.bool( False ),
    ghostTrackPriorDeltaR = cms.double( 0.03 ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 3 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumChiSquared = cms.double( 5.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    jetDirectionUsingTracks = cms.bool( False ),
    jetDirectionUsingGhostTrack = cms.bool( False ),
    useTrackQuality = cms.bool( False )
)
hltBLifetimeL25BJetTagsStartupU = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL25TagInfosStartupU' )
)
hltBLifetimeL25FilterStartupU = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL25BJetTagsStartupU" ),
    MinTag = cms.double( 2.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( False )
)
hltBLifetimeL3JetsStartupU = cms.EDProducer( "GetJetsFromHLTobject",
    jets = cms.InputTag( "hltBLifetimeL25FilterStartupU" )
)
hltBLifetimeL3AssociatorStartupU = cms.EDProducer( "JetTracksAssociatorAtVertex",
    jets = cms.InputTag( "hltBLifetimeL3JetsStartupU" ),
    tracks = cms.InputTag( "hltBLifetimeRegionalCtfWithMaterialTracksStartupU" ),
    coneSize = cms.double( 0.5 )
)
hltBLifetimeL3TagInfosStartupU = cms.EDProducer( "TrackIPProducer",
    jetTracks = cms.InputTag( "hltBLifetimeL3AssociatorStartupU" ),
    primaryVertex = cms.InputTag( "hltPixelVertices" ),
    computeProbabilities = cms.bool( False ),
    computeGhostTrack = cms.bool( False ),
    ghostTrackPriorDeltaR = cms.double( 0.03 ),
    minimumNumberOfPixelHits = cms.int32( 2 ),
    minimumNumberOfHits = cms.int32( 8 ),
    maximumTransverseImpactParameter = cms.double( 0.2 ),
    minimumTransverseMomentum = cms.double( 1.0 ),
    maximumChiSquared = cms.double( 20.0 ),
    maximumLongitudinalImpactParameter = cms.double( 17.0 ),
    jetDirectionUsingTracks = cms.bool( False ),
    jetDirectionUsingGhostTrack = cms.bool( False ),
    useTrackQuality = cms.bool( False )
)
hltBLifetimeL3BJetTagsStartupU = cms.EDProducer( "JetTagProducer",
    jetTagComputer = cms.string( "trackCounting3D2nd" ),
    tagInfos = cms.VInputTag( 'hltBLifetimeL3TagInfosStartupU' )
)
hltBLifetimeL3FilterStartupU = cms.EDFilter( "HLTJetTag",
    JetTag = cms.InputTag( "hltBLifetimeL3BJetTagsStartupU" ),
    MinTag = cms.double( 3.5 ),
    MaxTag = cms.double( 99999.0 ),
    MinJets = cms.int32( 1 ),
    SaveTag = cms.bool( True )
)
hltL1sZeroBias = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1GlobalDecision" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreZeroBias = cms.EDFilter( "HLTPrescaler" )
hltL1sMinBiasHcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleHfBitCountsRing1_1 OR L1_DoubleHfBitCountsRing1_P1N1 OR L1_SingleHfRingEtSumsRing1_4 OR L1_DoubleHfRingEtSumsRing1_P4N4 OR L1_SingleHfRingEtSumsRing2_4 OR L1_DoubleHfRingEtSumsRing2_P4N4" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreMinBiasHcal = cms.EDFilter( "HLTPrescaler" )
hltL1sMinBiasEcal = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleEG1 OR L1_SingleEG2 OR L1_DoubleEG1" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreMinBiasEcal = cms.EDFilter( "HLTPrescaler" )
hltL1sMinBiasPixel = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_MinBias_HTT10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreMinBiasPixel = cms.EDFilter( "HLTPrescaler" )
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
hltPreMinBiasPixelTrk5 = cms.EDFilter( "HLTPrescaler" )
hltPixelMBForAlignment = cms.EDFilter( "HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag( "hltPixelCands" ),
    MinPt = cms.double( 5.0 ),
    MinTrks = cms.uint32( 2 ),
    MinSep = cms.double( 1.0 ),
    MinIsol = cms.double( 0.05 )
)
hltL1sBackwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "38 OR 39" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreBackwardBSC = cms.EDFilter( "HLTPrescaler" )
hltL1sForwardBSC = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "36 OR 37" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreForwardBSC = cms.EDFilter( "HLTPrescaler" )
hltL1sCSCBeamHalo = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreCSCBeamHalo = cms.EDFilter( "HLTPrescaler" )
hltL1sCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing1 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( True ),
    ring2 = cms.bool( False ),
    fillHists = cms.bool( False )
)
hltL1sCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTPrescaler" )
hltOverlapsHLTCSCBeamHaloOverlapRing2 = cms.EDFilter( "HLTCSCOverlapFilter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 ),
    ring1 = cms.bool( False ),
    ring2 = cms.bool( True ),
    fillHists = cms.bool( False )
)
hltL1sCSCBeamHaloRing2or3 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMuBeamHalo" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreCSCBeamHaloRing2or3 = cms.EDFilter( "HLTPrescaler" )
hltFilter23HLTCSCBeamHaloRing2or3 = cms.EDFilter( "HLTCSCRing2or3Filter",
    input = cms.InputTag( "hltCsc2DRecHits" ),
    minHits = cms.uint32( 4 ),
    xWindow = cms.double( 2.0 ),
    yWindow = cms.double( 2.0 )
)
hltL1sTrackerCosmics = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( True ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "24 OR 25 OR 26 OR 27 OR 28" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreTrackerCosmics = cms.EDFilter( "HLTPrescaler" )
hltL1sAlCaEcalPhiSym = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleHfBitCountsRing1_1 OR L1_DoubleHfBitCountsRing1_P1N1 OR L1_SingleHfRingEtSumsRing1_4 OR L1_DoubleHfRingEtSumsRing1_P4N4 OR L1_SingleHfRingEtSumsRing1_4 OR L1_DoubleHfRingEtSumsRing2_P4N4 OR L1_SingleEG1 OR L1_SingleEG2 OR L1_DoubleEG1" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreAlCaEcalPhiSym = cms.EDFilter( "HLTPrescaler" )
hltAlCaPhiSymStream = cms.EDFilter( "HLTEcalPhiSymFilter",
    barrelHitCollection = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEB' ),
    endcapHitCollection = cms.InputTag( 'hltEcalRecHitAll','EcalRecHitsEE' ),
    phiSymBarrelHitCollection = cms.string( "phiSymEcalRecHitsEB" ),
    phiSymEndcapHitCollection = cms.string( "phiSymEcalRecHitsEE" ),
    eCut_barrel = cms.double( 0.15 ),
    eCut_endcap = cms.double( 0.75 )
)
hltL1sAlCaEcalPi0 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleEG1 OR L1_SingleEG2 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreAlCaEcalPi0 = cms.EDFilter( "HLTPrescaler" )
hltAlCaPi0RegRecHits = cms.EDFilter( "HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalPi0RecHit','EcalRecHitsEE' ),
    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    seleNRHMax = cms.int32( 1000 ),
    seleXtalMinEnergy = cms.double( -0.15 ),
    seleXtalMinEnergyEndCap = cms.double( -0.75 ),
    doSelForPi0Barrel = cms.bool( True ),
    selePtGamma = cms.double( 1.0 ),
    selePtPi0 = cms.double( 2.0 ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleS4S9Gamma = cms.double( 0.83 ),
    selePi0Iso = cms.double( 0.5 ),
    ptMinForIsolation = cms.double( 1.0 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    storeIsoClusRecHitPi0EB = cms.bool( True ),
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    doSelForPi0Endcap = cms.bool( True ),
    region1_Pi0EndCap = cms.double( 2.0 ),
    selePtGammaPi0EndCap_region1 = cms.double( 0.7 ),
    selePtPi0EndCap_region1 = cms.double( 3.0 ),
    region2_Pi0EndCap = cms.double( 2.5 ),
    selePtGammaPi0EndCap_region2 = cms.double( 0.5 ),
    selePtPi0EndCap_region2 = cms.double( 2.0 ),
    selePtGammaPi0EndCap_region3 = cms.double( 0.3 ),
    selePtPi0EndCap_region3 = cms.double( 1.2 ),
    seleS4S9GammaEndCap = cms.double( 0.9 ),
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),
    ptMinForIsolationEndCap = cms.double( 0.5 ),
    selePi0BeltDREndCap = cms.double( 0.2 ),
    selePi0BeltDetaEndCap = cms.double( 0.05 ),
    selePi0IsoEndCap = cms.double( 0.5 ),
    storeIsoClusRecHitPi0EE = cms.bool( True ),
    pi0EndcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
    doSelForEtaBarrel = cms.bool( False ),
    selePtGammaEta = cms.double( 1.2 ),
    selePtEta = cms.double( 4.0 ),
    seleS4S9GammaEta = cms.double( 0.9 ),
    seleS9S25GammaEta = cms.double( 0.8 ),
    seleMinvMaxEta = cms.double( 0.8 ),
    seleMinvMinEta = cms.double( 0.3 ),
    ptMinForIsolationEta = cms.double( 1.0 ),
    seleEtaIso = cms.double( 0.5 ),
    seleEtaBeltDR = cms.double( 0.3 ),
    seleEtaBeltDeta = cms.double( 0.1 ),
    storeIsoClusRecHitEtaEB = cms.bool( True ),
    removePi0CandidatesForEta = cms.bool( True ),
    massLowPi0Cand = cms.double( 0.104 ),
    massHighPi0Cand = cms.double( 0.163 ),
    etaBarrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
    store5x5RecHitEtaEB = cms.bool( True ),
    store5x5IsoClusRecHitEtaEB = cms.bool( True ),
    doSelForEtaEndcap = cms.bool( False ),
    region1_EtaEndCap = cms.double( 2.0 ),
    selePtGammaEtaEndCap_region1 = cms.double( 1.3 ),
    selePtEtaEndCap_region1 = cms.double( 5.0 ),
    region2_EtaEndCap = cms.double( 2.5 ),
    selePtGammaEtaEndCap_region2 = cms.double( 1.0 ),
    selePtEtaEndCap_region2 = cms.double( 4.0 ),
    selePtGammaEtaEndCap_region3 = cms.double( 0.7 ),
    selePtEtaEndCap_region3 = cms.double( 3.0 ),
    seleS4S9GammaEtaEndCap = cms.double( 0.9 ),
    seleS9S25GammaEtaEndCap = cms.double( 0.85 ),
    seleMinvMaxEtaEndCap = cms.double( 0.8 ),
    seleMinvMinEtaEndCap = cms.double( 0.3 ),
    ptMinForIsolationEtaEndCap = cms.double( 0.5 ),
    seleEtaIsoEndCap = cms.double( 0.5 ),
    seleEtaBeltDREndCap = cms.double( 0.3 ),
    seleEtaBeltDetaEndCap = cms.double( 0.1 ),
    storeIsoClusRecHitEtaEE = cms.bool( True ),
    etaEndcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
    store5x5RecHitEtaEE = cms.bool( True ),
    store5x5IsoClusRecHitEtaEE = cms.bool( True ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    storeRecHitES = cms.bool( True ),
    preshNclust = cms.int32( 4 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    addESEnergyToEECluster = cms.bool( True ),
    debugLevelES = cms.string( " " ),
    pi0ESCollection = cms.string( "pi0EcalRecHitsES" ),
    etaESCollection = cms.string( "etaEcalRecHitsES" ),
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 7.4 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sAlCaEcalPi0" ),
    debugLevel = cms.int32( 0 ),
    RegionalMatch = cms.untracked.bool( False ),
    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 )
)
hltL1sAlCaEcalEta = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleIsoEG5 OR L1_SingleIsoEG8 OR L1_SingleIsoEG10 OR L1_SingleIsoEG12 OR L1_SingleIsoEG15 OR L1_SingleEG1 OR L1_SingleEG2 OR L1_SingleEG5 OR L1_SingleEG8 OR L1_SingleEG10 OR L1_SingleEG12 OR L1_SingleEG15 OR L1_SingleEG20" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreAlCaEcalEta = cms.EDFilter( "HLTPrescaler" )
hltAlCaEtaRegRecHits = cms.EDFilter( "HLTPi0RecHitsFilter",
    barrelHits = cms.InputTag( 'hltEcalRegionalEtaRecHit','EcalRecHitsEB' ),
    endcapHits = cms.InputTag( 'hltEcalRegionalEtaRecHit','EcalRecHitsEE' ),
    clusSeedThr = cms.double( 0.5 ),
    clusSeedThrEndCap = cms.double( 1.0 ),
    clusEtaSize = cms.int32( 3 ),
    clusPhiSize = cms.int32( 3 ),
    seleNRHMax = cms.int32( 1000 ),
    seleXtalMinEnergy = cms.double( -0.15 ),
    seleXtalMinEnergyEndCap = cms.double( -0.75 ),
    doSelForPi0Barrel = cms.bool( False ),
    selePtGamma = cms.double( 1.0 ),
    selePtPi0 = cms.double( 2.0 ),
    seleMinvMaxPi0 = cms.double( 0.22 ),
    seleMinvMinPi0 = cms.double( 0.06 ),
    seleS4S9Gamma = cms.double( 0.83 ),
    selePi0Iso = cms.double( 0.5 ),
    ptMinForIsolation = cms.double( 1.0 ),
    selePi0BeltDR = cms.double( 0.2 ),
    selePi0BeltDeta = cms.double( 0.05 ),
    storeIsoClusRecHitPi0EB = cms.bool( True ),
    pi0BarrelHitCollection = cms.string( "pi0EcalRecHitsEB" ),
    doSelForPi0Endcap = cms.bool( False ),
    region1_Pi0EndCap = cms.double( 2.0 ),
    selePtGammaPi0EndCap_region1 = cms.double( 0.7 ),
    selePtPi0EndCap_region1 = cms.double( 3.0 ),
    region2_Pi0EndCap = cms.double( 2.5 ),
    selePtGammaPi0EndCap_region2 = cms.double( 0.5 ),
    selePtPi0EndCap_region2 = cms.double( 2.0 ),
    selePtGammaPi0EndCap_region3 = cms.double( 0.3 ),
    selePtPi0EndCap_region3 = cms.double( 1.2 ),
    seleS4S9GammaEndCap = cms.double( 0.9 ),
    seleMinvMaxPi0EndCap = cms.double( 0.3 ),
    seleMinvMinPi0EndCap = cms.double( 0.05 ),
    ptMinForIsolationEndCap = cms.double( 0.5 ),
    selePi0BeltDREndCap = cms.double( 0.2 ),
    selePi0BeltDetaEndCap = cms.double( 0.05 ),
    selePi0IsoEndCap = cms.double( 0.5 ),
    storeIsoClusRecHitPi0EE = cms.bool( True ),
    pi0EndcapHitCollection = cms.string( "pi0EcalRecHitsEE" ),
    doSelForEtaBarrel = cms.bool( True ),
    selePtGammaEta = cms.double( 1.2 ),
    selePtEta = cms.double( 4.0 ),
    seleS4S9GammaEta = cms.double( 0.9 ),
    seleS9S25GammaEta = cms.double( 0.8 ),
    seleMinvMaxEta = cms.double( 0.8 ),
    seleMinvMinEta = cms.double( 0.3 ),
    ptMinForIsolationEta = cms.double( 1.0 ),
    seleEtaIso = cms.double( 0.5 ),
    seleEtaBeltDR = cms.double( 0.3 ),
    seleEtaBeltDeta = cms.double( 0.1 ),
    storeIsoClusRecHitEtaEB = cms.bool( True ),
    removePi0CandidatesForEta = cms.bool( True ),
    massLowPi0Cand = cms.double( 0.104 ),
    massHighPi0Cand = cms.double( 0.163 ),
    etaBarrelHitCollection = cms.string( "etaEcalRecHitsEB" ),
    store5x5RecHitEtaEB = cms.bool( True ),
    store5x5IsoClusRecHitEtaEB = cms.bool( True ),
    doSelForEtaEndcap = cms.bool( True ),
    region1_EtaEndCap = cms.double( 2.0 ),
    selePtGammaEtaEndCap_region1 = cms.double( 1.3 ),
    selePtEtaEndCap_region1 = cms.double( 5.0 ),
    region2_EtaEndCap = cms.double( 2.5 ),
    selePtGammaEtaEndCap_region2 = cms.double( 1.0 ),
    selePtEtaEndCap_region2 = cms.double( 4.0 ),
    selePtGammaEtaEndCap_region3 = cms.double( 0.7 ),
    selePtEtaEndCap_region3 = cms.double( 3.0 ),
    seleS4S9GammaEtaEndCap = cms.double( 0.9 ),
    seleS9S25GammaEtaEndCap = cms.double( 0.85 ),
    seleMinvMaxEtaEndCap = cms.double( 0.8 ),
    seleMinvMinEtaEndCap = cms.double( 0.3 ),
    ptMinForIsolationEtaEndCap = cms.double( 0.5 ),
    seleEtaIsoEndCap = cms.double( 0.5 ),
    seleEtaBeltDREndCap = cms.double( 0.3 ),
    seleEtaBeltDetaEndCap = cms.double( 0.1 ),
    storeIsoClusRecHitEtaEE = cms.bool( True ),
    etaEndcapHitCollection = cms.string( "etaEcalRecHitsEE" ),
    store5x5RecHitEtaEE = cms.bool( True ),
    store5x5IsoClusRecHitEtaEE = cms.bool( True ),
    preshRecHitProducer = cms.InputTag( 'hltEcalPreshowerRecHit','EcalRecHitsES' ),
    storeRecHitES = cms.bool( True ),
    preshNclust = cms.int32( 4 ),
    preshClusterEnergyCut = cms.double( 0.0 ),
    preshStripEnergyCut = cms.double( 0.0 ),
    preshSeededNstrip = cms.int32( 15 ),
    preshCalibPlaneX = cms.double( 1.0 ),
    preshCalibPlaneY = cms.double( 0.7 ),
    preshCalibGamma = cms.double( 0.024 ),
    preshCalibMIP = cms.double( 9.0E-5 ),
    addESEnergyToEECluster = cms.bool( True ),
    debugLevelES = cms.string( " " ),
    pi0ESCollection = cms.string( "pi0EcalRecHitsES" ),
    etaESCollection = cms.string( "etaEcalRecHitsES" ),
    ParameterLogWeighted = cms.bool( True ),
    ParameterX0 = cms.double( 0.89 ),
    ParameterT0_barl = cms.double( 7.4 ),
    ParameterT0_endc = cms.double( 3.1 ),
    ParameterT0_endcPresh = cms.double( 1.2 ),
    ParameterW0 = cms.double( 4.2 ),
    l1IsolatedTag = cms.InputTag( 'l1extraParticles','Isolated' ),
    l1NonIsolatedTag = cms.InputTag( 'l1extraParticles','NonIsolated' ),
    l1SeedFilterTag = cms.InputTag( "hltL1sAlCaEcalEta" ),
    debugLevel = cms.int32( 0 ),
    RegionalMatch = cms.untracked.bool( False ),
    ptMinEMObj = cms.double( 2.0 ),
    EMregionEtaMargin = cms.double( 0.25 ),
    EMregionPhiMargin = cms.double( 0.4 )
)
hltL1sL1Mu14L1SingleEG10 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu14 AND L1_SingleEG10" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Mu14L1SingleEG10 = cms.EDFilter( "HLTPrescaler" )
hltL1sL1Mu14L1SingleJet6U = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu14 AND L1_SingleJet6U" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Mu14L1SingleJet6U = cms.EDFilter( "HLTPrescaler" )
hltL1sL1Mu14L1ETM30 = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding = cms.bool( False ),
    L1UseAliasesForSeeding = cms.bool( True ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleMu14 AND L1_ETM30" ),
    L1GtReadoutRecordTag = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag = cms.InputTag( "gtDigis" ),
    L1CollectionsTag = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag = cms.InputTag( "l1ParamMuons" )
)
hltPreL1Mu14L1ETM30 = cms.EDFilter( "HLTPrescaler" )
hltTriggerSummaryAOD = cms.EDProducer( "TriggerSummaryProducerAOD",
    processName = cms.string( "@" )
)
hltPreTriggerSummaryRAW = cms.EDFilter( "HLTPrescaler" )
hltTriggerSummaryRAW = cms.EDProducer( "TriggerSummaryProducerRAW",
    processName = cms.string( "@" )
)
hltBoolFinalPath = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)
hltTrigReport = cms.EDAnalyzer( "HLTrigReport",
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
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
    beamSpot = cms.InputTag( "offlineBeamSpot" )
)

HLTDoLocalHcalSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco + hltHoreco )
HLTDoCaloSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalRestFEDs + hltEcalRecHitAll + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForAll )
HLTDoJetRecoSequence = cms.Sequence( hltIterativeCone5CaloJets + hltMCJetCorJetIcone5 )
HLTDoHTRecoSequence = cms.Sequence( hltHtMet )
HLTRecoJetMETSequence = cms.Sequence( HLTDoCaloSequence + HLTDoJetRecoSequence + hltMet + HLTDoHTRecoSequence )
HLTL2muonrecoNocandSequence = cms.Sequence( cms.SequencePlaceholder("simMuonDTDigis") + hltDt1DRecHits + hltDt4DSegments + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltCscSegments + cms.SequencePlaceholder("simMuonRPCDigis") + hltRpcRecHits + hltL2MuonSeeds + hltL2Muons )
HLTL2muonrecoSequence = cms.Sequence( HLTL2muonrecoNocandSequence + hltL2MuonCandidates )
HLTL2muonisorecoSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalMuonsFEDs + hltEcalRegionalMuonsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForMuons + hltL2MuonIsolations )
HLTL3muonTkCandidateSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL3TrajectorySeed + hltL3TrackCandidateFromL2 )
HLTL3muonrecoNocandSequence = cms.Sequence( HLTL3muonTkCandidateSequence + hltL3TkTracksFromL2 + hltL3Muons )
HLTL3muonrecoSequence = cms.Sequence( HLTL3muonrecoNocandSequence + hltL3MuonCandidates )
HLTL3muonisorecoSequence = cms.Sequence( hltPixelTracks + hltL3MuonIsolations )
HLTDoRegionalEgammaEcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalEgammaFEDs + hltEcalRegionalEgammaRecHit + hltEcalPreshowerRecHit )
HLTMulti5x5SuperClusterL1Isolated = cms.Sequence( hltMulti5x5BasicClustersL1Isolated + hltMulti5x5SuperClustersL1Isolated + hltMulti5x5EndcapSuperClustersWithPreshowerL1Isolated + hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Isolated )
HLTL1IsolatedEcalClustersSequence = cms.Sequence( hltHybridSuperClustersL1Isolated + hltCorrectedHybridSuperClustersL1Isolated + HLTMulti5x5SuperClusterL1Isolated )
HLTMulti5x5SuperClusterL1NonIsolated = cms.Sequence( hltMulti5x5BasicClustersL1NonIsolated + hltMulti5x5SuperClustersL1NonIsolated + hltMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated + hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolatedTemp + hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1NonIsolated )
HLTL1NonIsolatedEcalClustersSequence = cms.Sequence( hltHybridSuperClustersL1NonIsolated + hltCorrectedHybridSuperClustersL1NonIsolatedTemp + hltCorrectedHybridSuperClustersL1NonIsolated + HLTMulti5x5SuperClusterL1NonIsolated )
HLTDoLocalHcalWithoutHOSequence = cms.Sequence( hltHcalDigis + hltHbhereco + hltHfreco )
HLTSingleElectronLWEt10L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt10HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt10PixelMatchFilter )
HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoLargeWindowElectronPixelSeeds + hltL1NonIsoLargeWindowElectronPixelSeeds + hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter )
HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional + hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedElectronHcalIsol + hltL1NonIsolatedElectronHcalIsol + hltL1NonIsoHLTNonIsoDoubleElectronEt5HcalIsolFilter + HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltL1IsoStartUpElectronPixelSeeds + hltL1NonIsoStartUpElectronPixelSeeds + hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter )
HLTSinglePhoton10L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt10L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt10EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt10HcalIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter )
HLTDoLocalTrackerSequence = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence )
HLTSinglePhoton15L1NonIsolatedHLTHTISequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonEt15HTIL1MatchFilterRegional + hltL1NonIsoSinglePhotonEt15HTIEtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonEt15HTIHcalIsolFilter + HLTDoLocalTrackerSequence + HLTL1IsoEgammaRegionalRecoTrackerSequence + HLTL1NonIsoEgammaRegionalRecoTrackerSequence + hltL1IsoPhotonHollowTrackIsol + hltL1NonIsoPhotonHollowTrackIsol + hltL1NonIsoSinglePhotonEt15HTITrackIsolFilter )
HLTSinglePhoton15L1NonIsolatedHLTLEISequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoSinglePhotonEt15LEIL1MatchFilterRegional + hltL1NonIsoSinglePhotonEt15LEIEtFilter + hltL1IsolatedPhotonEcalIsol + hltL1NonIsolatedPhotonEcalIsol + hltL1NonIsoSinglePhotonEt15LEIEcalIsolFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoSinglePhotonEt15LEIHcalIsolFilter )
HLTSinglePhoton20L1NonIsolatedHLTNonIsoSequence = cms.Sequence( HLTDoRegionalEgammaEcalSequence + HLTL1IsolatedEcalClustersSequence + HLTL1NonIsolatedEcalClustersSequence + hltL1IsoRecoEcalCandidate + hltL1NonIsoRecoEcalCandidate + hltL1NonIsoHLTNonIsoSinglePhotonEt20L1MatchFilterRegional + hltL1NonIsoHLTNonIsoSinglePhotonEt20EtFilter + HLTDoLocalHcalWithoutHOSequence + hltL1IsolatedPhotonHcalIsol + hltL1NonIsolatedPhotonHcalIsol + hltL1NonIsoHLTNonIsoSinglePhotonEt20HcalIsolFilter )
HLTCaloTausCreatorRegionalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalJetsFEDs + hltEcalRegionalJetsRecHit + hltEcalPreshowerRecHit + HLTDoLocalHcalSequence + hltTowerMakerForJets + hltCaloTowersTau1Regional + hltIconeTau1Regional + hltCaloTowersTau2Regional + hltIconeTau2Regional + hltCaloTowersTau3Regional + hltIconeTau3Regional + hltCaloTowersTau4Regional + hltIconeTau4Regional + hltCaloTowersCentral1Regional + hltIconeCentral1Regional + hltCaloTowersCentral2Regional + hltIconeCentral2Regional + hltCaloTowersCentral3Regional + hltIconeCentral3Regional + hltCaloTowersCentral4Regional + hltIconeCentral4Regional )
HLTL2TauJetsSequence = cms.Sequence( HLTCaloTausCreatorRegionalSequence + hltL2TauJets )
HLTL2TauEcalIsolationSequence = cms.Sequence( hltL2TauNarrowConeIsolationProducer + hltL2TauRelaxingIsolationSelector )
HLTBTagMuSequenceL25U = cms.Sequence( HLTL2muonrecoNocandSequence + hltSelector4JetsU + hltBSoftMuonL25JetsU + hltBSoftMuonL25TagInfosU + hltBSoftMuonL25BJetTagsUByDR )
HLTBTagMuSequenceL3U = cms.Sequence( HLTL3muonrecoNocandSequence + hltBSoftMuonL3TagInfosU + hltBSoftMuonL3BJetTagsUByPt + hltBSoftMuonL3BJetTagsUByDR )
HLTBTagIPSequenceL25StartupU = cms.Sequence( HLTDoLocalPixelSequence + HLTRecopixelvertexingSequence + hltSelector4JetsU + hltBLifetimeL25JetsStartupU + hltBLifetimeL25AssociatorStartupU + hltBLifetimeL25TagInfosStartupU + hltBLifetimeL25BJetTagsStartupU )
HLTBTagIPSequenceL3StartupU = cms.Sequence( HLTDoLocalPixelSequence + HLTDoLocalStripSequence + hltBLifetimeL3JetsStartupU + hltBLifetimeRegionalPixelSeedGeneratorStartupU + hltBLifetimeRegionalCkfTrackCandidatesStartupU + hltBLifetimeRegionalCtfWithMaterialTracksStartupU + hltBLifetimeL3AssociatorStartupU + hltBLifetimeL3TagInfosStartupU + hltBLifetimeL3BJetTagsStartupU )
HLTDoRegionalPi0EcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalPi0FEDs + hltEcalRegionalPi0RecHit + hltEcalPreshowerRecHit )
HLTDoRegionalEtaEcalSequence = cms.Sequence( hltEcalPreshowerDigis + hltEcalRawToRecHitFacility + hltEcalRegionalEtaFEDs + hltEcalRegionalEtaRecHit + hltEcalPreshowerRecHit )

HLTriggerFirstPath = cms.Path( HLTBeginSequence + hltGetRaw + hltPreFirstPath + hltBoolFirstPath )
HLT_L1Jet6U = cms.Path( HLTBeginSequence + hltL1sL1Jet6U + hltPreL1Jet6U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet15U = cms.Path( HLTBeginSequence + hltL1sJet15U + hltPreJet15U + HLTRecoJetMETSequence + hlt1jet15U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet30U = cms.Path( HLTBeginSequence + hltL1sJet30U + hltPreJet30U + HLTRecoJetMETSequence + hlt1jet30U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Jet50U = cms.Path( HLTBeginSequence + hltL1sJet50U + hltPreJet50U + HLTRecoJetMETSequence + hlt1jet50U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_FwdJet20U = cms.Path( HLTBeginSequence + hltL1sFwdJet20U + hltPreFwdJet20U + HLTRecoJetMETSequence + hltRapGap20U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve15U_8E29 = cms.Path( HLTBeginSequence + hltL1sDiJetAve15U8E29 + hltPreDiJetAve15U8E29 + HLTRecoJetMETSequence + hltDiJetAve15U8E29 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DiJetAve30U_8E29 = cms.Path( hltL1sDiJetAve30U8E29 + hltPreDiJetAve30U8E29 + HLTRecoJetMETSequence + hltDiJetAve30U8E29 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_QuadJet15U = cms.Path( HLTBeginSequence + hltL1sQuadJet15U + hltPreQuadJet15U + HLTRecoJetMETSequence + hlt4jet15U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1MET20 = cms.Path( HLTBeginSequence + hltL1sL1MET20 + hltPreL1MET20 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET35 = cms.Path( HLTBeginSequence + hltL1sMET35 + hltPreMET35 + HLTRecoJetMETSequence + hlt1MET35 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MET100 = cms.Path( HLTBeginSequence + hltL1sMET100 + hltPreMET100 + HLTRecoJetMETSequence + hlt1MET100 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1MuOpen = cms.Path( HLTBeginSequence + hltL1sL1MuOpen + hltPreL1MuOpen + hltL1MuOpenL1Filtered0 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu = cms.Path( HLTBeginSequence + hltL1sL1Mu + hltPreL1Mu + hltL1MuL1Filtered0 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu20 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu20 + hltPreL1Mu20 + hltL1Mu20L1Filtered20 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L2Mu9 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu7 + hltPreL2Mu9 + hltL1SingleMu7L1Filtered0 + HLTL2muonrecoSequence + hltL2Mu9L2Filtered9 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L2Mu11 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu7 + hltPreL2Mu11 + hltL1SingleMu7L1Filtered0 + HLTL2muonrecoSequence + hltL2Mu11L2Filtered11 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_IsoMu3 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3 + hltPreIsoMu3 + hltSingleMuIsoL1Filtered3 + HLTL2muonrecoSequence + hltSingleMuIsoL2PreFiltered3 + HLTL2muonisorecoSequence + hltSingleMuIsoL2IsoFiltered3 + HLTL3muonrecoSequence + hltSingleMuIsoL3PreFiltered3 + HLTL3muonisorecoSequence + hltSingleMuIsoL3IsoFiltered3 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu3 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3 + hltPreMu3 + hltL1SingleMu3L1Filtered0 + HLTL2muonrecoSequence + hltSingleMu3L2Filtered3 + HLTL3muonrecoSequence + hltSingleMu3L3Filtered3 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu5 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu3 + hltPreMu5 + hltL1SingleMu3L1Filtered0 + HLTL2muonrecoSequence + hltSingleMu5L2Filtered4 + HLTL3muonrecoSequence + hltSingleMu5L3Filtered5 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Mu9 = cms.Path( HLTBeginSequence + hltL1sL1SingleMu7 + hltPreMu9 + hltL1SingleMu7L1Filtered0 + HLTL2muonrecoSequence + hltSingleMu9L2Filtered7 + HLTL3muonrecoSequence + hltSingleMu9L3Filtered9 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1DoubleMuOpen = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpen + hltPreL1DoubleMuOpen + hltDoubleMuLevel1PathL1OpenFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu0 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMuOpen + hltPreDoubleMu0 + hltDiMuonL1Filtered0 + HLTL2muonrecoSequence + hltDiMuonL2PreFiltered0 + HLTL3muonrecoSequence + hltDiMuonL3PreFiltered0 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleMu3 = cms.Path( HLTBeginSequence + hltL1sL1DoubleMu3 + hltPreDoubleMu3 + hltDiMuonL1Filtered + HLTL2muonrecoSequence + hltDiMuonL2PreFiltered + HLTL3muonrecoSequence + hltDiMuonL3PreFiltered + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1SingleEG5 = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt5 + hltPreL1SingleEG5 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1SingleEG8 = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt8 + hltPreL1SingleEG8 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele10_LW_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt8 + hltPreEle10LWL1R + HLTSingleElectronLWEt10L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele15_LW_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPreEle15LWL1R + HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1DoubleEG5 = cms.Path( HLTBeginSequence + hltL1sRelaxedDoubleEgammaEt5 + hltPreL1DoubleEG5 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleEle5_SW_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedDoubleEgammaEt5 + hltPreDoubleEle5SWL1R + HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele15_SC10_LW_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPreEle15SC10LWL1R + HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESscWrapper + hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Ele25_LW_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPreEle25LWL1R + HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence + hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilterESet25 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon10_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt8 + hltPrePhoton10L1R + HLTSinglePhoton10L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon15_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPrePhoton15L1R + HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon15_TrackIso_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPrePhoton15HTIL1R + HLTSinglePhoton15L1NonIsolatedHLTHTISequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon15_LooseEcalIso_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPrePhoton15LEIL1R + HLTSinglePhoton15L1NonIsolatedHLTLEISequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon20_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPrePhoton20L1R + HLTSinglePhoton20L1NonIsolatedHLTNonIsoSequence + cms.SequencePlaceholder("HLTEndSequence") )
HLT_Photon30_L1R = cms.Path( HLTBeginSequence + hltL1sRelaxedSingleEgammaEt10 + hltPrePhoton30L1R + HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence + hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilterESet30 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_SingleLooseIsoTau20 = cms.Path( HLTBeginSequence + hltL1sSingleLooseIsoTau20 + hltPreSingleLooseIsoTau20 + HLTL2TauJetsSequence + hltFilterL2EtCutSingleLooseIsoTau20 + HLTL2TauEcalIsolationSequence + hltL1HLTSingleLooseIsoTau20JetsMatch + hltFilterL2EcalIsolationSingleLooseIsoTau20 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_DoubleLooseIsoTau15 = cms.Path( HLTBeginSequence + hltL1sDoubleLooseIsoTau15 + hltPreDoubleLooseIsoTau15 + HLTL2TauJetsSequence + hltFilterL2EtCutDoubleLooseIsoTau15 + HLTL2TauEcalIsolationSequence + hltL1HLTDoubleLooseIsoTau15JetsMatch + hltFilterL2EcalIsolationDoubleLooseIsoTau15 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagMu_Jet10U = cms.Path( HLTBeginSequence + hltL1sBTagMuJet10U + hltPreBTagMuJet10U + HLTRecoJetMETSequence + hltBJet10U + HLTBTagMuSequenceL25U + hltBSoftMuonL25FilterUByDR + HLTBTagMuSequenceL3U + hltBSoftMuonL3FilterUByDR + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BTagIP_Jet50U = cms.Path( HLTBeginSequence + hltL1sBTagIPJet50U + hltPreBTagIPJet50U + HLTRecoJetMETSequence + hltBJet50U + HLTBTagIPSequenceL25StartupU + hltBLifetimeL25FilterStartupU + HLTBTagIPSequenceL3StartupU + hltBLifetimeL3FilterStartupU + cms.SequencePlaceholder("HLTEndSequence") )
HLT_ZeroBias = cms.Path( HLTBeginSequence + hltL1sZeroBias + hltPreZeroBias + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasHcal = cms.Path( HLTBeginSequence + hltL1sMinBiasHcal + hltPreMinBiasHcal + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasEcal = cms.Path( HLTBeginSequence + hltL1sMinBiasEcal + hltPreMinBiasEcal + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasPixel = cms.Path( HLTBeginSequence + hltL1sMinBiasPixel + hltPreMinBiasPixel + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltMinBiasPixelFilter + cms.SequencePlaceholder("HLTEndSequence") )
HLT_MinBiasPixel_Trk5 = cms.Path( HLTBeginSequence + hltL1sMinBiasPixel + hltPreMinBiasPixelTrk5 + HLTDoLocalPixelSequence + HLTPixelTrackingForMinBiasSequence + hltPixelCands + hltPixelMBForAlignment + cms.SequencePlaceholder("HLTEndSequence") )
HLT_BackwardBSC = cms.Path( HLTBeginSequence + hltL1sBackwardBSC + hltPreBackwardBSC + cms.SequencePlaceholder("HLTEndSequence") )
HLT_ForwardBSC = cms.Path( HLTBeginSequence + hltL1sForwardBSC + hltPreForwardBSC + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHalo = cms.Path( HLTBeginSequence + hltL1sCSCBeamHalo + hltPreCSCBeamHalo + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloOverlapRing1 = cms.Path( HLTBeginSequence + hltL1sCSCBeamHaloOverlapRing1 + hltPreCSCBeamHaloOverlapRing1 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing1 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloOverlapRing2 = cms.Path( HLTBeginSequence + hltL1sCSCBeamHaloOverlapRing2 + hltPreCSCBeamHaloOverlapRing2 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltOverlapsHLTCSCBeamHaloOverlapRing2 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_CSCBeamHaloRing2or3 = cms.Path( HLTBeginSequence + hltL1sCSCBeamHaloRing2or3 + hltPreCSCBeamHaloRing2or3 + cms.SequencePlaceholder("simMuonCSCDigis") + hltCsc2DRecHits + hltFilter23HLTCSCBeamHaloRing2or3 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_TrackerCosmics = cms.Path( HLTBeginSequence + hltL1sTrackerCosmics + hltPreTrackerCosmics + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_EcalPhiSym = cms.Path( HLTBeginSequence + hltL1sAlCaEcalPhiSym + hltPreAlCaEcalPhiSym + hltEcalRawToRecHitFacility + hltEcalRegionalRestFEDs + hltEcalRecHitAll + hltAlCaPhiSymStream + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_EcalPi0 = cms.Path( HLTBeginSequence + hltL1sAlCaEcalPi0 + hltPreAlCaEcalPi0 + HLTDoRegionalPi0EcalSequence + hltAlCaPi0RegRecHits + cms.SequencePlaceholder("HLTEndSequence") )
AlCa_EcalEta = cms.Path( HLTBeginSequence + hltL1sAlCaEcalEta + hltPreAlCaEcalEta + HLTDoRegionalEtaEcalSequence + hltAlCaEtaRegRecHits + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu14_L1SingleEG10 = cms.Path( HLTBeginSequence + hltL1sL1Mu14L1SingleEG10 + hltPreL1Mu14L1SingleEG10 + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu14_L1SingleJet6U = cms.Path( HLTBeginSequence + hltL1sL1Mu14L1SingleJet6U + hltPreL1Mu14L1SingleJet6U + cms.SequencePlaceholder("HLTEndSequence") )
HLT_L1Mu14_L1ETM30 = cms.Path( HLTBeginSequence + hltL1sL1Mu14L1ETM30 + hltPreL1Mu14L1ETM30 + cms.SequencePlaceholder("HLTEndSequence") )
HLTriggerFinalPath = cms.Path( hltTriggerSummaryAOD + hltPreTriggerSummaryRAW + hltTriggerSummaryRAW + hltBoolFinalPath )
HLTAnalyzerEndpath = cms.EndPath( hltL1gtTrigReport + hltTrigReport )


HLTSchedule = cms.Schedule( HLTriggerFirstPath, HLT_L1Jet6U, HLT_Jet15U, HLT_Jet30U, HLT_Jet50U, HLT_FwdJet20U, HLT_DiJetAve15U_8E29, HLT_DiJetAve30U_8E29, HLT_QuadJet15U, HLT_L1MET20, HLT_MET35, HLT_MET100, HLT_L1MuOpen, HLT_L1Mu, HLT_L1Mu20, HLT_L2Mu9, HLT_L2Mu11, HLT_IsoMu3, HLT_Mu3, HLT_Mu5, HLT_Mu9, HLT_L1DoubleMuOpen, HLT_DoubleMu0, HLT_DoubleMu3, HLT_L1SingleEG5, HLT_L1SingleEG8, HLT_Ele10_LW_L1R, HLT_Ele15_LW_L1R, HLT_L1DoubleEG5, HLT_DoubleEle5_SW_L1R, HLT_Ele15_SC10_LW_L1R, HLT_Ele25_LW_L1R, HLT_Photon10_L1R, HLT_Photon15_L1R, HLT_Photon15_TrackIso_L1R, HLT_Photon15_LooseEcalIso_L1R, HLT_Photon20_L1R, HLT_Photon30_L1R, HLT_SingleLooseIsoTau20, HLT_DoubleLooseIsoTau15, HLT_BTagMu_Jet10U, HLT_BTagIP_Jet50U, HLT_ZeroBias, HLT_MinBiasHcal, HLT_MinBiasEcal, HLT_MinBiasPixel, HLT_MinBiasPixel_Trk5, HLT_BackwardBSC, HLT_ForwardBSC, HLT_CSCBeamHalo, HLT_CSCBeamHaloOverlapRing1, HLT_CSCBeamHaloOverlapRing2, HLT_CSCBeamHaloRing2or3, HLT_TrackerCosmics, AlCa_EcalPhiSym, AlCa_EcalPi0, AlCa_EcalEta, HLT_L1Mu14_L1SingleEG10, HLT_L1Mu14_L1SingleJet6U, HLT_L1Mu14_L1ETM30, HLTriggerFinalPath, HLTAnalyzerEndpath )
