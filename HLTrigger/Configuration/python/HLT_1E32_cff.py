import FWCore.ParameterSet.Config as cms

# /dev/CMSSW_2_0_0_pre8/HLT1032/V10  (CMSSW_2_0_0_pre8)
# Begin replace statements specific to the HLT
from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
hltTriggerSummaryAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    TriggerSummaryAOD,
    processName = cms.string('@')
)

BTagRecord = cms.ESSource("EmptyESSource",
    recordName = cms.string('JetTagComputerRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

DummyCorrector = cms.ESSource("SimpleJetCorrectionService",
    scale = cms.double(1.0),
    label = cms.string('DummyCorrector')
)

es_prefer_DummyCorrector = cms.ESPrefer("SimpleJetCorrectionService","DummyCorrector")
MCJetCorrectorIcone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_iterativeCone5'),
    label = cms.string('MCJetCorrectorIcone5')
)

MCJetCorrectorScone5 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone5'),
    label = cms.string('MCJetCorrectorScone5')
)

MCJetCorrectorScone7 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_sisCone7'),
    label = cms.string('MCJetCorrectorScone7')
)

MCJetCorrectorktjet4 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_fastjet4'),
    label = cms.string('MCJetCorrectorktjet4')
)

MCJetCorrectorktjet6 = cms.ESSource("MCJetCorrectionService",
    tagName = cms.string('CMSSW_152_fastjet6'),
    label = cms.string('MCJetCorrectorktjet6')
)

CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")

CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

Chi2EstimatorForL3Refit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForL3Refit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

Chi2EstimatorForMuonTrackLoader = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForMuonTrackLoader'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

Chi2EstimatorForRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

Chi2MeasurementEstimator = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)

CkfTrajectoryBuilder = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('CkfTrajectoryBuilder'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

FitterRK = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('FitterRK'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

FittingSmootherRK = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('FitterRK'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('SmootherRK'),
    ComponentName = cms.string('FittingSmootherRK'),
    RejectTracks = cms.bool(True)
)

FittingSmootherWithOutlierRejection = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('RKSmoother'),
    ComponentName = cms.string('FittingSmootherWithOutlierRejection'),
    RejectTracks = cms.bool(True)
)

GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

GroupedCkfTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('GroupedCkfTrajectoryBuilder'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5)
)

HcalTopologyIdealEP = cms.ESProducer("HcalTopologyIdealEP")

KFFitterForRefitInsideOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFFitterForRefitOutsideIn = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAnyOpposite'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmoother'),
    ComponentName = cms.string('KFFittingSmoother'),
    RejectTracks = cms.bool(True)
)

KFSmootherForMuonTrackLoader = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForMuonTrackLoader'),
    Estimator = cms.string('Chi2EstimatorForMuonTrackLoader'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

KFSmootherForRefitInsideOut = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

KFSmootherForRefitOutsideIn = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagator')
)

KFTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('PropagatorWithMaterial'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterial')
)

KFUpdatorESProducer = cms.ESProducer("KFUpdatorESProducer",
    ComponentName = cms.string('KFUpdator')
)

L3MuKFFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('L3MuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForL3Refit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

MaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('PropagatorWithMaterial'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)

MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(False),
    OnDemand = cms.bool(True),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string(''),
    stripClusterProducer = cms.string('hltSiStripClusters'),
    Regional = cms.bool(True),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    pixelClusterProducer = cms.string('hltSiPixelClusters'),
    stripLazyGetterProducer = cms.string('hltSiStripRawToClustersFacility'),
    UseStripModuleQualityDB = cms.bool(False),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)

MuonCkfTrajectoryBuilder = cms.ESProducer("MuonCkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('muonCkfTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('muonCkfTrajectoryBuilder'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    useSeedLayer = cms.bool(False),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    propagatorProximity = cms.string('SteppingHelixPropagatorAny'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    rescaleErrorIfFail = cms.double(1.0),
    intermediateCleaning = cms.bool(False),
    lostHitPenalty = cms.double(30.0)
)

MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer")

MuonTransientTrackingRecHitBuilderESProducer = cms.ESProducer("MuonTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('MuonRecHitBuilder')
)

OppositeMaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('PropagatorWithMaterialOpposite'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False)
)

PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    ComponentName = cms.string('PixelCPEGeneric'),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    PixelErrorParametrization = cms.string('NOTcmsim')
)

PixelCPEInitialESProducer = cms.ESProducer("PixelCPEInitialESProducer",
    ComponentName = cms.string('PixelCPEInitial'),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

PixelCPEParmErrorESProducer = cms.ESProducer("PixelCPEParmErrorESProducer",
    UseNewParametrization = cms.bool(True),
    ComponentName = cms.string('PixelCPEfromTrackAngle'),
    UseSigma = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

RKFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('RKSmoother'),
    ComponentName = cms.string('RKFittingSmoother'),
    RejectTracks = cms.bool(True)
)

RKTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RKTrackerPropagator'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(True)
)

RKTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('RKFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

RKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('RKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

RungeKuttaTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RungeKuttaTrackerPropagator'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(True)
)

SiStripRegionConnectivity = cms.ESProducer("SiStripRegionConnectivity",
    EtaDivisions = cms.untracked.uint32(20),
    PhiDivisions = cms.untracked.uint32(20),
    EtaMax = cms.untracked.double(2.5)
)

SmartPropagator = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagator'),
    TrackerPropagator = cms.string('PropagatorWithMaterial'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

SmartPropagatorAny = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAny'),
    TrackerPropagator = cms.string('PropagatorWithMaterial'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

SmartPropagatorAnyOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyOpposite'),
    TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    Epsilon = cms.double(5.0)
)

SmartPropagatorAnyRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyRK'),
    TrackerPropagator = cms.string('RKTrackerPropagator'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

SmartPropagatorOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorOpposite'),
    TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
    MuonPropagator = cms.string('SteppingHelixPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    Epsilon = cms.double(5.0)
)

SmartPropagatorRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorRK'),
    TrackerPropagator = cms.string('RKTrackerPropagator'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

SmootherRK = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('SmootherRK'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

SteppingHelixPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
    PropagationDirection = cms.string('alongMomentum'),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    sendLogWarning = cms.bool(False),
    useMatVolumes = cms.bool(True),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    ComponentName = cms.string('SteppingHelixPropagatorAlong')
)

SteppingHelixPropagatorAny = cms.ESProducer("SteppingHelixPropagatorESProducer",
    PropagationDirection = cms.string('anyDirection'),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    sendLogWarning = cms.bool(False),
    useMatVolumes = cms.bool(True),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    ComponentName = cms.string('SteppingHelixPropagatorAny')
)

SteppingHelixPropagatorOpposite = cms.ESProducer("SteppingHelixPropagatorESProducer",
    PropagationDirection = cms.string('oppositeToMomentum'),
    useTuningForL2Speed = cms.bool(False),
    useIsYokeFlag = cms.bool(True),
    NoErrorPropagation = cms.bool(False),
    SetVBFPointer = cms.bool(False),
    AssumeNoMaterial = cms.bool(False),
    returnTangentPlane = cms.bool(True),
    useInTeslaFromMagField = cms.bool(False),
    VBFName = cms.string('VolumeBasedMagneticField'),
    sendLogWarning = cms.bool(False),
    useMatVolumes = cms.bool(True),
    debug = cms.bool(False),
    ApplyRadX0Correction = cms.bool(True),
    useMagVolumes = cms.bool(True),
    ComponentName = cms.string('SteppingHelixPropagatorOpposite')
)

StripCPEESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentName = cms.string('SimpleStripCPE')
)

TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithAngleAndTemplate'),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    Matcher = cms.string('StandardMatcher')
)

TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")

TrajectoryBuilderForPixelMatchElectronsL1Iso = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('TrajectoryBuilderForPixelMatchElectronsL1Iso'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('egammaHLTChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

TrajectoryBuilderForPixelMatchElectronsL1IsoLargeWindow = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('TrajectoryBuilderForPixelMatchElectronsL1IsoLargeWindow'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('egammaHLTChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

TrajectoryBuilderForPixelMatchElectronsL1NonIso = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('TrajectoryBuilderForPixelMatchElectronsL1NonIso'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('egammaHLTChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

TrajectoryBuilderForPixelMatchElectronsL1NonIsoLargeWindow = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('TrajectoryBuilderForPixelMatchElectronsL1NonIsoLargeWindow'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('egammaHLTChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)

bJetRegionalTrajectoryBuilder = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('bJetRegionalTrajectoryFilter'),
    maxCand = cms.int32(1),
    ComponentName = cms.string('bJetRegionalTrajectoryBuilder'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(False),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

bJetRegionalTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(8),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(1.0)
    ),
    ComponentName = cms.string('bJetRegionalTrajectoryFilter')
)

beamHaloNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('BeamHaloNavigationSchool')
)

ckfBaseTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(0.9)
    ),
    ComponentName = cms.string('ckfBaseTrajectoryFilter')
)

compositeTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('compositeTrajectoryFilter'),
    filterNames = cms.vstring()
)

cosmicsNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('CosmicNavigationSchool')
)

egammaHLTChi2MeasurementEstimatorESProducer = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('egammaHLTChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(5.0)
)

hltCkfTrajectoryBuilderMumu = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('hltCkfTrajectoryFilterMumu'),
    maxCand = cms.int32(3),
    ComponentName = cms.string('hltCkfTrajectoryBuilderMumu'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(False),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

hltCkfTrajectoryBuilderMumuk = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('hltCkfTrajectoryFilterMumuk'),
    maxCand = cms.int32(3),
    ComponentName = cms.string('hltCkfTrajectoryBuilderMumuk'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(False),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

hltCkfTrajectoryFilterMumu = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(5),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(3.0)
    ),
    ComponentName = cms.string('hltCkfTrajectoryFilterMumu')
)

hltCkfTrajectoryFilterMumuk = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(5),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(3.0)
    ),
    ComponentName = cms.string('hltCkfTrajectoryFilterMumuk')
)

impactParameterMVAComputer = cms.ESProducer("GenericMVAJetTagESProducer",
    useCategories = cms.bool(False),
    calibrationRecord = cms.string('ImpactParameterMVA')
)

jetBProbability = cms.ESProducer("JetBProbabilityESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    trackIpSign = cms.int32(1),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    maximumDecayLength = cms.double(5.0)
)

jetProbability = cms.ESProducer("JetProbabilityESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(0.3),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    trackIpSign = cms.int32(1),
    minimumProbability = cms.double(0.005),
    maximumDecayLength = cms.double(5.0)
)

mixedlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    ComponentName = cms.string('MixedLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 
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
        'TEC2_neg+TEC3_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

mixedlayertriplets = cms.ESProducer("MixedLayerTripletsESProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg', 
        'BPix1+BPix2+TIB1', 
        'BPix1+BPix3+TIB1', 
        'BPix2+BPix3+TIB1'),
    ComponentName = cms.string('MixedLayerTriplets'),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

muonCkfTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(0.9)
    ),
    ComponentName = cms.string('muonCkfTrajectoryFilter')
)

muonRoadTrajectoryBuilderESProducer = cms.ESProducer("MuonRoadTrajectoryBuilderESProducer",
    minNumberOfHitOnCandidate = cms.uint32(4),
    maxTrajectories = cms.uint32(30),
    ComponentName = cms.string('muonRoadTrajectoryBuilder'),
    outputAllTraj = cms.bool(True),
    numberOfHitPerModuleThreshold = cms.vuint32(3, 1),
    measurementTrackerName = cms.string(''),
    dynamicMaxNumberOfHitPerModule = cms.bool(True),
    maxChi2Road = cms.double(40.0),
    maxChi2Hit = cms.double(40.0),
    propagatorName = cms.string('SteppingHelixPropagatorAny'),
    numberOfHitPerModule = cms.uint32(1000),
    maxTrajectoriesThreshold = cms.vuint32(10, 25)
)

myTTRHBuilderWithoutAngle = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('PixelTTRHBuilderWithoutAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

myTTRHBuilderWithoutAngle4MixedPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

myTTRHBuilderWithoutAngle4MixedTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

myTTRHBuilderWithoutAngle4PixelPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

myTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

navigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('SimpleNavigationSchool')
)

newTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('newTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('newTrajectoryBuilder'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    intermediateCleaning = cms.bool(True),
    minNrOfHitsForRebuild = cms.int32(5)
)

newTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(3),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(0.3)
    ),
    ComponentName = cms.string('newTrajectoryFilter')
)

pixellayerpairs = cms.ESProducer("PixelLayerPairsESProducer",
    ComponentName = cms.string('PixelLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 
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
        'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

pixellayertriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('PixelLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('hltSiPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

rings = cms.ESProducer("RingMakerESProducer",
    DetIdsDumpFileName = cms.untracked.string('tracker_detids.dat'),
    ComponentName = cms.string(''),
    RingAsciiFileName = cms.untracked.string('rings.dat')
)

roads = cms.ESProducer("RoadMapMakerESProducer",
    ComponentName = cms.string(''),
    SeedingType = cms.string('FourRingSeeds'),
    RingsLabel = cms.string(''),
    GeometryStructure = cms.string('FullDetector'),
    RoadMapAsciiFile = cms.untracked.string('roads.dat')
)

softLeptonByDistance = cms.ESProducer("LeptonTaggerByDistanceESProducer",
    distance = cms.double(0.5)
)

softLeptonByPt = cms.ESProducer("LeptonTaggerByPtESProducer")

templates = cms.ESProducer("PixelCPETemplateRecoESProducer",
    ComponentName = cms.string('PixelCPETemplateReco'),
    TanLorentzAnglePerTesla = cms.double(0.106),
    speed = cms.int32(0),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

trackCounting3D2nd = cms.ESProducer("TrackCountingESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(2)
)

trackCounting3D3rd = cms.ESProducer("TrackCountingESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(3)
)

trajBuilderL25 = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('trajFilterL25'),
    maxCand = cms.int32(1),
    ComponentName = cms.string('trajBuilderL25'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(False),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

trajBuilderL3 = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('trajFilterL3'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('trajBuilderL3'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(False),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0)
)

trajFilterL25 = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(7),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(5.0)
    ),
    ComponentName = cms.string('trajFilterL25')
)

trajFilterL3 = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        minimumNumberOfHits = cms.int32(5),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(7),
        maxConsecLostHits = cms.int32(1),
        chargeSignificance = cms.double(-1.0),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(0.9)
    ),
    ComponentName = cms.string('trajFilterL3')
)

trajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerBySharedHits')
)

ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('WithoutRefit'),
    PixelCPE = cms.string('Fake'),
    Matcher = cms.string('Fake')
)

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

hlt2GetRaw = cms.EDFilter("HLTGetRaw",
    RawDataCollection = cms.InputTag("rawDataCollector")
)

hltGtDigis = cms.EDFilter("L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32(813),
    DaqGtInputTag = cms.InputTag("rawDataCollector"),
    UnpackBxInEvent = cms.int32(1),
    ActiveBoardsMask = cms.uint32(257)
)

hltGctDigis = cms.EDFilter("GctRawToDigi",
    inputLabel = cms.InputTag("rawDataCollector"),
    unpackFibres = cms.untracked.bool(False),
    grenCompatibilityMode = cms.bool(False),
    gctFedId = cms.int32(745),
    unpackInternEm = cms.untracked.bool(False),
    hltMode = cms.bool(False)
)

hltL1GtObjectMap = cms.EDFilter("L1GlobalTrigger",
    ProduceL1GtDaqRecord = cms.bool(False),
    ReadTechnicalTriggerRecords = cms.bool(True),
    ProduceL1GtEvmRecord = cms.bool(False),
    GmtInputTag = cms.InputTag("hltGtDigis"),
    TechnicalTriggersInputTag = cms.InputTag("techTrigDigis"),
    ProduceL1GtObjectMapRecord = cms.bool(True),
    GctInputTag = cms.InputTag("hltGctDigis"),
    WritePsbL1GtDaqRecord = cms.bool(False)
)

hltL1extraParticles = cms.EDProducer("L1ExtraParticlesProd",
    muonSource = cms.InputTag("hltGtDigis"),
    etTotalSource = cms.InputTag("hltGctDigis"),
    nonIsolatedEmSource = cms.InputTag("hltGctDigis","nonIsoEm"),
    etMissSource = cms.InputTag("hltGctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("hltGctDigis","forJets"),
    centralJetSource = cms.InputTag("hltGctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("hltGctDigis","tauJets"),
    isolatedEmSource = cms.InputTag("hltGctDigis","isoEm"),
    etHadSource = cms.InputTag("hltGctDigis")
)

hltOfflineBeamSpot = cms.EDProducer("BeamSpotProducer")

hltL1s1jet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltEcalPreshowerDigis = cms.EDFilter("ESRawToDigi",
    InstanceES = cms.string(''),
    ESdigiCollection = cms.string(''),
    Label = cms.string('rawDataCollector')
)

hltEcalRegionalJetsFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    Jets = cms.untracked.bool(True),
    OutputLabel = cms.untracked.string(''),
    ForwardSource = cms.untracked.InputTag("hltL1extraParticles","Forward"),
    TauSource = cms.untracked.InputTag("hltL1extraParticles","Tau"),
    CentralSource = cms.untracked.InputTag("hltL1extraParticles","Central"),
    Ptmin_jets = cms.untracked.double(50.0),
    debug = cms.untracked.bool(False)
)

hltEcalRegionalJetsDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.string('hltEcalRegionalJetsFEDs'),
    syncCheck = cms.untracked.bool(False),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    DoRegional = cms.untracked.bool(True)
)

hltEcalRegionalJetsWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalRegionalJetsDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalRegionalJetsDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRegionalJetsRecHitTmp = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalRegionalJetsWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalRegionalJetsWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltEcalRegionalJetsRecHit = cms.EDFilter("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)

hltEcalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
    ESrechitCollection = cms.string('EcalRecHitsES'),
    ESdigiCollection = cms.InputTag("hltEcalPreshowerDigis")
)

hltHcalDigis = cms.EDFilter("HcalRawToDigi",
    UnpackZDC = cms.untracked.bool(True),
    FilterDataQuality = cms.bool(True),
    ExceptionEmptyData = cms.untracked.bool(False),
    InputLabel = cms.InputTag("rawDataCollector"),
    UnpackCalib = cms.untracked.bool(True),
    lastSample = cms.int32(9),
    firstSample = cms.int32(0)
)

hltHbhereco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hltHcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HBHE'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True)
)

hltHfreco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hltHcalDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('HF'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False)
)

hltHoreco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hltHcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HO'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True)
)

hltTowerMakerForJets = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRegionalJetsRecHit","EcalRecHitsEB"), cms.InputTag("hltEcalRegionalJetsRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltCaloTowersForJets = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("hltTowerMakerForJets"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

hltIterativeCone5CaloJetsRegional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersForJets"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltMCJetCorJetIcone5Regional = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("hltIterativeCone5CaloJetsRegional"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('corJetIcone5')
)

hlt1jet200 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(200.0),
    MinN = cms.int32(1)
)

hltBoolEnd = cms.EDFilter("HLTBool",
    result = cms.bool(True)
)

hltL1s2jet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt2jet150 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(150.0),
    MinN = cms.int32(2)
)

hltL1s3jet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre3jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt3jet85 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(85.0),
    MinN = cms.int32(3)
)

hltL1s4jet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre4jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltEcalRegionalRestFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    debug = cms.untracked.bool(False),
    OutputLabel = cms.untracked.string('')
)

hltEcalRegionalRestDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.string('hltEcalRegionalRestFEDs'),
    syncCheck = cms.untracked.bool(False),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    DoRegional = cms.untracked.bool(True)
)

hltEcalRegionalRestWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalRegionalRestDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalRegionalRestDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRegionalRestRecHitTmp = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalRegionalRestWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalRegionalRestWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltEcalRecHitAll = cms.EDFilter("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)

hltTowerMakerForAll = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"), cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltCaloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("hltTowerMakerForAll"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

hltIterativeCone5CaloJets = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowers"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltMCJetCorJetIcone5 = cms.EDProducer("JetCorrectionProducer",
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    correctors = cms.vstring('MCJetCorrectorIcone5'),
    alias = cms.untracked.string('MCJetCorJetIcone5')
)

hltMet = cms.EDProducer("METProducer",
    src = cms.InputTag("hltCaloTowers"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection')
)

hltHtMet = cms.EDProducer("METProducer",
    src = cms.InputTag("hltMCJetCorJetIcone5"),
    METType = cms.string('MET'),
    alias = cms.string('HTMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

hlt4jet60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(4)
)

hltL1s1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1MET65 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(65.0),
    MinN = cms.int32(1)
)

hltL1s2jetAco = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jetAco = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt2jet125 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(125.0),
    MinN = cms.int32(2)
)

hlt2jetAco = cms.EDFilter("HLT2JetJet",
    MinMinv = cms.double(0.0),
    MinN = cms.int32(1),
    MaxMinv = cms.double(-1.0),
    MinDeta = cms.double(0.0),
    inputTag1 = cms.InputTag("hlt2jet125"),
    inputTag2 = cms.InputTag("hlt2jet125"),
    MaxDphi = cms.double(2.1),
    MaxDeta = cms.double(-1.0),
    MinDphi = cms.double(0.0)
)

hltL1s1jet1METAco = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet1METAco = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1MET60 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(1)
)

hlt1jet100 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(1)
)

hlt1jet1METAco = cms.EDFilter("HLT2JetMET",
    MinMinv = cms.double(0.0),
    MinN = cms.int32(1),
    MaxMinv = cms.double(-1.0),
    MinDeta = cms.double(0.0),
    inputTag1 = cms.InputTag("hlt1jet100"),
    inputTag2 = cms.InputTag("hlt1MET60"),
    MaxDphi = cms.double(2.1),
    MaxDeta = cms.double(-1.0),
    MinDphi = cms.double(0.0)
)

hltL1s1jet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet1MET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1jet180 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(180.0),
    MinN = cms.int32(1)
)

hltL1s2jet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jet1MET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1s3jet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre3jet1MET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt3jet60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(3)
)

hltL1s4jet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre4jet1MET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt4jet35 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(4)
)

hltL1s1MET1HT = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_HTT300'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET1HT = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1HT350 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(350.0)
)

hltL1s1SumET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETT60'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET1SumET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1SumET120 = cms.EDFilter("HLTGlobalSumMET",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinN = cms.int32(1),
    Min = cms.double(120.0)
)

hltL1s1jetPE1 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet100'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jetPE1 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(10)
)

hlt1jet150 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(150.0),
    MinN = cms.int32(1)
)

hltL1s1jetPE3 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jetPE3 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1jet110 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(110.0),
    MinN = cms.int32(1)
)

hltL1s1jetPE5 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jetPE5 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1jet60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(1)
)

hltL1s1jetPE7 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jetPE7 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1jet30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(1)
)

hltL1s1METPre1 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1METPre1 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(100)
)

hlt1MET55 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(55.0),
    MinN = cms.int32(1)
)

hltL1s1METPre2 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1METPre2 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1MET30 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(1)
)

hltL1s1METPre3 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1METPre3 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1MET20 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(20.0),
    MinN = cms.int32(1)
)

hltL1sdijetave30 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave30 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltdijetave30 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltMCJetCorJetIcone5"),
    minEtAve = cms.double(30.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave60 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave60 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltdijetave60 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltMCJetCorJetIcone5"),
    minEtAve = cms.double(60.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave110 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave110 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltdijetave110 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltMCJetCorJetIcone5"),
    minEtAve = cms.double(110.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave150 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet100'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave150 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(10)
)

hltdijetave150 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltMCJetCorJetIcone5"),
    minEtAve = cms.double(150.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave200 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave200 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltdijetave200 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltMCJetCorJetIcone5"),
    minEtAve = cms.double(200.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1s2jetvbfMET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jetvbfMET = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt2jetvbf = cms.EDFilter("HLTJetVBFFilter",
    minDeltaEta = cms.double(2.5),
    minEt = cms.double(40.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5")
)

hltL1snvMET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrenv = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltnv = cms.EDFilter("HLTNVFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtJet2 = cms.double(20.0),
    minEtJet1 = cms.double(80.0),
    minNV = cms.double(0.1),
    inputMETTag = cms.InputTag("hlt1MET60")
)

hltL1sPhi2MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrephi2met = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPhi2metAco = cms.EDFilter("HLTPhi2METFilter",
    maxDeltaPhi = cms.double(3.1514),
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    inputMETTag = cms.InputTag("hlt1MET60"),
    minDeltaPhi = cms.double(0.377),
    minEtJet1 = cms.double(60.0),
    minEtJet2 = cms.double(60.0)
)

hltL1sPhiJet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrephijet1met = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hlt1MET70 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(70.0),
    MinN = cms.int32(1)
)

hltPhiJet1metAco = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(2.89),
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    Acoplanar = cms.string('Jet1Met'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.0),
    minEtJet1 = cms.double(60.0),
    minEtJet2 = cms.double(-1.0)
)

hltL1sPhiJet2MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrephijet2met = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPhiJet2metAco = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(3.141593),
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    Acoplanar = cms.string('Jet2Met'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.377),
    minEtJet1 = cms.double(50.0),
    minEtJet2 = cms.double(50.0)
)

hltL1sPhiJet1Jet2 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrephijet1jet2 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPhiJet1Jet2Aco = cms.EDFilter("HLTAcoFilter",
    maxDeltaPhi = cms.double(2.7646),
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    Acoplanar = cms.string('Jet1Jet2'),
    inputMETTag = cms.InputTag("hlt1MET70"),
    minDeltaPhi = cms.double(0.0),
    minEtJet1 = cms.double(40.0),
    minEtJet2 = cms.double(40.0)
)

hltL1RapGap = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_IsoEG10_Jet15_ForJet10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrerapgap = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltRapGap = cms.EDFilter("HLTRapGapFilter",
    maxEta = cms.double(5.0),
    minEta = cms.double(3.0),
    caloThresh = cms.double(20.0),
    inputTag = cms.InputTag("hltIterativeCone5CaloJets")
)

hltL1seedSingle = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleIsoEG12'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltEcalRegionalEgammaFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    EM_l1TagIsolated = cms.untracked.InputTag("hltL1extraParticles","Isolated"),
    OutputLabel = cms.untracked.string(''),
    Ptmin_noniso = cms.untracked.double(5.0),
    EM_l1TagNonIsolated = cms.untracked.InputTag("hltL1extraParticles","NonIsolated"),
    debug = cms.untracked.bool(False),
    EGamma = cms.untracked.bool(True),
    Ptmin_iso = cms.untracked.double(5.0)
)

hltEcalRegionalEgammaDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.string('hltEcalRegionalEgammaFEDs'),
    syncCheck = cms.untracked.bool(False),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    DoRegional = cms.untracked.bool(True)
)

hltEcalRegionalEgammaWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalRegionalEgammaDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalRegionalEgammaDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRegionalEgammaRecHitTmp = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalRegionalEgammaWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalRegionalEgammaWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltEcalRegionalEgammaRecHit = cms.EDFilter("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)

hltIslandBasicClustersEndcapL1Isolated = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(5.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("ecalRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(False)
)

hltIslandBasicClustersBarrelL1Isolated = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("ecalRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(5.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(False),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(True)
)

hltHybridSuperClustersL1Isolated = cms.EDProducer("EgammaHLTHybridClusterProducer",
    regionEtaMargin = cms.double(0.14),
    regionPhiMargin = cms.double(0.4),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(True),
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    l1LowerThr = cms.double(5.0),
    eseed = cms.double(0.35),
    ethresh = cms.double(0.1),
    ewing = cms.double(1.0),
    step = cms.int32(10),
    debugLevel = cms.string('INFO'),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    superclusterCollection = cms.string(''),
    posCalc_x0 = cms.double(0.89),
    HybridBarrelSeedThr = cms.double(1.5),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    posCalc_t0 = cms.double(7.4),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    ecalhitproducer = cms.InputTag("hltEcalRegionalEgammaRecHit")
)

hltIslandSuperClustersL1Isolated = cms.EDProducer("SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('islandBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    endcapClusterProducer = cms.string('hltIslandBasicClustersEndcapL1Isolated'),
    barrelPhiSearchRoad = cms.double(0.2),
    endcapPhiSearchRoad = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.5),
    endcapSuperclusterCollection = cms.string('islandEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    barrelClusterProducer = cms.string('hltIslandBasicClustersBarrelL1Isolated')
)

hltCorrectedIslandEndcapSuperClustersL1Isolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEE'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('hltIslandSuperClustersL1Isolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string('islandEndcapSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedIslandBarrelSuperClustersL1Isolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('hltIslandSuperClustersL1Isolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string('islandBarrelSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedHybridSuperClustersL1Isolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(5.0),
    rawSuperClusterProducer = cms.string('hltHybridSuperClustersL1Isolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string(''),
    isl_fCorrPset = cms.PSet(

    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(
        brLinearThr = cms.double(12.0),
        fBremVec = cms.vdouble(-0.01258, 0.03154, 0.9888, -0.0007973, 1.59),
        fEtEtaVec = cms.vdouble(1.0, -0.8206, 3.16, 0.8637, 44.88, 
            2.292, 1.023, 8.0)
    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    etThresh = cms.double(5.0),
    preshRecHitProducer = cms.string('hltEcalPreshowerRecHit'),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    assocSClusterCollection = cms.string(''),
    preshClusterEnergyCut = cms.double(0.0),
    endcapSClusterProducer = cms.string('hltCorrectedIslandEndcapSuperClustersL1Isolated'),
    preshNclust = cms.int32(4),
    endcapSClusterCollection = cms.string(''),
    debugLevel = cms.string(''),
    preshRecHitCollection = cms.string('EcalRecHitsES'),
    preshSeededNstrip = cms.int32(15)
)

hltL1IsoRecoEcalCandidate = cms.EDFilter("EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag("hltCorrectedHybridSuperClustersL1Isolated"),
    scIslandEndcapProducer = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    recoEcalCandidateCollection = cms.string('')
)

hltL1IsoSingleL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedSingle"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1IsoSingleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1IsoSingleL1MatchFilter")
)

hltL1IsolatedElectronHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.15),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1IsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoSingleElectronEtFilter")
)

hltSiPixelDigis = cms.EDFilter("SiPixelRawToDigi",
    InputLabel = cms.untracked.string('rawDataCollector')
)

hltSiPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    src = cms.InputTag("hltSiPixelDigis"),
    ChannelThreshold = cms.int32(2500),
    MissCalibrate = cms.untracked.bool(True),
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(3000),
    ClusterThreshold = cms.double(5050.0)
)

hltSiPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    src = cms.InputTag("hltSiPixelClusters"),
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    CPE = cms.string('PixelCPEGeneric'),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)
)

hltSiStripRawToClustersFacility = cms.EDFilter("SiStripRawToClusters",
    ProductLabel = cms.untracked.string('rawDataCollector'),
    ChannelThreshold = cms.untracked.double(2.0),
    MaxHolesInCluster = cms.untracked.uint32(0),
    ClusterizerAlgorithm = cms.untracked.string('ThreeThreshold'),
    SeedThreshold = cms.untracked.double(3.0),
    ClusterThreshold = cms.untracked.double(5.0)
)

hltSiStripClusters = cms.EDProducer("MeasurementTrackerSiStripRefGetterProducer",
    InputModuleLabel = cms.InputTag("hltSiStripRawToClustersFacility"),
    measurementTrackerName = cms.string('')
)

hltL1IsoElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    SeedAlgo = cms.string(''),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.08),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.025),
        PhiMin2 = cms.double(-0.001),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.05),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.015),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.001),
        r2MaxF = cms.double(0.08),
        pPhiMin1 = cms.double(-0.015),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.05),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1Isolated")
)

hltL1IsoSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoSingleElectronHcalIsolFilter")
)

hltCkfL1IsoTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1IsoWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1IsoTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltPixelMatchElectronsL1Iso = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrackProducer = cms.InputTag("hltCtfL1IsoWithMaterialTracks"),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    estimator = cms.string('egammaHLTChi2'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectronsL1Iso'),
    updator = cms.string('KFUpdator')
)

hltL1IsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoSingleElectronPixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1IsoElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(0.5),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(True),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    originRadius = cms.double(0.02)
)

hltL1IsoElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1IsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1IsoElectronsRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1IsoElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1IsoElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1IsoSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedRelaxedSingle = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltIslandBasicClustersEndcapL1NonIsolated = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(False),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(5.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(True),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("ecalRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(False)
)

hltIslandBasicClustersBarrelL1NonIsolated = cms.EDProducer("EgammaHLTIslandClusterProducer",
    endcapHitProducer = cms.InputTag("ecalRecHit"),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    regionEtaMargin = cms.double(0.3),
    regionPhiMargin = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(1.2),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(False),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5),
    l1LowerThr = cms.double(5.0),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    doEndcaps = cms.bool(False),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    barrelHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit"),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    posCalc_t0_barl = cms.double(7.4),
    doBarrel = cms.bool(True)
)

hltHybridSuperClustersL1NonIsolated = cms.EDProducer("EgammaHLTHybridClusterProducer",
    regionEtaMargin = cms.double(0.14),
    regionPhiMargin = cms.double(0.4),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    posCalc_logweight = cms.bool(True),
    doIsolated = cms.bool(False),
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    l1UpperThr = cms.double(999.0),
    l1LowerThr = cms.double(5.0),
    eseed = cms.double(0.35),
    ethresh = cms.double(0.1),
    ewing = cms.double(1.0),
    step = cms.int32(10),
    debugLevel = cms.string('INFO'),
    l1TagIsolated = cms.InputTag("hltL1extraParticles","Isolated"),
    superclusterCollection = cms.string(''),
    posCalc_x0 = cms.double(0.89),
    HybridBarrelSeedThr = cms.double(1.5),
    l1TagNonIsolated = cms.InputTag("hltL1extraParticles","NonIsolated"),
    posCalc_t0 = cms.double(7.4),
    l1LowerThrIgnoreIsolation = cms.double(999.0),
    ecalhitproducer = cms.InputTag("hltEcalRegionalEgammaRecHit")
)

hltIslandSuperClustersL1NonIsolated = cms.EDProducer("SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('islandBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    endcapClusterProducer = cms.string('hltIslandBasicClustersEndcapL1NonIsolated'),
    barrelPhiSearchRoad = cms.double(0.2),
    endcapPhiSearchRoad = cms.double(0.4),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.5),
    endcapSuperclusterCollection = cms.string('islandEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    barrelClusterProducer = cms.string('hltIslandBasicClustersBarrelL1NonIsolated')
)

hltCorrectedIslandEndcapSuperClustersL1NonIsolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEE'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('hltIslandSuperClustersL1NonIsolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string('islandEndcapSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedIslandBarrelSuperClustersL1NonIsolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('hltIslandSuperClustersL1NonIsolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string('islandBarrelSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(5.0),
    rawSuperClusterProducer = cms.string('hltHybridSuperClustersL1NonIsolated'),
    applyEnergyCorrection = cms.bool(True),
    applyOldCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    rawSuperClusterCollection = cms.string(''),
    isl_fCorrPset = cms.PSet(

    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(
        brLinearThr = cms.double(12.0),
        fBremVec = cms.vdouble(-0.01258, 0.03154, 0.9888, -0.0007973, 1.59),
        fEtEtaVec = cms.vdouble(1.0, -0.8206, 3.16, 0.8637, 44.88, 
            2.292, 1.023, 8.0)
    ),
    recHitProducer = cms.string('hltEcalRegionalEgammaRecHit')
)

hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    etThresh = cms.double(5.0),
    preshRecHitProducer = cms.string('hltEcalPreshowerRecHit'),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    assocSClusterCollection = cms.string(''),
    preshClusterEnergyCut = cms.double(0.0),
    endcapSClusterProducer = cms.string('hltCorrectedIslandEndcapSuperClustersL1Isolated'),
    preshNclust = cms.int32(4),
    endcapSClusterCollection = cms.string(''),
    debugLevel = cms.string(''),
    preshRecHitCollection = cms.string('EcalRecHitsES'),
    preshSeededNstrip = cms.int32(15)
)

hltL1NonIsoRecoEcalCandidate = cms.EDFilter("EgammaHLTRecoEcalCandidateProducers",
    scHybridBarrelProducer = cms.InputTag("hltCorrectedHybridSuperClustersL1NonIsolated"),
    scIslandEndcapProducer = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    recoEcalCandidateCollection = cms.string('')
)

hltL1NonIsoSingleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingle"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoSingleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(18.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1NonIsoSingleElectronL1MatchFilterRegional")
)

hltL1NonIsolatedElectronHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.15),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1NonIsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronEtFilter")
)

hltL1NonIsoElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    SeedAlgo = cms.string(''),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.08),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.025),
        PhiMin2 = cms.double(-0.001),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.05),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.015),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.001),
        r2MaxF = cms.double(0.08),
        pPhiMin1 = cms.double(-0.015),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.05),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1NonIsolated")
)

hltL1NonIsoSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronHcalIsolFilter")
)

hltCkfL1NonIsoTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1NonIsoWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1NonIsoTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltPixelMatchElectronsL1NonIso = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrackProducer = cms.InputTag("hltCtfL1NonIsoWithMaterialTracks"),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    estimator = cms.string('egammaHLTChi2'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectronsL1NonIso'),
    updator = cms.string('KFUpdator')
)

hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronPixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(0.5),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(True),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    originRadius = cms.double(0.02)
)

hltL1NonIsoElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1NonIsoElectronsRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1NonIsoElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1NonIsoSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedDouble = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleIsoEG8'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1IsoDoubleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoDoubleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional")
)

hltL1IsoDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronEtFilter")
)

hltL1IsoDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1IsoDoubleElectronHcalIsolFilter")
)

hltL1IsoDoubleElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1IsoDoubleElectronPixelMatchFilter")
)

hltL1IsoDoubleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronEoverpFilter")
)

hltDoubleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedRelaxedDouble = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleEG10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1NonIsoDoubleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDouble"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoDoubleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional")
)

hltL1NonIsoDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter")
)

hltL1NonIsoDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronHcalIsolFilter")
)

hltL1NonIsoDoubleElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronPixelMatchFilter")
)

hltL1NonIsoDoubleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter")
)

hltDoubleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoSinglePhotonL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedSingle"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1IsoSinglePhotonEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(30.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1IsoSinglePhotonL1MatchFilter")
)

hltL1IsolatedPhotonEcalIsol = cms.EDFilter("EgammaHLTEcalIsolationProducersRegional",
    egEcalIsoEtMin = cms.double(0.0),
    scIslandBarrelProducer = cms.InputTag("hltCorrectedIslandBarrelSuperClustersL1Isolated"),
    bcEndcapProducer = cms.InputTag("hltIslandBasicClustersEndcapL1Isolated","islandEndcapBasicClusters"),
    bcBarrelProducer = cms.InputTag("hltIslandBasicClustersBarrelL1Isolated","islandBarrelBasicClusters"),
    scIslandEndcapProducer = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    egEcalIsoConeSize = cms.double(0.3),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate")
)

hltL1IsoSinglePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(1),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonEtFilter")
)

hltL1IsolatedPhotonHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1IsoSinglePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter")
)

hltL1IsoEgammaRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(15.0),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(False),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("pixelMatchElectrons"),
    originRadius = cms.double(0.02)
)

hltL1IsoEgammaRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoEgammaRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1IsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1IsoEgammaRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1IsoPhotonTrackIsol = cms.EDFilter("EgammaHLTPhotonTrackIsolationProducersRegional",
    egTrkIsoVetoConeSize = cms.double(0.0),
    trackProducer = cms.InputTag("hltL1IsoEgammaRegionalCTFFinalFitWithMaterial"),
    egTrkIsoConeSize = cms.double(0.3),
    egTrkIsoRSpan = cms.double(999999.0),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(999999.0)
)

hltL1IsoSinglePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoSinglePhotonL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingle"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoSinglePhotonEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(40.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1NonIsoSinglePhotonL1MatchFilterRegional")
)

hltL1NonIsolatedPhotonEcalIsol = cms.EDFilter("EgammaHLTEcalIsolationProducersRegional",
    egEcalIsoEtMin = cms.double(0.0),
    scIslandBarrelProducer = cms.InputTag("hltCorrectedIslandBarrelSuperClustersL1NonIsolated"),
    bcEndcapProducer = cms.InputTag("hltIslandBasicClustersEndcapL1NonIsolated","islandEndcapBasicClusters"),
    bcBarrelProducer = cms.InputTag("hltIslandBasicClustersBarrelL1NonIsolated","islandBarrelBasicClusters"),
    scIslandEndcapProducer = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    egEcalIsoConeSize = cms.double(0.3),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoSinglePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEtFilter")
)

hltL1NonIsolatedPhotonHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1NonIsoSinglePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter")
)

hltL1NonIsoEgammaRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(15.0),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(False),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("pixelMatchElectrons"),
    originRadius = cms.double(0.02)
)

hltL1NonIsoEgammaRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoEgammaRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1NonIsoEgammaRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1NonIsoPhotonTrackIsol = cms.EDFilter("EgammaHLTPhotonTrackIsolationProducersRegional",
    egTrkIsoVetoConeSize = cms.double(0.0),
    trackProducer = cms.InputTag("hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial"),
    egTrkIsoConeSize = cms.double(0.3),
    egTrkIsoRSpan = cms.double(999999.0),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(999999.0)
)

hltL1NonIsoSinglePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoDoublePhotonL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoDoublePhotonEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1IsoDoublePhotonL1MatchFilterRegional")
)

hltL1IsoDoublePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(2),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonEtFilter")
)

hltL1IsoDoublePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter")
)

hltL1IsoDoublePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltL1IsoDoublePhotonDoubleEtFilter = cms.EDFilter("HLTEgammaDoubleEtFilter",
    etcut1 = cms.double(20.0),
    candTag = cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter"),
    etcut2 = cms.double(20.0),
    npaircut = cms.int32(1)
)

hltDoublePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoDoublePhotonL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDouble"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoDoublePhotonEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional")
)

hltL1NonIsoDoublePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonEtFilter")
)

hltL1NonIsoDoublePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter")
)

hltL1NonIsoDoublePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltL1NonIsoDoublePhotonDoubleEtFilter = cms.EDFilter("HLTEgammaDoubleEtFilter",
    etcut1 = cms.double(20.0),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter"),
    etcut2 = cms.double(20.0),
    npaircut = cms.int32(1)
)

hltDoublePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoSingleEMHighEtL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingle"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoSinglePhotonEMHighEtEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(80.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1NonIsoSingleEMHighEtL1MatchFilterRegional")
)

hltL1NonIsoSingleEMHighEtEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalisolcut = cms.double(5.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEMHighEtEtFilter")
)

hltL1NonIsoSingleEMHighEtHOEFilter = cms.EDFilter("HLTEgammaHOEFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(0.05),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtEcalIsolFilter")
)

hltHcalDoubleCone = cms.EDFilter("EgammaHLTHcalIsolationDoubleConeProducers",
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco"),
    egHcalExclusion = cms.double(0.15),
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    egHcalIsoPtMin = cms.double(0.0),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate")
)

hltL1NonIsoEMHcalDoubleCone = cms.EDFilter("EgammaHLTHcalIsolationDoubleConeProducers",
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco"),
    egHcalExclusion = cms.double(0.15),
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    egHcalIsoPtMin = cms.double(0.0),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoSingleEMHighEtHcalDBCFilter = cms.EDFilter("HLTEgammaHcalDBCFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoEMHcalDoubleCone"),
    hcalisolbarrelcut = cms.double(8.0),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltHcalDoubleCone"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtHOEFilter")
)

hltL1NonIsoSingleEMHighEtTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtHcalDBCFilter"),
    numtrackisolcut = cms.double(4.0)
)

hltSingleEMVHighEtL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingle"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(200.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional")
)

hltSingleEMVHEL1NonIsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoDoubleElectronZeeL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoDoubleElectronZeeEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1IsoDoubleElectronZeeL1MatchFilterRegional")
)

hltL1IsoDoubleElectronZeeHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeEtFilter")
)

hltL1IsoDoubleElectronZeePixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeHcalIsolFilter")
)

hltL1IsoDoubleElectronZeeEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeePixelMatchFilter")
)

hltL1IsoDoubleElectronZeeTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeEoverpFilter")
)

hltL1IsoDoubleElectronZeePMMassFilter = cms.EDFilter("HLTPMMassFilter",
    upperMassCut = cms.double(99999.9),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeTrackIsolFilter"),
    nZcandcut = cms.int32(1),
    lowerMassCut = cms.double(54.22)
)

hltZeeCounterPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedExclusiveDouble = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ExclusiveDoubleIsoEG6'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1IsoDoubleExclElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedExclusiveDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoDoubleExclElectronEtPhiFilter = cms.EDFilter("HLTEgammaDoubleEtPhiFilter",
    etcut1 = cms.double(6.0),
    etcut2 = cms.double(6.0),
    npaircut = cms.int32(1),
    MaxAcop = cms.double(0.6),
    MinEtBalance = cms.double(-1.0),
    MaxEtBalance = cms.double(10.0),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronL1MatchFilterRegional"),
    MinAcop = cms.double(-0.1)
)

hltL1IsoDoubleExclElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronEtPhiFilter")
)

hltL1IsoDoubleExclElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronHcalIsolFilter")
)

hltL1IsoDoubleExclElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronPixelMatchFilter")
)

hltL1IsoDoubleExclElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronEoverpFilter")
)

hltDoubleExclElectronL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoDoubleExclPhotonL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedExclusiveDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoDoubleExclPhotonEtPhiFilter = cms.EDFilter("HLTEgammaDoubleEtPhiFilter",
    etcut1 = cms.double(10.0),
    etcut2 = cms.double(10.0),
    npaircut = cms.int32(1),
    MaxAcop = cms.double(0.6),
    MinEtBalance = cms.double(-1.0),
    MaxEtBalance = cms.double(10.0),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonL1MatchFilterRegional"),
    MinAcop = cms.double(-0.1)
)

hltL1IsoDoubleExclPhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(2),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonEtPhiFilter")
)

hltL1IsoDoubleExclPhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonEcalIsolFilter")
)

hltL1IsoDoubleExclPhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltDoubleExclPhotonL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedSinglePrescaled = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleIsoEG10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1IsoSinglePhotonPrescaledL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedSinglePrescaled"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1IsoSinglePhotonPrescaledEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledL1MatchFilter")
)

hltL1IsoSinglePhotonPrescaledEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(1),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledEtFilter")
)

hltL1IsoSinglePhotonPrescaledHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledEcalIsolFilter")
)

hltL1IsoSinglePhotonPrescaledTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonPrescaledL1IsoPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoLargeWindowSingleL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedSingle"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1IsoLargeWindowSingleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1IsoLargeWindowSingleL1MatchFilter")
)

hltL1IsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronEtFilter")
)

hltL1IsoLargeWindowElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    SeedAlgo = cms.string(''),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.16),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.035),
        PhiMin2 = cms.double(-0.002),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.1),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.02),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.002),
        r2MaxF = cms.double(0.16),
        pPhiMin1 = cms.double(-0.02),
        pPhiMax1 = cms.double(0.035),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.1),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1Isolated")
)

hltL1IsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronHcalIsolFilter")
)

hltCkfL1IsoLargeWindowTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoLargeWindowElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1IsoLargeWindowWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1IsoLargeWindowTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltPixelMatchElectronsL1IsoLargeWindow = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrackProducer = cms.InputTag("hltCtfL1IsoLargeWindowWithMaterialTracks"),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    estimator = cms.string('egammaHLTChi2'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectronsL1IsoLargeWindow'),
    updator = cms.string('KFUpdator')
)

hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsL1NonIsoLargeWindowForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronPixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(0.5),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(True),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    originRadius = cms.double(0.02)
)

hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1IsoLargeWindowElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1IsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingle"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoLargeWindowSingleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(18.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional")
)

hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronEtFilter")
)

hltL1NonIsoLargeWindowElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    SeedAlgo = cms.string(''),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.16),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.035),
        PhiMin2 = cms.double(-0.002),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.1),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.02),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.002),
        r2MaxF = cms.double(0.16),
        pPhiMin1 = cms.double(-0.02),
        pPhiMax1 = cms.double(0.035),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.1),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1NonIsolated")
)

hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter")
)

hltCkfL1NonIsoLargeWindowTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoLargeWindowElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1NonIsoLargeWindowWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1NonIsoLargeWindowTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltPixelMatchElectronsL1NonIsoLargeWindow = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    TrackProducer = cms.InputTag("hltCtfL1NonIsoLargeWindowWithMaterialTracks"),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    estimator = cms.string('egammaHLTChi2'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchElectronsL1NonIsoLargeWindow'),
    updator = cms.string('KFUpdator')
)

hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3),
    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(0.5),
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    UseZInVertex = cms.bool(True),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3),
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    candTagEle = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    originRadius = cms.double(0.02)
)

hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('hltEgammaRegionalCTFFinalFitWithMaterial'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltL1NonIsoLargeWindowElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedDouble"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1IsoLargeWindowDoubleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional")
)

hltL1IsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronEtFilter")
)

hltL1IsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronHcalIsolFilter")
)

hltL1IsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronPixelMatchFilter")
)

hltL1IsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronEoverpFilter")
)

hltDoubleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDouble"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoLargeWindowDoubleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(2),
    inputTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional")
)

hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronEtFilter")
)

hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter")
)

hltL1NonIsoLargeWindowDoubleElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    eoverpendcapcut = cms.double(24500.0),
    ncandcut = cms.int32(2),
    eoverpbarrelcut = cms.double(15000.0),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter")
)

hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronEoverpFilter")
)

hltDoubleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPrescaleSingleMuIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuonDTDigis = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('DDU'),
    fedColl = cms.untracked.string('rawDataCollector'),
    fedbyType = cms.untracked.bool(False),
    readOutParameters = cms.PSet(
        debug = cms.untracked.bool(False),
        rosParameters = cms.PSet(
            writeSC = cms.untracked.bool(True),
            readingDDU = cms.untracked.bool(True),
            performDataIntegrityMonitor = cms.untracked.bool(False),
            readDDUIDfromDDU = cms.untracked.bool(True),
            debug = cms.untracked.bool(False),
            localDAQ = cms.untracked.bool(False)
        ),
        localDAQ = cms.untracked.bool(False),
        performDataIntegrityMonitor = cms.untracked.bool(False)
    ),
    rosParameters = cms.PSet(
        readingDDU = cms.untracked.bool(True),
        localDAQ = cms.untracked.bool(False)
    )
)

hltDt1DRecHits = cms.EDProducer("DTRecHitProducer",
    debug = cms.untracked.bool(False),
    recAlgoConfig = cms.PSet(
        tTrigMode = cms.string('DTTTrigSyncFromDB'),
        minTime = cms.double(-3.0),
        interpolate = cms.bool(True),
        debug = cms.untracked.bool(False),
        tTrigModeConfig = cms.PSet(
            vPropWire = cms.double(24.4),
            doTOFCorrection = cms.bool(True),
            tofCorrType = cms.int32(1),
            kFactor = cms.double(-2.0),
            wirePropCorrType = cms.int32(1),
            doWirePropCorrection = cms.bool(True),
            doT0Correction = cms.bool(True),
            debug = cms.untracked.bool(False)
        ),
        maxTime = cms.double(415.0)
    ),
    dtDigiLabel = cms.InputTag("hltMuonDTDigis"),
    recAlgo = cms.string('DTParametrizedDriftAlgo')
)

hltDt4DSegments = cms.EDProducer("DTRecSegment4DProducer",
    debug = cms.untracked.bool(False),
    Reco4DAlgoName = cms.string('DTCombinatorialPatternReco4D'),
    Reco4DAlgoConfig = cms.PSet(
        segmCleanerMode = cms.int32(1),
        Reco2DAlgoName = cms.string('DTCombinatorialPatternReco'),
        recAlgoConfig = cms.PSet(
            tTrigMode = cms.string('DTTTrigSyncFromDB'),
            minTime = cms.double(-3.0),
            interpolate = cms.bool(True),
            debug = cms.untracked.bool(False),
            tTrigModeConfig = cms.PSet(
                vPropWire = cms.double(24.4),
                doTOFCorrection = cms.bool(True),
                tofCorrType = cms.int32(1),
                kFactor = cms.double(-2.0),
                wirePropCorrType = cms.int32(1),
                doWirePropCorrection = cms.bool(True),
                doT0Correction = cms.bool(True),
                debug = cms.untracked.bool(False)
            ),
            maxTime = cms.double(415.0)
        ),
        nSharedHitsMax = cms.int32(2),
        debug = cms.untracked.bool(False),
        Reco2DAlgoConfig = cms.PSet(
            segmCleanerMode = cms.int32(1),
            recAlgo = cms.string('DTParametrizedDriftAlgo'),
            AlphaMaxPhi = cms.double(1.0),
            MaxAllowedHits = cms.uint32(50),
            nSharedHitsMax = cms.int32(2),
            AlphaMaxTheta = cms.double(0.1),
            debug = cms.untracked.bool(False),
            recAlgoConfig = cms.PSet(
                tTrigMode = cms.string('DTTTrigSyncFromDB'),
                minTime = cms.double(-3.0),
                interpolate = cms.bool(True),
                debug = cms.untracked.bool(False),
                tTrigModeConfig = cms.PSet(
                    vPropWire = cms.double(24.4),
                    doTOFCorrection = cms.bool(True),
                    tofCorrType = cms.int32(1),
                    kFactor = cms.double(-2.0),
                    wirePropCorrType = cms.int32(1),
                    doWirePropCorrection = cms.bool(True),
                    doT0Correction = cms.bool(True),
                    debug = cms.untracked.bool(False)
                ),
                maxTime = cms.double(415.0)
            ),
            nUnSharedHitsMin = cms.int32(2)
        ),
        recAlgo = cms.string('DTParametrizedDriftAlgo'),
        nUnSharedHitsMin = cms.int32(2),
        AllDTRecHits = cms.bool(True)
    ),
    recHits1DLabel = cms.InputTag("hltDt1DRecHits"),
    recHits2DLabel = cms.InputTag("dt2DSegments")
)

hltMuonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    ErrorMask = cms.untracked.uint32(3754946559),
    InputObjects = cms.InputTag("rawDataCollector"),
    ExaminerMask = cms.untracked.uint32(374076406),
    UseExaminer = cms.untracked.bool(False)
)

hltCsc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    CSCStripClusterSize = cms.untracked.int32(3),
    CSCStripPeakThreshold = cms.untracked.double(10.0),
    ConstSyst = cms.untracked.double(0.03),
    readBadChannels = cms.bool(False),
    CSCproduce1DHits = cms.untracked.bool(False),
    CSCStripxtalksOffset = cms.untracked.double(0.03),
    CSCstripWireDeltaTime = cms.untracked.int32(8),
    CSCUseCalibrations = cms.untracked.bool(True),
    XTasymmetry = cms.untracked.double(0.005),
    CSCStripDigiProducer = cms.string('hltMuonCSCDigis'),
    CSCWireDigiProducer = cms.string('hltMuonCSCDigis'),
    CSCDebug = cms.untracked.bool(False),
    NoiseLevel = cms.untracked.double(7.0),
    CSCWireClusterDeltaT = cms.untracked.int32(1),
    CSCStripClusterChargeCut = cms.untracked.double(25.0)
)

hltCscSegments = cms.EDProducer("CSCSegmentProducer",
    inputObjects = cms.InputTag("hltCsc2DRecHits"),
    algo_type = cms.int32(4),
    algo_psets = cms.VPSet(cms.PSet(
        chamber_types = cms.vstring('ME1/a', 
            'ME1/b', 
            'ME1/2', 
            'ME1/3', 
            'ME2/1', 
            'ME2/2', 
            'ME3/1', 
            'ME3/2', 
            'ME4/1'),
        algo_name = cms.string('CSCSegAlgoSK'),
        parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
            1, 1, 1, 1),
        algo_psets = cms.VPSet(cms.PSet(
            dPhiFineMax = cms.double(0.025),
            verboseInfo = cms.untracked.bool(True),
            chi2Max = cms.double(99999.0),
            dPhiMax = cms.double(0.003),
            wideSeg = cms.double(3.0),
            minLayersApart = cms.int32(2),
            dRPhiFineMax = cms.double(8.0),
            dRPhiMax = cms.double(8.0)
        ), 
            cms.PSet(
                dPhiFineMax = cms.double(0.025),
                verboseInfo = cms.untracked.bool(True),
                chi2Max = cms.double(99999.0),
                dPhiMax = cms.double(0.025),
                wideSeg = cms.double(3.0),
                minLayersApart = cms.int32(2),
                dRPhiFineMax = cms.double(3.0),
                dRPhiMax = cms.double(8.0)
            ))
    ), 
        cms.PSet(
            chamber_types = cms.vstring('ME1/a', 
                'ME1/b', 
                'ME1/2', 
                'ME1/3', 
                'ME2/1', 
                'ME2/2', 
                'ME3/1', 
                'ME3/2', 
                'ME4/1'),
            algo_name = cms.string('CSCSegAlgoTC'),
            parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
                1, 1, 1, 1),
            algo_psets = cms.VPSet(cms.PSet(
                dPhiFineMax = cms.double(0.02),
                verboseInfo = cms.untracked.bool(True),
                SegmentSorting = cms.int32(1),
                chi2Max = cms.double(6000.0),
                dPhiMax = cms.double(0.003),
                chi2ndfProbMin = cms.double(0.0001),
                minLayersApart = cms.int32(2),
                dRPhiFineMax = cms.double(6.0),
                dRPhiMax = cms.double(1.2)
            ), 
                cms.PSet(
                    dPhiFineMax = cms.double(0.013),
                    verboseInfo = cms.untracked.bool(True),
                    SegmentSorting = cms.int32(1),
                    chi2Max = cms.double(6000.0),
                    dPhiMax = cms.double(0.00198),
                    chi2ndfProbMin = cms.double(0.0001),
                    minLayersApart = cms.int32(2),
                    dRPhiFineMax = cms.double(3.0),
                    dRPhiMax = cms.double(0.6)
                ))
        ), 
        cms.PSet(
            chamber_types = cms.vstring('ME1/a', 
                'ME1/b', 
                'ME1/2', 
                'ME1/3', 
                'ME2/1', 
                'ME2/2', 
                'ME3/1', 
                'ME3/2', 
                'ME4/1'),
            algo_name = cms.string('CSCSegAlgoDF'),
            parameters_per_chamber_type = cms.vint32(3, 1, 2, 2, 1, 
                2, 1, 2, 1),
            algo_psets = cms.VPSet(cms.PSet(
                tanThetaMax = cms.double(1.2),
                maxRatioResidualPrune = cms.double(3.0),
                dPhiFineMax = cms.double(0.025),
                tanPhiMax = cms.double(0.5),
                dXclusBoxMax = cms.double(8.0),
                preClustering = cms.untracked.bool(False),
                chi2Max = cms.double(5000.0),
                minHitsPerSegment = cms.int32(3),
                minHitsForPreClustering = cms.int32(10),
                minLayersApart = cms.int32(2),
                dRPhiFineMax = cms.double(8.0),
                nHitsPerClusterIsShower = cms.int32(20),
                CSCSegmentDebug = cms.untracked.bool(False),
                Pruning = cms.untracked.bool(False),
                dYclusBoxMax = cms.double(8.0)
            ), 
                cms.PSet(
                    tanThetaMax = cms.double(2.0),
                    maxRatioResidualPrune = cms.double(3.0),
                    dPhiFineMax = cms.double(0.025),
                    tanPhiMax = cms.double(0.8),
                    dXclusBoxMax = cms.double(8.0),
                    preClustering = cms.untracked.bool(False),
                    chi2Max = cms.double(5000.0),
                    minHitsPerSegment = cms.int32(3),
                    minHitsForPreClustering = cms.int32(10),
                    minLayersApart = cms.int32(2),
                    dRPhiFineMax = cms.double(12.0),
                    nHitsPerClusterIsShower = cms.int32(20),
                    CSCSegmentDebug = cms.untracked.bool(False),
                    Pruning = cms.untracked.bool(False),
                    dYclusBoxMax = cms.double(12.0)
                ), 
                cms.PSet(
                    tanThetaMax = cms.double(1.2),
                    maxRatioResidualPrune = cms.double(3.0),
                    dPhiFineMax = cms.double(0.025),
                    tanPhiMax = cms.double(0.5),
                    dXclusBoxMax = cms.double(8.0),
                    preClustering = cms.untracked.bool(False),
                    chi2Max = cms.double(5000.0),
                    minHitsPerSegment = cms.int32(3),
                    minHitsForPreClustering = cms.int32(30),
                    minLayersApart = cms.int32(2),
                    dRPhiFineMax = cms.double(8.0),
                    nHitsPerClusterIsShower = cms.int32(20),
                    CSCSegmentDebug = cms.untracked.bool(False),
                    Pruning = cms.untracked.bool(False),
                    dYclusBoxMax = cms.double(8.0)
                ))
        ), 
        cms.PSet(
            chamber_types = cms.vstring('ME1/a', 
                'ME1/b', 
                'ME1/2', 
                'ME1/3', 
                'ME2/1', 
                'ME2/2', 
                'ME3/1', 
                'ME3/2', 
                'ME4/1'),
            algo_name = cms.string('CSCSegAlgoST'),
            parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
                1, 1, 1, 1),
            algo_psets = cms.VPSet(cms.PSet(
                preClustering = cms.untracked.bool(True),
                minHitsPerSegment = cms.untracked.int32(3),
                yweightPenaltyThreshold = cms.untracked.double(1.0),
                curvePenalty = cms.untracked.double(2.0),
                dXclusBoxMax = cms.untracked.double(4.0),
                hitDropLimit5Hits = cms.untracked.double(0.8),
                yweightPenalty = cms.untracked.double(1.5),
                BrutePruning = cms.untracked.bool(False),
                curvePenaltyThreshold = cms.untracked.double(0.85),
                hitDropLimit4Hits = cms.untracked.double(0.6),
                Pruning = cms.untracked.bool(False),
                onlyBestSegment = cms.untracked.bool(False),
                CSCDebug = cms.untracked.bool(False),
                maxRecHitsInCluster = cms.untracked.int32(20),
                hitDropLimit6Hits = cms.untracked.double(0.3333),
                dYclusBoxMax = cms.untracked.double(8.0)
            ), 
                cms.PSet(
                    preClustering = cms.untracked.bool(True),
                    minHitsPerSegment = cms.untracked.int32(3),
                    yweightPenaltyThreshold = cms.untracked.double(1.0),
                    curvePenalty = cms.untracked.double(2.0),
                    dXclusBoxMax = cms.untracked.double(4.0),
                    hitDropLimit5Hits = cms.untracked.double(0.8),
                    yweightPenalty = cms.untracked.double(1.5),
                    BrutePruning = cms.untracked.bool(False),
                    curvePenaltyThreshold = cms.untracked.double(0.85),
                    hitDropLimit4Hits = cms.untracked.double(0.6),
                    Pruning = cms.untracked.bool(False),
                    onlyBestSegment = cms.untracked.bool(False),
                    CSCDebug = cms.untracked.bool(False),
                    maxRecHitsInCluster = cms.untracked.int32(24),
                    hitDropLimit6Hits = cms.untracked.double(0.3333),
                    dYclusBoxMax = cms.untracked.double(8.0)
                ))
        ))
)

hltMuonRPCDigis = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("rawDataCollector")
)

hltRpcRecHits = cms.EDProducer("RPCRecHitProducer",
    recAlgoConfig = cms.PSet(

    ),
    recAlgo = cms.string('RPCRecHitStandardAlgo'),
    rpcDigiLabel = cms.InputTag("hltMuonRPCDigis")
)

hltL2MuonSeeds = cms.EDFilter("L2MuonSeedGenerator",
    L1MinPt = cms.double(0.0),
    InputObjects = cms.InputTag("hltL1extraParticles"),
    L1MaxEta = cms.double(2.5),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'SteppingHelixPropagatorAlong', 
            'SteppingHelixPropagatorOpposite', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite', 
            'SmartPropagator', 
            'SmartPropagatorOpposite', 
            'SmartPropagatorAnyOpposite', 
            'SmartPropagatorAny', 
            'SmartPropagatorRK', 
            'SmartPropagatorAnyRK'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    L1MinQuality = cms.uint32(1),
    GMTReadoutCollection = cms.InputTag("hltGtDigis"),
    Propagator = cms.string('SteppingHelixPropagatorAny')
)

hltL2Muons = cms.EDProducer("L2MuonProducer",
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'SteppingHelixPropagatorAlong', 
            'SteppingHelixPropagatorOpposite', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite', 
            'SmartPropagator', 
            'SmartPropagatorOpposite', 
            'SmartPropagatorAnyOpposite', 
            'SmartPropagatorAny', 
            'SmartPropagatorRK', 
            'SmartPropagatorAnyRK'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    L2TrajBuilderParameters = cms.PSet(
        SeedPropagator = cms.string('SteppingHelixPropagatorAny'),
        NavigationType = cms.string('Standard'),
        SmootherParameters = cms.PSet(

        ),
        SeedPosition = cms.string('in'),
        BWFilterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            BWSeedType = cms.string('fromGenerator'),
            FitDirection = cms.string('outsideIn'),
            DTRecSegmentLabel = cms.InputTag("hltDt4DSegments"),
            MaxChi2 = cms.double(25.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(25.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                Granularity = cms.int32(2)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("hltCscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("hltRpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        RefitterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            FitDirection = cms.string('insideOut'),
            DTRecSegmentLabel = cms.InputTag("hltDt4DSegments"),
            MaxChi2 = cms.double(1000.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(1000.0),
                RescaleError = cms.bool(False),
                RescaleErrorFactor = cms.double(100.0),
                Granularity = cms.int32(0)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("hltCscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("hltRpcRecHits"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        DoSmoothing = cms.bool(False),
        DoBackwardRefit = cms.bool(True)
    ),
    InputObjects = cms.InputTag("hltL2MuonSeeds"),
    TrackLoaderParameters = cms.PSet(
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(False),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        VertexConstraint = cms.bool(True)
    )
)

hltL2MuonCandidates = cms.EDProducer("L2MuonCandidateProducer",
    InputObjects = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)

hltSingleMuIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL1Filtered"),
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltEcalRegionalMuonsFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    debug = cms.untracked.bool(False),
    Muon = cms.untracked.bool(True),
    MuonSource = cms.untracked.InputTag("hltL1extraParticles"),
    OutputLabel = cms.untracked.string('')
)

hltEcalRegionalMuonsDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.string('hltEcalRegionalMuonsFEDs'),
    syncCheck = cms.untracked.bool(False),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    DoRegional = cms.untracked.bool(True)
)

hltEcalRegionalMuonsWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalRegionalMuonsDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalRegionalMuonsDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRegionalMuonsRecHitTmp = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalRegionalMuonsWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalRegionalMuonsWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltEcalRegionalMuonsRecHit = cms.EDFilter("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)

hltTowerMakerForMuons = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRegionalMuonsRecHit","EcalRecHitsEB"), cms.InputTag("hltEcalRegionalMuonsRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltCaloTowersForMuons = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("hltTowerMakerForMuons"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

hltL2MuonIsolations = cms.EDProducer("L2MuonIsolationProducer",
    ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24, 0.24, 0.24, 0.24, 0.24, 
        0.24),
    Thresholds = cms.vdouble(4.0, 3.7, 4.0, 3.5, 3.4, 
        3.4, 3.2, 3.4, 3.1, 2.9, 
        2.9, 2.7, 3.1, 3.0, 2.4, 
        2.1, 2.0, 2.3, 2.2, 2.4, 
        2.5, 2.5, 2.6, 2.9, 3.1, 
        2.9),
    StandAloneCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
        0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
        0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
        1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
        1.785, 1.88, 1.9865, 2.1075, 2.247, 
        2.411),
    OutputMuIsoDeposits = cms.bool(True),
    ExtractorPSet = cms.PSet(
        DR_Veto_H = cms.double(0.1),
        Vertex_Constraint_Z = cms.bool(False),
        Threshold_H = cms.double(0.5),
        ComponentName = cms.string('CaloExtractor'),
        Threshold_E = cms.double(0.2),
        DR_Max = cms.double(0.24),
        DR_Veto_E = cms.double(0.07),
        Weight_E = cms.double(1.5),
        Vertex_Constraint_XY = cms.bool(False),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        CaloTowerCollectionLabel = cms.InputTag("hltTowerMakerForMuons"),
        Weight_H = cms.double(1.0)
    )
)

hltSingleMuIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltL3TrajectorySeed = cms.EDFilter("TSGFromL2Muon",
    tkSeedGenerator = cms.string('TSGFromCombinedHits'),
    TSGFromCombinedHits = cms.PSet(
        firstTSG = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitTripletGenerator'),
                SeedingLayers = cms.string('PixelLayerTriplets'),
                GeneratorPSet = cms.PSet(
                    useBending = cms.bool(True),
                    useFixedPreFiltering = cms.bool(False),
                    ComponentName = cms.string('PixelTripletHLTGenerator'),
                    extraHitRPhitolerance = cms.double(0.06),
                    useMultScattering = cms.bool(True),
                    phiPreFiltering = cms.double(0.3),
                    extraHitRZtolerance = cms.double(0.06)
                )
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        ComponentName = cms.string('CombinedTSG'),
        thirdTSG = cms.PSet(
            PSetNames = cms.vstring('endcapTSG', 
                'barrelTSG'),
            ComponentName = cms.string('DualByEtaTSG'),
            endcapTSG = cms.PSet(
                ComponentName = cms.string('TSGFromOrderedHits'),
                OrderedHitsFactoryPSet = cms.PSet(
                    ComponentName = cms.string('StandardHitPairGenerator'),
                    SeedingLayers = cms.string('MixedLayerPairs')
                ),
                TTRHBuilder = cms.string('WithTrackAngle')
            ),
            etaSeparation = cms.double(2.0),
            barrelTSG = cms.PSet(

            )
        ),
        secondTSG = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitPairGenerator'),
                SeedingLayers = cms.string('PixelLayerPairs')
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        PSetNames = cms.vstring('firstTSG', 
            'secondTSG')
    ),
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'SteppingHelixPropagatorAlong', 
            'SteppingHelixPropagatorOpposite', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite', 
            'SmartPropagator', 
            'SmartPropagatorOpposite', 
            'SmartPropagatorAnyOpposite', 
            'SmartPropagatorAny', 
            'SmartPropagatorRK', 
            'SmartPropagatorAnyRK'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    TSGFromPropagation = cms.PSet(
        ErrorRescaling = cms.double(10.0),
        ComponentName = cms.string('TSGFromPropagation'),
        UpdateState = cms.bool(False),
        UseSecondMeasurements = cms.bool(False),
        MaxChi2 = cms.double(30.0),
        UseVertexState = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyOpposite')
    ),
    TSGFromPixelTriplets = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitTripletGenerator'),
            SeedingLayers = cms.string('PixelLayerTriplets'),
            GeneratorPSet = cms.PSet(
                useBending = cms.bool(True),
                useFixedPreFiltering = cms.bool(False),
                ComponentName = cms.string('PixelTripletHLTGenerator'),
                extraHitRPhitolerance = cms.double(0.06),
                useMultScattering = cms.bool(True),
                phiPreFiltering = cms.double(0.3),
                extraHitRZtolerance = cms.double(0.06)
            )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    TSGForRoadSearchOI = cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
        option = cms.uint32(3),
        ComponentName = cms.string('TSGForRoadSearch'),
        errorMatrixPset = cms.PSet(
            action = cms.string('use'),
            atIP = cms.bool(True),
            errorMatrixValuesPSet = cms.PSet(
                pf3_V12 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V13 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V11 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 4.593, 
                        5.253, 1.895, 1.985, 2.344, 5.37, 
                        2.059, 2.423, 1.985, 2.054, 2.071, 
                        2.232, 2.159, 2.1, 2.355, 3.862, 
                        1.855, 2.311, 1.784, 1.766, 1.999, 
                        2.18, 2.071, 2.03, 2.212, 2.266, 
                        1.693, 1.984, 1.664, 1.602, 1.761, 
                        2.007, 1.985, 1.982, 2.118, 1.734, 
                        1.647, 1.705, 1.56, 1.542, 1.699, 
                        2.058, 2.037, 1.934, 2.067, 1.555, 
                        1.566, 1.638, 1.51, 1.486, 1.635, 
                        1.977, 1.944, 1.865, 1.925, 1.415, 
                        1.542, 1.571, 1.499, 1.468, 1.608, 
                        1.899, 1.893, 1.788, 1.851, 1.22, 
                        1.49, 1.54, 1.493, 1.457, 1.572, 
                        1.876, 1.848, 1.751, 1.827, 1.223, 
                        1.51, 1.583, 1.486, 1.431, 1.534, 
                        1.79, 1.802, 1.65, 1.755, 1.256, 
                        1.489, 1.641, 1.464, 1.438, 1.48, 
                        1.888, 1.839, 1.657, 1.903, 1.899)
                ),
                pf3_V25 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V14 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V15 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V34 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                yAxis = cms.vdouble(0.0, 0.2, 0.3, 0.7, 0.9, 
                    1.15, 1.35, 1.55, 1.75, 2.2, 
                    2.5),
                pf3_V45 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V44 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.622, 2.139, 2.08, 
                        1.178, 1.044, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.01, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.002, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.001, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.001, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.001, 1.0, 1.0, 1.011, 
                        1.001, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.002, 
                        1.0, 1.002, 1.013, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.005, 1.0, 
                        1.004, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.009, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                xAxis = cms.vdouble(0.0, 3.16, 6.69, 10.695, 15.319, 
                    20.787, 27.479, 36.106, 48.26, 69.03, 
                    200.0),
                pf3_V23 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V22 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 66.152, 3.219, 66.051, 
                        1.298, 1.186, 1.197, 1.529, 2.807, 
                        1.056, 1.092, 1.15, 1.158, 1.163, 
                        1.05, 1.191, 1.287, 1.371, 2.039, 
                        1.02, 1.059, 1.048, 1.087, 1.087, 
                        1.041, 1.072, 1.118, 1.097, 1.229, 
                        1.042, 1.07, 1.071, 1.063, 1.039, 
                        1.038, 1.061, 1.052, 1.058, 1.188, 
                        1.099, 1.075, 1.082, 1.055, 1.084, 
                        1.024, 1.058, 1.069, 1.022, 1.184, 
                        1.117, 1.105, 1.093, 1.082, 1.086, 
                        1.053, 1.097, 1.07, 1.044, 1.125, 
                        1.141, 1.167, 1.136, 1.133, 1.146, 
                        1.089, 1.081, 1.117, 1.085, 1.075, 
                        1.212, 1.199, 1.186, 1.212, 1.168, 
                        1.125, 1.127, 1.119, 1.114, 1.062, 
                        1.273, 1.229, 1.272, 1.293, 1.172, 
                        1.124, 1.141, 1.123, 1.158, 1.115, 
                        1.419, 1.398, 1.425, 1.394, 1.278, 
                        1.132, 1.132, 1.115, 1.26, 1.096)
                ),
                pf3_V55 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 27.275, 15.167, 13.818, 
                        1.0, 1.0, 1.0, 1.037, 1.129, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.023, 1.028, 1.063, 1.08, 
                        1.077, 1.054, 1.068, 1.065, 1.047, 
                        1.025, 1.046, 1.064, 1.082, 1.078, 
                        1.137, 1.12, 1.163, 1.158, 1.112, 
                        1.072, 1.054, 1.095, 1.101, 1.092, 
                        1.219, 1.167, 1.186, 1.203, 1.144, 
                        1.096, 1.095, 1.109, 1.111, 1.105, 
                        1.236, 1.187, 1.203, 1.262, 1.2, 
                        1.086, 1.106, 1.112, 1.138, 1.076, 
                        1.287, 1.255, 1.241, 1.334, 1.244, 
                        1.112, 1.083, 1.111, 1.127, 1.025, 
                        1.309, 1.257, 1.263, 1.393, 1.23, 
                        1.091, 1.075, 1.078, 1.135, 1.042, 
                        1.313, 1.303, 1.295, 1.436, 1.237, 
                        1.064, 1.078, 1.075, 1.149, 1.037, 
                        1.329, 1.509, 1.369, 1.546, 1.269, 
                        1.079, 1.084, 1.047, 1.183, 1.008)
                ),
                zAxis = cms.vdouble(-3.14159, 3.14159),
                pf3_V35 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V33 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 6.174, 56.89, 1.019, 
                        2.206, 1.694, 1.698, 1.776, 3.563, 
                        2.141, 2.432, 1.898, 1.834, 1.763, 
                        1.797, 1.944, 1.857, 2.068, 2.894, 
                        1.76, 2.185, 1.664, 1.656, 1.761, 
                        1.964, 1.925, 1.89, 2.012, 2.014, 
                        1.651, 1.825, 1.573, 1.534, 1.634, 
                        1.856, 1.962, 1.879, 1.95, 1.657, 
                        1.556, 1.639, 1.481, 1.433, 1.605, 
                        1.943, 1.99, 1.885, 1.916, 1.511, 
                        1.493, 1.556, 1.445, 1.457, 1.543, 
                        1.897, 1.919, 1.884, 1.797, 1.394, 
                        1.489, 1.571, 1.436, 1.425, 1.534, 
                        1.796, 1.845, 1.795, 1.763, 1.272, 
                        1.472, 1.484, 1.452, 1.412, 1.508, 
                        1.795, 1.795, 1.773, 1.741, 1.207, 
                        1.458, 1.522, 1.437, 1.399, 1.485, 
                        1.747, 1.739, 1.741, 1.716, 1.187, 
                        1.463, 1.589, 1.411, 1.404, 1.471, 
                        1.92, 1.86, 1.798, 1.867, 1.436)
                ),
                pf3_V24 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                )
            )
        ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        maxChi2 = cms.double(40.0)
    ),
    MuonTrackingRegionBuilder = cms.PSet(
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        DeltaR = cms.double(0.2),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        Rescale_Dz = cms.double(3.0),
        vertexCollection = cms.InputTag("pixelVertices"),
        Rescale_phi = cms.double(3.0),
        Rescale_eta = cms.double(3.0),
        DeltaZ_Region = cms.double(15.9),
        Phi_min = cms.double(0.05),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        Eta_min = cms.double(0.05),
        UseFixedRegion = cms.bool(False),
        EscapePt = cms.double(1.5),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        Etafixed = cms.double(0.2),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phifixed = cms.double(0.2),
        UseVertex = cms.bool(False)
    ),
    TSGFromMixedPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('MixedLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TrackerSeedCleaner = cms.PSet(

    ),
    PtCut = cms.double(1.0),
    TSGForRoadSearchIOpxl = cms.PSet(
        propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
        option = cms.uint32(4),
        ComponentName = cms.string('TSGForRoadSearch'),
        errorMatrixPset = cms.PSet(
            action = cms.string('use'),
            atIP = cms.bool(True),
            errorMatrixValuesPSet = cms.PSet(
                pf3_V12 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V13 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V11 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 4.593, 
                        5.253, 1.895, 1.985, 2.344, 5.37, 
                        2.059, 2.423, 1.985, 2.054, 2.071, 
                        2.232, 2.159, 2.1, 2.355, 3.862, 
                        1.855, 2.311, 1.784, 1.766, 1.999, 
                        2.18, 2.071, 2.03, 2.212, 2.266, 
                        1.693, 1.984, 1.664, 1.602, 1.761, 
                        2.007, 1.985, 1.982, 2.118, 1.734, 
                        1.647, 1.705, 1.56, 1.542, 1.699, 
                        2.058, 2.037, 1.934, 2.067, 1.555, 
                        1.566, 1.638, 1.51, 1.486, 1.635, 
                        1.977, 1.944, 1.865, 1.925, 1.415, 
                        1.542, 1.571, 1.499, 1.468, 1.608, 
                        1.899, 1.893, 1.788, 1.851, 1.22, 
                        1.49, 1.54, 1.493, 1.457, 1.572, 
                        1.876, 1.848, 1.751, 1.827, 1.223, 
                        1.51, 1.583, 1.486, 1.431, 1.534, 
                        1.79, 1.802, 1.65, 1.755, 1.256, 
                        1.489, 1.641, 1.464, 1.438, 1.48, 
                        1.888, 1.839, 1.657, 1.903, 1.899)
                ),
                pf3_V25 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V14 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V15 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V34 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                yAxis = cms.vdouble(0.0, 0.2, 0.3, 0.7, 0.9, 
                    1.15, 1.35, 1.55, 1.75, 2.2, 
                    2.5),
                pf3_V45 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V44 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.622, 2.139, 2.08, 
                        1.178, 1.044, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.01, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.002, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.001, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.001, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.001, 1.0, 1.0, 1.011, 
                        1.001, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.002, 
                        1.0, 1.002, 1.013, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.005, 1.0, 
                        1.004, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.009, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                xAxis = cms.vdouble(0.0, 3.16, 6.69, 10.695, 15.319, 
                    20.787, 27.479, 36.106, 48.26, 69.03, 
                    200.0),
                pf3_V23 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V22 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 66.152, 3.219, 66.051, 
                        1.298, 1.186, 1.197, 1.529, 2.807, 
                        1.056, 1.092, 1.15, 1.158, 1.163, 
                        1.05, 1.191, 1.287, 1.371, 2.039, 
                        1.02, 1.059, 1.048, 1.087, 1.087, 
                        1.041, 1.072, 1.118, 1.097, 1.229, 
                        1.042, 1.07, 1.071, 1.063, 1.039, 
                        1.038, 1.061, 1.052, 1.058, 1.188, 
                        1.099, 1.075, 1.082, 1.055, 1.084, 
                        1.024, 1.058, 1.069, 1.022, 1.184, 
                        1.117, 1.105, 1.093, 1.082, 1.086, 
                        1.053, 1.097, 1.07, 1.044, 1.125, 
                        1.141, 1.167, 1.136, 1.133, 1.146, 
                        1.089, 1.081, 1.117, 1.085, 1.075, 
                        1.212, 1.199, 1.186, 1.212, 1.168, 
                        1.125, 1.127, 1.119, 1.114, 1.062, 
                        1.273, 1.229, 1.272, 1.293, 1.172, 
                        1.124, 1.141, 1.123, 1.158, 1.115, 
                        1.419, 1.398, 1.425, 1.394, 1.278, 
                        1.132, 1.132, 1.115, 1.26, 1.096)
                ),
                pf3_V55 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 27.275, 15.167, 13.818, 
                        1.0, 1.0, 1.0, 1.037, 1.129, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.023, 1.028, 1.063, 1.08, 
                        1.077, 1.054, 1.068, 1.065, 1.047, 
                        1.025, 1.046, 1.064, 1.082, 1.078, 
                        1.137, 1.12, 1.163, 1.158, 1.112, 
                        1.072, 1.054, 1.095, 1.101, 1.092, 
                        1.219, 1.167, 1.186, 1.203, 1.144, 
                        1.096, 1.095, 1.109, 1.111, 1.105, 
                        1.236, 1.187, 1.203, 1.262, 1.2, 
                        1.086, 1.106, 1.112, 1.138, 1.076, 
                        1.287, 1.255, 1.241, 1.334, 1.244, 
                        1.112, 1.083, 1.111, 1.127, 1.025, 
                        1.309, 1.257, 1.263, 1.393, 1.23, 
                        1.091, 1.075, 1.078, 1.135, 1.042, 
                        1.313, 1.303, 1.295, 1.436, 1.237, 
                        1.064, 1.078, 1.075, 1.149, 1.037, 
                        1.329, 1.509, 1.369, 1.546, 1.269, 
                        1.079, 1.084, 1.047, 1.183, 1.008)
                ),
                zAxis = cms.vdouble(-3.14159, 3.14159),
                pf3_V35 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                pf3_V33 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 6.174, 56.89, 1.019, 
                        2.206, 1.694, 1.698, 1.776, 3.563, 
                        2.141, 2.432, 1.898, 1.834, 1.763, 
                        1.797, 1.944, 1.857, 2.068, 2.894, 
                        1.76, 2.185, 1.664, 1.656, 1.761, 
                        1.964, 1.925, 1.89, 2.012, 2.014, 
                        1.651, 1.825, 1.573, 1.534, 1.634, 
                        1.856, 1.962, 1.879, 1.95, 1.657, 
                        1.556, 1.639, 1.481, 1.433, 1.605, 
                        1.943, 1.99, 1.885, 1.916, 1.511, 
                        1.493, 1.556, 1.445, 1.457, 1.543, 
                        1.897, 1.919, 1.884, 1.797, 1.394, 
                        1.489, 1.571, 1.436, 1.425, 1.534, 
                        1.796, 1.845, 1.795, 1.763, 1.272, 
                        1.472, 1.484, 1.452, 1.412, 1.508, 
                        1.795, 1.795, 1.773, 1.741, 1.207, 
                        1.458, 1.522, 1.437, 1.399, 1.485, 
                        1.747, 1.739, 1.741, 1.716, 1.187, 
                        1.463, 1.589, 1.411, 1.404, 1.471, 
                        1.92, 1.86, 1.798, 1.867, 1.436)
                ),
                pf3_V24 = cms.PSet(
                    values = cms.vdouble(1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0, 
                        1.0, 1.0, 1.0, 1.0, 1.0)
                )
            )
        ),
        propagatorName = cms.string('SteppingHelixPropagatorAlong'),
        manySeeds = cms.bool(False),
        copyMuonRecHit = cms.bool(False),
        maxChi2 = cms.double(40.0)
    ),
    TSGFromPixelPairs = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('PixelLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

hltL3TrackCandidateFromL2 = cms.EDFilter("CkfTrajectoryMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    trackCandidateAlso = cms.bool(True),
    SeedProducer = cms.string('hltL3TrajectorySeed'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('muonCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL3Muons = cms.EDProducer("L3MuonProducer",
    ServiceParameters = cms.PSet(
        Propagators = cms.untracked.vstring('SteppingHelixPropagatorAny', 
            'SteppingHelixPropagatorAlong', 
            'SteppingHelixPropagatorOpposite', 
            'PropagatorWithMaterial', 
            'PropagatorWithMaterialOpposite', 
            'SmartPropagator', 
            'SmartPropagatorOpposite', 
            'SmartPropagatorAnyOpposite', 
            'SmartPropagatorAny', 
            'SmartPropagatorRK', 
            'SmartPropagatorAnyRK'),
        RPCLayers = cms.bool(True),
        UseMuonNavigation = cms.untracked.bool(True)
    ),
    L3TrajBuilderParameters = cms.PSet(
        MuonHitsOption = cms.int32(1),
        CSCRecSegmentLabel = cms.InputTag("hltCscSegments"),
        StateOnTrackerBoundOutPropagator = cms.string('SmartPropagatorAny'),
        Chi2ProbabilityCut = cms.double(30.0),
        Direction = cms.int32(0),
        HitThreshold = cms.int32(1),
        TrackRecHitBuilder = cms.string('WithTrackAngle'),
        MuonTrackingRegionBuilder = cms.PSet(
            EtaR_UpperLimit_Par1 = cms.double(0.25),
            DeltaR = cms.double(0.2),
            beamSpot = cms.InputTag("hltOfflineBeamSpot"),
            Rescale_Dz = cms.double(3.0),
            vertexCollection = cms.InputTag("pixelVertices"),
            Rescale_phi = cms.double(3.0),
            Rescale_eta = cms.double(3.0),
            DeltaZ_Region = cms.double(15.9),
            Phi_min = cms.double(0.05),
            PhiR_UpperLimit_Par2 = cms.double(0.2),
            Eta_min = cms.double(0.05),
            UseFixedRegion = cms.bool(False),
            EscapePt = cms.double(1.5),
            PhiR_UpperLimit_Par1 = cms.double(0.6),
            Etafixed = cms.double(0.2),
            EtaR_UpperLimit_Par2 = cms.double(0.15),
            Phifixed = cms.double(0.2),
            UseVertex = cms.bool(False)
        ),
        TkTrackBuilder = cms.string('muonCkfTrajectoryBuilder'),
        TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
        DTRecSegmentLabel = cms.InputTag("hltDt4DSegments"),
        Chi2CutCSC = cms.double(150.0),
        Chi2CutRPC = cms.double(1.0),
        GlobalMuonTrackMatcher = cms.PSet(
            MinP = cms.double(2.5),
            Chi2Cut = cms.double(50.0),
            MinPt = cms.double(1.0),
            DeltaDCut = cms.double(10.0),
            DeltaRCut = cms.double(0.2)
        ),
        RPCRecSegmentLabel = cms.InputTag("hltRpcRecHits"),
        tkTrajLabel = cms.InputTag("hltL3TrackCandidateFromL2"),
        SeedGeneratorParameters = cms.PSet(
            ComponentName = cms.string('TSGFromOrderedHits'),
            OrderedHitsFactoryPSet = cms.PSet(
                ComponentName = cms.string('StandardHitPairGenerator'),
                SeedingLayers = cms.string('PixelLayerPairs')
            ),
            TTRHBuilder = cms.string('WithTrackAngle')
        ),
        l3SeedLabel = cms.InputTag(""),
        Chi2CutDT = cms.double(10.0),
        TrackTransformer = cms.PSet(
            Fitter = cms.string('KFFitterForRefitInsideOut'),
            TrackerRecHitBuilder = cms.string('WithTrackAngle'),
            Smoother = cms.string('KFSmootherForRefitInsideOut'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            RefitDirection = cms.string('insideOut'),
            RefitRPCHits = cms.bool(True)
        ),
        PtCut = cms.double(1.0),
        KFFitter = cms.string('L3MuKFFitter')
    ),
    TrackLoaderParameters = cms.PSet(
        PutTkTrackIntoEvent = cms.untracked.bool(True),
        SmoothTkTrack = cms.untracked.bool(False),
        MuonSeededTracksInstance = cms.untracked.string('L2Seeded'),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        VertexConstraint = cms.bool(False),
        DoSmoothing = cms.bool(True)
    ),
    MuonCollectionLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx")
)

hltL3MuonCandidates = cms.EDProducer("L3MuonCandidateProducer",
    InputObjects = cms.InputTag("hltL3Muons")
)

hltSingleMuIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL2IsoFiltered"),
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPixelTracks = cms.EDProducer("PixelTrackProducer",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    FilterPSet = cms.PSet(
        chi2 = cms.double(1000.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        ptMin = cms.double(0.0),
        tipMax = cms.double(1.0)
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(22.7),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.06),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.06)
        )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits')
    )
)

hltL3MuonIsolations = cms.EDProducer("L3MuonIsolationProducer",
    inputMuonCollection = cms.InputTag("hltL3Muons"),
    CutsPSet = cms.PSet(
        ConeSizes = cms.vdouble(0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24, 0.24, 0.24, 0.24, 0.24, 
            0.24),
        ComponentName = cms.string('SimpleCuts'),
        EtaBounds = cms.vdouble(0.0435, 0.1305, 0.2175, 0.3045, 0.3915, 
            0.4785, 0.5655, 0.6525, 0.7395, 0.8265, 
            0.9135, 1.0005, 1.0875, 1.1745, 1.2615, 
            1.3485, 1.4355, 1.5225, 1.6095, 1.6965, 
            1.785, 1.88, 1.9865, 2.1075, 2.247, 
            2.411),
        Thresholds = cms.vdouble(1.1, 1.1, 1.1, 1.1, 1.2, 
            1.1, 1.2, 1.1, 1.2, 1.0, 
            1.1, 1.0, 1.0, 1.1, 1.0, 
            1.0, 1.1, 0.9, 1.1, 0.9, 
            1.1, 1.0, 1.0, 0.9, 0.8, 
            0.1)
    ),
    TrackPt_Min = cms.double(-1.0),
    OutputMuIsoDeposits = cms.bool(True),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("hltPixelTracks"),
        BeamSpotLabel = cms.InputTag("hltOfflineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(0.24),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string('PXLS'),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

hltSingleMuIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescaleSingleMuNoIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuNoIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuNoIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuNoIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuNoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltDiMuonIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltDiMuonIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltDiMuonIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltDiMuonIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonIsoL1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltDiMuonIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltDiMuonIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltDiMuonIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonIsoL2IsoFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltDiMuonIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltDiMuonIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescaleDiMuonNoIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltDiMuonNoIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltDiMuonNoIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltDiMuonNoIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltDiMuonNoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltDiMuonNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleJPsiMM = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltJpsiMMLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltJpsiMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltJpsiMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltJpsiMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltJpsiMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(5.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(1.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltJpsiMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltJpsiMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(3.4),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(2.8),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleUpsilonMM = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltUpsilonMMLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltUpsilonMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltUpsilonMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltUpsilonMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltUpsilonMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(13.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(6.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltUpsilonMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltUpsilonMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(11.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(8.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleZMM = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltZMMLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltZMML1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltZMMLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltZMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(7.0),
    PreviousCandTag = cms.InputTag("hltZMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(9999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(7.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltZMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(7.0),
    PreviousCandTag = cms.InputTag("hltZMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(1e+30),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(7.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(70.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleMultiMuonNoIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMultiMuonNoIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_TripleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMultiMuonNoIsoL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMultiMuonNoIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(3),
    MinQuality = cms.int32(-1)
)

hltMultiMuonNoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMultiMuonNoIsoL1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(3),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMultiMuonNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMultiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(3),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSameSignMu = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSameSignMuLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSameSignMuL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSameSignMuLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltSameSignMuL2PreFiltered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltSameSignMuL1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(1),
    MaxInvMass = cms.double(9999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltSameSignMuL3PreFiltered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltSameSignMuL2PreFiltered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(1),
    MaxInvMass = cms.double(9999.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleSingleMuPrescale3 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuPrescale3Level1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuPrescale3L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuPrescale3Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuPrescale3L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale3L1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuPrescale3L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale3L2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuPrescale5 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuPrescale5Level1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu5'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuPrescale5L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuPrescale5Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuPrescale5L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale5L1Filtered"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuPrescale5L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale5L2PreFiltered"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPreSingleMuPrescale77 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(400)
)

hltSingleMuPrescale77Level1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuPrescale77L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuPrescale77Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuPrescale77L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale77L1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuPrescale77L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale77L2PreFiltered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPreSingleMuPrescale710 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(100)
)

hltSingleMuPrescale710Level1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuPrescale710L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuPrescale710Level1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuPrescale710L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale710L1Filtered"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuPrescale710L3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuPrescale710L2PreFiltered"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleMuLevel1Path = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1000)
)

hltMuLevel1PathLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7 OR L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuLevel1PathL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuLevel1PathLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltPrescaleSingleMuNoIsoRelaxedVtx2cm = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuNoIsoRelaxedVtx2mm = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(16.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(0.2),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonNoIsoRelaxedVtx2cm = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonNoIsoRelaxedVtx2mm = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(0.2),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescalerBLifetime1jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBLifetimeL1seeds = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet40 OR L1_HTT300'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltBLifetime1jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(180.0),
    MinN = cms.int32(1)
)

hltBLifetimeHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    maxNumber = cms.uint32(4)
)

hltBLifetimeL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBLifetimeHighestEtJets"),
    etMin = cms.double(35.0)
)

hltPixelVertices = cms.EDProducer("PixelVertexProducer",
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Verbosity = cms.int32(0),
    UseError = cms.bool(True),
    TrackCollection = cms.InputTag("hltPixelTracks"),
    ZSeparation = cms.double(0.05),
    NTrkMin = cms.int32(2),
    Method2 = cms.bool(True),
    Finder = cms.string('DivisiveVertexFinder'),
    PtMin = cms.double(1.0)
)

hltBLifetimeL25Associator = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltBLifetimeL25Jets"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltBLifetimeL25TagInfos = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(3),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("hltPixelVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("hltBLifetimeL25Associator"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(False),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(5.0)
)

hltBLifetimeL25BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBLifetimeL25TagInfos"),
    jetTagComputer = cms.string('trackCounting3D2nd')
)

hltBLifetimeL25filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL25BJetTags"),
    MinTag = cms.double(3.5),
    MaxTag = cms.double(99999.0),
    MinN = cms.int32(1)
)

hltBLifetimeL3Jets = cms.EDFilter("GetJetsFromHLTobject",
    jets = cms.InputTag("hltBLifetimeL25filter")
)

hltBLifetimeRegionalPixelSeedGenerator = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.25),
            originHalfLength = cms.double(0.2),
            originZPos = cms.double(0.0),
            deltaEtaRegion = cms.double(0.25),
            ptMin = cms.double(1.0),
            JetSrc = cms.InputTag("hltBLifetimeL3Jets"),
            originRadius = cms.double(0.2),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltBLifetimeRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltBLifetimeRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('bJetRegionalTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltBLifetimeRegionalCtfWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltBLifetimeRegionalCkfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltBLifetimeL3Associator = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltBLifetimeL3Jets"),
    tracks = cms.InputTag("hltBLifetimeRegionalCtfWithMaterialTracks"),
    coneSize = cms.double(0.5)
)

hltBLifetimeL3TagInfos = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("hltPixelVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("hltBLifetimeL3Associator"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(False),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(5.0)
)

hltBLifetimeL3BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBLifetimeL3TagInfos"),
    jetTagComputer = cms.string('trackCounting3D2nd')
)

hltBLifetimeL3filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL3BJetTags"),
    MinTag = cms.double(6.0),
    MaxTag = cms.double(99999.0),
    MinN = cms.int32(1)
)

hltPrescalerBLifetime2jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBLifetime2jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(120.0),
    MinN = cms.int32(2)
)

hltPrescalerBLifetime3jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBLifetime3jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(70.0),
    MinN = cms.int32(3)
)

hltPrescalerBLifetime4jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBLifetime4jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(4)
)

hltPrescalerBLifetimeHT = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBLifetimeHTL2filter = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(470.0)
)

hltPrescalerBSoftmuon1jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(20)
)

hltBSoftmuonNjetL1seeds = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_Jet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltBSoftmuon1jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(20.0),
    MinN = cms.int32(1)
)

hltBSoftmuonHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    maxNumber = cms.uint32(2)
)

hltBSoftmuonL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBSoftmuonHighestEtJets"),
    etMin = cms.double(20.0)
)

hltBSoftmuonL25TagInfos = cms.EDFilter("SoftLepton",
    leptons = cms.InputTag("hltL2Muons"),
    primaryVertex = cms.InputTag("nominal"),
    refineJetAxis = cms.uint32(0),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("hltBSoftmuonL25Jets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(0.0)
)

hltBSoftmuonL25BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL25TagInfos"),
    jetTagComputer = cms.string('softLeptonByDistance')
)

hltBSoftmuonL25filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL25BJetTags"),
    MinTag = cms.double(0.5),
    MaxTag = cms.double(99999.0),
    MinN = cms.int32(1)
)

hltBSoftmuonL3TagInfos = cms.EDFilter("SoftLepton",
    leptons = cms.InputTag("hltL3Muons"),
    primaryVertex = cms.InputTag("nominal"),
    refineJetAxis = cms.uint32(0),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("hltBSoftmuonL25Jets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(0.0)
)

hltBSoftmuonL3BJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    jetTagComputer = cms.string('softLeptonByPt')
)

hltBSoftmuonL3BJetTagsByDR = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    jetTagComputer = cms.string('softLeptonByDistance')
)

hltBSoftmuonByDRL3filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL3BJetTagsByDR"),
    MinTag = cms.double(0.5),
    MaxTag = cms.double(99999.0),
    MinN = cms.int32(1)
)

hltPrescalerBSoftmuon2jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBSoftmuon2jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(120.0),
    MinN = cms.int32(2)
)

hltBSoftmuonL3filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL3BJetTags"),
    MinTag = cms.double(0.7),
    MaxTag = cms.double(99999.0),
    MinN = cms.int32(1)
)

hltPrescalerBSoftmuon3jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBSoftmuon3jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(70.0),
    MinN = cms.int32(3)
)

hltPrescalerBSoftmuon4jet = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBSoftmuon4jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(4)
)

hltPrescalerBSoftmuonHT = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltBSoftmuonHTL1seeds = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_HTT300'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltBSoftmuonHTL2filter = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(370.0)
)

hltJpsitoMumuL1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltJpsitoMumuL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltJpsitoMumuL1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltJpsitoMumuL2Filtered = cms.EDFilter("HLTMuonDimuonFilter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(4.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(-1),
    MaxInvMass = cms.double(10.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(4.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(2.0)
)

hltMumuPixelSeedFromL2Candidate = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('L3MumuTrackingRegion'),
        RegionPSet = cms.PSet(
            deltaPhiRegion = cms.double(0.15),
            TrkSrc = cms.InputTag("hltL2Muons"),
            originHalfLength = cms.double(1.0),
            deltaEtaRegion = cms.double(0.15),
            vertexZDefault = cms.double(0.0),
            vertexSrc = cms.string('hltPixelVertices'),
            originRadius = cms.double(1.0),
            ptMin = cms.double(3.0)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesMumu = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltMumuPixelSeedFromL2Candidate'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('hltCkfTrajectoryBuilderMumu'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksMumu = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesMumu"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltMuTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumu"),
    particleType = cms.string('mu-')
)

hltDisplacedJpsitoMumuFilter = cms.EDFilter("HLTDisplacedmumuFilter",
    Src = cms.InputTag("hltMuTracks"),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(4.0),
    ChargeOpt = cms.int32(-1),
    MaxEta = cms.double(2.5),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(6.0),
    MinPtPair = cms.double(4.0),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(1.0)
)

hltMuMukL1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuMukL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuMukL1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltMuMukL2Filtered = cms.EDFilter("HLTMuonDimuonFilter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(1000.0),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    MinPtPair = cms.double(0.0),
    MaxDr = cms.double(100.0),
    CandTag = cms.InputTag("hltL2MuonCandidates"),
    MinInvMass = cms.double(0.2),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(2.0)
)

hltDisplacedMuMukFilter = cms.EDFilter("HLTDisplacedmumuFilter",
    Src = cms.InputTag("hltMuTracks"),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(3.0),
    ChargeOpt = cms.int32(0),
    MaxEta = cms.double(2.5),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(3.0),
    MinPtPair = cms.double(0.0),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(0.2)
)

hltMumukPixelSeedFromL2Candidate = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('L3MumuTrackingRegion'),
        RegionPSet = cms.PSet(
            deltaPhiRegion = cms.double(0.15),
            TrkSrc = cms.InputTag("hltL2Muons"),
            originHalfLength = cms.double(1.0),
            deltaEtaRegion = cms.double(0.15),
            vertexZDefault = cms.double(0.0),
            vertexSrc = cms.string('hltPixelVertices'),
            originRadius = cms.double(1.0),
            ptMin = cms.double(3.0)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesMumuk = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltMumukPixelSeedFromL2Candidate'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('hltCkfTrajectoryBuilderMumuk'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksMumuk = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesMumuk"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltMumukAllConeTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumuk"),
    particleType = cms.string('mu-')
)

hltmmkFilter = cms.EDFilter("HLTmmkFilter",
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(3.0),
    TrackCand = cms.InputTag("hltMumukAllConeTracks"),
    MaxEta = cms.double(2.5),
    ThirdTrackMass = cms.double(0.106),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(2.2),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(1.2),
    MuCand = cms.InputTag("hltMuTracks")
)

hltElectronBPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltElectronBL1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_IsoEG10_Jet20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltElBElectronL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltElectronBL1Seed"),
    candNonIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltElBElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltElBElectronL1MatchFilter")
)

hltElBElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltElBElectronEtFilter")
)

hltElBElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltElBElectronHcalIsolFilter")
)

hltElBElectronEoverpFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltElBElectronPixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltElBElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltElBElectronEoverpFilter")
)

hltMuBPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMuBLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_Jet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuBLifetimeL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuBLevel1Seed"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuBLifetimeIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuBLifetimeL1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuBLifetimeIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuBLifetimeIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltMuBLifetimeIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuBLifetimeIsoL2IsoFiltered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuBLifetimeIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuBLifetimeIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuBsoftMuPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMuBSoftL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuBLevel1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltMuBSoftIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuBSoftL1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuBSoftIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuBSoftIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltMuBSoftIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuBSoftIsoL2IsoFiltered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuBSoftIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuBSoftIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltL1seedEJet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_IsoEG10_Jet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1IsoEJetSingleEEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltL1IsoSingleL1MatchFilter")
)

hltL1IsoEJetSingleEHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEtFilter")
)

hltL1IsoEJetSingleEPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoEJetSingleEHcalIsolFilter")
)

hltL1IsoEJetSingleEEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(2.45),
    ncandcut = cms.int32(1),
    eoverpbarrelcut = cms.double(2.0),
    candTag = cms.InputTag("hltL1IsoEJetSingleEPixelMatchFilter")
)

hltL1IsoEJetSingleETrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEoverpFilter")
)

hltej1jet40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(1)
)

hltej2jet80 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(80.0),
    MinN = cms.int32(2)
)

hltej3jet60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(3)
)

hltej4jet35 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(4)
)

hltMuJetsPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMuJetsLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_Jet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuJetsL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuJetsLevel1Seed"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuJetsL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuJetsL1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuJetsL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuJetsL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltMuJetsL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuJetsL2IsoFiltered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuJetsL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuJetsL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuJetsHLT1jet40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(1)
)

hltMuNoL2IsoJetsPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMuNoL2IsoJetsLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_Jet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuNoL2IsoJetsL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuNoL2IsoJetsLevel1Seed"),
    MinPt = cms.double(8.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuNoL2IsoJetsL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoL2IsoJetsL1Filtered"),
    MinPt = cms.double(8.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuNoL2IsoJetsL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoL2IsoJetsL2PreFiltered"),
    MinPt = cms.double(8.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuNoL2IsoJetsL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuNoL2IsoJetsL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuNoL2IsoJetsHLT1jet40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(1)
)

hltMuNoIsoJetsPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltMuNoIsoJetsLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_Jet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuNoIsoJetsL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuNoIsoJetsLevel1Seed"),
    MinPt = cms.double(14.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuNoIsoJetsL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoIsoJetsL1Filtered"),
    MinPt = cms.double(14.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuNoIsoJetsL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoIsoJetsL2PreFiltered"),
    MinPt = cms.double(14.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuNoIsoJetsHLT1jet50 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(50.0),
    MinN = cms.int32(1)
)

hltemuPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltEMuonLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu3_IsoEG5'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltEMuL1MuonFilter = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltEMuonLevel1Seed"),
    MinPt = cms.double(4.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltemuL1IsoSingleL1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltEMuonLevel1Seed"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltemuL1IsoSingleElectronEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltemuL1IsoSingleL1MatchFilter")
)

hltemuL1IsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronEtFilter")
)

hltEMuL2MuonPreFilter = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltEMuL1MuonFilter"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltEMuL2MuonIsoFilter = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltEMuL2MuonPreFilter"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltemuL1IsoSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronHcalIsolFilter")
)

hltemuL1IsoSingleElectronEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(2.45),
    ncandcut = cms.int32(1),
    eoverpbarrelcut = cms.double(1.5),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronPixelMatchFilter")
)

hltEMuL3MuonPreFilter = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltEMuL2MuonIsoFilter"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltEMuL3MuonIsoFilter = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltEMuL3MuonPreFilter"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltemuL1IsoSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronEoverpFilter")
)

hltemuNonIsoPrescale = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltemuNonIsoLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu3_EG12'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltNonIsoEMuL1MuonFilter = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltemuNonIsoLevel1Seed"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltemuNonIsoL1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltemuNonIsoLevel1Seed"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltemuNonIsoL1IsoEtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltemuNonIsoL1MatchFilterRegional")
)

hltemuNonIsoL1HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltemuNonIsoL1IsoEtFilter")
)

hltNonIsoEMuL2MuonPreFilter = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltNonIsoEMuL1MuonFilter"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltemuNonIsoL1IsoPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltemuNonIsoL1HcalIsolFilter")
)

hltemuNonIsoL1IsoEoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(2.45),
    ncandcut = cms.int32(1),
    eoverpbarrelcut = cms.double(1.5),
    candTag = cms.InputTag("hltemuNonIsoL1IsoPixelMatchFilter")
)

hltNonIsoEMuL3MuonPreFilter = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltNonIsoEMuL2MuonPreFilter"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltemuNonIsoL1IsoTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltemuNonIsoL1IsoEoverpFilter")
)

hltPrescalerElectronTau = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltLevel1GTSeedElectronTau = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_IsoEG10_TauJet20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltEgammaL1MatchFilterRegionalElectronTau = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltLevel1GTSeedElectronTau"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltEgammaEtFilterElectronTau = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(1),
    inputTag = cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau")
)

hltEgammaHcalIsolFilterElectronTau = cms.EDFilter("HLTEgammaHcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltEgammaEtFilterElectronTau")
)

hltElectronPixelMatchFilterElectronTau = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltEgammaHcalIsolFilterElectronTau")
)

hltElectronOneOEMinusOneOPFilterElectronTau = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsL1NonIsoForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltElectronPixelMatchFilterElectronTau"),
    endcapcut = cms.double(999.03)
)

hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltElectronOneOEMinusOneOPFilterElectronTau")
)

hltEcalRegionalTausFEDs = cms.EDProducer("EcalListOfFEDSProducer",
    JETS_doForward = cms.untracked.bool(False),
    OutputLabel = cms.untracked.string(''),
    JETS_doCentral = cms.untracked.bool(False),
    ForwardSource = cms.untracked.InputTag("hltL1extraParticles","Forward"),
    TauSource = cms.untracked.InputTag("hltL1extraParticles","Tau"),
    CentralSource = cms.untracked.InputTag("hltL1extraParticles","Central"),
    Ptmin_jets = cms.untracked.double(20.0),
    debug = cms.untracked.bool(False),
    Jets = cms.untracked.bool(True)
)

hltEcalRegionalTausDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    FedLabel = cms.untracked.string('hltEcalRegionalTausFEDs'),
    syncCheck = cms.untracked.bool(False),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    DoRegional = cms.untracked.bool(True)
)

hltEcalRegionalTausWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalRegionalTausDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalRegionalTausDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRegionalTausRecHitTmp = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalRegionalTausWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalRegionalTausWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltEcalRegionalTausRecHit = cms.EDFilter("EcalRecHitsMerger",
    EgammaSource_EB = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEB"),
    MuonsSource_EB = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EB = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEB"),
    JetsSource_EE = cms.untracked.InputTag("hltEcalRegionalJetsRecHitTmp","EcalRecHitsEE"),
    MuonsSource_EE = cms.untracked.InputTag("hltEcalRegionalMuonsRecHitTmp","EcalRecHitsEE"),
    EcalRecHitCollectionEB = cms.untracked.string('EcalRecHitsEB'),
    RestSource_EE = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEE"),
    RestSource_EB = cms.untracked.InputTag("hltEcalRegionalRestRecHitTmp","EcalRecHitsEB"),
    TausSource_EB = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEB"),
    TausSource_EE = cms.untracked.InputTag("hltEcalRegionalTausRecHitTmp","EcalRecHitsEE"),
    debug = cms.untracked.bool(False),
    EcalRecHitCollectionEE = cms.untracked.string('EcalRecHitsEE'),
    OutputLabel_EE = cms.untracked.string('EcalRecHitsEE'),
    EgammaSource_EE = cms.untracked.InputTag("hltEcalRegionalEgammaRecHitTmp","EcalRecHitsEE"),
    OutputLabel_EB = cms.untracked.string('EcalRecHitsEB')
)

hltTowerMakerForTaus = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEB"), cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltCaloTowersTau1Regional = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForTaus"),
    TauId = cms.int32(0),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau1Regional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau1Regional"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau2Regional = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForTaus"),
    TauId = cms.int32(1),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau2Regional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau2Regional"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau3Regional = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForTaus"),
    TauId = cms.int32(2),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau3Regional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau3Regional"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau4Regional = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForTaus"),
    TauId = cms.int32(3),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau4Regional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau4Regional"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltL2TauJetsProviderElectronTau = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1Regional"), cms.InputTag("hltIcone5Tau2Regional"), cms.InputTag("hltIcone5Tau3Regional"), cms.InputTag("hltIcone5Tau4Regional")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltLevel1GTSeedElectronTau")
)

hltJetCrystalsAssociatorElectronTau = cms.EDFilter("JetCrystalsAssociator",
    jets = cms.InputTag("hltL2TauJetsProviderElectronTau"),
    EBRecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"),
    coneSize = cms.double(0.5),
    EERecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE")
)

hltEcalIsolationElectronTau = cms.EDFilter("EcalIsolation",
    JetForFilter = cms.InputTag("hltJetCrystalsAssociatorElectronTau"),
    SmallCone = cms.double(0.13),
    BigCone = cms.double(0.4),
    Pisol = cms.double(5.0)
)

hltEMIsolatedTauJetsSelectorElectronTau = cms.EDFilter("EMIsolatedTauJetsSelector",
    TauSrc = cms.VInputTag(cms.InputTag("hltEcalIsolationElectronTau"))
)

hltFilterEcalIsolatedTauJetsElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltEMIsolatedTauJetsSelectorElectronTau","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltJetsPixelTracksAssociatorElectronTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltEMIsolatedTauJetsSelectorElectronTau","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltPixelTrackConeIsolationElectronTau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(0.03),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltJetsPixelTracksAssociatorElectronTau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltPixelTrackIsolatedTauJetsSelectorElectronTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltPixelTrackConeIsolationElectronTau")),
    IsolationCone = cms.double(0.3),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterPixelTrackIsolatedTauJetsElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1)
)

hltLevel1seedHLTBackwardBSC = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('2'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTBackwardBSC = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltLevel1seedHLTForwardBSC = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('1'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTForwardBSC = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltLevel1seedHLTCSCBeamHalo = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltG1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTCSCBeamHalo = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltLevel1seedHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltOverlapsHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTCSCOverlapFilter",
    fillHists = cms.bool(False),
    minHits = cms.uint32(4),
    ring2 = cms.bool(False),
    ring1 = cms.bool(True),
    yWindow = cms.double(2.0),
    input = cms.InputTag("hltCsc2DRecHits"),
    xWindow = cms.double(2.0)
)

hltLevel1seedHLTCSCBeamHaloOverlapRing2 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTCSCBeamHaloOverlapRing2 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltOverlapsHLTCSCBeamHaloOverlapRing2 = cms.EDFilter("HLTCSCOverlapFilter",
    fillHists = cms.bool(False),
    minHits = cms.uint32(4),
    ring2 = cms.bool(True),
    ring1 = cms.bool(False),
    yWindow = cms.double(2.0),
    input = cms.InputTag("hltCsc2DRecHits"),
    xWindow = cms.double(2.0)
)

hltLevel1seedHLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltFilter23HLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTCSCRing2or3Filter",
    input = cms.InputTag("hltCsc2DRecHits"),
    xWindow = cms.double(2.0),
    minHits = cms.uint32(4),
    yWindow = cms.double(2.0)
)

hltLevel1seedHLTTrackerCosmics = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('0'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTTrackerCosmics = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPrePi0Ecal = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1sEcalPi0 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15 OR L1_SingleJet20 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltAlCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleNRHMax = cms.int32(75),
    seleMinvMaxPi0 = cms.double(0.16),
    gammaCandPhiSize = cms.int32(21),
    clusPhiSize = cms.int32(3),
    gammaCandEtaSize = cms.int32(21),
    clusEtaSize = cms.int32(3),
    barrelHits = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB"),
    seleMinvMinPi0 = cms.double(0.09),
    selePtGammaTwo = cms.double(1.0),
    selePtPi0 = cms.double(2.5),
    seleXtalMinEnergy = cms.double(0.0),
    selePtGammaOne = cms.double(1.0),
    clusSeedThr = cms.double(0.5)
)

hltL1sEcalPhiSym = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltEcalPhiSymPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltEcalDigis = cms.EDFilter("EcalRawToDigiDev",
    orderedDCCIdList = cms.untracked.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 
        21, 22, 23, 24, 25, 
        26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 
        36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 
        46, 47, 48, 49, 50, 
        51, 52, 53, 54),
    orderedFedList = cms.untracked.vint32(601, 602, 603, 604, 605, 
        606, 607, 608, 609, 610, 
        611, 612, 613, 614, 615, 
        616, 617, 618, 619, 620, 
        621, 622, 623, 624, 625, 
        626, 627, 628, 629, 630, 
        631, 632, 633, 634, 635, 
        636, 637, 638, 639, 640, 
        641, 642, 643, 644, 645, 
        646, 647, 648, 649, 650, 
        651, 652, 653, 654),
    eventPut = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('rawDataCollector'),
    syncCheck = cms.untracked.bool(False)
)

hltEcalWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("hltEcalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("hltEcalDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

hltEcalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("hltEcalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("hltEcalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

hltAlCaPhiSymStream = cms.EDFilter("HLTEcalPhiSymFilter",
    endcapHitCollection = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
    eCut_barrel = cms.double(0.15),
    eCut_endcap = cms.double(0.75),
    phiSymBarrelHitCollection = cms.string('phiSymEcalRecHitsEB'),
    barrelHitCollection = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    phiSymEndcapHitCollection = cms.string('phiSymEcalRecHitsEE')
)

hltL1sHcalPhiSym = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltHcalPhiSymPresc = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltAlCaHcalPhiSymStream = cms.EDFilter("HLTHcalPhiSymFilter",
    eCut_HE = cms.double(-10.0),
    eCut_HF = cms.double(-10.0),
    eCut_HB = cms.double(-10.0),
    eCut_HO = cms.double(-10.0),
    phiSymHOHitCollection = cms.string('phiSymHcalRecHitsHO'),
    HFHitCollection = cms.InputTag("hltHfreco"),
    phiSymHBHEHitCollection = cms.string('phiSymHcalRecHitsHBHE'),
    HOHitCollection = cms.InputTag("hltHoreco"),
    phiSymHFHitCollection = cms.string('phiSymHcalRecHitsHF'),
    HBHEHitCollection = cms.InputTag("hltHbhereco")
)

hltL1sIsolTrack = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet100 OR L1_SingleTauJet100'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPreIsolTrack = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltEcalIsolPartProd = cms.EDProducer("EcalIsolatedParticleCandidateProducer",
    ECHitEnergyThreshold = cms.double(0.05),
    L1eTauJetsSource = cms.InputTag("hltL1extraParticles","Tau"),
    L1GTSeedLabel = cms.InputTag("hltL1sIsolTrack"),
    EBrecHitCollectionLabel = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    ECHitCountEnergyThreshold = cms.double(0.5),
    EcalInnerConeSize = cms.double(0.3),
    EcalOuterConeSize = cms.double(0.7),
    EErecHitCollectionLabel = cms.InputTag("hltEcalRecHit","EcalRecHitsEE")
)

hltEcalIsolFilter = cms.EDFilter("HLTEcalIsolationFilter",
    MaxNhitInnerCone = cms.int32(1000),
    MaxNhitOuterCone = cms.int32(0),
    EcalIsolatedParticleSource = cms.InputTag("hltEcalIsolPartProd"),
    MaxEnergyOuterCone = cms.double(10000.0),
    MaxEtaCandidate = cms.double(1.3),
    MaxEnergyInnerCone = cms.double(10000.0)
)

hltIsolPixelTrackProd = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    L1GtObjectMapSource = cms.InputTag("l1GtEmulDigis"),
    L1eTauJetsSource = cms.InputTag("hltL1extraParticles","Tau"),
    tauAssociationCone = cms.double(0.5),
    PixelTracksSource = cms.InputTag("hltPixelTracks"),
    L1GTSeedLabel = cms.InputTag("hltL1sIsolTrack"),
    tauUnbiasCone = cms.double(0.0),
    PixelIsolationConeSize = cms.double(0.2),
    ecalFilterLabel = cms.InputTag("aaa")
)

hltIsolPixelTrackFilter = cms.EDFilter("HLTPixelIsolTrackFilter",
    MaxPtNearby = cms.double(2.0),
    candTag = cms.InputTag("hltIsolPixelTrackProd"),
    MaxEtaTrack = cms.double(1.3),
    MinPtTrack = cms.double(20.0)
)

hltPreIsolTrackNoEcalIso = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPreMinBiasPixel = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltL1seedMinBiasPixel = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPixelTracksForMinBias = cms.EDProducer("PixelTrackProducer",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    FilterPSet = cms.PSet(
        chi2 = cms.double(1000.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        ptMin = cms.double(0.0),
        tipMax = cms.double(1.0)
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(22.7),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.2),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.06),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.06)
        )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits')
    )
)

hltPixelCands = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltPixelTracksForMinBias"),
    particleType = cms.string('pi+')
)

hltMinBiasPixelFilter = cms.EDFilter("HLTPixlMBFilt",
    pixlTag = cms.InputTag("hltPixelCands"),
    MinTrks = cms.uint32(2),
    MinPt = cms.double(0.0),
    MinSep = cms.double(1.0)
)

hltPreMBForAlignment = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPixelMBForAlignment = cms.EDFilter("HLTPixlMBForAlignmentFilter",
    pixlTag = cms.InputTag("hltPixelCands"),
    MinIsol = cms.double(0.05),
    MinTrks = cms.uint32(2),
    MinPt = cms.double(5.0),
    MinSep = cms.double(1.0)
)

hltl1sMin = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_MinBias_HTT10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltpreMin = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltl1sZero = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltpreZero = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltPrescaleTriggerType = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltFilterTriggerType = cms.EDFilter("TriggerTypeFilter",
    TriggerFedId = cms.int32(812),
    InputLabel = cms.string('rawDataCollector'),
    SelectedTriggerType = cms.int32(2)
)

hltL1gtTrigReport = cms.EDFilter("L1GtTrigReport",
    L1GtDaqInputTag = cms.InputTag("hltGtDigis"),
    UseL1GlobalTriggerRecord = cms.bool(False),
    L1GtRecordInputTag = cms.InputTag("hltGtDigis")
)

hltTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

hltL2ElectronTauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(False),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2TauJetsProviderElectronTau"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2ElectronTauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltJetTracksAssociatorAtVertexL25ElectronTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25ElectronTau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltJetsPixelTracksAssociatorElectronTau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedTauJetsSelectorL25ElectronTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25ElectronTau")),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(5.0),
    UseVertex = cms.bool(False)
)

hltFilterIsolatedTauJetsL25ElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltPrescalerMuonTau = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltLevel1GTSeedMuonTau = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu5_TauJet20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuonTauL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltLevel1GTSeedMuonTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuonTauIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuonTauL1Filtered"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(3.9),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuonTauIsoL2IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuonTauIsoL2PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltL2TauJetsProviderMuonTau = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1Regional"), cms.InputTag("hltIcone5Tau2Regional"), cms.InputTag("hltIcone5Tau3Regional"), cms.InputTag("hltIcone5Tau4Regional")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltLevel1GTSeedMuonTau")
)

hltL2MuonTauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(False),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2TauJetsProviderMuonTau"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2MuonTauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2MuonTauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltFilterEcalIsolatedTauJetsMuonTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltJetsPixelTracksAssociatorMuonTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2MuonTauIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltPixelTrackConeIsolationMuonTau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltJetsPixelTracksAssociatorMuonTau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    useVertex = cms.bool(True)
)

hltPixelTrackIsolatedTauJetsSelectorMuonTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltPixelTrackConeIsolationMuonTau")),
    IsolationCone = cms.double(0.5),
    MinimumTransverseMomentumLeadingTrack = cms.double(5.0),
    UseVertex = cms.bool(False)
)

hltFilterPixelTrackIsolatedTauJetsMuonTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1)
)

hltMuonTauIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuonTauIsoL2IsoFiltered"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    NSigmaPt = cms.double(2.2),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuonTauIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuonTauIsoL3PreFiltered"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltSingleTauMETPrescaler = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleTauMETL1SeedFilter = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_TauJet30_ETM30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltCaloTowersTau1 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(0),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau1 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau1"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau2 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(1),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau2 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau2"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau3 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(2),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau3 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau3"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hltCaloTowersTau4 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(3),
    TauTrigger = cms.InputTag("hltL1extraParticles","Tau"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Tau4 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersTau4"),
    verbose = cms.untracked.bool(False),
    jetPtMin = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

hlt1METSingleTauMET = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(1)
)

hltL2SingleTauMETJets = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1"), cms.InputTag("hltIcone5Tau2"), cms.InputTag("hltIcone5Tau3"), cms.InputTag("hltIcone5Tau4")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltSingleTauMETL1SeedFilter")
)

hltL2SingleTauMETIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(False),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2SingleTauMETJets"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2SingleTauMETIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2SingleTauMETIsolationProducer","L2TauIsolationInfoAssociator")
)

hltFilterSingleTauMETEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2SingleTauMETIsolationSelector","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltAssociatorL25SingleTauMET = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2SingleTauMETIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25SingleTauMET = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25SingleTauMET"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedL25SingleTauMET = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25SingleTauMET")),
    IsolationCone = cms.double(0.5),
    MinimumTransverseMomentumLeadingTrack = cms.double(5.0),
    UseVertex = cms.bool(False)
)

hltFilterL25SingleTauMET = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL25SingleTauMET"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1)
)

hltL3SingleTauMETPixelSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.1),
            originHalfLength = cms.double(0.2),
            originRadius = cms.double(0.2),
            deltaEtaRegion = cms.double(0.1),
            ptMin = cms.double(10.0),
            JetSrc = cms.InputTag("hltIsolatedL25SingleTauMET"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesL3SingleTauMET = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL3SingleTauMETPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('trajBuilderL3'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksL3SingleTauMET = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesL3SingleTauMET"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltAssociatorL3SingleTauMET = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedL25SingleTauMET"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3SingleTauMET"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL3SingleTauMET = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(5),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL3SingleTauMET"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedL3SingleTauMET = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3SingleTauMET")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(15.0),
    UseVertex = cms.bool(False)
)

hltFilterL3SingleTauMET = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL3SingleTauMET"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1)
)

hltDoubleTauPrescaler = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltDoubleTauL1SeedFilter = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleTauJet40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL2DoubleTauJets = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1Regional"), cms.InputTag("hltIcone5Tau2Regional"), cms.InputTag("hltIcone5Tau3Regional"), cms.InputTag("hltIcone5Tau4Regional")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltDoubleTauL1SeedFilter")
)

hltL2DoubleTauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(False),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRegionalTausRecHit","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2DoubleTauJets"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2DoubleTauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2DoubleTauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltFilterDoubleTauEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(2)
)

hltAssociatorL25PixelTauIsolated = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2DoubleTauIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25PixelTauIsolated = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25PixelTauIsolated"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedL25PixelTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25PixelTauIsolated")),
    IsolationCone = cms.double(0.3),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL25PixelTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL25PixelTau"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(2)
)

hltSingleTauPrescaler = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltSingleTauL1SeedFilter = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleTauJet80'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hlt1METSingleTau = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(65.0),
    MinN = cms.int32(1)
)

hltL2SingleTauJets = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1"), cms.InputTag("hltIcone5Tau2"), cms.InputTag("hltIcone5Tau3"), cms.InputTag("hltIcone5Tau4")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltSingleTauL1SeedFilter")
)

hltL2SingleTauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(False),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2SingleTauJets"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2SingleTauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltFilterSingleTauEcalIsolation = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltAssociatorL25SingleTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2SingleTauIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25SingleTau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25SingleTau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedL25SingleTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25SingleTau")),
    IsolationCone = cms.double(0.5),
    MinimumTransverseMomentumLeadingTrack = cms.double(5.0),
    UseVertex = cms.bool(False)
)

hltFilterL25SingleTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL25SingleTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltL3SingleTauPixelSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.1),
            originHalfLength = cms.double(0.2),
            originRadius = cms.double(0.2),
            deltaEtaRegion = cms.double(0.1),
            ptMin = cms.double(10.0),
            JetSrc = cms.InputTag("hltIsolatedL25SingleTau"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesL3SingleTau = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL3SingleTauPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('trajBuilderL3'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksL3SingleTau = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesL3SingleTau"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltAssociatorL3SingleTau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedL25SingleTau"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3SingleTau"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL3SingleTau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(5),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL3SingleTau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("hltOfflineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    useVertex = cms.bool(True)
)

hltIsolatedL3SingleTau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3SingleTau")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(20.0),
    UseVertex = cms.bool(False)
)

hltFilterL3SingleTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL3SingleTau"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltTriggerSummaryRAWprescaler = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltTriggerSummaryRAW = cms.EDFilter("TriggerSummaryProducerRAW",
    processName = cms.string('@')
)

hltBoolFinal = cms.EDFilter("HLTBool",
    result = cms.bool(False)
)

# End replace statements specific to the HLT
HLTBeginSequence = cms.Sequence(hlt2GetRaw+hltGtDigis+hltGctDigis+hltL1GtObjectMap+hltL1extraParticles+hltOfflineBeamSpot)
HLTRecoJetRegionalSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalJetsFEDs+hltEcalRegionalJetsDigis+hltEcalRegionalJetsWeightUncalibRecHit+hltEcalRegionalJetsRecHitTmp+hltEcalRegionalJetsRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForJets+hltCaloTowersForJets+hltIterativeCone5CaloJetsRegional+hltMCJetCorJetIcone5Regional)
HLTDoLocalHcalSequence = cms.Sequence(hltHcalDigis+hltHbhereco+hltHfreco+hltHoreco)
HLTEndSequence = cms.Sequence(hltBoolEnd)
HLTRecoJetMETSequence = cms.Sequence(HLTDoCaloSequence+HLTDoJetRecoSequence+hltMet+HLTDoHTRecoSequence)
HLTDoCaloSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalRestFEDs+hltEcalRegionalRestDigis+hltEcalRegionalRestWeightUncalibRecHit+hltEcalRegionalRestRecHitTmp+hltEcalRecHitAll+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForAll+hltCaloTowers)
HLTDoJetRecoSequence = cms.Sequence(hltIterativeCone5CaloJets+hltMCJetCorJetIcone5)
HLTDoHTRecoSequence = cms.Sequence(hltHtMet)
HLTDoRegionalEgammaEcalSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalEgammaFEDs+hltEcalRegionalEgammaDigis+hltEcalRegionalEgammaWeightUncalibRecHit+hltEcalRegionalEgammaRecHitTmp+hltEcalRegionalEgammaRecHit+hltEcalPreshowerRecHit)
HLTL1IsolatedEcalClustersSequence = cms.Sequence(hltIslandBasicClustersEndcapL1Isolated+hltIslandBasicClustersBarrelL1Isolated+hltHybridSuperClustersL1Isolated+hltIslandSuperClustersL1Isolated+hltCorrectedIslandEndcapSuperClustersL1Isolated+hltCorrectedIslandBarrelSuperClustersL1Isolated+hltCorrectedHybridSuperClustersL1Isolated+hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated)
HLTDoLocalHcalWithoutHOSequence = cms.Sequence(hltHcalDigis+hltHbhereco+hltHfreco)
HLTDoLocalPixelSequence = cms.Sequence(hltSiPixelDigis+hltSiPixelClusters+hltSiPixelRecHits)
HLTDoLocalStripSequence = cms.Sequence(hltSiStripRawToClustersFacility+hltSiStripClusters)
HLTPixelMatchElectronL1IsoSequence = cms.Sequence(hltL1IsoElectronPixelSeeds)
HLTPixelMatchElectronL1IsoTrackingSequence = cms.Sequence(hltCkfL1IsoTrackCandidates+hltCtfL1IsoWithMaterialTracks+hltPixelMatchElectronsL1Iso)
HLTL1IsoElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1IsoElectronsRegionalPixelSeedGenerator+hltL1IsoElectronsRegionalCkfTrackCandidates+hltL1IsoElectronsRegionalCTFFinalFitWithMaterial)
HLTL1NonIsolatedEcalClustersSequence = cms.Sequence(hltIslandBasicClustersEndcapL1NonIsolated+hltIslandBasicClustersBarrelL1NonIsolated+hltHybridSuperClustersL1NonIsolated+hltIslandSuperClustersL1NonIsolated+hltCorrectedIslandEndcapSuperClustersL1NonIsolated+hltCorrectedIslandBarrelSuperClustersL1NonIsolated+hltCorrectedHybridSuperClustersL1NonIsolated+hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated)
HLTPixelMatchElectronL1NonIsoSequence = cms.Sequence(hltL1NonIsoElectronPixelSeeds)
HLTPixelMatchElectronL1NonIsoTrackingSequence = cms.Sequence(hltCkfL1NonIsoTrackCandidates+hltCtfL1NonIsoWithMaterialTracks+hltPixelMatchElectronsL1NonIso)
HLTL1NonIsoElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1NonIsoElectronsRegionalPixelSeedGenerator+hltL1NonIsoElectronsRegionalCkfTrackCandidates+hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial)
HLTDoLocalTrackerSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence)
HLTL1IsoEgammaRegionalRecoTrackerSequence = cms.Sequence(hltL1IsoEgammaRegionalPixelSeedGenerator+hltL1IsoEgammaRegionalCkfTrackCandidates+hltL1IsoEgammaRegionalCTFFinalFitWithMaterial)
HLTL1NonIsoEgammaRegionalRecoTrackerSequence = cms.Sequence(hltL1NonIsoEgammaRegionalPixelSeedGenerator+hltL1NonIsoEgammaRegionalCkfTrackCandidates+hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial)
HLTPixelMatchElectronL1IsoLargeWindowSequence = cms.Sequence(hltL1IsoLargeWindowElectronPixelSeeds)
HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence = cms.Sequence(hltCkfL1IsoLargeWindowTrackCandidates+hltCtfL1IsoLargeWindowWithMaterialTracks+hltPixelMatchElectronsL1IsoLargeWindow)
HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator+hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates+hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)
HLTPixelMatchElectronL1NonIsoLargeWindowSequence = cms.Sequence(hltL1NonIsoLargeWindowElectronPixelSeeds)
HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence = cms.Sequence(hltCkfL1NonIsoLargeWindowTrackCandidates+hltCtfL1NonIsoLargeWindowWithMaterialTracks+hltPixelMatchElectronsL1NonIsoLargeWindow)
HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator+hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates+hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial)
HLTL1muonrecoSequence = cms.Sequence(HLTBeginSequence)
HLTL2muonrecoSequence = cms.Sequence(HLTL2muonrecoNocandSequence+hltL2MuonCandidates)
HLTL2muonrecoNocandSequence = cms.Sequence(hltMuonDTDigis+hltDt1DRecHits+hltDt4DSegments+hltMuonCSCDigis+hltCsc2DRecHits+hltCscSegments+hltMuonRPCDigis+hltRpcRecHits+hltL2MuonSeeds+hltL2Muons)
HLTL2muonisorecoSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalMuonsFEDs+hltEcalRegionalMuonsDigis+hltEcalRegionalMuonsWeightUncalibRecHit+hltEcalRegionalMuonsRecHitTmp+hltEcalRegionalMuonsRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForMuons+hltCaloTowersForMuons+hltL2MuonIsolations)
HLTL3muonrecoSequence = cms.Sequence(HLTL3muonrecoNocandSequence+hltL3MuonCandidates)
HLTL3muonrecoNocandSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL3TrajectorySeed+hltL3TrackCandidateFromL2+hltL3Muons)
HLTL3muonisorecoSequence = cms.Sequence(hltPixelTracks+hltL3MuonIsolations)
HLTBCommonL2recoSequence = cms.Sequence(HLTDoCaloSequence+HLTDoJetRecoSequence+HLTDoHTRecoSequence)
HLTBLifetimeL25recoSequence = cms.Sequence(hltBLifetimeHighestEtJets+hltBLifetimeL25Jets+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltBLifetimeL25Associator+hltBLifetimeL25TagInfos+hltBLifetimeL25BJetTags)
HLTRecopixelvertexingSequence = cms.Sequence(hltPixelTracks+hltPixelVertices)
HLTBLifetimeL3recoSequence = cms.Sequence(hltBLifetimeL3Jets+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltBLifetimeRegionalPixelSeedGenerator+hltBLifetimeRegionalCkfTrackCandidates+hltBLifetimeRegionalCtfWithMaterialTracks+hltBLifetimeL3Associator+hltBLifetimeL3TagInfos+hltBLifetimeL3BJetTags)
HLTBSoftmuonL25recoSequence = cms.Sequence(hltBSoftmuonHighestEtJets+hltBSoftmuonL25Jets+HLTL2muonrecoNocandSequence+hltBSoftmuonL25TagInfos+hltBSoftmuonL25BJetTags)
HLTBSoftmuonL3recoSequence = cms.Sequence(HLTL3muonrecoNocandSequence+hltBSoftmuonL3TagInfos+hltBSoftmuonL3BJetTags+hltBSoftmuonL3BJetTagsByDR)
HLTL3displacedMumurecoSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTRecopixelvertexingSequence+hltMumuPixelSeedFromL2Candidate+hltCkfTrackCandidatesMumu+hltCtfWithMaterialTracksMumu+hltMuTracks)
HLTL1EplusJetSequence = cms.Sequence(HLTBeginSequence+hltL1seedEJet)
HLTEJetElectronSequence = cms.Sequence(HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoSingleL1MatchFilter+hltL1IsoEJetSingleEEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoEJetSingleEHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoEJetSingleEPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoEJetSingleEEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoEJetSingleETrackIsolFilter+hltSingleElectronL1IsoPresc)
HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence = cms.Sequence(HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltEgammaL1MatchFilterRegionalElectronTau+hltEgammaEtFilterElectronTau+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltEgammaHcalIsolFilterElectronTau+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltElectronPixelMatchFilterElectronTau+HLTPixelMatchElectronL1IsoTrackingSequence+hltElectronOneOEMinusOneOPFilterElectronTau+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau)
HLTL2TauJetsElectronTauSequence = cms.Sequence(HLTCaloTausCreatorRegionalSequence+hltL2TauJetsProviderElectronTau)
HLTCaloTausCreatorRegionalSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalTausFEDs+hltEcalRegionalTausDigis+hltEcalRegionalTausWeightUncalibRecHit+hltEcalRegionalTausRecHitTmp+hltEcalRegionalTausRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForTaus+hltCaloTowersTau1Regional+hltIcone5Tau1Regional+hltCaloTowersTau2Regional+hltIcone5Tau2Regional+hltCaloTowersTau3Regional+hltIcone5Tau3Regional+hltCaloTowersTau4Regional+hltIcone5Tau4Regional)
HLTL1SeedFilterSequence = cms.Sequence(hltL1sIsolTrack)
HLTL3PixelIsolFilterSequence = cms.Sequence(HLTDoLocalPixelSequence+hltPixelTracks+hltIsolPixelTrackProd+hltIsolPixelTrackFilter)
HLTPixelTrackingForMinBiasSequence = cms.Sequence(hltPixelTracksForMinBias)
HLTL2TauJetsElectronTauSequnce = cms.Sequence(HLTCaloTausCreatorRegionalSequence+hltL2TauJetsProviderElectronTau)
HLTCaloTausCreatorSequence = cms.Sequence(HLTDoCaloSequence+hltCaloTowersTau1+hltIcone5Tau1+hltCaloTowersTau2+hltIcone5Tau2+hltCaloTowersTau3+hltIcone5Tau3+hltCaloTowersTau4+hltIcone5Tau4)
HLT1jet = cms.Path(HLTBeginSequence+hltL1s1jet+hltPre1jet+HLTRecoJetRegionalSequence+hlt1jet200+HLTEndSequence)
HLT2jet = cms.Path(HLTBeginSequence+hltL1s2jet+hltPre2jet+HLTRecoJetRegionalSequence+hlt2jet150+HLTEndSequence)
HLT3jet = cms.Path(HLTBeginSequence+hltL1s3jet+hltPre3jet+HLTRecoJetRegionalSequence+hlt3jet85+HLTEndSequence)
HLT4jet = cms.Path(HLTBeginSequence+hltL1s4jet+hltPre4jet+HLTRecoJetMETSequence+hlt4jet60+HLTEndSequence)
HLT1MET = cms.Path(HLTBeginSequence+hltL1s1MET+hltPre1MET+HLTRecoJetMETSequence+hlt1MET65+HLTEndSequence)
HLT2jetAco = cms.Path(HLTBeginSequence+hltL1s2jetAco+hltPre2jetAco+HLTRecoJetRegionalSequence+hlt2jet125+hlt2jetAco+HLTEndSequence)
HLT1jet1METAco = cms.Path(HLTBeginSequence+hltL1s1jet1METAco+hltPre1jet1METAco+HLTRecoJetMETSequence+hlt1MET60+hlt1jet100+hlt1jet1METAco+HLTEndSequence)
HLT1jet1MET = cms.Path(HLTBeginSequence+hltL1s1jet1MET+hltPre1jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt1jet180+HLTEndSequence)
HLT2jet1MET = cms.Path(HLTBeginSequence+hltL1s2jet1MET+hltPre2jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt2jet125+HLTEndSequence)
HLT3jet1MET = cms.Path(HLTBeginSequence+hltL1s3jet1MET+hltPre3jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt3jet60+HLTEndSequence)
HLT4jet1MET = cms.Path(HLTBeginSequence+hltL1s4jet1MET+hltPre4jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt4jet35+HLTEndSequence)
HLT1MET1HT = cms.Path(HLTBeginSequence+hltL1s1MET1HT+hltPre1MET1HT+HLTRecoJetMETSequence+hlt1MET65+hlt1HT350+HLTEndSequence)
CandHLT1SumET = cms.Path(HLTBeginSequence+hltL1s1SumET+hltPre1MET1SumET+HLTRecoJetMETSequence+hlt1SumET120+HLTEndSequence)
HLT1jetPE1 = cms.Path(HLTBeginSequence+hltL1s1jetPE1+hltPre1jetPE1+HLTRecoJetRegionalSequence+hlt1jet150+HLTEndSequence)
HLT1jetPE3 = cms.Path(HLTBeginSequence+hltL1s1jetPE3+hltPre1jetPE3+HLTRecoJetRegionalSequence+hlt1jet110+HLTEndSequence)
HLT1jetPE5 = cms.Path(HLTBeginSequence+hltL1s1jetPE5+hltPre1jetPE5+HLTRecoJetMETSequence+hlt1jet60+HLTEndSequence)
HLT1jetPE7 = cms.Path(HLTBeginSequence+hltL1s1jetPE7+hltPre1jetPE7+HLTRecoJetMETSequence+hlt1jet30+HLTEndSequence)
HLT1METPre1 = cms.Path(HLTBeginSequence+hltL1s1METPre1+hltPre1METPre1+HLTRecoJetMETSequence+hlt1MET55+HLTEndSequence)
HLT1METPre2 = cms.Path(HLTBeginSequence+hltL1s1METPre2+hltPre1METPre2+HLTRecoJetMETSequence+hlt1MET30+HLTEndSequence)
HLT1METPre3 = cms.Path(HLTBeginSequence+hltL1s1METPre3+hltPre1METPre3+HLTRecoJetMETSequence+hlt1MET20+HLTEndSequence)
HLT2jetAve30 = cms.Path(HLTBeginSequence+hltL1sdijetave30+hltPredijetave30+HLTRecoJetMETSequence+hltdijetave30+HLTEndSequence)
HLT2jetAve60 = cms.Path(HLTBeginSequence+hltL1sdijetave60+hltPredijetave60+HLTRecoJetMETSequence+hltdijetave60+HLTEndSequence)
HLT2jetAve110 = cms.Path(HLTBeginSequence+hltL1sdijetave110+hltPredijetave110+HLTRecoJetRegionalSequence+hltdijetave110+HLTEndSequence)
HLT2jetAve150 = cms.Path(HLTBeginSequence+hltL1sdijetave150+hltPredijetave150+HLTRecoJetRegionalSequence+hltdijetave150+HLTEndSequence)
HLT2jetAve200 = cms.Path(HLTBeginSequence+hltL1sdijetave200+hltPredijetave200+HLTRecoJetRegionalSequence+hltdijetave200+HLTEndSequence)
HLT2jetvbfMET = cms.Path(HLTBeginSequence+hltL1s2jetvbfMET+hltPre2jetvbfMET+HLTRecoJetMETSequence+hlt1MET60+hlt2jetvbf+HLTEndSequence)
HLTS2jet1METNV = cms.Path(HLTBeginSequence+hltL1snvMET+hltPrenv+HLTRecoJetMETSequence+hlt1MET60+hltnv+HLTEndSequence)
HLTS2jet1METAco = cms.Path(HLTBeginSequence+hltL1sPhi2MET+hltPrephi2met+HLTRecoJetMETSequence+hlt1MET60+hltPhi2metAco+HLTEndSequence)
HLTSjet1MET1Aco = cms.Path(HLTBeginSequence+hltL1sPhiJet1MET+hltPrephijet1met+HLTRecoJetMETSequence+hlt1MET70+hltPhiJet1metAco+HLTEndSequence)
HLTSjet2MET1Aco = cms.Path(HLTBeginSequence+hltL1sPhiJet2MET+hltPrephijet2met+HLTRecoJetMETSequence+hlt1MET70+hltPhiJet2metAco+HLTEndSequence)
HLTS2jetMET1Aco = cms.Path(HLTBeginSequence+hltL1sPhiJet1Jet2+hltPrephijet1jet2+HLTRecoJetMETSequence+hlt1MET70+hltPhiJet1Jet2Aco+HLTEndSequence)
HLTJetMETRapidityGap = cms.Path(HLTBeginSequence+hltL1RapGap+hltPrerapgap+HLTRecoJetMETSequence+hltRapGap+HLTEndSequence)
HLT1Electron = cms.Path(HLTBeginSequence+hltL1seedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoSingleL1MatchFilter+hltL1IsoSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoSingleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoSingleElectronHOneOEMinusOneOPFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoSingleElectronTrackIsolFilter+hltSingleElectronL1IsoPresc+HLTEndSequence)
HLT1ElectronRelaxed = cms.Path(HLTBeginSequence+hltL1seedRelaxedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoSingleElectronL1MatchFilterRegional+hltL1NonIsoSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoSingleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+HLTPixelMatchElectronL1NonIsoSequence+hltL1NonIsoSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltL1NonIsoSingleElectronTrackIsolFilter+hltSingleElectronL1NonIsoPresc+HLTEndSequence)
HLT2Electron = cms.Path(HLTBeginSequence+hltL1seedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoDoubleElectronL1MatchFilterRegional+hltL1IsoDoubleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoDoubleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoDoubleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoDoubleElectronEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoDoubleElectronTrackIsolFilter+hltDoubleElectronL1IsoPresc+HLTEndSequence)
HLT2ElectronRelaxed = cms.Path(HLTBeginSequence+hltL1seedRelaxedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoDoubleElectronL1MatchFilterRegional+hltL1NonIsoDoubleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoDoubleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+HLTPixelMatchElectronL1NonIsoSequence+hltL1NonIsoDoubleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltL1NonIsoDoubleElectronEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltL1NonIsoDoubleElectronTrackIsolFilter+hltDoubleElectronL1NonIsoPresc+HLTEndSequence)
HLT1Photon = cms.Path(HLTBeginSequence+hltL1seedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoSinglePhotonL1MatchFilter+hltL1IsoSinglePhotonEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1IsoSinglePhotonEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1IsoSinglePhotonHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1IsoSinglePhotonTrackIsolFilter+hltSinglePhotonL1IsoPresc+HLTEndSequence)
HLT1PhotonRelaxed = cms.Path(HLTBeginSequence+hltL1seedRelaxedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoSinglePhotonL1MatchFilterRegional+hltL1NonIsoSinglePhotonEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoSinglePhotonEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoSinglePhotonHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoSinglePhotonTrackIsolFilter+hltSinglePhotonL1NonIsoPresc+HLTEndSequence)
HLT2Photon = cms.Path(HLTBeginSequence+hltL1seedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoDoublePhotonL1MatchFilterRegional+hltL1IsoDoublePhotonEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1IsoDoublePhotonEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1IsoDoublePhotonHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1IsoDoublePhotonTrackIsolFilter+hltL1IsoDoublePhotonDoubleEtFilter+hltDoublePhotonL1IsoPresc+HLTEndSequence)
HLT2PhotonRelaxed = cms.Path(HLTBeginSequence+hltL1seedRelaxedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoDoublePhotonL1MatchFilterRegional+hltL1NonIsoDoublePhotonEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoDoublePhotonEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoDoublePhotonHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoDoublePhotonTrackIsolFilter+hltL1NonIsoDoublePhotonDoubleEtFilter+hltDoublePhotonL1NonIsoPresc+HLTEndSequence)
HLT1EMHighEt = cms.Path(HLTBeginSequence+hltL1seedRelaxedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoSingleEMHighEtL1MatchFilterRegional+hltL1NonIsoSinglePhotonEMHighEtEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoSingleEMHighEtEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1NonIsolatedElectronHcalIsol+hltL1IsolatedElectronHcalIsol+hltL1NonIsoSingleEMHighEtHOEFilter+hltHcalDoubleCone+hltL1NonIsoEMHcalDoubleCone+hltL1NonIsoSingleEMHighEtHcalDBCFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoSingleEMHighEtTrackIsolFilter+hltSingleEMVHighEtL1NonIsoPresc+HLTEndSequence)
HLT1EMVeryHighEt = cms.Path(HLTBeginSequence+hltL1seedRelaxedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional+hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter+hltSingleEMVHEL1NonIsoPresc+HLTEndSequence)
HLT2ElectronZCounter = cms.Path(HLTBeginSequence+hltL1seedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoDoubleElectronZeeL1MatchFilterRegional+hltL1IsoDoubleElectronZeeEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoDoubleElectronZeeHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoDoubleElectronZeePixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoDoubleElectronZeeEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoDoubleElectronZeeTrackIsolFilter+hltL1IsoDoubleElectronZeePMMassFilter+hltZeeCounterPresc+HLTEndSequence)
HLT2ElectronExclusive = cms.Path(HLTBeginSequence+hltL1seedExclusiveDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoDoubleExclElectronL1MatchFilterRegional+hltL1IsoDoubleExclElectronEtPhiFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoDoubleExclElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoDoubleExclElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoDoubleExclElectronEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoDoubleExclElectronTrackIsolFilter+hltDoubleExclElectronL1IsoPresc+HLTEndSequence)
HLT2PhotonExclusive = cms.Path(HLTBeginSequence+hltL1seedExclusiveDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoDoubleExclPhotonL1MatchFilterRegional+hltL1IsoDoubleExclPhotonEtPhiFilter+hltL1IsolatedPhotonEcalIsol+hltL1IsoDoubleExclPhotonEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1IsoDoubleExclPhotonHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1IsoDoubleExclPhotonTrackIsolFilter+hltDoubleExclPhotonL1IsoPresc+HLTEndSequence)
HLT1PhotonL1Isolated = cms.Path(HLTBeginSequence+hltL1seedSinglePrescaled+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoSinglePhotonPrescaledL1MatchFilter+hltL1IsoSinglePhotonPrescaledEtFilter+hltL1IsolatedPhotonEcalIsol+hltL1IsoSinglePhotonPrescaledEcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1IsoSinglePhotonPrescaledHcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1IsoSinglePhotonPrescaledTrackIsolFilter+hltSinglePhotonPrescaledL1IsoPresc+HLTEndSequence)
CandHLT1ElectronStartup = cms.Path(HLTBeginSequence+hltL1seedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoLargeWindowSingleL1MatchFilter+hltL1IsoLargeWindowSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoLargeWindowSingleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoLargeWindowSequence+hltL1IsoLargeWindowSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1IsoLargeWindowSingleElectronTrackIsolFilter+hltSingleElectronL1IsoLargeWindowPresc+HLTEndSequence)
CandHLT1ElectronRelaxedStartup = cms.Path(HLTBeginSequence+hltL1seedRelaxedSingle+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional+hltL1NonIsoLargeWindowSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoLargeWindowSequence+HLTPixelMatchElectronL1NonIsoLargeWindowSequence+hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter+hltSingleElectronL1NonIsoLargeWindowPresc+HLTEndSequence)
CandHLT2ElectronStartup = cms.Path(HLTBeginSequence+hltL1seedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional+hltL1IsoLargeWindowDoubleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoLargeWindowDoubleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoLargeWindowSequence+hltL1IsoLargeWindowDoubleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+hltL1IsoLargeWindowDoubleElectronEoverpFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1IsoLargeWindowDoubleElectronTrackIsolFilter+hltDoubleElectronL1IsoLargeWindowPresc+HLTEndSequence)
CandHLT2ElectronRelaxedStartup = cms.Path(HLTBeginSequence+hltL1seedRelaxedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional+hltL1NonIsoLargeWindowDoubleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTPixelMatchElectronL1IsoLargeWindowSequence+HLTPixelMatchElectronL1NonIsoLargeWindowSequence+hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoLargeWindowDoubleElectronEoverpFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter+hltDoubleElectronL1NonIsoLargeWindowPresc+HLTEndSequence)
HLT1MuonIso = cms.Path(hltPrescaleSingleMuIso+HLTL1muonrecoSequence+hltSingleMuIsoLevel1Seed+hltSingleMuIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuIsoL2PreFiltered+HLTL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered+HLTL3muonrecoSequence+hltSingleMuIsoL3PreFiltered+HLTL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered+HLTEndSequence)
HLT1MuonNonIso = cms.Path(hltPrescaleSingleMuNoIso+HLTL1muonrecoSequence+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonIso = cms.Path(hltPrescaleDiMuonIso+HLTL1muonrecoSequence+hltDiMuonIsoLevel1Seed+hltDiMuonIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonIsoL2PreFiltered+HLTL2muonisorecoSequence+hltDiMuonIsoL2IsoFiltered+HLTL3muonrecoSequence+hltDiMuonIsoL3PreFiltered+HLTL3muonisorecoSequence+hltDiMuonIsoL3IsoFiltered+HLTEndSequence)
HLT2MuonNonIso = cms.Path(hltPrescaleDiMuonNoIso+HLTL1muonrecoSequence+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonJPsi = cms.Path(hltPrescaleJPsiMM+HLTL1muonrecoSequence+hltJpsiMMLevel1Seed+hltJpsiMML1Filtered+HLTL2muonrecoSequence+hltJpsiMML2Filtered+HLTL3muonrecoSequence+hltJpsiMML3Filtered+HLTEndSequence)
HLT2MuonUpsilon = cms.Path(hltPrescaleUpsilonMM+HLTL1muonrecoSequence+hltUpsilonMMLevel1Seed+hltUpsilonMML1Filtered+HLTL2muonrecoSequence+hltUpsilonMML2Filtered+HLTL3muonrecoSequence+hltUpsilonMML3Filtered+HLTEndSequence)
HLT2MuonZ = cms.Path(hltPrescaleZMM+HLTL1muonrecoSequence+hltZMMLevel1Seed+hltZMML1Filtered+HLTL2muonrecoSequence+hltZMML2Filtered+HLTL3muonrecoSequence+hltZMML3Filtered+HLTEndSequence)
HLTNMuonNonIso = cms.Path(hltPrescaleMultiMuonNoIso+HLTL1muonrecoSequence+hltMultiMuonNoIsoLevel1Seed+hltMultiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltMultiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltMultiMuonNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonSameSign = cms.Path(hltPrescaleSameSignMu+HLTL1muonrecoSequence+hltSameSignMuLevel1Seed+hltSameSignMuL1Filtered+HLTL2muonrecoSequence+hltSameSignMuL2PreFiltered+HLTL3muonrecoSequence+hltSameSignMuL3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt3 = cms.Path(hltPrescaleSingleMuPrescale3+HLTL1muonrecoSequence+hltSingleMuPrescale3Level1Seed+hltSingleMuPrescale3L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale3L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale3L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt5 = cms.Path(hltPrescaleSingleMuPrescale5+HLTL1muonrecoSequence+hltSingleMuPrescale5Level1Seed+hltSingleMuPrescale5L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale5L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale5L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt7x7 = cms.Path(hltPreSingleMuPrescale77+HLTL1muonrecoSequence+hltSingleMuPrescale77Level1Seed+hltSingleMuPrescale77L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale77L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale77L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt7x10 = cms.Path(hltPreSingleMuPrescale710+HLTL1muonrecoSequence+hltSingleMuPrescale710Level1Seed+hltSingleMuPrescale710L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale710L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale710L3PreFiltered+HLTEndSequence)
HLT1MuonLevel1 = cms.Path(hltPrescaleMuLevel1Path+HLTL1muonrecoSequence+hltMuLevel1PathLevel1Seed+hltMuLevel1PathL1Filtered+HLTEndSequence)
CandHLT1MuonPrescaleVtx2cm = cms.Path(hltPrescaleSingleMuNoIsoRelaxedVtx2cm+HLTL1muonrecoSequence+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm+HLTEndSequence)
CandHLT1MuonPrescaleVtx2mm = cms.Path(hltPrescaleSingleMuNoIsoRelaxedVtx2mm+HLTL1muonrecoSequence+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm+HLTEndSequence)
CandHLT2MuonPrescaleVtx2cm = cms.Path(hltPrescaleDiMuonNoIsoRelaxedVtx2cm+HLTL1muonrecoSequence+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm+HLTEndSequence)
CandHLT2MuonPrescaleVtx2mm = cms.Path(hltPrescaleDiMuonNoIsoRelaxedVtx2mm+HLTL1muonrecoSequence+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm+HLTEndSequence)
HLTB1Jet = cms.Path(hltPrescalerBLifetime1jet+HLTBeginSequence+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime1jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB2Jet = cms.Path(hltPrescalerBLifetime2jet+HLTBeginSequence+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime2jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB3Jet = cms.Path(hltPrescalerBLifetime3jet+HLTBeginSequence+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime3jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB4Jet = cms.Path(hltPrescalerBLifetime4jet+HLTBeginSequence+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime4jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTBHT = cms.Path(hltPrescalerBLifetimeHT+HLTBeginSequence+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetimeHTL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB1JetMu = cms.Path(hltPrescalerBSoftmuon1jet+HLTBeginSequence+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon1jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonByDRL3filter+HLTEndSequence)
HLTB2JetMu = cms.Path(hltPrescalerBSoftmuon2jet+HLTBeginSequence+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon2jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTB3JetMu = cms.Path(hltPrescalerBSoftmuon3jet+HLTBeginSequence+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon3jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTB4JetMu = cms.Path(hltPrescalerBSoftmuon4jet+HLTBeginSequence+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon4jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTBHTMu = cms.Path(hltPrescalerBSoftmuonHT+HLTBeginSequence+hltBSoftmuonHTL1seeds+HLTBCommonL2recoSequence+hltBSoftmuonHTL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTBJPsiMuMu = cms.Path(HLTBeginSequence+hltJpsitoMumuL1Seed+hltJpsitoMumuL1Filtered+HLTL2muonrecoSequence+hltJpsitoMumuL2Filtered+HLTL3displacedMumurecoSequence+hltDisplacedJpsitoMumuFilter+HLTEndSequence)
CandHLTBToMuMuK = cms.Path(HLTBeginSequence+hltMuMukL1Seed+hltMuMukL1Filtered+HLTL2muonrecoSequence+hltMuMukL2Filtered+HLTL3displacedMumurecoSequence+hltDisplacedMuMukFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTRecopixelvertexingSequence+hltMumukPixelSeedFromL2Candidate+hltCkfTrackCandidatesMumuk+hltCtfWithMaterialTracksMumuk+hltMumukAllConeTracks+hltmmkFilter+HLTEndSequence)
HLTXElectronBJet = cms.Path(hltElectronBPrescale+HLTBeginSequence+hltElectronBL1Seed+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltElBElectronL1MatchFilter+hltElBElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltElBElectronHcalIsolFilter+HLTBCommonL2recoSequence+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1NonIsoSequence+HLTPixelMatchElectronL1IsoSequence+hltElBElectronPixelMatchFilter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltElBElectronEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltElBElectronTrackIsolFilter+HLTEndSequence)
HLTXMuonBJet = cms.Path(hltMuBPrescale+HLTBeginSequence+hltMuBLevel1Seed+hltMuBLifetimeL1Filtered+HLTL2muonrecoSequence+hltMuBLifetimeIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuBLifetimeIsoL2IsoFiltered+HLTBCommonL2recoSequence+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTL3muonrecoSequence+hltMuBLifetimeIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuBLifetimeIsoL3IsoFiltered+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTXMuonBJetSoftMuon = cms.Path(hltMuBsoftMuPrescale+HLTBeginSequence+hltMuBLevel1Seed+hltMuBSoftL1Filtered+HLTL2muonrecoSequence+hltMuBSoftIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuBSoftIsoL2IsoFiltered+HLTBCommonL2recoSequence+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTL3muonrecoSequence+hltMuBSoftIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuBSoftIsoL3IsoFiltered+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTXElectron1Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej1jet40+HLTEndSequence)
HLTXElectron2Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej2jet80+HLTEndSequence)
HLTXElectron3Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej3jet60+HLTEndSequence)
HLTXElectron4Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej4jet35+HLTEndSequence)
HLTXMuonJets = cms.Path(hltMuJetsPrescale+HLTBeginSequence+HLTL1muonrecoSequence+hltMuJetsLevel1Seed+hltMuJetsL1Filtered+HLTL2muonrecoSequence+hltMuJetsL2PreFiltered+HLTL2muonisorecoSequence+hltMuJetsL2IsoFiltered+HLTL3muonrecoSequence+hltMuJetsL3PreFiltered+HLTL3muonisorecoSequence+hltMuJetsL3IsoFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuJetsHLT1jet40+HLTEndSequence)
CandHLTXMuonNoL2IsoJets = cms.Path(hltMuNoL2IsoJetsPrescale+HLTBeginSequence+HLTL1muonrecoSequence+hltMuNoL2IsoJetsLevel1Seed+hltMuNoL2IsoJetsL1Filtered+HLTL2muonrecoSequence+hltMuNoL2IsoJetsL2PreFiltered+HLTL3muonrecoSequence+hltMuNoL2IsoJetsL3PreFiltered+HLTL3muonisorecoSequence+hltMuNoL2IsoJetsL3IsoFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuNoL2IsoJetsHLT1jet40+HLTEndSequence)
CandHLTXMuonNoIsoJets = cms.Path(hltMuNoIsoJetsPrescale+HLTBeginSequence+HLTL1muonrecoSequence+hltMuNoIsoJetsLevel1Seed+hltMuNoIsoJetsL1Filtered+HLTL2muonrecoSequence+hltMuNoIsoJetsL2PreFiltered+HLTL3muonrecoSequence+hltMuNoIsoJetsL3PreFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuNoIsoJetsHLT1jet50+HLTEndSequence)
HLTXElectronMuon = cms.Path(hltemuPrescale+HLTBeginSequence+hltEMuonLevel1Seed+hltEMuL1MuonFilter+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltemuL1IsoSingleL1MatchFilter+hltemuL1IsoSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltemuL1IsoSingleElectronHcalIsolFilter+HLTL2muonrecoSequence+hltEMuL2MuonPreFilter+HLTL2muonisorecoSequence+hltEMuL2MuonIsoFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltemuL1IsoSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltemuL1IsoSingleElectronEoverpFilter+HLTL3muonrecoSequence+hltEMuL3MuonPreFilter+HLTL3muonisorecoSequence+hltEMuL3MuonIsoFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltemuL1IsoSingleElectronTrackIsolFilter+HLTEndSequence)
HLTXElectronMuonRelaxed = cms.Path(hltemuNonIsoPrescale+HLTBeginSequence+hltemuNonIsoLevel1Seed+hltNonIsoEMuL1MuonFilter+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltemuNonIsoL1MatchFilterRegional+hltemuNonIsoL1IsoEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltemuNonIsoL1HcalIsolFilter+HLTL2muonrecoSequence+hltNonIsoEMuL2MuonPreFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+HLTPixelMatchElectronL1NonIsoSequence+hltemuNonIsoL1IsoPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltemuNonIsoL1IsoEoverpFilter+HLTL3muonrecoSequence+hltNonIsoEMuL3MuonPreFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltemuNonIsoL1IsoTrackIsolFilter+HLTEndSequence)
CandHLTXElectronTauPixel = cms.Path(hltPrescalerElectronTau+HLTBeginSequence+hltLevel1GTSeedElectronTau+HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence+HLTL2TauJetsElectronTauSequence+hltJetCrystalsAssociatorElectronTau+hltEcalIsolationElectronTau+hltEMIsolatedTauJetsSelectorElectronTau+hltFilterEcalIsolatedTauJetsElectronTau+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTRecopixelvertexingSequence+hltJetsPixelTracksAssociatorElectronTau+hltPixelTrackConeIsolationElectronTau+hltPixelTrackIsolatedTauJetsSelectorElectronTau+hltFilterPixelTrackIsolatedTauJetsElectronTau+HLTEndSequence)
CandHLTBackwardBSC = cms.Path(hltLevel1seedHLTBackwardBSC+hltPrescaleHLTBackwardBSC+HLTEndSequence)
CandHLTForwardBSC = cms.Path(hltLevel1seedHLTForwardBSC+hltPrescaleHLTForwardBSC+HLTEndSequence)
CandHLTCSCBeamHalo = cms.Path(hltLevel1seedHLTCSCBeamHalo+hltPrescaleHLTCSCBeamHalo+HLTEndSequence)
CandHLTCSCBeamHaloOverlapRing1 = cms.Path(hltLevel1seedHLTCSCBeamHaloOverlapRing1+hltPrescaleHLTCSCBeamHaloOverlapRing1+hltMuonCSCDigis+hltCsc2DRecHits+hltOverlapsHLTCSCBeamHaloOverlapRing1+HLTEndSequence)
CandHLTCSCBeamHaloOverlapRing2 = cms.Path(hltLevel1seedHLTCSCBeamHaloOverlapRing2+hltPrescaleHLTCSCBeamHaloOverlapRing2+hltMuonCSCDigis+hltCsc2DRecHits+hltOverlapsHLTCSCBeamHaloOverlapRing2+HLTEndSequence)
CandHLTCSCBeamHaloRing2or3 = cms.Path(hltLevel1seedHLTCSCBeamHaloRing2or3+hltPrescaleHLTCSCBeamHaloRing2or3+hltMuonCSCDigis+hltCsc2DRecHits+hltFilter23HLTCSCBeamHaloRing2or3+HLTEndSequence)
CandHLTTrackerCosmics = cms.Path(hltLevel1seedHLTTrackerCosmics+hltPrescaleHLTTrackerCosmics+HLTEndSequence)
CandHLTEcalPi0 = cms.Path(HLTBeginSequence+hltPrePi0Ecal+hltL1sEcalPi0+HLTDoRegionalEgammaEcalSequence+hltAlCaPi0RegRecHits+HLTEndSequence)
CandHLTEcalPhiSym = cms.Path(HLTBeginSequence+hltL1sEcalPhiSym+hltEcalPhiSymPresc+hltEcalDigis+hltEcalWeightUncalibRecHit+hltEcalRecHit+hltAlCaPhiSymStream+HLTEndSequence)
CandHLTHcalPhiSym = cms.Path(HLTBeginSequence+hltL1sHcalPhiSym+hltHcalPhiSymPresc+HLTDoLocalHcalSequence+hltAlCaHcalPhiSymStream+HLTEndSequence)
HLTHcalIsolatedTrack = cms.Path(HLTBeginSequence+HLTL1SeedFilterSequence+hltPreIsolTrack+hltEcalDigis+hltEcalPreshowerDigis+hltEcalWeightUncalibRecHit+hltEcalRecHit+hltEcalPreshowerRecHit+hltEcalIsolPartProd+hltEcalIsolFilter+HLTL3PixelIsolFilterSequence+HLTEndSequence)
CandHLTHcalIsolatedTrackNoEcalIsol = cms.Path(HLTBeginSequence+HLTL1SeedFilterSequence+hltPreIsolTrackNoEcalIso+HLTL3PixelIsolFilterSequence+HLTEndSequence)
HLTMinBiasPixel = cms.Path(HLTBeginSequence+hltPreMinBiasPixel+hltL1seedMinBiasPixel+HLTDoLocalPixelSequence+HLTPixelTrackingForMinBiasSequence+hltPixelCands+hltMinBiasPixelFilter+HLTEndSequence)
CandHLTMinBiasForAlignment = cms.Path(HLTBeginSequence+hltPreMBForAlignment+hltL1seedMinBiasPixel+HLTDoLocalPixelSequence+HLTPixelTrackingForMinBiasSequence+hltPixelCands+hltPixelMBForAlignment+HLTEndSequence)
HLTMinBias = cms.Path(HLTBeginSequence+hltl1sMin+hltpreMin+HLTEndSequence)
HLTZeroBias = cms.Path(HLTBeginSequence+hltl1sZero+hltpreZero+HLTEndSequence)
HLTriggerType = cms.Path(HLTBeginSequence+hltPrescaleTriggerType+hltFilterTriggerType+HLTEndSequence)
HLTEndpath1 = cms.EndPath(hltL1gtTrigReport+hltTrigReport)
HLTXElectronTau = cms.Path(hltPrescalerElectronTau+HLTBeginSequence+hltLevel1GTSeedElectronTau+HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence+HLTL2TauJetsElectronTauSequnce+hltL2ElectronTauIsolationProducer+hltL2ElectronTauIsolationSelector+hltFilterEcalIsolatedTauJetsElectronTau+HLTRecopixelvertexingSequence+hltJetTracksAssociatorAtVertexL25ElectronTau+hltConeIsolationL25ElectronTau+hltIsolatedTauJetsSelectorL25ElectronTau+hltFilterIsolatedTauJetsL25ElectronTau+HLTEndSequence)
HLTXMuonTau = cms.Path(hltPrescalerMuonTau+HLTBeginSequence+hltLevel1GTSeedMuonTau+hltMuonTauL1Filtered+HLTL2muonrecoSequence+hltMuonTauIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuonTauIsoL2IsoFiltered+HLTCaloTausCreatorRegionalSequence+hltL2TauJetsProviderMuonTau+hltL2MuonTauIsolationProducer+hltL2MuonTauIsolationSelector+hltFilterEcalIsolatedTauJetsMuonTau+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltJetsPixelTracksAssociatorMuonTau+hltPixelTrackConeIsolationMuonTau+hltPixelTrackIsolatedTauJetsSelectorMuonTau+hltFilterPixelTrackIsolatedTauJetsMuonTau+HLTDoLocalStripSequence+HLTL3muonrecoSequence+hltMuonTauIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuonTauIsoL3IsoFiltered+HLTEndSequence)
HLT1Tau1MET = cms.Path(hltSingleTauMETPrescaler+HLTBeginSequence+hltSingleTauMETL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTauMET+hltL2SingleTauMETJets+hltL2SingleTauMETIsolationProducer+hltL2SingleTauMETIsolationSelector+hltFilterSingleTauMETEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTauMET+hltConeIsolationL25SingleTauMET+hltIsolatedL25SingleTauMET+hltFilterL25SingleTauMET+HLTDoLocalStripSequence+hltL3SingleTauMETPixelSeeds+hltCkfTrackCandidatesL3SingleTauMET+hltCtfWithMaterialTracksL3SingleTauMET+hltAssociatorL3SingleTauMET+hltConeIsolationL3SingleTauMET+hltIsolatedL3SingleTauMET+hltFilterL3SingleTauMET+HLTEndSequence)
HLT2TauPixel = cms.Path(hltDoubleTauPrescaler+HLTBeginSequence+hltDoubleTauL1SeedFilter+HLTCaloTausCreatorRegionalSequence+hltL2DoubleTauJets+hltL2DoubleTauIsolationProducer+hltL2DoubleTauIsolationSelector+hltFilterDoubleTauEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25PixelTauIsolated+hltConeIsolationL25PixelTauIsolated+hltIsolatedL25PixelTau+hltFilterL25PixelTau+HLTEndSequence)
HLT1Tau = cms.Path(hltSingleTauPrescaler+HLTBeginSequence+hltSingleTauL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTau+hltL2SingleTauJets+hltL2SingleTauIsolationProducer+hltL2SingleTauIsolationSelector+hltFilterSingleTauEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTau+hltConeIsolationL25SingleTau+hltIsolatedL25SingleTau+hltFilterL25SingleTau+HLTDoLocalStripSequence+hltL3SingleTauPixelSeeds+hltCkfTrackCandidatesL3SingleTau+hltCtfWithMaterialTracksL3SingleTau+hltAssociatorL3SingleTau+hltConeIsolationL3SingleTau+hltIsolatedL3SingleTau+hltFilterL3SingleTau+HLTEndSequence)
HLTriggerFinalPath = cms.Path(hltTriggerSummaryAOD+hltTriggerSummaryRAWprescaler+hltTriggerSummaryRAW+hltBoolFinal)

