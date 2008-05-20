import FWCore.ParameterSet.Config as cms

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

CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")

Chi2EstimatorForL2Refit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForL2Refit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(1000.0)
)

KFTrajectoryFitterForL2Muon = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFTrajectoryFitterForL2Muon'),
    Estimator = cms.string('Chi2EstimatorForL2Refit'),
    Propagator = cms.string('SteppingHelixPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

KFTrajectorySmootherForL2Muon = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFTrajectorySmootherForL2Muon'),
    Estimator = cms.string('Chi2EstimatorForL2Refit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SteppingHelixPropagatorOpposite')
)

KFFitterSmootherForL2Muon = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('KFTrajectoryFitterForL2Muon'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFTrajectorySmootherForL2Muon'),
    ComponentName = cms.string('KFFitterSmootherForL2Muon'),
    RejectTracks = cms.bool(True)
)

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

KFFitterForRefitInsideOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
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

PixelCPEParmErrorESProducer = cms.ESProducer("PixelCPEParmErrorESProducer",
    UseNewParametrization = cms.bool(True),
    ComponentName = cms.string('PixelCPEfromTrackAngle'),
    UseSigma = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

RKTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RKTrackerPropagator'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(True)
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

myTTRHBuilderWithoutAngle = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('PixelTTRHBuilderWithoutAngle'),
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

softLeptonByDistance = cms.ESProducer("LeptonTaggerByDistanceESProducer",
    distance = cms.double(0.5)
)

softLeptonByPt = cms.ESProducer("LeptonTaggerByPtESProducer")

trackCounting3D2nd = cms.ESProducer("TrackCountingESProducer",
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    impactParameterType = cms.int32(0),
    trackQualityClass = cms.string('any'),
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(2)
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

ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

UpdaterService = cms.Service("UpdaterService")

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
    EmulateBxInEvent = cms.int32(1),
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
    etHadSource = cms.InputTag("hltGctDigis"),
    etTotalSource = cms.InputTag("hltGctDigis"),
    centralBxOnly = cms.bool(True),
    etMissSource = cms.InputTag("hltGctDigis"),
    produceMuonParticles = cms.bool(True),
    forwardJetSource = cms.InputTag("hltGctDigis","forJets"),
    centralJetSource = cms.InputTag("hltGctDigis","cenJets"),
    produceCaloParticles = cms.bool(True),
    tauJetSource = cms.InputTag("hltGctDigis","tauJets"),
    isolatedEmSource = cms.InputTag("hltGctDigis","isoEm"),
    nonIsolatedEmSource = cms.InputTag("hltGctDigis","nonIsoEm")
)

hltOfflineBeamSpot = cms.EDProducer("BeamSpotProducer")

hltBoolFirst = cms.EDFilter("HLTBool",
    result = cms.bool(False)
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

hltPre2jet = cms.EDFilter("HLTPrescaler")

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
    FedLabel = cms.untracked.InputTag("hltEcalRegionalJetsFEDs"),
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
    MomEmDepth = cms.double(0.0),
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    MomConstrMethod = cms.int32(0),
    HcalThreshold = cms.double(-1000.0),
    HF1Threshold = cms.double(1.2),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    MomHadDepth = cms.double(0.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    MomTotDepth = cms.double(0.0),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRegionalJetsRecHit","EcalRecHitsEB"), cms.InputTag("hltEcalRegionalJetsRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltIterativeCone5CaloJetsRegional = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltTowerMakerForJets"),
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

hltPre3jet = cms.EDFilter("HLTPrescaler")

hlt3jet85 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(85.0),
    MinN = cms.int32(3)
)

hltL1s4jet = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre4jet = cms.EDFilter("HLTPrescaler")

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
    FedLabel = cms.untracked.InputTag("hltEcalRegionalRestFEDs"),
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
    MomEmDepth = cms.double(0.0),
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    MomConstrMethod = cms.int32(0),
    HcalThreshold = cms.double(-1000.0),
    HF1Threshold = cms.double(1.2),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    MomHadDepth = cms.double(0.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    MomTotDepth = cms.double(0.0),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"), cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

hltIterativeCone5CaloJets = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltTowerMakerForAll"),
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
    src = cms.InputTag("hltTowerMakerForAll"),
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

hltL1s2jetAco = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jetAco = cms.EDFilter("HLTPrescaler")

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

hltPre1jet1METAco = cms.EDFilter("HLTPrescaler")

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

hltPre1jet1MET = cms.EDFilter("HLTPrescaler")

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

hltPre2jet1MET = cms.EDFilter("HLTPrescaler")

hlt2jet125New = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(125.0),
    MinN = cms.int32(2)
)

hltL1s3jet1MET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre3jet1MET = cms.EDFilter("HLTPrescaler")

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

hltPre4jet1MET = cms.EDFilter("HLTPrescaler")

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

hltPre1MET1HT = cms.EDFilter("HLTPrescaler")

hlt1MET65 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(65.0),
    MinN = cms.int32(1)
)

hlt1HT350 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(350.0)
)

hltL1s2jetvbfMET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre2jetvbfMET = cms.EDFilter("HLTPrescaler")

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

hltPrenv = cms.EDFilter("HLTPrescaler")

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

hltPrephi2met = cms.EDFilter("HLTPrescaler")

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

hltPrephijet1met = cms.EDFilter("HLTPrescaler")

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

hltPrephijet2met = cms.EDFilter("HLTPrescaler")

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

hltPrephijet1jet2 = cms.EDFilter("HLTPrescaler")

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

hltPrerapgap = cms.EDFilter("HLTPrescaler")

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
    FedLabel = cms.untracked.InputTag("hltEcalRegionalEgammaFEDs"),
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
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hltIslandSuperClustersL1Isolated","islandEndcapSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0),
        corrF = cms.vint32(0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEE")
)

hltCorrectedIslandBarrelSuperClustersL1Isolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hltIslandSuperClustersL1Isolated","islandBarrelSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0),
        corrF = cms.vint32(0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB")
)

hltCorrectedHybridSuperClustersL1Isolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    etThresh = cms.double(5.0),
    rawSuperClusterProducer = cms.InputTag("hltHybridSuperClustersL1Isolated"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(

    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.7),
        fBremVec = cms.vdouble(-0.01217, 0.031, 0.9887, -0.0003776, 1.598),
        brLinearHighThr = cms.double(8.0),
        fEtEtaVec = cms.vdouble(1.001, -0.8654, 3.131, 0.0, 0.735, 
            20.72, 1.169, 8.0, 1.023, -0.00181, 
            0.0),
        corrF = cms.vint32(1, 1, 0)
    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB")
)

hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    etThresh = cms.double(5.0),
    preshRecHitProducer = cms.InputTag("hltEcalPreshowerRecHit","EcalRecHitsES"),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    assocSClusterCollection = cms.string(''),
    endcapSClusterProducer = cms.InputTag("hltCorrectedIslandEndcapSuperClustersL1Isolated"),
    preshNclust = cms.int32(4),
    debugLevel = cms.string(''),
    preshClusterEnergyCut = cms.double(0.0),
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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoSingleL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsolatedElectronHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.15),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1IsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoSingleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltSiPixelDigis = cms.EDFilter("SiPixelRawToDigi",
    InputLabel = cms.untracked.string('rawDataCollector')
)

hltSiPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    src = cms.InputTag("hltSiPixelDigis"),
    ChannelThreshold = cms.int32(2500),
    MissCalibrate = cms.untracked.bool(True),
    VCaltoElectronGain = cms.int32(65),
    VCaltoElectronOffset = cms.int32(0),
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(3000),
    ClusterThreshold = cms.double(5050.0)
)

hltSiPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    src = cms.InputTag("hltSiPixelClusters"),
    CPE = cms.string('PixelCPEGeneric')
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
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.08),
        pPhiMin1 = cms.double(-0.015),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.05),
        fromTrackerSeeds = cms.bool(False),
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1IsoWithMaterialTracks")
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hltIslandSuperClustersL1NonIsolated","islandEndcapSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0),
        corrF = cms.vint32(0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEE")
)

hltCorrectedIslandBarrelSuperClustersL1NonIsolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.InputTag("hltIslandSuperClustersL1NonIsolated","islandBarrelSuperClusters"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        brLinearHighThr = cms.double(0.0),
        fEtEtaVec = cms.vdouble(0.0),
        corrF = cms.vint32(0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(

    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB")
)

hltCorrectedHybridSuperClustersL1NonIsolated = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    etThresh = cms.double(5.0),
    rawSuperClusterProducer = cms.InputTag("hltHybridSuperClustersL1NonIsolated"),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(

    ),
    isl_fCorrPset = cms.PSet(

    ),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(

    ),
    hyb_fCorrPset = cms.PSet(
        brLinearLowThr = cms.double(0.7),
        fBremVec = cms.vdouble(-0.01217, 0.031, 0.9887, -0.0003776, 1.598),
        brLinearHighThr = cms.double(8.0),
        fEtEtaVec = cms.vdouble(1.001, -0.8654, 3.131, 0.0, 0.735, 
            20.72, 1.169, 8.0, 1.023, -0.00181, 
            0.0),
        corrF = cms.vint32(1, 1, 0)
    ),
    recHitProducer = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB")
)

hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    etThresh = cms.double(5.0),
    preshRecHitProducer = cms.InputTag("hltEcalPreshowerRecHit","EcalRecHitsES"),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    assocSClusterCollection = cms.string(''),
    endcapSClusterProducer = cms.InputTag("hltCorrectedIslandEndcapSuperClustersL1NonIsolated"),
    preshNclust = cms.int32(4),
    debugLevel = cms.string(''),
    preshClusterEnergyCut = cms.double(0.0),
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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoSingleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsolatedElectronHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.15),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1NonIsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
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
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.08),
        pPhiMin1 = cms.double(-0.015),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.05),
        fromTrackerSeeds = cms.bool(False),
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1NonIsoWithMaterialTracks")
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
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoDoubleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoDoubleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronEoverpFilter")
)

hltDoubleElectronL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoDoubleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoubleElectronEoverpFilter")
)

hltDoubleElectronL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoSinglePhotonL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsolatedPhotonEcalIsol = cms.EDFilter("EgammaHLTEcalIsolationProducersRegional",
    egEcalIsoEtMin = cms.double(0.0),
    SCAlgoType = cms.int32(0),
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
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1IsolatedPhotonHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1IsoSinglePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoSinglePhotonEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoSinglePhotonL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsolatedPhotonEcalIsol = cms.EDFilter("EgammaHLTEcalIsolationProducersRegional",
    egEcalIsoEtMin = cms.double(0.0),
    SCAlgoType = cms.int32(0),
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
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsolatedPhotonHcalIsol = cms.EDFilter("EgammaHLTHcalIsolationProducersRegional",
    hfRecHitProducer = cms.InputTag("hltHfreco"),
    recoEcalCandidateProducer = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    egHcalIsoPtMin = cms.double(0.0),
    egHcalIsoConeSize = cms.double(0.3),
    hbRecHitProducer = cms.InputTag("hltHbhereco")
)

hltL1NonIsoSinglePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoDoublePhotonL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoDoublePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1IsoDoublePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoDoublePhotonEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1IsoDoublePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltL1IsoDoublePhotonDoubleEtFilter = cms.EDFilter("HLTEgammaDoubleEtFilter",
    etcut1 = cms.double(20.0),
    etcut2 = cms.double(20.0),
    npaircut = cms.int32(1),
    relaxed = cms.untracked.bool(False),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    candTag = cms.InputTag("hltL1IsoDoublePhotonTrackIsolFilter")
)

hltDoublePhotonL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoDoublePhotonL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoDoublePhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoDoublePhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoDoublePhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltL1NonIsoDoublePhotonDoubleEtFilter = cms.EDFilter("HLTEgammaDoubleEtFilter",
    etcut1 = cms.double(20.0),
    etcut2 = cms.double(20.0),
    npaircut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    candTag = cms.InputTag("hltL1NonIsoDoublePhotonTrackIsolFilter")
)

hltDoublePhotonL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoSingleEMHighEtL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoSingleEMHighEtEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(5.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEMHighEtEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtHcalDBCFilter"),
    numtrackisolcut = cms.double(4.0)
)

hltSingleEMVHighEtL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoSingleEMVeryHighEtL1MatchFilterRegional"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    etcut = cms.double(200.0)
)

hltSingleEMVHEL1NonIsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoDoubleElectronZeeL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoDoubleElectronZeeHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoDoubleElectronZeePixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeEoverpFilter")
)

hltL1IsoDoubleElectronZeePMMassFilter = cms.EDFilter("HLTPMMassFilter",
    lowerMassCut = cms.double(54.22),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    relaxed = cms.untracked.bool(False),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    upperMassCut = cms.double(99999.9),
    candTag = cms.InputTag("hltL1IsoDoubleElectronZeeTrackIsolFilter"),
    nZcandcut = cms.int32(1)
)

hltZeeCounterPresc = cms.EDFilter("HLTPrescaler")

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
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronEtPhiFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoDoubleExclElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclElectronEoverpFilter")
)

hltDoubleExclElectronL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(2.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonEtPhiFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1IsoDoubleExclPhotonHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(8.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1IsoDoubleExclPhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoDoubleExclPhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltDoubleExclPhotonL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoSinglePhotonPrescaledEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1IsoSinglePhotonPrescaledHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledEcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1IsoSinglePhotonPrescaledTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1IsoSinglePhotonPrescaledHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltSinglePhotonPrescaledL1IsoPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoLargeWindowSingleL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoLargeWindowElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.3),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.045),
        PhiMin2 = cms.double(-0.01),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.2),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.03),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.2),
        PhiMax2 = cms.double(0.01),
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.3),
        pPhiMin1 = cms.double(-0.03),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.045),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.2),
        fromTrackerSeeds = cms.bool(False),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.2),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1Isolated")
)

hltL1IsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1IsoLargeWindowWithMaterialTracks")
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
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoLargeWindowSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoLargeWindowElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.3),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.045),
        PhiMin2 = cms.double(-0.01),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.2),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.03),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.2),
        PhiMax2 = cms.double(0.01),
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.3),
        pPhiMin1 = cms.double(-0.03),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.045),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.2),
        fromTrackerSeeds = cms.bool(False),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.2),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1NonIsolated")
)

hltL1NonIsoLargeWindowSingleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1NonIsoLargeWindowWithMaterialTracks")
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
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowSingleElectronHOneOEMinusOneOPFilter")
)

hltSingleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.4),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoLargeWindowDoubleElectronEoverpFilter")
)

hltDoubleElectronL1IsoLargeWindowPresc = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronEoverpFilter")
)

hltDoubleElectronL1NonIsoLargeWindowPresc = cms.EDFilter("HLTPrescaler")

hltPrescaleSingleMuIso = cms.EDFilter("HLTPrescaler")

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
    ErrorMask = cms.untracked.uint32(0),
    InputObjects = cms.InputTag("rawDataCollector"),
    ExaminerMask = cms.untracked.uint32(535557110),
    UseExaminer = cms.untracked.bool(False)
)

hltCsc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    XTasymmetry_ME1b = cms.untracked.double(0.0),
    XTasymmetry_ME1a = cms.untracked.double(0.0),
    ConstSyst_ME1a = cms.untracked.double(0.022),
    ConstSyst_ME1b = cms.untracked.double(0.02),
    XTasymmetry_ME41 = cms.untracked.double(0.025),
    CSCStripxtalksOffset = cms.untracked.double(0.03),
    CSCUseCalibrations = cms.untracked.bool(True),
    XTasymmetry_ME22 = cms.untracked.double(0.025),
    XTasymmetry_ME21 = cms.untracked.double(0.025),
    ConstSyst_ME21 = cms.untracked.double(0.06),
    XTasymmetry_ME32 = cms.untracked.double(0.025),
    CSCStripClusterChargeCut = cms.untracked.double(25.0),
    ConstSyst_ME32 = cms.untracked.double(0.06),
    NoiseLevel_ME13 = cms.untracked.double(7.0),
    NoiseLevel_ME12 = cms.untracked.double(7.0),
    NoiseLevel_ME32 = cms.untracked.double(7.0),
    NoiseLevel_ME31 = cms.untracked.double(7.0),
    ConstSyst_ME22 = cms.untracked.double(0.06),
    ConstSyst_ME41 = cms.untracked.double(0.06),
    CSCStripPeakThreshold = cms.untracked.double(10.0),
    readBadChannels = cms.bool(False),
    XTasymmetry_ME13 = cms.untracked.double(0.025),
    XTasymmetry_ME12 = cms.untracked.double(0.025),
    wireDigiTag = cms.InputTag("hltMuonCSCDigis","MuonCSCWireDigi"),
    ConstSyst_ME12 = cms.untracked.double(0.045),
    ConstSyst_ME13 = cms.untracked.double(0.065),
    XTasymmetry_ME31 = cms.untracked.double(0.025),
    ConstSyst_ME31 = cms.untracked.double(0.06),
    NoiseLevel_ME1a = cms.untracked.double(7.0),
    NoiseLevel_ME1b = cms.untracked.double(7.0),
    CSCWireClusterDeltaT = cms.untracked.int32(1),
    stripDigiTag = cms.InputTag("hltMuonCSCDigis","MuonCSCStripDigi"),
    CSCstripWireDeltaTime = cms.untracked.int32(8),
    NoiseLevel_ME21 = cms.untracked.double(7.0),
    NoiseLevel_ME22 = cms.untracked.double(7.0),
    NoiseLevel_ME41 = cms.untracked.double(7.0)
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
        DoRefit = cms.bool(False),
        NavigationType = cms.string('Standard'),
        DoBackwardFilter = cms.bool(True),
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
            FitterName = cms.string('KFFitterSmootherForL2Muon'),
            Option = cms.int32(1)
        ),
        FilterParameters = cms.PSet(
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
        )
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
    MinPt = cms.double(9.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    FedLabel = cms.untracked.InputTag("hltEcalRegionalMuonsFEDs"),
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
    MomEmDepth = cms.double(0.0),
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    MomConstrMethod = cms.int32(0),
    HcalThreshold = cms.double(-1000.0),
    HF1Threshold = cms.double(1.2),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    MomHadDepth = cms.double(0.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    MomTotDepth = cms.double(0.0),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("hltEcalRegionalMuonsRecHit","EcalRecHitsEB"), cms.InputTag("hltEcalRegionalMuonsRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
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
        Eta_fixed = cms.double(0.2),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        Rescale_Dz = cms.double(3.0),
        vertexCollection = cms.InputTag("pixelVertices"),
        Rescale_phi = cms.double(3.0),
        DeltaR = cms.double(0.2),
        DeltaZ_Region = cms.double(15.9),
        Rescale_eta = cms.double(3.0),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        Eta_min = cms.double(0.1),
        Phi_fixed = cms.double(0.2),
        EscapePt = cms.double(1.5),
        UseFixedRegion = cms.bool(False),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phi_min = cms.double(0.1),
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
        cleanerFromSharedHits = cms.bool(True),
        ptCleaner = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        directionCleaner = cms.bool(False)
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
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
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
        TransformerOutPropagator = cms.string('SmartPropagatorAny'),
        Chi2ProbabilityCut = cms.double(30.0),
        Direction = cms.int32(0),
        HitThreshold = cms.int32(1),
        TrackRecHitBuilder = cms.string('WithTrackAngle'),
        MuonTrackingRegionBuilder = cms.PSet(
            EtaR_UpperLimit_Par1 = cms.double(0.25),
            Eta_fixed = cms.double(0.2),
            beamSpot = cms.InputTag("hltOfflineBeamSpot"),
            Rescale_Dz = cms.double(3.0),
            vertexCollection = cms.InputTag("pixelVertices"),
            Rescale_phi = cms.double(3.0),
            DeltaR = cms.double(0.2),
            DeltaZ_Region = cms.double(15.9),
            Rescale_eta = cms.double(3.0),
            PhiR_UpperLimit_Par2 = cms.double(0.2),
            Eta_min = cms.double(0.05),
            Phi_fixed = cms.double(0.2),
            EscapePt = cms.double(1.5),
            UseFixedRegion = cms.bool(False),
            PhiR_UpperLimit_Par1 = cms.double(0.6),
            EtaR_UpperLimit_Par2 = cms.double(0.15),
            Phi_min = cms.double(0.05),
            UseVertex = cms.bool(False)
        ),
        TkTrackBuilder = cms.string('muonCkfTrajectoryBuilder'),
        TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
        DTRecSegmentLabel = cms.InputTag("hltDt4DSegments"),
        Chi2CutCSC = cms.double(150.0),
        Chi2CutRPC = cms.double(1.0),
        MatcherOutPropagator = cms.string('SmartPropagator'),
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
    NSigmaPt = cms.double(0.0),
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
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
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

hltPrescaleSingleMuNoIso = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(12.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonIso = cms.EDFilter("HLTPrescaler")

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
    NSigmaPt = cms.double(0.0),
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
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltDiMuonIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltDiMuonIsoL3PreFiltered"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescaleDiMuonNoIso = cms.EDFilter("HLTPrescaler")

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
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleJPsiMM = cms.EDFilter("HLTPrescaler")

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
    SaveTag = cms.untracked.bool(True),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MaxDr = cms.double(2.0),
    MinInvMass = cms.double(2.8),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleUpsilonMM = cms.EDFilter("HLTPrescaler")

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
    SaveTag = cms.untracked.bool(True),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MaxDr = cms.double(2.0),
    MinInvMass = cms.double(8.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleZMM = cms.EDFilter("HLTPrescaler")

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
    SaveTag = cms.untracked.bool(True),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MaxDr = cms.double(2.0),
    MinInvMass = cms.double(70.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleMultiMuonNoIso = cms.EDFilter("HLTPrescaler")

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
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSameSignMu = cms.EDFilter("HLTPrescaler")

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
    SaveTag = cms.untracked.bool(True),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MaxDr = cms.double(2.0),
    MinInvMass = cms.double(0.0),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPrescaleSingleMuPrescale3 = cms.EDFilter("HLTPrescaler")

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
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuPrescale5 = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(3.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPreSingleMuPrescale77 = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPreSingleMuPrescale710 = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(8.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleMuLevel1Path = cms.EDFilter("HLTPrescaler")

hltMuLevel1PathLevel1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu7 OR L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuLevel1PathL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    SaveTag = cms.untracked.bool(True),
    CandTag = cms.InputTag("hltMuLevel1PathLevel1Seed"),
    MinQuality = cms.int32(-1)
)

hltPrescaleSingleMuNoIsoRelaxedVtx2cm = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuNoIsoRelaxedVtx2mm = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(0.2),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonNoIsoRelaxedVtx2cm = cms.EDFilter("HLTPrescaler")

hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleDiMuonNoIsoRelaxedVtx2mm = cms.EDFilter("HLTPrescaler")

hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltDiMuonNoIsoL2PreFiltered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(0.2),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescalerBLifetime1jet = cms.EDFilter("HLTPrescaler")

hltBLifetimeL1seeds = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300'),
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
    jetTagComputer = cms.string('trackCounting3D2nd'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBLifetimeL25TagInfos"))
)

hltBLifetimeL25filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL25BJetTags"),
    MinTag = cms.double(3.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(False),
    MinJets = cms.int32(1)
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
    jetTagComputer = cms.string('trackCounting3D2nd'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBLifetimeL3TagInfos"))
)

hltBLifetimeL3filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL3BJetTags"),
    MinTag = cms.double(6.0),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(True),
    MinJets = cms.int32(1)
)

hltPrescalerBLifetime2jet = cms.EDFilter("HLTPrescaler")

hltBLifetime2jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(120.0),
    MinN = cms.int32(2)
)

hltPrescalerBLifetime3jet = cms.EDFilter("HLTPrescaler")

hltBLifetime3jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(70.0),
    MinN = cms.int32(3)
)

hltPrescalerBLifetime4jet = cms.EDFilter("HLTPrescaler")

hltBLifetime4jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(4)
)

hltPrescalerBLifetimeHT = cms.EDFilter("HLTPrescaler")

hltBLifetimeHTL2filter = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(470.0)
)

hltPrescalerBSoftmuon1jet = cms.EDFilter("HLTPrescaler")

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
    jetTagComputer = cms.string('softLeptonByDistance'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBSoftmuonL25TagInfos"))
)

hltBSoftmuonL25filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL25BJetTags"),
    MinTag = cms.double(0.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(False),
    MinJets = cms.int32(1)
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
    jetTagComputer = cms.string('softLeptonByPt'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBSoftmuonL3TagInfos"))
)

hltBSoftmuonL3BJetTagsByDR = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBSoftmuonL3TagInfos"),
    jetTagComputer = cms.string('softLeptonByDistance'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBSoftmuonL3TagInfos"))
)

hltBSoftmuonByDRL3filter = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL3BJetTagsByDR"),
    MinTag = cms.double(0.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(True),
    MinJets = cms.int32(1)
)

hltPrescalerBSoftmuon2jet = cms.EDFilter("HLTPrescaler")

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
    SaveTag = cms.bool(True),
    MinJets = cms.int32(1)
)

hltPrescalerBSoftmuon3jet = cms.EDFilter("HLTPrescaler")

hltBSoftmuon3jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(70.0),
    MinN = cms.int32(3)
)

hltPrescalerBSoftmuon4jet = cms.EDFilter("HLTPrescaler")

hltBSoftmuon4jetL2filter = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(4)
)

hltPrescalerBSoftmuonHT = cms.EDFilter("HLTPrescaler")

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

hltElectronBPrescale = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltElBElectronL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltElBElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltElBElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltElBElectronPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltElBElectronEoverpFilter")
)

hltMuBPrescale = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuBsoftMuPrescale = cms.EDFilter("HLTPrescaler")

hltMuBSoftL1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuBLevel1Seed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltMuBSoftIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuBSoftL1Filtered"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoSingleL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoEJetSingleEHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoEJetSingleEPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
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

hltMuJetsPrescale = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuJetsHLT1jet40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(1)
)

hltMuNoL2IsoJetsPrescale = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(6.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltMuNoL2IsoJetsHLT1jet40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(1)
)

hltMuNoIsoJetsPrescale = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    SaveTag = cms.untracked.bool(True),
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

hltemuPrescale = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltemuL1IsoSingleL1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltemuL1IsoSingleElectronHcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltEMuL2MuonPreFilter = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltEMuL1MuonFilter"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltEMuL3MuonIsoFilter = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltEMuL3MuonPreFilter"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltemuL1IsoSingleElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltemuL1IsoSingleElectronEoverpFilter")
)

hltemuNonIsoPrescale = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltemuNonIsoL1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltemuNonIsoL1HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltemuNonIsoL1IsoEtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltNonIsoEMuL2MuonPreFilter = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltNonIsoEMuL1MuonFilter"),
    MinPt = cms.double(8.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltemuNonIsoL1IsoPixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltemuNonIsoL1IsoTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltemuNonIsoL1IsoEoverpFilter")
)

hltLevel1seedHLTBackwardBSC = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('38 OR 39'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTBackwardBSC = cms.EDFilter("HLTPrescaler")

hltLevel1seedHLTForwardBSC = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('36 OR 37'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTForwardBSC = cms.EDFilter("HLTPrescaler")

hltLevel1seedHLTCSCBeamHalo = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMuBeamHalo'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrescaleHLTCSCBeamHalo = cms.EDFilter("HLTPrescaler")

hltLevel1seedHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMuBeamHalo'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrescaleHLTCSCBeamHaloOverlapRing1 = cms.EDFilter("HLTPrescaler")

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
    L1SeedsLogicalExpression = cms.string('L1_SingleMuBeamHalo'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrescaleHLTCSCBeamHaloOverlapRing2 = cms.EDFilter("HLTPrescaler")

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
    L1SeedsLogicalExpression = cms.string('L1_SingleMuBeamHalo'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPrescaleHLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTPrescaler")

hltFilter23HLTCSCBeamHaloRing2or3 = cms.EDFilter("HLTCSCRing2or3Filter",
    input = cms.InputTag("hltCsc2DRecHits"),
    xWindow = cms.double(2.0),
    minHits = cms.uint32(4),
    yWindow = cms.double(2.0)
)

hltLevel1seedHLTTrackerCosmics = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('24 OR 25 OR 26 OR 27 OR 28'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(True)
)

hltPrescaleHLTTrackerCosmics = cms.EDFilter("HLTPrescaler")

hltPreMinBiasPixel = cms.EDFilter("HLTPrescaler")

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
        nSigmaInvPtTolerance = cms.double(0.0),
        nSigmaTipMaxTolerance = cms.double(0.0),
        ComponentName = cms.string('PixelTrackFilterByKinematics'),
        chi2 = cms.double(1000.0),
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

hltPreMBForAlignment = cms.EDFilter("HLTPrescaler")

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

hltpreMin = cms.EDFilter("HLTPrescaler")

hltl1sZero = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltpreZero = cms.EDFilter("HLTPrescaler")

hltPrescaleTriggerType = cms.EDFilter("HLTPrescaler")

hltFilterTriggerType = cms.EDFilter("TriggerTypeFilter",
    TriggerFedId = cms.int32(812),
    InputLabel = cms.string('rawDataCollector'),
    SelectedTriggerType = cms.int32(2)
)

hltL1gtTrigReport = cms.EDFilter("L1GtTrigReport",
    UseL1GlobalTriggerRecord = cms.bool(False),
    L1GtRecordInputTag = cms.InputTag("hltGtDigis")
)

hltTrigReport = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT")
)

hltPrescalerElectronTau = cms.EDFilter("HLTPrescaler")

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
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltEgammaHcalIsolFilterElectronTau = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltEgammaEtFilterElectronTau"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltElectronPixelMatchFilterElectronTau = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
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
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
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
    FedLabel = cms.untracked.InputTag("hltEcalRegionalTausFEDs"),
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
    MomEmDepth = cms.double(0.0),
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hltHfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(1.1),
    HBThreshold = cms.double(0.9),
    EBThreshold = cms.double(0.09),
    MomConstrMethod = cms.int32(0),
    HcalThreshold = cms.double(-1000.0),
    HF1Threshold = cms.double(1.2),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HF1Weight = cms.double(1.0),
    MomHadDepth = cms.double(0.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hltHbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("hltHoreco"),
    MomTotDepth = cms.double(0.0),
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

hltFilterEcalIsolatedTauJetsElectronTau = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
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
    JetTrackSrc = cms.InputTag("hltJetTracksAssociatorAtVertexL25ElectronTau"),
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
    MinN = cms.int32(1),
    inputTag = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),
    MinPt = cms.double(1.0),
    saveTag = cms.untracked.bool(True)
)

hltPrescalerMuonTau = cms.EDFilter("HLTPrescaler")

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
    MinPt = cms.double(12.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
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

hltMuonTauIsoL3IsoFiltered = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltMuonTauIsoL3PreFiltered"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltSingleTauMETPrescaler = cms.EDFilter("HLTPrescaler")

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
    MinN = cms.int32(1),
    inputTag = cms.InputTag("hltIsolatedL3SingleTauMET"),
    MinPt = cms.double(10.0),
    saveTag = cms.untracked.bool(True)
)

hltSingleTauPrescaler = cms.EDFilter("HLTPrescaler")

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
    MinN = cms.int32(1),
    inputTag = cms.InputTag("hltIsolatedL3SingleTau"),
    MinPt = cms.double(1.0),
    saveTag = cms.untracked.bool(True)
)

hltSingleElectronEt10L1NonIsoHLTNonIsoPresc = cms.EDFilter("HLTPrescaler")

hltL1seedRelaxedSingleEt8 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG8'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt8"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.0),
    HoverEcut = cms.double(999999.0),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoStartUpElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated"),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.096),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.025),
        PhiMin2 = cms.double(-0.005),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.06),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.015),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.005),
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.096),
        pPhiMin1 = cms.double(-0.015),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.06),
        fromTrackerSeeds = cms.bool(False),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1Isolated")
)

hltL1NonIsoStartUpElectronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated"),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.096),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.025),
        PhiMin2 = cms.double(-0.005),
        LowPtThreshold = cms.double(5.0),
        z2MinB = cms.double(-0.06),
        dynamicPhiRoad = cms.bool(False),
        ePhiMax1 = cms.double(0.015),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.11),
        PhiMax2 = cms.double(0.005),
        preFilteredSeeds = cms.bool(False),
        r2MaxF = cms.double(0.096),
        pPhiMin1 = cms.double(-0.015),
        initialSeeds = cms.InputTag("globalMixedSeeds"),
        pPhiMax1 = cms.double(0.025),
        hbheModule = cms.string('hbhereco'),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.06),
        fromTrackerSeeds = cms.bool(False),
        hcalRecHits = cms.InputTag("hltHbhereco"),
        maxHOverE = cms.double(0.2),
        hbheInstance = cms.string(''),
        rMinI = cms.double(-0.11),
        hOverEConeSize = cms.double(0.1)
    ),
    barrelSuperClusters = cms.InputTag("hltCorrectedHybridSuperClustersL1NonIsolated")
)

hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter")
)

hltCkfL1IsoStartUpTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoStartUpElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1IsoStartUpWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1IsoStartUpTrackCandidates"),
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

hltPixelMatchStartUpElectronsL1Iso = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1IsoStartUpWithMaterialTracks")
)

hltCkfL1NonIsoStartUpTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoStartUpElectronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfL1NonIsoStartUpWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfL1NonIsoStartUpTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("hltOfflineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

hltPixelMatchStartUpElectronsL1NonIso = cms.EDFilter("EgammaHLTPixelMatchElectronProducers",
    BSProducer = cms.InputTag("hltOfflineBeamSpot"),
    TrackProducer = cms.InputTag("hltCtfL1NonIsoStartUpWithMaterialTracks")
)

hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1IsoStartUpElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
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
    candTagEle = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    originRadius = cms.double(0.02)
)

hltL1IsoStartUpElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1IsoStartUpElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1IsoStartUpElectronsRegionalCkfTrackCandidates"),
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

hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator = cms.EDFilter("EgammaHLTRegionalPixelSeedGeneratorProducers",
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
    candTagEle = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    originRadius = cms.double(0.02)
)

hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates"),
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

hltL1IsoStartUpElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1NonIsoStartupElectronTrackIsol = cms.EDFilter("EgammaHLTElectronTrackIsolationProducers",
    egTrkIsoVetoConeSize = cms.double(0.02),
    trackProducer = cms.InputTag("hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial"),
    electronProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    egTrkIsoConeSize = cms.double(0.2),
    egTrkIsoRSpan = cms.double(999999.0),
    egTrkIsoPtMin = cms.double(1.5),
    egTrkIsoZSpan = cms.double(0.1)
)

hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.0),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter")
)

hltSingleElectronEt8L1NonIsoHLTnoIsoPresc = cms.EDFilter("HLTPrescaler")

hltL1seedRelaxedSingleEt5 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG5'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.0),
    HoverEcut = cms.double(999999.0),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter")
)

hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.0),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter")
)

hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc = cms.EDFilter("HLTPrescaler")

hltL1seedRelaxedDoubleEt5 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleEG5'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(5.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.0),
    HoverEcut = cms.double(999999.0),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter")
)

hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.0),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter")
)

hltSinglePhotonEt10L1NonIsoPresc = cms.EDFilter("HLTPrescaler")

hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt8"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoSinglePhotonEt10EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoSinglePhotonEt10EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEt10EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoSinglePhotonEt10HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEt10EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoSinglePhotonEt10TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoSinglePhotonEt10HcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

hltL1sIsolTrack = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleTauJet30 OR L1_SingleTauJet40 OR L1_SingleTauJet60 OR L1_SingleTauJet80 '),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPreIsolTrackNoEcalIso = cms.EDFilter("HLTPrescaler")

hltIsolPixelTrackProd = cms.EDProducer("IsolatedPixelTrackCandidateProducer",
    PixelTracksSource = cms.InputTag("hltPixelTracks"),
    L1eTauJetsSource = cms.InputTag("hltL1extraParticles","Tau"),
    ecalFilterLabel = cms.InputTag("aaa"),
    L1GTSeedLabel = cms.InputTag("hltL1sIsolTrack")
)

hltIsolPixelTrackFilter = cms.EDFilter("HLTPixelIsolTrackFilter",
    MaxPtNearby = cms.double(2.0),
    MinEnergyTrack = cms.double(15.0),
    MinPtTrack = cms.double(20.0),
    MaxEtaTrack = cms.double(2.1),
    candTag = cms.InputTag("hltIsolPixelTrackProd"),
    filterTrackEnergy = cms.bool(False)
)

hltSiStripRegFED = cms.EDFilter("SiStripRegFEDSelector",
    rawInputLabel = cms.InputTag("rawDataCollector"),
    regSeedLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    delta = cms.double(1.0)
)

hltEcalRegFED = cms.EDFilter("ECALRegFEDSelector",
    rawInputLabel = cms.InputTag("rawDataCollector"),
    regSeedLabel = cms.InputTag("hltIsolPixelTrackFilter"),
    delta = cms.double(1.0)
)

hltSubdetFED = cms.EDFilter("SubdetFEDSelector",
    rawInputLabel = cms.InputTag("rawDataCollector"),
    getMuon = cms.bool(False),
    getHCAL = cms.bool(True),
    getECAL = cms.bool(False),
    getSiPixel = cms.bool(True),
    getTrigger = cms.bool(True),
    getSiStrip = cms.bool(False)
)

hltL1sHcalPhiSym = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltHcalPhiSymPresc = cms.EDFilter("HLTPrescaler")

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

hltL1sEcalPhiSym = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ZeroBias OR L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleEG2 OR L1_DoubleEG1 OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltEcalPhiSymPresc = cms.EDFilter("HLTPrescaler")

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

hltPrePi0Ecal = cms.EDFilter("HLTPrescaler")

hltL1sEcalPi0 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15 OR L1_SingleJet30 OR L1_SingleJet50 OR L1_SingleJet70 OR L1_SingleJet100 OR L1_SingleJet150 OR L1_SingleJet200 OR L1_DoubleJet70 OR L1_DoubleJet100'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltAlCaPi0RegRecHits = cms.EDFilter("HLTPi0RecHitsFilter",
    seleS4S9GammaOne = cms.double(0.85),
    seleMinvMaxPi0 = cms.double(0.22),
    gammaCandPhiSize = cms.int32(21),
    selePtGammaOne = cms.double(0.9),
    ParameterX0 = cms.double(0.89),
    seleXtalMinEnergy = cms.double(0.0),
    selePtPi0 = cms.double(2.5),
    clusSeedThr = cms.double(0.5),
    clusPhiSize = cms.int32(3),
    selePi0BeltDR = cms.double(0.2),
    clusEtaSize = cms.int32(3),
    selePi0Iso = cms.double(0.5),
    ParameterW0 = cms.double(4.2),
    seleNRHMax = cms.int32(1000),
    selePi0BeltDeta = cms.double(0.05),
    barrelHits = cms.InputTag("hltEcalRegionalEgammaRecHit","EcalRecHitsEB"),
    ParameterLogWeighted = cms.bool(True),
    seleS4S9GammaTwo = cms.double(0.85),
    pi0BarrelHitCollection = cms.string('pi0EcalRecHitsEB'),
    seleMinvMinPi0 = cms.double(0.06),
    gammaCandEtaSize = cms.int32(9),
    selePtGammaTwo = cms.double(0.9),
    ParameterT0_barl = cms.double(5.7)
)

hltPrescaleSingleMuLevel2 = cms.EDFilter("HLTPrescaler")

hltSingleMuLevel2NoIsoL2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered"),
    MinPt = cms.double(9.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltJpsitoMumuL1SeedRelaxed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleMu3'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltJpsitoMumuL1FilteredRelaxed = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltJpsitoMumuL1SeedRelaxed"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(2),
    MinQuality = cms.int32(-1)
)

hltDisplacedJpsitoMumuFilterRelaxed = cms.EDFilter("HLTDisplacedmumuFilter",
    Src = cms.InputTag("hltMuTracks"),
    MinLxySignificance = cms.double(3.0),
    MinPt = cms.double(3.0),
    ChargeOpt = cms.int32(-1),
    MaxEta = cms.double(2.5),
    FastAccept = cms.bool(False),
    MaxInvMass = cms.double(6.0),
    MinPtPair = cms.double(4.0),
    MinCosinePointingAngle = cms.double(0.9),
    MaxNormalisedChi2 = cms.double(10.0),
    MinInvMass = cms.double(1.0)
)

PreHLT2Photon10L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT2Photon8L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(8.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT1ElectronLWEt12L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt8"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.9),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter")
)

hltL1seedRelaxedSingleEt10 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

PreHLT1ElectronLWEt15L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt10"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter")
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.9),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter")
)

hltL1seedRelaxedSingleEt15 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

PreHLT1Photon20L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(3.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(12.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

PreHLT1Photon15L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt10"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT1Photon25L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(25.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT1Electron18L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(18.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.0),
    HoverEcut = cms.double(999999.0),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter")
)

hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.0),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter")
)

hltL1seedRelaxedSingleEt12 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG12'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

PreHLT1Electron15L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt12"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.0),
    HoverEcut = cms.double(999999.0),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter")
)

hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(9999999.0),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter")
)

PreHLT1ElectronLW12L1IHIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt8"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTIsoSingleElectronEt12EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter")
)

hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter")
)

PreHLT1ElectronLWEt18L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(18.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.12),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter")
)

PreHLT1ElectronLWEt15L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt12"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter")
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    pttrackisolcut = cms.double(0.12),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter")
)

PreHLT1Photon40L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(40.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT1Photon30L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(30.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

PreHLT1Photon45L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(45.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(3.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(12.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

PreHLT1Photon30L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(30.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(3.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(12.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

PreHLT1Photon25L1RHIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(25.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

PreHLT1Photon20L1RHIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

PreHLT1Photon15L1RHIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt12"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(4.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)

PreHLT2ElectronLWonlyPMEt8L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(8.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter")
)

PreHLT2ElectronLWonlyPMEt10L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt5"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(10.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter")
)

hltL1seedRelaxedDoubleEt10 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleEG10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

PreHLT2ElectronLWonlyPMEt12L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt10"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(12.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(2),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter")
)

PreHLT2Photon20L1RHNonIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt10"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(9999999.9),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(9999999.9),
    HoverEcut = cms.double(9999999.9),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(9999999.9),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter"),
    numtrackisolcut = cms.double(9999999.0)
)

hltl1sMinHcal = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJetCountsHFTow OR L1_DoubleJetCountsHFTow OR L1_SingleJetCountsHFRing0Sum3 OR L1_DoubleJetCountsHFRing0Sum3 OR L1_SingleJetCountsHFRing0Sum6 OR L1_DoubleJetCountsHFRing0Sum6'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltpreMinHcal = cms.EDFilter("HLTPrescaler")

hltl1sMinEcal = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleEG2 OR L1_DoubleEG1'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltpreMinEcal = cms.EDFilter("HLTPrescaler")

PreHLT1Electron15L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt12"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(15.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(6.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(6.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    doIsolated = cms.bool(False),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter")
)

hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter = cms.EDFilter("HLTElectronOneOEMinusOneOPFilterRegional",
    doIsolated = cms.bool(False),
    electronNonIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    barrelcut = cms.double(999.03),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter"),
    endcapcut = cms.double(999.03)
)

hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    pttrackisolcut = cms.double(0.12),
    L1NonIsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter")
)

PreHLT1Photon40L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedSingleEt15"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(40.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(3.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(12.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

PreHLT2Photon20L1RHLooseIso = cms.EDFilter("HLTPrescaler")

hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(False),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedRelaxedDoubleEt10"),
    candNonIsolatedTag = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(2)
)

hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(20.0),
    ncandcut = cms.int32(2),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    ncandcut = cms.int32(2),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(3.0),
    isoTag = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)

hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(12.0),
    HoverEcut = cms.double(0.1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(2),
    doIsolated = cms.bool(False),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter"),
    isoTag = cms.InputTag("hltL1IsolatedPhotonHcalIsol")
)

hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(2),
    isoTag = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    candTag = cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter"),
    numtrackisolcut = cms.double(3.0)
)

hltL1s4jet30 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_QuadJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre4jet30 = cms.EDFilter("HLTPrescaler")

hlt4jet30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(4)
)

hlt1METSingleTauRelaxed = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(1)
)

hltL2SingleTauIsolationSelectorRelaxed = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(10000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2SingleTauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltFilterSingleTauEcalIsolationRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2SingleTauIsolationSelectorRelaxed","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltAssociatorL25SingleTauRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2SingleTauIsolationSelectorRelaxed","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25SingleTauRelaxed = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25SingleTauRelaxed"),
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

hltIsolatedL25SingleTauRelaxed = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25SingleTauRelaxed")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL25SingleTauRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL25SingleTauRelaxed"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltL3SingleTauPixelSeedsRelaxed = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
            JetSrc = cms.InputTag("hltIsolatedL25SingleTauRelaxed"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesL3SingleTauRelaxed = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL3SingleTauPixelSeedsRelaxed'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('trajBuilderL3'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksL3SingleTauRelaxed = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesL3SingleTauRelaxed"),
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

hltAssociatorL3SingleTauRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedL25SingleTauRelaxed"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3SingleTauRelaxed"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL3SingleTauRelaxed = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(5),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL3SingleTauRelaxed"),
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

hltIsolatedL3SingleTauRelaxed = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3SingleTauRelaxed")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL3SingleTauRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    MinN = cms.int32(1),
    inputTag = cms.InputTag("hltIsolatedL3SingleTauRelaxed"),
    MinPt = cms.double(1.0),
    saveTag = cms.untracked.bool(True)
)

hlt1METSingleTauMETRelaxed = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(1)
)

hltL2SingleTauMETIsolationSelectorRelaxed = cms.EDFilter("L2TauIsolationSelector",
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

hltFilterSingleTauMETEcalIsolationRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2SingleTauMETIsolationSelectorRelaxed","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(1)
)

hltAssociatorL25SingleTauMETRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2SingleTauMETIsolationSelectorRelaxed","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25SingleTauMETRelaxed = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25SingleTauMETRelaxed"),
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

hltIsolatedL25SingleTauMETRelaxed = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25SingleTauMETRelaxed")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL25SingleTauMETRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltIsolatedL25SingleTauMETRelaxed"),
    MinPt = cms.double(10.0),
    MinN = cms.int32(1)
)

hltL3SingleTauMETPixelSeedsRelaxed = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
            JetSrc = cms.InputTag("hltIsolatedL25SingleTauMETRelaxed"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesL3SingleTauMETRelaxed = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltL3SingleTauMETPixelSeedsRelaxed'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('trajBuilderL3'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksL3SingleTauMETRelaxed = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltCkfTrackCandidatesL3SingleTauMETRelaxed"),
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

hltAssociatorL3SingleTauMETRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedL25SingleTauMETRelaxed"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3SingleTauMETRelaxed"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL3SingleTauMETRelaxed = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(5),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL3SingleTauMETRelaxed"),
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

hltIsolatedL3SingleTauMETRelaxed = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3SingleTauMETRelaxed")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL3SingleTauMETRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    MinN = cms.int32(1),
    inputTag = cms.InputTag("hltIsolatedL3SingleTauMETRelaxed"),
    MinPt = cms.double(10.0),
    saveTag = cms.untracked.bool(True)
)

hltDoubleTauPrescaler = cms.EDFilter("HLTPrescaler")

hltDoubleTauL1SeedFilterRelaxed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_DoubleTauJet20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL2DoubleTauJetsRelaxed = cms.EDFilter("L2TauJetsProvider",
    L1Particles = cms.InputTag("hltL1extraParticles","Tau"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1Regional"), cms.InputTag("hltIcone5Tau2Regional"), cms.InputTag("hltIcone5Tau3Regional"), cms.InputTag("hltIcone5Tau4Regional")),
    EtMin = cms.double(15.0),
    L1TauTrigger = cms.InputTag("hltDoubleTauL1SeedFilterRelaxed")
)

hltL2DoubleTauIsolationProducerRelaxed = cms.EDProducer("L2TauIsolationProducer",
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
    L2TauJetCollection = cms.InputTag("hltL2DoubleTauJetsRelaxed"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(False),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2DoubleTauIsolationSelectorRelaxed = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(15.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(5.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2DoubleTauIsolationProducerRelaxed","L2TauIsolationInfoAssociator")
)

hltFilterDoubleTauEcalIsolationRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltL2DoubleTauIsolationSelectorRelaxed","Isolated"),
    MinPt = cms.double(1.0),
    MinN = cms.int32(2)
)

hltAssociatorL25PixelTauIsolatedRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2DoubleTauIsolationSelectorRelaxed","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25PixelTauIsolatedRelaxed = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25PixelTauIsolatedRelaxed"),
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

hltIsolatedL25PixelTauRelaxed = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(False),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.2),
    MatchingCone = cms.double(0.1),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25PixelTauIsolatedRelaxed")),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(3.0),
    UseVertex = cms.bool(False)
)

hltFilterL25PixelTauRelaxed = cms.EDFilter("HLT1Tau",
    MaxEta = cms.double(5.0),
    MinN = cms.int32(2),
    inputTag = cms.InputTag("hltIsolatedL25PixelTauRelaxed"),
    MinPt = cms.double(0.0),
    saveTag = cms.untracked.bool(True)
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
    MinN = cms.int32(2),
    inputTag = cms.InputTag("hltIsolatedL25PixelTau"),
    MinPt = cms.double(0.0),
    saveTag = cms.untracked.bool(True)
)

hltPre1Level1jet15 = cms.EDFilter("HLTPrescaler")

hltL1s1Level1jet15 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1s1jet30 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet30 = cms.EDFilter("HLTPrescaler")

hlt1jet30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(1)
)

hltL1s1jet50 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet50 = cms.EDFilter("HLTPrescaler")

hlt1jet50 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(50.0),
    MinN = cms.int32(1)
)

hltL1s1jet80 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet50'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet80 = cms.EDFilter("HLTPrescaler")

hlt1jet80 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(80.0),
    MinN = cms.int32(1)
)

hltL1s1jet110 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet110 = cms.EDFilter("HLTPrescaler")

hlt1jet110 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(110.0),
    MinN = cms.int32(1)
)

hltL1s1jet250 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet250 = cms.EDFilter("HLTPrescaler")

hlt1jet250 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(250.0),
    MinN = cms.int32(1)
)

hltL1s1SumET = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETT60'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET1SumET = cms.EDFilter("HLTPrescaler")

hlt1SumET120 = cms.EDFilter("HLTGlobalSumMET",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinN = cms.int32(1),
    Min = cms.double(120.0)
)

hltL1s1jet180 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1jet180 = cms.EDFilter("HLTPrescaler")

hlt1jet180regional = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5Regional"),
    MinPt = cms.double(180.0),
    MinN = cms.int32(1)
)

hltPreLevel1MET20 = cms.EDFilter("HLTPrescaler")

hltL1sLevel1MET20 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1s1MET25 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM20'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET25 = cms.EDFilter("HLTPrescaler")

hlt1MET25 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(25.0),
    MinN = cms.int32(1)
)

hltL1s1MET35 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET35 = cms.EDFilter("HLTPrescaler")

hlt1MET35 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(1)
)

hltL1s1MET50 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM40'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET50 = cms.EDFilter("HLTPrescaler")

hlt1MET50 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(50.0),
    MinN = cms.int32(1)
)

hltL1s1MET65 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM50'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET65 = cms.EDFilter("HLTPrescaler")

hltL1s1MET75 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_ETM50'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPre1MET75 = cms.EDFilter("HLTPrescaler")

hlt1MET75 = cms.EDFilter("HLT1CaloMET",
    MaxEta = cms.double(-1.0),
    inputTag = cms.InputTag("hltMet"),
    MinPt = cms.double(75.0),
    MinN = cms.int32(1)
)

hltL1sdijetave15 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave15 = cms.EDFilter("HLTPrescaler")

hltdijetave15 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(15.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave30 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave30 = cms.EDFilter("HLTPrescaler")

hltdijetave30 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(30.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave50 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet50'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave50 = cms.EDFilter("HLTPrescaler")

hltdijetave50 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(50.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave70 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave70 = cms.EDFilter("HLTPrescaler")

hltdijetave70 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(70.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave130 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave130 = cms.EDFilter("HLTPrescaler")

hltdijetave130 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(130.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltL1sdijetave220 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet70'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltPredijetave220 = cms.EDFilter("HLTPrescaler")

hltdijetave220 = cms.EDFilter("HLTDiJetAveFilter",
    inputJetTag = cms.InputTag("hltIterativeCone5CaloJets"),
    minEtAve = cms.double(220.0),
    minDphi = cms.double(0.0),
    minEtJet3 = cms.double(3000.0)
)

hltPrescalerBLifetime1jet120 = cms.EDFilter("HLTPrescaler")

hltBLifetimeL1seedsLowEnergy = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltBLifetime1jetL2filter120 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(120.0),
    MinN = cms.int32(1)
)

hltBLifetimeL25JetsRelaxed = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBLifetimeHighestEtJets"),
    etMin = cms.double(30.0)
)

hltBLifetimeL25AssociatorRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltBLifetimeL25JetsRelaxed"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltBLifetimeL25TagInfosRelaxed = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(3),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("hltPixelVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("hltBLifetimeL25AssociatorRelaxed"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(False),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(5.0)
)

hltBLifetimeL25BJetTagsRelaxed = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBLifetimeL25TagInfosRelaxed"),
    jetTagComputer = cms.string('trackCounting3D2nd'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBLifetimeL25TagInfosRelaxed"))
)

hltBLifetimeL25filterRelaxed = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL25BJetTagsRelaxed"),
    MinTag = cms.double(2.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(False),
    MinJets = cms.int32(1)
)

hltBLifetimeL3JetsRelaxed = cms.EDFilter("GetJetsFromHLTobject",
    jets = cms.InputTag("hltBLifetimeL25filterRelaxed")
)

hltBLifetimeRegionalPixelSeedGeneratorRelaxed = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
            deltaPhiRegion = cms.double(0.5),
            originHalfLength = cms.double(0.2),
            originZPos = cms.double(0.0),
            deltaEtaRegion = cms.double(0.5),
            ptMin = cms.double(1.0),
            JetSrc = cms.InputTag("hltBLifetimeL3JetsRelaxed"),
            originRadius = cms.double(0.2),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltBLifetimeRegionalCkfTrackCandidatesRelaxed = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('hltBLifetimeRegionalPixelSeedGeneratorRelaxed'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('bJetRegionalTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltBLifetimeRegionalCtfWithMaterialTracksRelaxed = cms.EDProducer("TrackProducer",
    src = cms.InputTag("hltBLifetimeRegionalCkfTrackCandidatesRelaxed"),
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

hltBLifetimeL3AssociatorRelaxed = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltBLifetimeL3JetsRelaxed"),
    tracks = cms.InputTag("hltBLifetimeRegionalCtfWithMaterialTracksRelaxed"),
    coneSize = cms.double(0.5)
)

hltBLifetimeL3TagInfosRelaxed = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("hltPixelVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("hltBLifetimeL3AssociatorRelaxed"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(False),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(20.0)
)

hltBLifetimeL3BJetTagsRelaxed = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("hltBLifetimeL3TagInfosRelaxed"),
    jetTagComputer = cms.string('trackCounting3D2nd'),
    tagInfos = cms.VInputTag(cms.InputTag("hltBLifetimeL3TagInfosRelaxed"))
)

hltBLifetimeL3filterRelaxed = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBLifetimeL3BJetTagsRelaxed"),
    MinTag = cms.double(3.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(True),
    MinJets = cms.int32(1)
)

hltPrescalerBLifetime1jet160 = cms.EDFilter("HLTPrescaler")

hltBLifetime1jetL2filter160 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(160.0),
    MinN = cms.int32(1)
)

hltPrescalerBLifetime2jet100 = cms.EDFilter("HLTPrescaler")

hltBLifetime2jetL2filter100 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(2)
)

hltPrescalerBLifetime2jet60 = cms.EDFilter("HLTPrescaler")

hltBLifetime2jetL2filter60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(2)
)

hltPrescalerBSoftmuon2jet100 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon2jetL2filter100 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(100.0),
    MinN = cms.int32(2)
)

hltBSoftmuonL3filterRelaxed = cms.EDFilter("HLTJetTag",
    JetTag = cms.InputTag("hltBSoftmuonL3BJetTags"),
    MinTag = cms.double(0.5),
    MaxTag = cms.double(99999.0),
    SaveTag = cms.bool(True),
    MinJets = cms.int32(1)
)

hltPrescalerBSoftmuon2jet60 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon2jetL2filter60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(2)
)

hltPrescalerBLifetime3jet40 = cms.EDFilter("HLTPrescaler")

hltBLifetime3jetL2filter40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(3)
)

hltPrescalerBLifetime3jet60 = cms.EDFilter("HLTPrescaler")

hltBLifetime3jetL2filter60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(3)
)

hltPrescalerBSoftmuon3jet40 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon3jetL2filter40 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(40.0),
    MinN = cms.int32(3)
)

hltPrescalerBSoftmuon3jet60 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon3jetL2filter60 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(60.0),
    MinN = cms.int32(3)
)

hltPrescalerBLifetime4jet30 = cms.EDFilter("HLTPrescaler")

hltBLifetime4jetL2filter30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(4)
)

hltPrescalerBLifetime4jet35 = cms.EDFilter("HLTPrescaler")

hltBLifetime4jetL2filter35 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(4)
)

hltPrescalerBSoftmuon4jet30 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon4jetL2filter30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(4)
)

hltPrescalerBSoftmuon4jet35 = cms.EDFilter("HLTPrescaler")

hltBSoftmuon4jetL2filter35 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(35.0),
    MinN = cms.int32(4)
)

hltPrescalerBLifetimeHT320 = cms.EDFilter("HLTPrescaler")

hltBLifetimeHTL2filter320 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(320.0)
)

hltPrescalerBLifetimeHT420 = cms.EDFilter("HLTPrescaler")

hltBLifetimeHTL2filter420 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(420.0)
)

hltPrescalerBSoftmuonHT250 = cms.EDFilter("HLTPrescaler")

hltBSoftmuonHTL1seedsLowEnergy = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_HTT200'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltBSoftmuonHTL2filter250 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(250.0)
)

hltPrescalerBSoftmuonHT330 = cms.EDFilter("HLTPrescaler")

hltBSoftmuonHTL2filter330 = cms.EDFilter("HLTGlobalSumHT",
    observable = cms.string('sumEt'),
    Max = cms.double(-1.0),
    inputTag = cms.InputTag("hltHtMet"),
    MinN = cms.int32(1),
    Min = cms.double(330.0)
)

hltL1seedEJet30 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_EG5_TripleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL1IsoSingleEJet30L1MatchFilter = cms.EDFilter("HLTEgammaL1MatchFilterRegional",
    doIsolated = cms.bool(True),
    region_eta_size_ecap = cms.double(1.0),
    endcap_end = cms.double(2.65),
    region_eta_size = cms.double(0.522),
    l1IsolatedTag = cms.InputTag("hltL1extraParticles","Isolated"),
    candIsolatedTag = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    region_phi_size = cms.double(1.044),
    barrel_end = cms.double(1.4791),
    L1SeedFilterTag = cms.InputTag("hltL1seedEJet30"),
    candNonIsolatedTag = cms.InputTag("hltRecoNonIsolatedEcalCandidate"),
    l1NonIsolatedTag = cms.InputTag("hltL1extraParticles","NonIsolated"),
    ncandcut = cms.int32(1)
)

hltL1IsoEJetSingleEEt5Filter = cms.EDFilter("HLTEgammaEtFilter",
    etcut = cms.double(5.0),
    ncandcut = cms.int32(1),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    inputTag = cms.InputTag("hltL1IsoSingleEJet30L1MatchFilter"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate")
)

hltL1IsoEJetSingleEEt5HcalIsolFilter = cms.EDFilter("HLTEgammaHcalIsolFilter",
    HoverEt2cut = cms.double(0.0),
    nonIsoTag = cms.InputTag("hltSingleEgammaHcalNonIsol"),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    hcalisolbarrelcut = cms.double(3.0),
    HoverEcut = cms.double(0.05),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    hcalisolendcapcut = cms.double(3.0),
    ncandcut = cms.int32(1),
    doIsolated = cms.bool(True),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEt5Filter"),
    isoTag = cms.InputTag("hltL1IsolatedElectronHcalIsol")
)

hltL1IsoEJetSingleEEt5PixelMatchFilter = cms.EDFilter("HLTElectronPixelMatchFilter",
    L1IsoPixelSeedsTag = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    doIsolated = cms.bool(True),
    L1NonIsoCand = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    L1NonIsoPixelSeedsTag = cms.InputTag("electronPixelSeeds"),
    L1IsoCand = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    npixelmatchcut = cms.double(1.0),
    ncandcut = cms.int32(1),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEt5HcalIsolFilter")
)

hltL1IsoEJetSingleEEt5EoverpFilter = cms.EDFilter("HLTElectronEoverpFilterRegional",
    doIsolated = cms.bool(True),
    electronNonIsolatedProducer = cms.InputTag("pixelMatchElectronsForHLT"),
    electronIsolatedProducer = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    eoverpendcapcut = cms.double(2.45),
    ncandcut = cms.int32(1),
    eoverpbarrelcut = cms.double(2.0),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEt5PixelMatchFilter")
)

hltL1IsoEJetSingleEEt5TrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    L1NonIsoCand = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    L1IsoCand = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    SaveTag = cms.untracked.bool(True),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltL1IsoElectronTrackIsol"),
    candTag = cms.InputTag("hltL1IsoEJetSingleEEt5EoverpFilter")
)

hltej3jet30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(3)
)

hltMuNoIsoJets30Level1Seed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_Mu3_TripleJet15'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuNoIsoJetsMinPt4L1Filtered = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltMuNoIsoJets30Level1Seed"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltMuNoIsoJetsMinPt4L2PreFiltered = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoIsoJetsMinPt4L1Filtered"),
    MinPt = cms.double(3.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltMuNoIsoJetsMinPtL3PreFiltered = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltMuNoIsoJetsMinPt4L2PreFiltered"),
    MinPt = cms.double(5.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltMuNoIsoHLTJets3jet30 = cms.EDFilter("HLT1CaloJet",
    MaxEta = cms.double(5.0),
    inputTag = cms.InputTag("hltMCJetCorJetIcone5"),
    MinPt = cms.double(30.0),
    MinN = cms.int32(3)
)

hltPrescaleMuLevel1Open = cms.EDFilter("HLTPrescaler")

hltMuLevel1PathLevel1OpenSeed = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMuOpen OR L1_SingleMu3 OR L1_SingleMu5'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltMuLevel1PathL1OpenFiltered = cms.EDFilter("HLTMuonL1Filter",
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    SaveTag = cms.untracked.bool(True),
    CandTag = cms.InputTag("hltMuLevel1PathLevel1OpenSeed"),
    MinQuality = cms.int32(-1)
)

hltPrescaleSingleMuNoIso9 = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoL2PreFiltered7 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered9 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered7"),
    MinPt = cms.double(9.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuNoIso11 = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoL2PreFiltered9 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered"),
    MinPt = cms.double(9.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered11 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered9"),
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuNoIso13 = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoLevel1Seed10 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuNoIsoL1Filtered10 = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuNoIsoLevel1Seed10"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuNoIsoL2PreFiltered11 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered10"),
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered13 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered11"),
    MinPt = cms.double(13.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuNoIso15 = cms.EDFilter("HLTPrescaler")

hltSingleMuNoIsoL2PreFiltered12 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL1Filtered10"),
    MinPt = cms.double(12.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuNoIsoL3PreFiltered15 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuNoIsoL2PreFiltered12"),
    MinPt = cms.double(15.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MaxDr = cms.double(2.0),
    CandTag = cms.InputTag("hltL3MuonCandidates")
)

hltPrescaleSingleMuIso9 = cms.EDFilter("HLTPrescaler")

hltSingleMuIsoL2PreFiltered7 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL1Filtered"),
    MinPt = cms.double(7.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuIsoL2IsoFiltered7 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL2PreFiltered7"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltSingleMuIsoL3PreFiltered9 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL2IsoFiltered7"),
    MinPt = cms.double(9.0),
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

hltSingleMuIsoL3IsoFiltered9 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL3PreFiltered9"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescaleSingleMuIso13 = cms.EDFilter("HLTPrescaler")

hltSingleMuIsoLevel1Seed10 = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleMu10'),
    L1MuonCollectionTag = cms.InputTag("hltL1extraParticles"),
    L1GtReadoutRecordTag = cms.InputTag("hltGtDigis"),
    L1CollectionsTag = cms.InputTag("hltL1extraParticles"),
    L1GtObjectMapTag = cms.InputTag("hltL1GtObjectMap"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltSingleMuIsoL1Filtered10 = cms.EDFilter("HLTMuonL1Filter",
    MaxEta = cms.double(2.5),
    CandTag = cms.InputTag("hltSingleMuIsoLevel1Seed10"),
    MinPt = cms.double(0.0),
    MinN = cms.int32(1),
    MinQuality = cms.int32(-1)
)

hltSingleMuIsoL2PreFiltered11 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL1Filtered10"),
    MinPt = cms.double(11.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuIsoL2IsoFiltered11 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL2PreFiltered11"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltSingleMuIsoL3PreFiltered13 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL2IsoFiltered11"),
    MinPt = cms.double(13.0),
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

hltSingleMuIsoL3IsoFiltered13 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL3PreFiltered13"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescaleSingleMuIso15 = cms.EDFilter("HLTPrescaler")

hltSingleMuIsoL2PreFiltered12 = cms.EDFilter("HLTMuonL2PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL1Filtered10"),
    MinPt = cms.double(12.0),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    NSigmaPt = cms.double(0.0),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    SeedTag = cms.InputTag("hltL2MuonSeeds"),
    MaxDr = cms.double(9999.0),
    CandTag = cms.InputTag("hltL2MuonCandidates")
)

hltSingleMuIsoL2IsoFiltered12 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL2PreFiltered12"),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL2MuonIsolations")
)

hltSingleMuIsoL3PreFiltered15 = cms.EDFilter("HLTMuonL3PreFilter",
    PreviousCandTag = cms.InputTag("hltSingleMuIsoL2IsoFiltered12"),
    MinPt = cms.double(15.0),
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

hltSingleMuIsoL3IsoFiltered15 = cms.EDFilter("HLTMuonIsoFilter",
    CandTag = cms.InputTag("hltSingleMuIsoL3PreFiltered15"),
    SaveTag = cms.untracked.bool(True),
    MinN = cms.int32(1),
    IsoTag = cms.InputTag("hltL3MuonIsolations")
)

hltPrescalePsi2SMM = cms.EDFilter("HLTPrescaler")

hltPsi2SMML2Filtered = cms.EDFilter("HLTMuonDimuonL2Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltJpsiMML1Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(5.6),
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
    MinInvMass = cms.double(1.6),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltPsi2SMML3Filtered = cms.EDFilter("HLTMuonDimuonL3Filter",
    MinPtBalance = cms.double(-1.0),
    MinPtMax = cms.double(3.0),
    PreviousCandTag = cms.InputTag("hltPsi2SMML2Filtered"),
    MaxPtBalance = cms.double(999999.0),
    ChargeOpt = cms.int32(0),
    MaxInvMass = cms.double(3.9),
    MaxEta = cms.double(2.5),
    MinPtMin = cms.double(3.0),
    MinNhits = cms.int32(0),
    LinksTag = cms.InputTag("hltL3Muons"),
    SaveTag = cms.untracked.bool(True),
    MaxAcop = cms.double(3.15),
    FastAccept = cms.bool(False),
    MaxDz = cms.double(9999.0),
    BeamSpotTag = cms.InputTag("hltOfflineBeamSpot"),
    MinPtPair = cms.double(0.0),
    CandTag = cms.InputTag("hltL3MuonCandidates"),
    MaxDr = cms.double(2.0),
    MinInvMass = cms.double(3.5),
    NSigmaPt = cms.double(0.0),
    MinAcop = cms.double(-1.0)
)

hltTriggerSummaryAOD = cms.EDFilter("TriggerSummaryProducerAOD",
    processName = cms.string('@')
)

hltTriggerSummaryRAWprescaler = cms.EDFilter("HLTPrescaler")

hltTriggerSummaryRAW = cms.EDFilter("TriggerSummaryProducerRAW",
    processName = cms.string('@')
)

hltBoolFinal = cms.EDFilter("HLTBool",
    result = cms.bool(False)
)

# /dev/CMSSW_2_1_0_pre4/HLT/V10  (CMSSW_2_1_X_2008-05-19-0200)
# Begin replace statements specific to the HLT
#
# End replace statements specific to the HLT
HLTBeginSequence = cms.Sequence(hlt2GetRaw+hltGtDigis+hltGctDigis+hltL1GtObjectMap+hltL1extraParticles+hltOfflineBeamSpot)
HLTEndSequence = cms.Sequence(hltBoolEnd)
HLTRecoJetRegionalSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalJetsFEDs+hltEcalRegionalJetsDigis+hltEcalRegionalJetsWeightUncalibRecHit+hltEcalRegionalJetsRecHitTmp+hltEcalRegionalJetsRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForJets+hltIterativeCone5CaloJetsRegional+hltMCJetCorJetIcone5Regional)
HLTDoLocalHcalSequence = cms.Sequence(hltHcalDigis+hltHbhereco+hltHfreco+hltHoreco)
HLTRecoJetMETSequence = cms.Sequence(HLTDoCaloSequence+HLTDoJetRecoSequence+hltMet+HLTDoHTRecoSequence)
HLTDoCaloSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalRestFEDs+hltEcalRegionalRestDigis+hltEcalRegionalRestWeightUncalibRecHit+hltEcalRegionalRestRecHitTmp+hltEcalRecHitAll+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForAll)
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
HLTL2muonisorecoSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalMuonsFEDs+hltEcalRegionalMuonsDigis+hltEcalRegionalMuonsWeightUncalibRecHit+hltEcalRegionalMuonsRecHitTmp+hltEcalRegionalMuonsRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForMuons+hltL2MuonIsolations)
HLTL3muonrecoSequence = cms.Sequence(HLTL3muonrecoNocandSequence+hltL3MuonCandidates)
HLTL3muonrecoNocandSequence = cms.Sequence(HLTL3muonTkCandidateSequence+hltL3Muons)
HLTL3muonTkCandidateSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL3TrajectorySeed+hltL3TrackCandidateFromL2)
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
HLTPixelTrackingForMinBiasSequence = cms.Sequence(hltPixelTracksForMinBias)
HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence = cms.Sequence(HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltEgammaL1MatchFilterRegionalElectronTau+hltEgammaEtFilterElectronTau+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltEgammaHcalIsolFilterElectronTau+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltElectronPixelMatchFilterElectronTau+HLTPixelMatchElectronL1IsoTrackingSequence+hltElectronOneOEMinusOneOPFilterElectronTau+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau)
HLTL2TauJetsElectronTauSequnce = cms.Sequence(HLTCaloTausCreatorRegionalSequence+hltL2TauJetsProviderElectronTau)
HLTCaloTausCreatorRegionalSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalTausFEDs+hltEcalRegionalTausDigis+hltEcalRegionalTausWeightUncalibRecHit+hltEcalRegionalTausRecHitTmp+hltEcalRegionalTausRecHit+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForTaus+hltCaloTowersTau1Regional+hltIcone5Tau1Regional+hltCaloTowersTau2Regional+hltIcone5Tau2Regional+hltCaloTowersTau3Regional+hltIcone5Tau3Regional+hltCaloTowersTau4Regional+hltIcone5Tau4Regional)
HLTCaloTausCreatorSequence = cms.Sequence(HLTDoCaloSequence+hltCaloTowersTau1+hltIcone5Tau1+hltCaloTowersTau2+hltIcone5Tau2+hltCaloTowersTau3+hltIcone5Tau3+hltCaloTowersTau4+hltIcone5Tau4)
HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence = cms.Sequence(HLTBeginSequence+hltSingleElectronEt10L1NonIsoHLTNonIsoPresc+hltL1seedRelaxedSingleEt8+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSingleElectronEt10L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSingleElectronEt10EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonHLTnonIsoIsoSingleElectronEt10HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTNonIsoSingleElectronEt10PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTnonIsoSingleElectronEt10HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter)
HLTPixelMatchStartUpElectronL1IsoTrackingSequence = cms.Sequence(hltCkfL1IsoStartUpTrackCandidates+hltCtfL1IsoStartUpWithMaterialTracks+hltPixelMatchStartUpElectronsL1Iso)
HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence = cms.Sequence(hltCkfL1NonIsoStartUpTrackCandidates+hltCtfL1NonIsoStartUpWithMaterialTracks+hltPixelMatchStartUpElectronsL1NonIso)
HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1IsoStartUpElectronsRegionalPixelSeedGenerator+hltL1IsoStartUpElectronsRegionalCkfTrackCandidates+hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial)
HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence = cms.Sequence(hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator+hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates+hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial)
HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltSingleElectronEt8L1NonIsoHLTnoIsoPresc+hltL1seedRelaxedSingleEt5+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTnoIsoSingleElectronEt8L1MatchFilterRegional+hltL1NonIsoHLTnoIsoSingleElectronEt8EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTnoIsoSingleElectronEt8HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTnoIsoSingleElectronEt8PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTnoIsoSingleElectronEt8HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter)
HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence = cms.Sequence(HLTBeginSequence+hltDoubleElectronEt5L1NonIsoHLTNonIsoPresc+hltL1seedRelaxedDoubleEt5+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoubleElectronEt5L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoubleElectronEt5EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonHLTnonIsoIsoDoubleElectronEt5HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTNonIsoDoubleElectronEt5PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTnonIsoDoubleElectronEt5HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter)
HLTSinglePhotonEt10L1NonIsolatedSequence = cms.Sequence(HLTBeginSequence+hltSinglePhotonEt10L1NonIsoPresc+hltL1seedRelaxedSingleEt8+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoSinglePhotonEt10L1MatchFilterRegional+hltL1NonIsoSinglePhotonEt10EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoSinglePhotonEt10EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoSinglePhotonEt10HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoSinglePhotonEt10TrackIsolFilter)
HLTL1SeedFilterSequence = cms.Sequence(hltL1sIsolTrack)
HLTL3PixelIsolFilterSequence = cms.Sequence(HLTDoLocalPixelSequence+hltPixelTracks+hltIsolPixelTrackProd+hltIsolPixelTrackFilter)
HLTIsoTrRegFEDSelection = cms.Sequence(hltSiStripRegFED+hltEcalRegFED+hltSubdetFED)
HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt5+PreHLT2Photon10L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoublePhotonEt10L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoublePhotonEt10EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt10EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter)
HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt5+PreHLT2Photon8L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoublePhotonEt8L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoublePhotonEt8EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt8EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt8HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter)
HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt8+PreHLT1ElectronLWEt12L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSingleElectronLWEt12L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSingleElectronLWEt12EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoSingleElectronLWEt12HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTNonIsoSingleElectronLWEt12PixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoHLTNonIsoSingleElectronLWEt12HOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter)
HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt10+PreHLT1ElectronLWEt15L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSingleElectronLWEt15L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSingleElectronLWEt15EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoSingleElectronLWEt15HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTNonIsoSingleElectronLWEt15PixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoHLTNonIsoSingleElectronLWEt15HOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter)
HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon20L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSinglePhotonEt20L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSinglePhotonEt20EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt20EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt20HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter)
HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt10+PreHLT1Photon15L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSinglePhotonEt15L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSinglePhotonEt15EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt15EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt15HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter)
HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon25L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSinglePhotonEt25L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSinglePhotonEt25EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt25EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt25HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter)
HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Electron18L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSingleElectronEt18L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSingleElectronEt18EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoSingleElectronEt18HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTNonIsoSingleElectronEt18PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTNonIsoSingleElectronEt18HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter)
HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt12+PreHLT1Electron15L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSingleElectronEt15L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSingleElectronEt15EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoSingleElectronEt15HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTNonIsoSingleElectronEt15PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTNonIsoSingleElectronEt15HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter)
HLTSingleElectronEt12L1NonIsoHLTIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt8+PreHLT1ElectronLW12L1IHIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTIsoSingleElectronEt12L1MatchFilterRegional+hltL1NonIsoHLTIsoSingleElectronEt12EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTIsoSingleElectronEt12HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTIsoSingleElectronEt12PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTIsoSingleElectronEt12HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter)
HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1ElectronLWEt18L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18PixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18HOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter)
HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt12+PreHLT1ElectronLWEt15L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15PixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15HOneOEMinusOneOPFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter)
HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon40L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSinglePhotonEt40L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSinglePhotonEt40EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt40EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt40HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter)
HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon30L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoSinglePhotonEt30L1MatchFilterRegional+hltL1NonIsoHLTNonIsoSinglePhotonEt30EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt30EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt30HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter)
HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon45L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSinglePhotonEt45L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSinglePhotonEt45EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt45EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt45HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter)
HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon30L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSinglePhotonEt30L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSinglePhotonEt30EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt30EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt30HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter)
HLTSinglePhoton25L1NonIsolatedHLTIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon25L1RHIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTIsoSinglePhotonEt25L1MatchFilterRegional+hltL1NonIsoHLTIsoSinglePhotonEt25EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt25EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt25HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter)
HLTSinglePhoton20L1NonIsolatedHLTIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon20L1RHIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTIsoSinglePhotonEt20L1MatchFilterRegional+hltL1NonIsoHLTIsoSinglePhotonEt20EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt20EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt20HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter)
HLTSinglePhoton15L1NonIsolatedHLTIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt12+PreHLT1Photon15L1RHIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTIsoSinglePhotonEt15L1MatchFilterRegional+hltL1NonIsoHLTIsoSinglePhotonEt15EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt15EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTIsoSinglePhotonEt15HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter)
HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt5+PreHLT2ElectronLWonlyPMEt8L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter)
HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt5+PreHLT2ElectronLWonlyPMEt10L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter)
HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt10+PreHLT2ElectronLWonlyPMEt12L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoLargeWindowElectronPixelSeeds+hltL1NonIsoLargeWindowElectronPixelSeeds+hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter)
HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt10+PreHLT2Photon20L1RHNonIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTNonIsoDoublePhotonEt20L1MatchFilterRegional+hltL1NonIsoHLTNonIsoDoublePhotonEt20EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt20EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt20HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter)
HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt12+PreHLT1Electron15L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSingleElectronEt15L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSingleElectronEt15EtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoHLTLooseIsoSingleElectronEt15HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltL1IsoStartUpElectronPixelSeeds+hltL1NonIsoStartUpElectronPixelSeeds+hltL1NonIsoHLTLooseIsoSingleElectronEt15PixelMatchFilter+HLTPixelMatchStartUpElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence+hltL1NonIsoHLTLooseIsoSingleElectronEt15HOneOEMinusOneOPFilter+HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence+HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence+hltL1IsoStartUpElectronTrackIsol+hltL1NonIsoStartupElectronTrackIsol+hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter)
HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedSingleEt15+PreHLT1Photon40L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoSinglePhotonEt40L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoSinglePhotonEt40EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt40EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt40HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter)
HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence = cms.Sequence(HLTBeginSequence+hltL1seedRelaxedDoubleEt10+PreHLT2Photon20L1RHLooseIso+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoHLTLooseIsoDoublePhotonEt20L1MatchFilterRegional+hltL1NonIsoHLTLooseIsoDoublePhotonEt20EtFilter+hltL1IsolatedPhotonEcalIsol+hltL1NonIsolatedPhotonEcalIsol+hltL1NonIsoHLTLooseIsoDoublePhotonEt20EcalIsolFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedPhotonHcalIsol+hltL1NonIsolatedPhotonHcalIsol+hltL1NonIsoHLTLooseIsoDoublePhotonEt20HcalIsolFilter+HLTDoLocalTrackerSequence+HLTL1IsoEgammaRegionalRecoTrackerSequence+HLTL1NonIsoEgammaRegionalRecoTrackerSequence+hltL1IsoPhotonTrackIsol+hltL1NonIsoPhotonTrackIsol+hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter)
HLTBLifetimeL25recoSequenceRelaxed = cms.Sequence(hltBLifetimeHighestEtJets+hltBLifetimeL25JetsRelaxed+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltBLifetimeL25AssociatorRelaxed+hltBLifetimeL25TagInfosRelaxed+hltBLifetimeL25BJetTagsRelaxed)
HLTBLifetimeL3recoSequenceRelaxed = cms.Sequence(hltBLifetimeL3JetsRelaxed+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+hltBLifetimeRegionalPixelSeedGeneratorRelaxed+hltBLifetimeRegionalCkfTrackCandidatesRelaxed+hltBLifetimeRegionalCtfWithMaterialTracksRelaxed+hltBLifetimeL3AssociatorRelaxed+hltBLifetimeL3TagInfosRelaxed+hltBLifetimeL3BJetTagsRelaxed)
HLTL1EplusJet30Sequence = cms.Sequence(HLTBeginSequence+hltL1seedEJet30)
HLTE3Jet30ElectronSequence = cms.Sequence(HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1IsoSingleEJet30L1MatchFilter+hltL1IsoEJetSingleEEt5Filter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1IsoEJetSingleEEt5HcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltL1IsoEJetSingleEEt5PixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltL1IsoEJetSingleEEt5EoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1IsoEJetSingleEEt5TrackIsolFilter+hltSingleElectronL1IsoPresc)
HLTriggerFirstPath = cms.Path(HLTBeginSequence+hltBoolFirst+HLTEndSequence)
HLT2jet = cms.Path(HLTBeginSequence+hltL1s2jet+hltPre2jet+HLTRecoJetRegionalSequence+hlt2jet150+HLTEndSequence)
HLT3jet = cms.Path(HLTBeginSequence+hltL1s3jet+hltPre3jet+HLTRecoJetRegionalSequence+hlt3jet85+HLTEndSequence)
HLT4jet = cms.Path(HLTBeginSequence+hltL1s4jet+hltPre4jet+HLTRecoJetMETSequence+hlt4jet60+HLTEndSequence)
HLT2jetAco = cms.Path(HLTBeginSequence+hltL1s2jetAco+hltPre2jetAco+HLTRecoJetRegionalSequence+hlt2jet125+hlt2jetAco+HLTEndSequence)
HLT1jet1METAco = cms.Path(HLTBeginSequence+hltL1s1jet1METAco+hltPre1jet1METAco+HLTRecoJetMETSequence+hlt1MET60+hlt1jet100+hlt1jet1METAco+HLTEndSequence)
HLT1jet1MET = cms.Path(HLTBeginSequence+hltL1s1jet1MET+hltPre1jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt1jet180+HLTEndSequence)
HLT2jet1MET = cms.Path(HLTBeginSequence+hltL1s2jet1MET+hltPre2jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt2jet125New+HLTEndSequence)
HLT3jet1MET = cms.Path(HLTBeginSequence+hltL1s3jet1MET+hltPre3jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt3jet60+HLTEndSequence)
HLT4jet1MET = cms.Path(HLTBeginSequence+hltL1s4jet1MET+hltPre4jet1MET+HLTRecoJetMETSequence+hlt1MET60+hlt4jet35+HLTEndSequence)
HLT1MET1HT = cms.Path(HLTBeginSequence+hltL1s1MET1HT+hltPre1MET1HT+HLTRecoJetMETSequence+hlt1MET65+hlt1HT350+HLTEndSequence)
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
CandHLT2ElectronRelaxedStartup = cms.Path(HLTBeginSequence+hltL1seedRelaxedDouble+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltL1NonIsoLargeWindowDoubleElectronL1MatchFilterRegional+hltL1NonIsoLargeWindowDoubleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltL1NonIsoLargeWindowDoubleElectronHcalIsolFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoLargeWindowSequence+HLTPixelMatchElectronL1NonIsoLargeWindowSequence+hltL1NonIsoLargeWindowDoubleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoLargeWindowTrackingSequence+HLTPixelMatchElectronL1NonIsoLargeWindowTrackingSequence+hltL1NonIsoLargeWindowDoubleElectronEoverpFilter+HLTL1IsoLargeWindowElectronsRegionalRecoTrackerSequence+HLTL1NonIsoLargeWindowElectronsRegionalRecoTrackerSequence+hltL1IsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowElectronTrackIsol+hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter+hltDoubleElectronL1NonIsoLargeWindowPresc+HLTEndSequence)
HLT1MuonIso = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuIso+hltSingleMuIsoLevel1Seed+hltSingleMuIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuIsoL2PreFiltered+HLTL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered+HLTL3muonrecoSequence+hltSingleMuIsoL3PreFiltered+HLTL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered+HLTEndSequence)
HLT1MuonNonIso = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIso+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonIso = cms.Path(HLTL1muonrecoSequence+hltPrescaleDiMuonIso+hltDiMuonIsoLevel1Seed+hltDiMuonIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonIsoL2PreFiltered+HLTL2muonisorecoSequence+hltDiMuonIsoL2IsoFiltered+HLTL3muonrecoSequence+hltDiMuonIsoL3PreFiltered+HLTL3muonisorecoSequence+hltDiMuonIsoL3IsoFiltered+HLTEndSequence)
HLT2MuonNonIso = cms.Path(HLTL1muonrecoSequence+hltPrescaleDiMuonNoIso+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonJPsi = cms.Path(HLTL1muonrecoSequence+hltPrescaleJPsiMM+hltJpsiMMLevel1Seed+hltJpsiMML1Filtered+HLTL2muonrecoSequence+hltJpsiMML2Filtered+HLTL3muonrecoSequence+hltJpsiMML3Filtered+HLTEndSequence)
HLT2MuonUpsilon = cms.Path(HLTL1muonrecoSequence+hltPrescaleUpsilonMM+hltUpsilonMMLevel1Seed+hltUpsilonMML1Filtered+HLTL2muonrecoSequence+hltUpsilonMML2Filtered+HLTL3muonrecoSequence+hltUpsilonMML3Filtered+HLTEndSequence)
HLT2MuonZ = cms.Path(HLTL1muonrecoSequence+hltPrescaleZMM+hltZMMLevel1Seed+hltZMML1Filtered+HLTL2muonrecoSequence+hltZMML2Filtered+HLTL3muonrecoSequence+hltZMML3Filtered+HLTEndSequence)
HLTNMuonNonIso = cms.Path(HLTL1muonrecoSequence+hltPrescaleMultiMuonNoIso+hltMultiMuonNoIsoLevel1Seed+hltMultiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltMultiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltMultiMuonNoIsoL3PreFiltered+HLTEndSequence)
HLT2MuonSameSign = cms.Path(HLTL1muonrecoSequence+hltPrescaleSameSignMu+hltSameSignMuLevel1Seed+hltSameSignMuL1Filtered+HLTL2muonrecoSequence+hltSameSignMuL2PreFiltered+HLTL3muonrecoSequence+hltSameSignMuL3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt3 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuPrescale3+hltSingleMuPrescale3Level1Seed+hltSingleMuPrescale3L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale3L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale3L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt5 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuPrescale5+hltSingleMuPrescale5Level1Seed+hltSingleMuPrescale5L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale5L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale5L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt7x7 = cms.Path(HLTL1muonrecoSequence+hltPreSingleMuPrescale77+hltSingleMuPrescale77Level1Seed+hltSingleMuPrescale77L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale77L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale77L3PreFiltered+HLTEndSequence)
HLT1MuonPrescalePt7x10 = cms.Path(HLTL1muonrecoSequence+hltPreSingleMuPrescale710+hltSingleMuPrescale710Level1Seed+hltSingleMuPrescale710L1Filtered+HLTL2muonrecoSequence+hltSingleMuPrescale710L2PreFiltered+HLTL3muonrecoSequence+hltSingleMuPrescale710L3PreFiltered+HLTEndSequence)
HLT1MuonLevel1 = cms.Path(HLTL1muonrecoSequence+hltPrescaleMuLevel1Path+hltMuLevel1PathLevel1Seed+hltMuLevel1PathL1Filtered+HLTEndSequence)
CandHLT1MuonPrescaleVtx2cm = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIsoRelaxedVtx2cm+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm+HLTEndSequence)
CandHLT1MuonPrescaleVtx2mm = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIsoRelaxedVtx2mm+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm+HLTEndSequence)
CandHLT2MuonPrescaleVtx2cm = cms.Path(HLTL1muonrecoSequence+hltPrescaleDiMuonNoIsoRelaxedVtx2cm+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm+HLTEndSequence)
CandHLT2MuonPrescaleVtx2mm = cms.Path(HLTL1muonrecoSequence+hltPrescaleDiMuonNoIsoRelaxedVtx2mm+hltDiMuonNoIsoLevel1Seed+hltDiMuonNoIsoL1Filtered+HLTL2muonrecoSequence+hltDiMuonNoIsoL2PreFiltered+HLTL3muonrecoSequence+hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm+HLTEndSequence)
HLTB1Jet = cms.Path(HLTBeginSequence+hltPrescalerBLifetime1jet+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime1jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB2Jet = cms.Path(HLTBeginSequence+hltPrescalerBLifetime2jet+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime2jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB3Jet = cms.Path(HLTBeginSequence+hltPrescalerBLifetime3jet+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime3jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB4Jet = cms.Path(HLTBeginSequence+hltPrescalerBLifetime4jet+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime4jetL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTBHT = cms.Path(HLTBeginSequence+hltPrescalerBLifetimeHT+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetimeHTL2filter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTB1JetMu = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon1jet+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon1jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonByDRL3filter+HLTEndSequence)
HLTB2JetMu = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon2jet+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon2jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTB3JetMu = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon3jet+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon3jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTB4JetMu = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon4jet+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon4jetL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTBHTMu = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuonHT+hltBSoftmuonHTL1seeds+HLTBCommonL2recoSequence+hltBSoftmuonHTL2filter+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTBJPsiMuMu = cms.Path(HLTBeginSequence+hltJpsitoMumuL1Seed+hltJpsitoMumuL1Filtered+HLTL2muonrecoSequence+HLTL3displacedMumurecoSequence+hltDisplacedJpsitoMumuFilter+HLTEndSequence)
HLTTauTo3Mu = cms.Path(HLTBeginSequence+hltMuMukL1Seed+hltMuMukL1Filtered+HLTL2muonrecoSequence+HLTL3displacedMumurecoSequence+hltDisplacedMuMukFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTRecopixelvertexingSequence+hltMumukPixelSeedFromL2Candidate+hltCkfTrackCandidatesMumuk+hltCtfWithMaterialTracksMumuk+hltMumukAllConeTracks+hltmmkFilter+HLTEndSequence)
HLTXElectronBJet = cms.Path(HLTBeginSequence+hltElectronBPrescale+hltElectronBL1Seed+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltElBElectronL1MatchFilter+hltElBElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltElBElectronHcalIsolFilter+HLTBCommonL2recoSequence+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1NonIsoSequence+HLTPixelMatchElectronL1IsoSequence+hltElBElectronPixelMatchFilter+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltElBElectronEoverpFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltElBElectronTrackIsolFilter+HLTEndSequence)
HLTXMuonBJet = cms.Path(HLTBeginSequence+hltMuBPrescale+hltMuBLevel1Seed+hltMuBLifetimeL1Filtered+HLTL2muonrecoSequence+hltMuBLifetimeIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuBLifetimeIsoL2IsoFiltered+HLTBCommonL2recoSequence+HLTBLifetimeL25recoSequence+hltBLifetimeL25filter+HLTL3muonrecoSequence+hltMuBLifetimeIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuBLifetimeIsoL3IsoFiltered+HLTBLifetimeL3recoSequence+hltBLifetimeL3filter+HLTEndSequence)
HLTXMuonBJetSoftMuon = cms.Path(HLTBeginSequence+hltMuBsoftMuPrescale+hltMuBLevel1Seed+hltMuBSoftL1Filtered+HLTL2muonrecoSequence+hltMuBSoftIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuBSoftIsoL2IsoFiltered+HLTBCommonL2recoSequence+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTL3muonrecoSequence+hltMuBSoftIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuBSoftIsoL3IsoFiltered+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filter+HLTEndSequence)
HLTXElectron1Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej1jet40+HLTEndSequence)
HLTXElectron2Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej2jet80+HLTEndSequence)
HLTXElectron3Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej3jet60+HLTEndSequence)
HLTXElectron4Jet = cms.Path(HLTL1EplusJetSequence+HLTEJetElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej4jet35+HLTEndSequence)
HLTXMuonJets = cms.Path(HLTL1muonrecoSequence+hltMuJetsPrescale+hltMuJetsLevel1Seed+hltMuJetsL1Filtered+HLTL2muonrecoSequence+hltMuJetsL2PreFiltered+HLTL2muonisorecoSequence+hltMuJetsL2IsoFiltered+HLTL3muonrecoSequence+hltMuJetsL3PreFiltered+HLTL3muonisorecoSequence+hltMuJetsL3IsoFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuJetsHLT1jet40+HLTEndSequence)
CandHLTXMuonNoL2IsoJets = cms.Path(HLTL1muonrecoSequence+hltMuNoL2IsoJetsPrescale+hltMuNoL2IsoJetsLevel1Seed+hltMuNoL2IsoJetsL1Filtered+HLTL2muonrecoSequence+hltMuNoL2IsoJetsL2PreFiltered+HLTL3muonrecoSequence+hltMuNoL2IsoJetsL3PreFiltered+HLTL3muonisorecoSequence+hltMuNoL2IsoJetsL3IsoFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuNoL2IsoJetsHLT1jet40+HLTEndSequence)
CandHLTXMuonNoIsoJets = cms.Path(HLTL1muonrecoSequence+hltMuNoIsoJetsPrescale+hltMuNoIsoJetsLevel1Seed+hltMuNoIsoJetsL1Filtered+HLTL2muonrecoSequence+hltMuNoIsoJetsL2PreFiltered+HLTL3muonrecoSequence+hltMuNoIsoJetsL3PreFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuNoIsoJetsHLT1jet50+HLTEndSequence)
HLTXElectronMuon = cms.Path(HLTBeginSequence+hltemuPrescale+hltEMuonLevel1Seed+hltEMuL1MuonFilter+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltemuL1IsoSingleL1MatchFilter+hltemuL1IsoSingleElectronEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltemuL1IsoSingleElectronHcalIsolFilter+HLTL2muonrecoSequence+hltEMuL2MuonPreFilter+HLTL2muonisorecoSequence+hltEMuL2MuonIsoFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+hltemuL1IsoSingleElectronPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+hltemuL1IsoSingleElectronEoverpFilter+HLTL3muonrecoSequence+hltEMuL3MuonPreFilter+HLTL3muonisorecoSequence+hltEMuL3MuonIsoFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltemuL1IsoSingleElectronTrackIsolFilter+HLTEndSequence)
HLTXElectronMuonRelaxed = cms.Path(HLTBeginSequence+hltemuNonIsoPrescale+hltemuNonIsoLevel1Seed+hltNonIsoEMuL1MuonFilter+HLTDoRegionalEgammaEcalSequence+HLTL1IsolatedEcalClustersSequence+HLTL1NonIsolatedEcalClustersSequence+hltL1IsoRecoEcalCandidate+hltL1NonIsoRecoEcalCandidate+hltemuNonIsoL1MatchFilterRegional+hltemuNonIsoL1IsoEtFilter+HLTDoLocalHcalWithoutHOSequence+hltL1IsolatedElectronHcalIsol+hltL1NonIsolatedElectronHcalIsol+hltemuNonIsoL1HcalIsolFilter+HLTL2muonrecoSequence+hltNonIsoEMuL2MuonPreFilter+HLTDoLocalPixelSequence+HLTDoLocalStripSequence+HLTPixelMatchElectronL1IsoSequence+HLTPixelMatchElectronL1NonIsoSequence+hltemuNonIsoL1IsoPixelMatchFilter+HLTPixelMatchElectronL1IsoTrackingSequence+HLTPixelMatchElectronL1NonIsoTrackingSequence+hltemuNonIsoL1IsoEoverpFilter+HLTL3muonrecoSequence+hltNonIsoEMuL3MuonPreFilter+HLTL1IsoElectronsRegionalRecoTrackerSequence+HLTL1NonIsoElectronsRegionalRecoTrackerSequence+hltL1IsoElectronTrackIsol+hltL1NonIsoElectronTrackIsol+hltemuNonIsoL1IsoTrackIsolFilter+HLTEndSequence)
CandHLTBackwardBSC = cms.Path(HLTBeginSequence+hltLevel1seedHLTBackwardBSC+hltPrescaleHLTBackwardBSC+HLTEndSequence)
CandHLTForwardBSC = cms.Path(HLTBeginSequence+hltLevel1seedHLTForwardBSC+hltPrescaleHLTForwardBSC+HLTEndSequence)
CandHLTCSCBeamHalo = cms.Path(HLTBeginSequence+hltLevel1seedHLTCSCBeamHalo+hltPrescaleHLTCSCBeamHalo+HLTEndSequence)
CandHLTCSCBeamHaloOverlapRing1 = cms.Path(HLTBeginSequence+hltLevel1seedHLTCSCBeamHaloOverlapRing1+hltPrescaleHLTCSCBeamHaloOverlapRing1+hltMuonCSCDigis+hltCsc2DRecHits+hltOverlapsHLTCSCBeamHaloOverlapRing1+HLTEndSequence)
CandHLTCSCBeamHaloOverlapRing2 = cms.Path(HLTBeginSequence+hltLevel1seedHLTCSCBeamHaloOverlapRing2+hltPrescaleHLTCSCBeamHaloOverlapRing2+hltMuonCSCDigis+hltCsc2DRecHits+hltOverlapsHLTCSCBeamHaloOverlapRing2+HLTEndSequence)
CandHLTCSCBeamHaloRing2or3 = cms.Path(HLTBeginSequence+hltLevel1seedHLTCSCBeamHaloRing2or3+hltPrescaleHLTCSCBeamHaloRing2or3+hltMuonCSCDigis+hltCsc2DRecHits+hltFilter23HLTCSCBeamHaloRing2or3+HLTEndSequence)
CandHLTTrackerCosmics = cms.Path(HLTBeginSequence+hltLevel1seedHLTTrackerCosmics+hltPrescaleHLTTrackerCosmics+HLTEndSequence)
HLTMinBiasPixel = cms.Path(HLTBeginSequence+hltPreMinBiasPixel+hltL1seedMinBiasPixel+HLTDoLocalPixelSequence+HLTPixelTrackingForMinBiasSequence+hltPixelCands+hltMinBiasPixelFilter+HLTEndSequence)
CandHLTMinBiasForAlignment = cms.Path(HLTBeginSequence+hltPreMBForAlignment+hltL1seedMinBiasPixel+HLTDoLocalPixelSequence+HLTPixelTrackingForMinBiasSequence+hltPixelCands+hltPixelMBForAlignment+HLTEndSequence)
HLTMinBias = cms.Path(HLTBeginSequence+hltl1sMin+hltpreMin+HLTEndSequence)
HLTZeroBias = cms.Path(HLTBeginSequence+hltl1sZero+hltpreZero+HLTEndSequence)
HLTriggerType = cms.Path(HLTBeginSequence+hltPrescaleTriggerType+hltFilterTriggerType+HLTEndSequence)
HLTEndpath1 = cms.EndPath(hltL1gtTrigReport+hltTrigReport)
HLTXElectronTau = cms.Path(HLTBeginSequence+hltPrescalerElectronTau+hltLevel1GTSeedElectronTau+HLTETauSingleElectronL1IsolatedHOneOEMinusOneOPFilterSequence+HLTL2TauJetsElectronTauSequnce+hltL2ElectronTauIsolationProducer+hltL2ElectronTauIsolationSelector+hltFilterEcalIsolatedTauJetsElectronTau+HLTRecopixelvertexingSequence+hltJetTracksAssociatorAtVertexL25ElectronTau+hltConeIsolationL25ElectronTau+hltIsolatedTauJetsSelectorL25ElectronTau+hltFilterIsolatedTauJetsL25ElectronTau+HLTEndSequence)
HLTXMuonTau = cms.Path(HLTBeginSequence+hltPrescalerMuonTau+hltLevel1GTSeedMuonTau+hltMuonTauL1Filtered+HLTL2muonrecoSequence+hltMuonTauIsoL2PreFiltered+HLTL2muonisorecoSequence+hltMuonTauIsoL2IsoFiltered+HLTCaloTausCreatorRegionalSequence+hltL2TauJetsProviderMuonTau+hltL2MuonTauIsolationProducer+hltL2MuonTauIsolationSelector+hltFilterEcalIsolatedTauJetsMuonTau+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltJetsPixelTracksAssociatorMuonTau+hltPixelTrackConeIsolationMuonTau+hltPixelTrackIsolatedTauJetsSelectorMuonTau+hltFilterPixelTrackIsolatedTauJetsMuonTau+HLTDoLocalStripSequence+HLTL3muonrecoSequence+hltMuonTauIsoL3PreFiltered+HLTL3muonisorecoSequence+hltMuonTauIsoL3IsoFiltered+HLTEndSequence)
HLT1Tau1MET = cms.Path(HLTBeginSequence+hltSingleTauMETPrescaler+hltSingleTauMETL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTauMET+hltL2SingleTauMETJets+hltL2SingleTauMETIsolationProducer+hltL2SingleTauMETIsolationSelector+hltFilterSingleTauMETEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTauMET+hltConeIsolationL25SingleTauMET+hltIsolatedL25SingleTauMET+hltFilterL25SingleTauMET+HLTDoLocalStripSequence+hltL3SingleTauMETPixelSeeds+hltCkfTrackCandidatesL3SingleTauMET+hltCtfWithMaterialTracksL3SingleTauMET+hltAssociatorL3SingleTauMET+hltConeIsolationL3SingleTauMET+hltIsolatedL3SingleTauMET+hltFilterL3SingleTauMET+HLTEndSequence)
HLT1Tau = cms.Path(HLTBeginSequence+hltSingleTauPrescaler+hltSingleTauL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTau+hltL2SingleTauJets+hltL2SingleTauIsolationProducer+hltL2SingleTauIsolationSelector+hltFilterSingleTauEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTau+hltConeIsolationL25SingleTau+hltIsolatedL25SingleTau+hltFilterL25SingleTau+HLTDoLocalStripSequence+hltL3SingleTauPixelSeeds+hltCkfTrackCandidatesL3SingleTau+hltCtfWithMaterialTracksL3SingleTau+hltAssociatorL3SingleTau+hltConeIsolationL3SingleTau+hltIsolatedL3SingleTau+hltFilterL3SingleTau+HLTEndSequence)
HLT1Electron10_L1R_NI = cms.Path(HLTSingleElectronEt10L1NonIsoHLTnonIsoSequence+HLTEndSequence)
HLT1Electron8_L1R_NI = cms.Path(HLTSingleElectronEt8L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT2Electron5_L1R_NI = cms.Path(HLTDoubleElectronEt5L1NonIsoHLTnonIsoSequence+HLTEndSequence)
HLT1Photon10_L1R = cms.Path(HLTSinglePhotonEt10L1NonIsolatedSequence+HLTEndSequence)
AlCaIsoTrack = cms.Path(HLTBeginSequence+HLTL1SeedFilterSequence+hltPreIsolTrackNoEcalIso+HLTL3PixelIsolFilterSequence+HLTIsoTrRegFEDSelection+HLTEndSequence)
AlCaHcalPhiSym = cms.Path(HLTBeginSequence+hltL1sHcalPhiSym+hltHcalPhiSymPresc+HLTDoLocalHcalSequence+hltAlCaHcalPhiSymStream+HLTEndSequence)
AlCaEcalPhiSym = cms.Path(HLTBeginSequence+hltL1sEcalPhiSym+hltEcalPhiSymPresc+hltEcalDigis+hltEcalWeightUncalibRecHit+hltEcalRecHit+hltAlCaPhiSymStream+HLTEndSequence)
AlCaEcalPi0 = cms.Path(HLTBeginSequence+hltPrePi0Ecal+hltL1sEcalPi0+HLTDoRegionalEgammaEcalSequence+hltAlCaPi0RegRecHits+HLTEndSequence)
HLT1MuonLevel2 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuLevel2+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuLevel2NoIsoL2PreFiltered+HLTEndSequence)
HLTBJPsiMuMuRelaxed = cms.Path(HLTBeginSequence+hltJpsitoMumuL1SeedRelaxed+hltJpsitoMumuL1FilteredRelaxed+HLTL2muonrecoSequence+HLTL3displacedMumurecoSequence+hltDisplacedJpsitoMumuFilterRelaxed+HLTEndSequence)
HLT2PhotonEt10_L1R_NI = cms.Path(HLTDoublePhoton10L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT2PhotonEt8_L1R_NI = cms.Path(HLTDoublePhoton8L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT1ElectronLWEt12_L1R_NI = cms.Path(HLTSingleElectronLWEt12L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT1ElectronLWEt15_L1R_NI = cms.Path(HLTSingleElectronLWEt15L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT1PhotonEt20_L1R_LI = cms.Path(HLTSinglePhoton20L1NonIsolatedHLTLooseIsoSequence+HLTEndSequence)
HLT1PhotonEt15_L1R_NI = cms.Path(HLTSinglePhoton15L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT1PhotonEt25_L1R_NI = cms.Path(HLTSinglePhoton25L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT1ElectronEt18_L1R_NI = cms.Path(HLTSingleElectronEt18L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT1ElectronEt15_L1R_NI = cms.Path(HLTSingleElectronEt15L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT1ElectronEt12_L1R_HI = cms.Path(HLTSingleElectronEt12L1NonIsoHLTIsoSequence+HLTEndSequence)
HLT1ElectronLWEt18_L1R_LI = cms.Path(HLTSingleElectronLWEt18L1NonIsoHLTLooseIsoSequence+HLTEndSequence)
HLT1ElectronLWEt15_L1R_LI = cms.Path(HLTSingleElectronLWEt15L1NonIsoHLTLooseIsoSequence+HLTEndSequence)
HLT1PhotonEt40_L1R_NI = cms.Path(HLTSinglePhoton40L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT1PhotonEt30_L1R_NI = cms.Path(HLTSinglePhoton30L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLT1PhotonEt45_L1R_LI = cms.Path(HLTSinglePhoton45L1NonIsolatedHLTLooseIsoSequence+HLTEndSequence)
HLT1PhotonEt30_L1R_LI = cms.Path(HLTSinglePhoton30L1NonIsolatedHLTLooseIsoSequence+HLTEndSequence)
HLT1PhotonEt25_L1R_HI = cms.Path(HLTSinglePhoton25L1NonIsolatedHLTIsoSequence+HLTEndSequence)
HLT1PhotonEt20_L1R_HI = cms.Path(HLTSinglePhoton20L1NonIsolatedHLTIsoSequence+HLTEndSequence)
HLT1PhotonEt15_L1R_HI = cms.Path(HLTSinglePhoton15L1NonIsolatedHLTIsoSequence+HLTEndSequence)
HLT2ElectronLWonlyPMEt8_L1R_NI = cms.Path(HLTDoubleElectronLWonlyPMEt8L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT2ElectronLWonlyPMEt10_L1R_NI = cms.Path(HLTDoubleElectronLWonlyPMEt10L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT2ElectronLWonlyPMEt12_L1R_NI = cms.Path(HLTDoubleElectronLWonlyPMEt12L1NonIsoHLTNonIsoSequence+HLTEndSequence)
HLT2PhotonEt20_L1R_NI = cms.Path(HLTDoublePhoton20L1NonIsolatedHLTNonIsoSequence+HLTEndSequence)
HLTMinBiasHcal = cms.Path(HLTBeginSequence+hltl1sMinHcal+hltpreMinHcal+HLTEndSequence)
HLTMinBiasEcal = cms.Path(HLTBeginSequence+hltl1sMinEcal+hltpreMinEcal+HLTEndSequence)
HLT1ElectronEt15_L1R_LI = cms.Path(HLTSingleElectronEt15L1NonIsoHLTLooseIsoSequence+HLTEndSequence)
HLT1PhotonEt40_L1R_LI = cms.Path(HLTSinglePhoton40L1NonIsolatedHLTLooseIsoSequence+HLTEndSequence)
HLT2PhotonEt20_L1R_LI = cms.Path(HLTDoublePhoton20L1NonIsolatedHLTLooseIsoSequence+HLTEndSequence)
HLT4jet30 = cms.Path(HLTBeginSequence+hltL1s4jet30+hltPre4jet30+HLTRecoJetMETSequence+hlt4jet30+HLTEndSequence)
HLT1TauRelaxed = cms.Path(HLTBeginSequence+hltSingleTauPrescaler+hltSingleTauL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTauRelaxed+hltL2SingleTauJets+hltL2SingleTauIsolationProducer+hltL2SingleTauIsolationSelectorRelaxed+hltFilterSingleTauEcalIsolationRelaxed+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTauRelaxed+hltConeIsolationL25SingleTauRelaxed+hltIsolatedL25SingleTauRelaxed+hltFilterL25SingleTauRelaxed+HLTDoLocalStripSequence+hltL3SingleTauPixelSeedsRelaxed+hltCkfTrackCandidatesL3SingleTauRelaxed+hltCtfWithMaterialTracksL3SingleTauRelaxed+hltAssociatorL3SingleTauRelaxed+hltConeIsolationL3SingleTauRelaxed+hltIsolatedL3SingleTauRelaxed+hltFilterL3SingleTauRelaxed+HLTEndSequence)
HLT1Tau1METRelaxed = cms.Path(HLTBeginSequence+hltSingleTauMETPrescaler+hltSingleTauMETL1SeedFilter+HLTCaloTausCreatorSequence+hltMet+hlt1METSingleTauMETRelaxed+hltL2SingleTauMETJets+hltL2SingleTauMETIsolationProducer+hltL2SingleTauMETIsolationSelectorRelaxed+hltFilterSingleTauMETEcalIsolationRelaxed+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25SingleTauMETRelaxed+hltConeIsolationL25SingleTauMETRelaxed+hltIsolatedL25SingleTauMETRelaxed+hltFilterL25SingleTauMETRelaxed+HLTDoLocalStripSequence+hltL3SingleTauMETPixelSeedsRelaxed+hltCkfTrackCandidatesL3SingleTauMETRelaxed+hltCtfWithMaterialTracksL3SingleTauMETRelaxed+hltAssociatorL3SingleTauMETRelaxed+hltConeIsolationL3SingleTauMETRelaxed+hltIsolatedL3SingleTauMETRelaxed+hltFilterL3SingleTauMETRelaxed+HLTEndSequence)
HLT2TauPixelRelaxed = cms.Path(HLTBeginSequence+hltDoubleTauPrescaler+hltDoubleTauL1SeedFilterRelaxed+HLTCaloTausCreatorRegionalSequence+hltL2DoubleTauJetsRelaxed+hltL2DoubleTauIsolationProducerRelaxed+hltL2DoubleTauIsolationSelectorRelaxed+hltFilterDoubleTauEcalIsolationRelaxed+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25PixelTauIsolatedRelaxed+hltConeIsolationL25PixelTauIsolatedRelaxed+hltIsolatedL25PixelTauRelaxed+hltFilterL25PixelTauRelaxed+HLTEndSequence)
HLT2TauPixel = cms.Path(HLTBeginSequence+hltDoubleTauPrescaler+hltDoubleTauL1SeedFilter+HLTCaloTausCreatorRegionalSequence+hltL2DoubleTauJets+hltL2DoubleTauIsolationProducer+hltL2DoubleTauIsolationSelector+hltFilterDoubleTauEcalIsolation+HLTDoLocalPixelSequence+HLTRecopixelvertexingSequence+hltAssociatorL25PixelTauIsolated+hltConeIsolationL25PixelTauIsolated+hltIsolatedL25PixelTau+hltFilterL25PixelTau+HLTEndSequence)
HLT1Level1jet15 = cms.Path(HLTBeginSequence+hltPre1Level1jet15+hltL1s1Level1jet15+HLTEndSequence)
HLT1jet30 = cms.Path(HLTBeginSequence+hltL1s1jet30+hltPre1jet30+HLTRecoJetMETSequence+hlt1jet30+HLTEndSequence)
HLT1jet50 = cms.Path(HLTBeginSequence+hltL1s1jet50+hltPre1jet50+HLTRecoJetMETSequence+hlt1jet50+HLTEndSequence)
HLT1jet80 = cms.Path(HLTBeginSequence+hltL1s1jet80+hltPre1jet80+HLTRecoJetRegionalSequence+hlt1jet80+HLTEndSequence)
HLT1jet110 = cms.Path(HLTBeginSequence+hltL1s1jet110+hltPre1jet110+HLTRecoJetRegionalSequence+hlt1jet110+HLTEndSequence)
HLT1jet250 = cms.Path(HLTBeginSequence+hltL1s1jet250+hltPre1jet250+HLTRecoJetRegionalSequence+hlt1jet250+HLTEndSequence)
HLT1SumET = cms.Path(HLTBeginSequence+hltL1s1SumET+hltPre1MET1SumET+HLTRecoJetMETSequence+hlt1SumET120+HLTEndSequence)
HLT1jet180 = cms.Path(HLTBeginSequence+hltL1s1jet180+hltPre1jet180+HLTRecoJetRegionalSequence+hlt1jet180regional+HLTEndSequence)
HLT1Level1MET20 = cms.Path(HLTBeginSequence+hltPreLevel1MET20+hltL1sLevel1MET20+HLTEndSequence)
HLT1MET25 = cms.Path(HLTBeginSequence+hltL1s1MET25+hltPre1MET25+HLTRecoJetMETSequence+hlt1MET25+HLTEndSequence)
HLT1MET35 = cms.Path(HLTBeginSequence+hltL1s1MET35+hltPre1MET35+HLTRecoJetMETSequence+hlt1MET35+HLTEndSequence)
HLT1MET50 = cms.Path(HLTBeginSequence+hltL1s1MET50+hltPre1MET50+HLTRecoJetMETSequence+hlt1MET50+HLTEndSequence)
HLT1MET65 = cms.Path(HLTBeginSequence+hltL1s1MET65+hltPre1MET65+HLTRecoJetMETSequence+hlt1MET65+HLTEndSequence)
HLT1MET75 = cms.Path(HLTBeginSequence+hltL1s1MET75+hltPre1MET75+HLTRecoJetMETSequence+hlt1MET75+HLTEndSequence)
HLT2jetAve15 = cms.Path(HLTBeginSequence+hltL1sdijetave15+hltPredijetave15+HLTRecoJetMETSequence+hltdijetave15+HLTEndSequence)
HLT2jetAve30 = cms.Path(HLTBeginSequence+hltL1sdijetave30+hltPredijetave30+HLTRecoJetMETSequence+hltdijetave30+HLTEndSequence)
HLT2jetAve50 = cms.Path(HLTBeginSequence+hltL1sdijetave50+hltPredijetave50+HLTRecoJetMETSequence+hltdijetave50+HLTEndSequence)
HLT2jetAve70 = cms.Path(HLTBeginSequence+hltL1sdijetave70+hltPredijetave70+HLTRecoJetMETSequence+hltdijetave70+HLTEndSequence)
HLT2jetAve130 = cms.Path(HLTBeginSequence+hltL1sdijetave130+hltPredijetave130+HLTRecoJetMETSequence+hltdijetave130+HLTEndSequence)
HLT2jetAve220 = cms.Path(HLTBeginSequence+hltL1sdijetave220+hltPredijetave220+HLTRecoJetMETSequence+hltdijetave220+HLTEndSequence)
HLTB1Jet120 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime1jet120+hltBLifetimeL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBLifetime1jetL2filter120+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB1Jet160 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime1jet160+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime1jetL2filter160+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB2Jet100 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime2jet100+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime2jetL2filter100+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB2Jet60 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime2jet60+hltBLifetimeL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBLifetime2jetL2filter60+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB2JetMu100 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon2jet100+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon2jetL2filter100+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTB2JetMu60 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon2jet60+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon2jetL2filter60+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTB3Jet40 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime3jet40+hltBLifetimeL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBLifetime3jetL2filter40+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB3Jet60 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime3jet60+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime3jetL2filter60+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB3JetMu40 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon3jet40+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon3jetL2filter40+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTB3JetMu60 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon3jet60+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon3jetL2filter60+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTB4Jet30 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime4jet30+hltBLifetimeL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBLifetime4jetL2filter30+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB4Jet35 = cms.Path(HLTBeginSequence+hltPrescalerBLifetime4jet35+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetime4jetL2filter35+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTB4JetMu30 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon4jet30+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon4jetL2filter30+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTB4JetMu35 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuon4jet35+hltBSoftmuonNjetL1seeds+HLTBCommonL2recoSequence+hltBSoftmuon4jetL2filter35+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTBHT320 = cms.Path(HLTBeginSequence+hltPrescalerBLifetimeHT320+hltBLifetimeL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBLifetimeHTL2filter320+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTBHT420 = cms.Path(HLTBeginSequence+hltPrescalerBLifetimeHT420+hltBLifetimeL1seeds+HLTBCommonL2recoSequence+hltBLifetimeHTL2filter420+HLTBLifetimeL25recoSequenceRelaxed+hltBLifetimeL25filterRelaxed+HLTBLifetimeL3recoSequenceRelaxed+hltBLifetimeL3filterRelaxed+HLTEndSequence)
HLTBHTMu250 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuonHT250+hltBSoftmuonHTL1seedsLowEnergy+HLTBCommonL2recoSequence+hltBSoftmuonHTL2filter250+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTBHTMu330 = cms.Path(HLTBeginSequence+hltPrescalerBSoftmuonHT330+hltBSoftmuonHTL1seeds+HLTBCommonL2recoSequence+hltBSoftmuonHTL2filter330+HLTBSoftmuonL25recoSequence+hltBSoftmuonL25filter+HLTBSoftmuonL3recoSequence+hltBSoftmuonL3filterRelaxed+HLTEndSequence)
HLTXElectron3Jet30 = cms.Path(HLTL1EplusJet30Sequence+HLTE3Jet30ElectronSequence+HLTDoCaloSequence+HLTDoJetRecoSequence+hltej3jet30+HLTEndSequence)
HLTXMuonNoIso3Jets30 = cms.Path(HLTL1muonrecoSequence+hltMuNoIsoJetsPrescale+hltMuNoIsoJets30Level1Seed+hltMuNoIsoJetsMinPt4L1Filtered+HLTL2muonrecoSequence+hltMuNoIsoJetsMinPt4L2PreFiltered+HLTL3muonrecoSequence+hltMuNoIsoJetsMinPtL3PreFiltered+HLTDoCaloSequence+HLTDoJetRecoSequence+hltMuNoIsoHLTJets3jet30+HLTEndSequence)
HLT1MuonL1Open = cms.Path(HLTL1muonrecoSequence+hltPrescaleMuLevel1Open+hltMuLevel1PathLevel1OpenSeed+hltMuLevel1PathL1OpenFiltered+HLTEndSequence)
HLT1MuonNonIso9 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIso9+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered7+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered9+HLTEndSequence)
HLT1MuonNonIso11 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIso11+hltSingleMuNoIsoLevel1Seed+hltSingleMuNoIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered9+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered11+HLTEndSequence)
HLT1MuonNonIso13 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIso13+hltSingleMuNoIsoLevel1Seed10+hltSingleMuNoIsoL1Filtered10+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered11+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered13+HLTEndSequence)
HLT1MuonNonIso15 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuNoIso15+hltSingleMuNoIsoLevel1Seed10+hltSingleMuNoIsoL1Filtered10+HLTL2muonrecoSequence+hltSingleMuNoIsoL2PreFiltered12+HLTL3muonrecoSequence+hltSingleMuNoIsoL3PreFiltered15+HLTEndSequence)
HLT1MuonIso9 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuIso9+hltSingleMuIsoLevel1Seed+hltSingleMuIsoL1Filtered+HLTL2muonrecoSequence+hltSingleMuIsoL2PreFiltered7+HLTL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered7+HLTL3muonrecoSequence+hltSingleMuIsoL3PreFiltered9+HLTL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered9+HLTEndSequence)
HLT1MuonIso13 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuIso13+hltSingleMuIsoLevel1Seed10+hltSingleMuIsoL1Filtered10+HLTL2muonrecoSequence+hltSingleMuIsoL2PreFiltered11+HLTL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered11+HLTL3muonrecoSequence+hltSingleMuIsoL3PreFiltered13+HLTL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered13+HLTEndSequence)
HLT1MuonIso15 = cms.Path(HLTL1muonrecoSequence+hltPrescaleSingleMuIso15+hltSingleMuIsoLevel1Seed10+hltSingleMuIsoL1Filtered10+HLTL2muonrecoSequence+hltSingleMuIsoL2PreFiltered12+HLTL2muonisorecoSequence+hltSingleMuIsoL2IsoFiltered12+HLTL3muonrecoSequence+hltSingleMuIsoL3PreFiltered15+HLTL3muonisorecoSequence+hltSingleMuIsoL3IsoFiltered15+HLTEndSequence)
CandHLT2MuonPsi2S = cms.Path(HLTBeginSequence+hltPrescalePsi2SMM+hltJpsiMMLevel1Seed+hltJpsiMML1Filtered+HLTL2muonrecoSequence+hltPsi2SMML2Filtered+HLTL3muonrecoSequence+hltPsi2SMML3Filtered+HLTEndSequence)
HLTriggerFinalPath = cms.Path(hltTriggerSummaryAOD+hltTriggerSummaryRAWprescaler+hltTriggerSummaryRAW+hltBoolFinal)

