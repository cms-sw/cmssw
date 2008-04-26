import FWCore.ParameterSet.Config as cms

process = cms.Process("RAW2DIGIRECO")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:raw.root')
)

process.siPixelClusters = cms.EDProducer("SiPixelClusterProducer",
    src = cms.InputTag("siPixelDigis"),
    ChannelThreshold = cms.int32(2500),
    MissCalibrate = cms.untracked.bool(True),
    payloadType = cms.string('Offline'),
    SeedThreshold = cms.int32(3000),
    ClusterThreshold = cms.double(5050.0)
)

process.thPLSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('ThLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(22.7),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.3),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.genMet = cms.EDProducer("METProducer",
    src = cms.InputTag("genCandidatesForMET"),
    METType = cms.string('GenMET'),
    alias = cms.string('GenMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

process.muParamGlobalIsoDepositCalByAssociatorHits = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(True),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        NoiseTow_EE = cms.double(0.15),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(True),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(True),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(False),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(True)
        ),
        Threshold_HO = cms.double(0.1),
        Noise_EB = cms.double(0.025),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_E = cms.double(0.025),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(True),
        Threshold_H = cms.double(0.1),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        DR_Veto_E = cms.double(0.07),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    )
)

process.particleFlow = cms.EDProducer("PFProducer",
    pf_mergedPhotons_mvaWeightFile = cms.string('RecoParticleFlow/PFProducer/data/MVAnalysis_MLP.weights.txt'),
    blocks = cms.InputTag("particleFlowBlock"),
    verbose = cms.untracked.bool(False),
    pf_clusterRecovery = cms.bool(False),
    pf_calib_ECAL_HCAL_eslope = cms.double(1.05),
    pf_mergedPhotons_mvaCut = cms.double(0.5),
    pf_calib_HCAL_offset = cms.double(1.73),
    pf_calib_HCAL_slope = cms.double(2.17),
    pf_calib_HCAL_damping = cms.double(2.49),
    pf_nsigma_ECAL = cms.double(3.0),
    pf_calib_ECAL_HCAL_hslope = cms.double(1.06),
    pf_calib_ECAL_HCAL_offset = cms.double(6.11),
    pf_mergedPhotons_PSCut = cms.double(0.001),
    debug = cms.untracked.bool(False),
    pf_calib_ECAL_offset = cms.double(0.0),
    pf_nsigma_HCAL = cms.double(1.7),
    pf_calib_ECAL_slope = cms.double(1.0)
)

process.pixelVertices = cms.EDProducer("PixelVertexProducer",
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Verbosity = cms.int32(0),
    UseError = cms.bool(True),
    TrackCollection = cms.InputTag("pixelTracks"),
    ZSeparation = cms.double(0.05),
    NTrkMin = cms.int32(2),
    Method2 = cms.bool(True),
    Finder = cms.string('DivisiveVertexFinder'),
    PtMin = cms.double(1.0)
)

process.genEventScale = cms.EDProducer("GenEventScaleProducer",
    src = cms.InputTag("source")
)

process.htMetIC5 = cms.EDProducer("METProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETIC5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.photons = cms.EDProducer("PhotonProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    endcapHitProducer = cms.string('ecalRecHit'),
    minR9 = cms.double(0.93),
    usePrimaryVertex = cms.bool(True),
    scIslandEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower'),
    primaryVertexProducer = cms.string('offlinePrimaryVerticesFromCTFTracks'),
    conversionCollection = cms.string(''),
    endcapClusterShapeMapCollection = cms.string('islandEndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_endcPresh = cms.double(3.6),
    posCalc_logweight = cms.bool(True),
    scIslandEndcapCollection = cms.string(''),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    posCalc_w0 = cms.double(4.2),
    photonCollection = cms.string(''),
    pixelSeedProducer = cms.string('electronPixelSeeds'),
    conversionProducer = cms.string('conversions'),
    hbheInstance = cms.string(''),
    scHybridBarrelCollection = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    posCalc_t0_endc = cms.double(6.3),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    minSCEt = cms.double(5.0),
    maxHOverE = cms.double(0.2),
    hOverEConeSize = cms.double(0.1),
    posCalc_x0 = cms.double(0.89),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    endcapClusterShapeMapProducer = cms.string('islandBasicClusters'),
    barrelHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_barl = cms.double(7.7)
)

process.correctedEndcapSuperClustersWithPreshower = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    assocSClusterCollection = cms.string(''),
    etThresh = cms.double(0.0),
    preshRecHitProducer = cms.string('ecalPreshowerRecHit'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    preshClusterEnergyCut = cms.double(0.0),
    endcapSClusterProducer = cms.string('correctedIslandEndcapSuperClusters'),
    preshNclust = cms.int32(4),
    endcapSClusterCollection = cms.string(''),
    debugLevel = cms.string(''),
    preshRecHitCollection = cms.string('EcalRecHitsES'),
    preshSeededNstrip = cms.int32(15)
)

process.MinProd = cms.EDProducer("AlCaEcalHcalReadoutsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco")
)

process.genParticleCandidates = cms.EDProducer("FastGenParticleCandidateProducer",
    saveBarCodes = cms.untracked.bool(False),
    src = cms.InputTag("source"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)

process.htMetKT4 = cms.EDProducer("METProducer",
    src = cms.InputTag("kt4CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETKT4'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.htMetKT6 = cms.EDProducer("METProducer",
    src = cms.InputTag("kt6CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETKT6'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.siStripElectrons = cms.EDProducer("SiStripElectronProducer",
    siStereoHitCollection = cms.string('stereoRecHit'),
    maxHitsOnDetId = cms.int32(4),
    minHits = cms.int32(5),
    trackCandidatesLabel = cms.string(''),
    superClusterProducer = cms.string('correctedHybridSuperClusters'),
    phiBandWidth = cms.double(0.01),
    siStripElectronsLabel = cms.string(''),
    siRphiHitCollection = cms.string('rphiRecHit'),
    siHitProducer = cms.string('siStripMatchedRecHits'),
    maxReducedChi2 = cms.double(10000.0),
    originUncertainty = cms.double(15.0),
    maxNormResid = cms.double(10.0),
    siMatchedHitCollection = cms.string('matchedRecHit'),
    superClusterCollection = cms.string('')
)

process.fixedMatrixPreshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    preshStripEnergyCut = cms.double(0.0),
    preshRecHitProducer = cms.string('ecalPreshowerRecHit'),
    preshPi0Nstrip = cms.int32(5),
    endcapSClusterProducer = cms.string('fixedMatrixSuperClusters'),
    PreshowerClusterShapeCollectionX = cms.string('fixedMatrixPreshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('fixedMatrixPreshowerYClustersShape'),
    endcapSClusterCollection = cms.string('fixedMatrixEndcapSuperClusters'),
    debugLevel = cms.string('INFO'),
    preshRecHitCollection = cms.string('EcalRecHitsES')
)

process.fixedMatrixBasicClusters = cms.EDProducer("FixedMatrixClusterProducer",
    posCalc_x0 = cms.double(0.89),
    endcapHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_endcPresh = cms.double(1.2),
    barrelClusterCollection = cms.string('fixedMatrixBarrelBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    barrelShapeAssociation = cms.string('fixedMatrixBarrelShapeAssoc'),
    posCalc_w0 = cms.double(4.2),
    posCalc_logweight = cms.bool(True),
    clustershapecollectionEE = cms.string('fixedMatrixEndcapShape'),
    clustershapecollectionEB = cms.string('fixedMatrixBarrelShape'),
    VerbosityLevel = cms.string('ERROR'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    barrelHitProducer = cms.string('ecalRecHit'),
    endcapShapeAssociation = cms.string('fixedMatrixEndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_barl = cms.double(7.4),
    endcapClusterCollection = cms.string('fixedMatrixEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5)
)

process.trackCountingHighPurBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('trackCounting3D3rd')
)

process.pixelMatchGsfFit = cms.EDProducer("GsfTrackProducer",
    src = cms.InputTag("egammaCkfTrackCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    producer = cms.string(''),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdGsfElectronPropagator')
)

process.globalMixedSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('MixedLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.preFilterCmsTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("newTrackCandidateMaker"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherWithOutlierRejection'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithAngleAndTemplate'),
    AlgorithmName = cms.string('ctf'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.hoCalibProducer = cms.EDProducer("AlCaHOCalibProducer",
    lastTS = cms.untracked.int32(8),
    hotime = cms.untracked.bool(False),
    hbinfo = cms.untracked.bool(False),
    sigma = cms.untracked.double(1.0),
    digiInput = cms.untracked.bool(False),
    RootFileName = cms.untracked.string('test.root'),
    m_scale = cms.untracked.double(4.0),
    debug = cms.untracked.bool(False),
    muons = cms.untracked.InputTag("standAloneMuons"),
    firstTS = cms.untracked.int32(5),
    PedestalFile = cms.untracked.string('peds_mtcc2_4333.log')
)

process.htMetSC5 = cms.EDProducer("METProducer",
    src = cms.InputTag("sisCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETSC5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.htMetSC7 = cms.EDProducer("METProducer",
    src = cms.InputTag("sisCone7CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMETSC7'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.gsfPFtracks = cms.EDProducer("GsfTrackProducer",
    src = cms.InputTag("gsfElCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    producer = cms.string(''),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdElectronPropagator')
)

process.muParamGlobalIsoDepositCalEcal = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('track'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        DR_Veto_H = cms.double(0.1),
        Vertex_Constraint_Z = cms.bool(False),
        Threshold_H = cms.double(0.5),
        ComponentName = cms.string('CaloExtractor'),
        Threshold_E = cms.double(0.2),
        DR_Max = cms.double(1.0),
        DR_Veto_E = cms.double(0.07),
        Weight_E = cms.double(1.0),
        Vertex_Constraint_XY = cms.bool(False),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        Weight_H = cms.double(0.0)
    )
)

process.Fastjet10PFJets = cms.EDProducer("FastJetProducer",
    src = cms.InputTag("particleFlowJetCandidates"),
    inputEtMin = cms.double(0.0),
    inputEMin = cms.double(0.0),
    FJ_ktRParam = cms.double(1.0),
    jetType = cms.untracked.string('PFJet'),
    towerThreshold = cms.double(0.5),
    PtMin = cms.double(1.0)
)

process.GammaJetProd = cms.EDProducer("AlCaGammaJetProducer",
    hbheInput = cms.InputTag("hbhereco"),
    correctedIslandBarrelSuperClusterCollection = cms.string(''),
    correctedIslandEndcapSuperClusterCollection = cms.string(''),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    correctedIslandEndcapSuperClusterProducer = cms.string('correctedIslandEndcapSuperClusters'),
    correctedIslandBarrelSuperClusterProducer = cms.string('correctedIslandBarrelSuperClusters'),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    srcCalo = cms.VInputTag(cms.InputTag("iterativeCone7CaloJets")),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.conversions = cms.EDProducer("ConvertedPhotonProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    scHybridBarrelCollection = cms.string(''),
    convertedPhotonCollection = cms.string(''),
    scIslandEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower'),
    inOutTrackSCAssociation = cms.string('inOutTrackSCAssociationCollection'),
    outInTrackCollection = cms.string(''),
    barrelClusterShapeMapCollection = cms.string('hybridShapeAssoc'),
    endcapClusterShapeMapProducer = cms.string('islandBasicClusters'),
    bcEndcapCollection = cms.string('islandEndcapBasicClusters'),
    conversionIOTrackProducer = cms.string('ckfInOutTracksFromConversions'),
    bcBarrelCollection = cms.string('islandBarrelBasicClusters'),
    inOutTrackCollection = cms.string(''),
    bcProducer = cms.string('islandBasicClusters'),
    outInTrackSCAssociation = cms.string('outInTrackSCAssociationCollection'),
    scIslandEndcapCollection = cms.string(''),
    barrelClusterShapeMapProducer = cms.string('hybridSuperClusters'),
    conversionOITrackProducer = cms.string('ckfOutInTracksFromConversions'),
    endcapClusterShapeMapCollection = cms.string('islandEndcapShapeAssoc')
)

process.globalSeedsFromPairsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('MixedLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            nSigmaZ = cms.double(3.0),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.string('pixelVertices'),
            ptMin = cms.double(0.9),
            useFoundVertices = cms.bool(True),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.particleFlowClusterECAL = cms.EDProducer("PFClusterProducer",
    thresh_Seed_Endcap = cms.double(0.8),
    verbose = cms.untracked.bool(False),
    showerSigma = cms.double(5.0),
    thresh_Seed_Barrel = cms.double(0.23),
    depthCor_Mode = cms.int32(1),
    posCalcNCrystal = cms.int32(9),
    depthCor_B_preshower = cms.double(4.0),
    nNeighbours = cms.int32(8),
    PFRecHits = cms.InputTag("particleFlowRecHitECAL"),
    thresh_Barrel = cms.double(0.08),
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    depthCor_A = cms.double(0.89),
    thresh_Endcap = cms.double(0.3),
    posCalcP1 = cms.double(-1.0)
)

process.sisCone5GenJets = cms.EDProducer("SISConeJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("genParticlesForJets"),
    protojetPtMin = cms.double(0.0),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    inputEtMin = cms.double(0.0),
    coneOverlapThreshold = cms.double(0.75),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('SISC5GenJet'),
    inputEMin = cms.double(0.0),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    jetType = cms.untracked.string('GenJet'),
    UE_Subtraction = cms.string('no'),
    splitMergeScale = cms.string('pttilde'),
    JetPtMin = cms.double(1.0),
    GhostArea = cms.double(1.0)
)

process.genMetIC5GenJets = cms.EDProducer("METProducer",
    src = cms.InputTag("iterativeCone5GenJets"),
    METType = cms.string('MET'),
    alias = cms.string('GenMETIC5'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

process.offlinePrimaryVertices = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.02),
        minVertexFitProb = cms.double(0.01)
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(7),
        maxD0Significance = cms.double(5.0),
        minPt = cms.double(0.0),
        minPixelHits = cms.int32(2)
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(False),
    VtxFinderParameters = cms.PSet(
        minTrackCompatibilityToOtherVertex = cms.double(0.01),
        minTrackCompatibilityToMainVertex = cms.double(0.05),
        maxNbVertices = cms.int32(0)
    ),
    TkClusParameters = cms.PSet(
        zSeparation = cms.double(0.1)
    )
)

process.jetProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('jetProbability')
)

process.impactParameterMVABJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('impactParameterMVAComputer')
)

process.alCaIsolatedElectrons = cms.EDProducer("AlCaElectronsProducer",
    electronLabel = cms.InputTag("electronFilter"),
    alcaEndcapHitCollection = cms.string('alcaEndcapHits'),
    phiSize = cms.int32(11),
    etaSize = cms.int32(5),
    ebRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    alcaBarrelHitCollection = cms.string('alcaBarrelHits'),
    eeRecHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)

process.fixedMatrixSuperClustersWithPreshower = cms.EDProducer("PreshowerClusterProducer",
    preshCalibGamma = cms.double(0.024),
    preshStripEnergyCut = cms.double(0.0),
    preshClusterCollectionY = cms.string('preshowerYClusters'),
    assocSClusterCollection = cms.string(''),
    etThresh = cms.double(0.0),
    preshRecHitProducer = cms.string('ecalPreshowerRecHit'),
    preshClusterCollectionX = cms.string('preshowerXClusters'),
    preshCalibPlaneY = cms.double(0.7),
    preshCalibPlaneX = cms.double(1.0),
    preshCalibMIP = cms.double(9e-05),
    preshClusterEnergyCut = cms.double(0.0),
    endcapSClusterProducer = cms.string('fixedMatrixSuperClusters'),
    preshNclust = cms.int32(4),
    endcapSClusterCollection = cms.string('fixedMatrixEndcapSuperClusters'),
    debugLevel = cms.string(''),
    preshRecHitCollection = cms.string('EcalRecHitsES'),
    preshSeededNstrip = cms.int32(15)
)

process.impactParameterTagInfos = cms.EDProducer("TrackIPProducer",
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumNumberOfHits = cms.int32(8),
    minimumTransverseMomentum = cms.double(1.0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    maximumDecayLength = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetTracks = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
    minimumNumberOfPixelHits = cms.int32(2),
    jetDirectionUsingTracks = cms.bool(False),
    computeProbabilities = cms.bool(True),
    maximumDistanceToJetAxis = cms.double(0.07),
    maximumChiSquared = cms.double(5.0)
)

process.genMetNoNuBSM = cms.EDProducer("METProducer",
    src = cms.InputTag("genParticlesForJets"),
    METType = cms.string('GenMET'),
    alias = cms.string('GenMETNoNuBSM'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

process.muIsoDepositCalByAssociatorHits = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("muons"),
        MultipleDepositsFlag = cms.bool(True),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        NoiseTow_EE = cms.double(0.15),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(True),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(True),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(False),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(True)
        ),
        Threshold_HO = cms.double(0.1),
        Noise_EB = cms.double(0.025),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_E = cms.double(0.025),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(True),
        Threshold_H = cms.double(0.1),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        DR_Veto_E = cms.double(0.07),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    )
)

process.muParamGlobalIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(True),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        NoiseTow_EE = cms.double(0.15),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        Threshold_HO = cms.double(0.5),
        Noise_EB = cms.double(0.025),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_E = cms.double(0.2),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(False),
        Threshold_H = cms.double(0.5),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        DR_Veto_E = cms.double(0.07),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    )
)

process.particleFlowSimParticle = cms.EDProducer("PFSimParticleProducer",
    verbose = cms.untracked.bool(False),
    Fitter = cms.string('KFFittingSmoother'),
    process_RecTracks = cms.untracked.bool(False),
    ParticleFilter = cms.PSet(
        EProton = cms.double(5000.0),
        etaMax = cms.double(5.0),
        pTMin = cms.double(0.0),
        EMin = cms.double(0.0)
    ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    sim = cms.InputTag("g4SimHits"),
    process_Particles = cms.untracked.bool(True),
    Propagator = cms.string('PropagatorWithMaterial'),
    VertexGenerator = cms.PSet(
        type = cms.string('None')
    )
)

process.electrontruth = cms.EDProducer("TrackingElectronProducer")

process.muParamGlobalIsoDepositCtfTk = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("ctfGSWithMaterialTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

process.pfRecoTauTagInfoProducer = cms.EDProducer("PFRecoTauTagInfoProducer",
    ChargedHadrCand_tkmaxipt = cms.double(0.03),
    tkminTrackerHitsn = cms.int32(8),
    tkminPixelHitsn = cms.int32(2),
    NeutrHadrCand_HcalclusminE = cms.double(1.0),
    ChargedHadrCand_tkminPixelHitsn = cms.int32(2),
    PVProducer = cms.string('offlinePrimaryVertices'),
    ChargedHadrCand_tkminTrackerHitsn = cms.int32(3),
    tkmaxChi2 = cms.double(100.0),
    PFCandidateProducer = cms.string('particleFlow'),
    ChargedHadrCand_tkmaxChi2 = cms.double(100.0),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    ChargedHadrCand_tkminPt = cms.double(1.0),
    smearedPVsigmaZ = cms.double(0.005),
    tkPVmaxDZ = cms.double(0.2),
    GammaCand_EcalclusminE = cms.double(1.0),
    tkminPt = cms.double(1.0),
    UsePVconstraint = cms.bool(False),
    PFJetTracksAssociatorProducer = cms.string('ic5PFJetTracksAssociatorAtVertex'),
    tkmaxipt = cms.double(0.03),
    ChargedHadrCand_tkPVmaxDZ = cms.double(0.2)
)

process.ecalWeightUncalibRecHit = cms.EDProducer("EcalWeightUncalibRecHitProducer",
    EBdigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EEhitCollection = cms.string('EcalUncalibRecHitsEE'),
    EEdigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EBhitCollection = cms.string('EcalUncalibRecHitsEB')
)

process.met = cms.EDProducer("METProducer",
    src = cms.InputTag("caloTowers"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection')
)

process.pfTrackElec = cms.EDProducer("PFElecTkProducer",
    TrajInEvents = cms.bool(True),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    GsfTrackCandidateModuleLabel = cms.string('gsfElCandidates'),
    ModeMomentum = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdElectronPropagator'),
    GsfTrackModuleLabel = cms.string('gsfPFtracks')
)

process.genParticles = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("source"),
    abortOnUnknownPDGCode = cms.untracked.bool(True)
)

process.kt6GenJets = cms.EDProducer("KtJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    JetPtMin = cms.double(1.0),
    UE_Subtraction = cms.string('no'),
    alias = cms.untracked.string('KT6GenJet'),
    FJ_ktRParam = cms.double(0.6),
    jetType = cms.untracked.string('GenJet'),
    Strategy = cms.string('Best'),
    GhostArea = cms.double(1.0),
    inputEMin = cms.double(0.0)
)

process.particleFlowBlock = cms.EDProducer("PFBlockProducer",
    PFClustersPS = cms.InputTag("particleFlowClusterPS"),
    pf_resolution_map_HCAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_eta.dat'),
    pf_chi2_ECAL_PS = cms.double(100.0),
    pf_chi2_PS_Track = cms.double(100.0),
    pf_DPtoverPt_Cut = cms.double(999.9),
    PFClustersHCAL = cms.InputTag("particleFlowClusterHCAL"),
    pf_resolution_map_HCAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_HCAL_phi.dat'),
    useNuclear = cms.untracked.bool(False),
    pf_multilink = cms.bool(True),
    pf_resolution_map_ECAL_eta = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    PFNuclear = cms.InputTag("pfNuclear"),
    pf_chi2_ECAL_Track = cms.double(100.0),
    pf_chi2_PSH_PSV = cms.double(5.0),
    pf_chi2_HCAL_Track = cms.double(100.0),
    debug = cms.untracked.bool(False),
    PFClustersECAL = cms.InputTag("particleFlowClusterECAL"),
    RecTracks = cms.InputTag("elecpreid"),
    pf_resolution_map_ECAL_phi = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
    pf_chi2_ECAL_HCAL = cms.double(10.0),
    verbose = cms.untracked.bool(False)
)

process.genEventWeight = cms.EDProducer("GenEventWeightProducer",
    src = cms.InputTag("source")
)

process.btagSoftElectrons = cms.EDProducer("SoftElectronProducer",
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(False),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    ),
    TrackTag = cms.InputTag("generalTracks"),
    BasicClusterTag = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    BasicClusterShapeTag = cms.InputTag("islandBasicClusters","islandBarrelShape"),
    HBHERecHitTag = cms.InputTag("hbhereco"),
    DiscriminatorCut = cms.double(0.9),
    HOverEConeSize = cms.double(0.3)
)

process.htMet = cms.EDProducer("METProducer",
    src = cms.InputTag("midPointCone5CaloJets"),
    METType = cms.string('MET'),
    alias = cms.string('HTMET'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(5.0),
    InputType = cms.string('CaloJetCollection')
)

process.islandBasicClusters = cms.EDProducer("IslandClusterProducer",
    posCalc_x0 = cms.double(0.89),
    endcapHitProducer = cms.string('ecalRecHit'),
    posCalc_t0_endcPresh = cms.double(1.2),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    IslandEndcapSeedThr = cms.double(0.18),
    posCalc_t0_endc = cms.double(3.1),
    barrelShapeAssociation = cms.string('islandBarrelShapeAssoc'),
    posCalc_w0 = cms.double(4.2),
    posCalc_logweight = cms.bool(True),
    clustershapecollectionEE = cms.string('islandEndcapShape'),
    clustershapecollectionEB = cms.string('islandBarrelShape'),
    VerbosityLevel = cms.string('ERROR'),
    endcapHitCollection = cms.string('EcalRecHitsEE'),
    barrelHitProducer = cms.string('ecalRecHit'),
    endcapShapeAssociation = cms.string('islandEndcapShapeAssoc'),
    barrelHitCollection = cms.string('EcalRecHitsEB'),
    posCalc_t0_barl = cms.double(7.4),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    IslandBarrelSeedThr = cms.double(0.5)
)

process.dt2DSegments = cms.EDProducer("DTRecSegment2DProducer",
    debug = cms.untracked.bool(False),
    Reco2DAlgoConfig = cms.PSet(
        segmCleanerMode = cms.int32(1),
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
        AlphaMaxPhi = cms.double(1.0),
        MaxAllowedHits = cms.uint32(50),
        nSharedHitsMax = cms.int32(2),
        AlphaMaxTheta = cms.double(0.1),
        debug = cms.untracked.bool(False),
        recAlgo = cms.string('DTParametrizedDriftAlgo'),
        nUnSharedHitsMin = cms.int32(2)
    ),
    recHits1DLabel = cms.InputTag("dt1DRecHits"),
    Reco2DAlgoName = cms.string('DTCombinatorialPatternReco')
)

process.fixedMatrixSuperClusters = cms.EDProducer("FixedMatrixSuperClusterProducer",
    barrelSuperclusterCollection = cms.string('fixedMatrixBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('fixedMatrixBarrelBasicClusters'),
    dynamicPhiRoad = cms.bool(True),
    endcapClusterProducer = cms.string('fixedMatrixBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.0),
    doBarrel = cms.bool(False),
    endcapSuperclusterCollection = cms.string('fixedMatrixEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryVec = cms.vint32(16, 13, 11, 10, 9, 
                8, 7, 6, 5, 4, 
                3),
            cryMin = cms.int32(2),
            etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0, 
                40.0, 45.0, 55.0, 135.0, 195.0, 
                225.0)
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            c = cms.double(0.1201),
            b = cms.double(108.8)
        )
    ),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('fixedMatrixEndcapBasicClusters'),
    barrelClusterProducer = cms.string('fixedMatrixBasicClusters')
)

process.muParamGlobalIsoDepositGsTk = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('track'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("ctfGSWithMaterialTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

process.ecalTriggerPrimitiveDigis = cms.EDProducer("EcalTrigPrimProducer",
    BarrelOnly = cms.bool(False),
    InstanceEB = cms.string(''),
    InstanceEE = cms.string(''),
    Famos = cms.bool(False),
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Label = cms.string('ecalUnsuppressedDigis')
)

process.muIsoDepositJets = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("muons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        PrintTimeReport = cms.untracked.bool(False),
        ExcludeMuonVeto = cms.bool(True),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(0.5),
            dREcal = cms.double(0.5),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(0.5),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(0.5),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        ComponentName = cms.string('JetExtractor'),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),
        DR_Veto = cms.double(0.1),
        Threshold = cms.double(5.0)
    )
)

process.sisCone7GenJets = cms.EDProducer("SISConeJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("genParticlesForJets"),
    protojetPtMin = cms.double(0.0),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    inputEtMin = cms.double(0.0),
    coneOverlapThreshold = cms.double(0.75),
    coneRadius = cms.double(0.7),
    alias = cms.untracked.string('SISC7GenJet'),
    inputEMin = cms.double(0.0),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    jetType = cms.untracked.string('GenJet'),
    UE_Subtraction = cms.string('no'),
    splitMergeScale = cms.string('pttilde'),
    JetPtMin = cms.double(1.0),
    GhostArea = cms.double(1.0)
)

process.iterativeCone5PFJets = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("particleFlowJetCandidates"),
    inputEtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    towerThreshold = cms.double(0.5),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('PFJet'),
    inputEMin = cms.double(0.0),
    seedThreshold = cms.double(1.0)
)

process.muIsoDepositTk = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("muons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

process.muParamGlobalIsoDepositTk = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('track'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)

process.pixelMatchGsfElectrons = cms.EDProducer("GsfElectronProducer",
    endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower","electronPixelSeeds"),
    maxDeltaPhi = cms.double(0.1),
    minEOverPEndcaps = cms.double(0.0),
    maxEOverPEndcaps = cms.double(10000.0),
    minEOverPBarrel = cms.double(0.0),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters","electronPixelSeeds"),
    applyEtaCorrection = cms.bool(False),
    tracks = cms.InputTag("pixelMatchGsfFit"),
    maxDeltaEta = cms.double(0.02),
    ElectronType = cms.string(''),
    maxEOverPBarrel = cms.double(10000.0),
    highPtPreselection = cms.bool(False),
    hcalRecHits = cms.InputTag("hbhereco"),
    endcapClusterShapes = cms.InputTag("islandBasicClusters","islandEndcapShapeAssoc"),
    highPtMin = cms.double(150.0),
    barrelClusterShapes = cms.InputTag("hybridSuperClusters","hybridShapeAssoc"),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.sisCone7CaloJets = cms.EDProducer("SISConeJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("caloTowers"),
    protojetPtMin = cms.double(0.0),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    JetPtMin = cms.double(1.0),
    jetPtMin = cms.double(0.0),
    coneRadius = cms.double(0.7),
    coneOverlapThreshold = cms.double(0.75),
    alias = cms.untracked.string('SISC7CaloJet'),
    inputEtMin = cms.double(0.5),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    UE_Subtraction = cms.string('no'),
    splitMergeScale = cms.string('pttilde'),
    inputEMin = cms.double(0.0),
    GhostArea = cms.double(1.0)
)

process.secondaryVertexTagInfos = cms.EDProducer("SecondaryVertexProducer",
    vertexReco = cms.PSet(
        primcut = cms.double(1.8),
        seccut = cms.double(6.0),
        smoothing = cms.bool(False),
        finder = cms.string('avr'),
        minweight = cms.double(0.5),
        weightthreshold = cms.double(0.001)
    ),
    vertexSelection = cms.PSet(
        sortCriterium = cms.string('dist3dError')
    ),
    useBeamConstraint = cms.bool(True),
    vertexCuts = cms.PSet(
        fracPV = cms.double(0.65),
        distSig3dMax = cms.double(99999.9),
        distVal2dMax = cms.double(2.5),
        useTrackWeights = cms.bool(True),
        maxDeltaRToJetAxis = cms.double(0.5),
        v0Filter = cms.PSet(
            k0sMassWindow = cms.double(0.05)
        ),
        distSig2dMin = cms.double(3.0),
        multiplicityMin = cms.uint32(2),
        distVal2dMin = cms.double(0.01),
        distSig2dMax = cms.double(99999.9),
        distVal3dMax = cms.double(99999.9),
        minimumTrackWeight = cms.double(0.5),
        distVal3dMin = cms.double(-99999.9),
        massMax = cms.double(6.5),
        distSig3dMin = cms.double(-99999.9)
    ),
    trackIPTagInfos = cms.InputTag("impactParameterTagInfos"),
    minimumTrackWeight = cms.double(0.5),
    usePVError = cms.bool(True),
    trackSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    trackSort = cms.string('sip3dSig')
)

process.pfRecoTauProducerHighEfficiency = cms.EDProducer("PFRecoTauProducer",
    LeadTrack_minPt = cms.double(5.0),
    PVProducer = cms.string('offlinePrimaryVertices'),
    ECALSignalConeSizeFormula = cms.string('0.15'),
    TrackerIsolConeMetric = cms.string('DR'),
    TrackerSignalConeMetric = cms.string('DR'),
    ECALSignalConeSize_min = cms.double(0.0),
    Track_minPt = cms.double(1.0),
    MatchingConeMetric = cms.string('DR'),
    TrackerSignalConeSizeFormula = cms.string('5.0/ET'),
    MatchingConeSizeFormula = cms.string('0.1'),
    TrackerIsolConeSize_min = cms.double(0.0),
    GammaCand_minPt = cms.double(1.5),
    HCALSignalConeMetric = cms.string('DR'),
    ChargedHadrCandLeadChargedHadrCand_tksmaxDZ = cms.double(0.2),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.15),
    MatchingConeSize_min = cms.double(0.0),
    TrackerSignalConeSize_min = cms.double(0.07),
    ECALIsolConeSize_max = cms.double(0.6),
    HCALIsolConeSizeFormula = cms.string('0.50'),
    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    Track_IsolAnnulus_minNhits = cms.uint32(8),
    ChargedHadrCand_IsolAnnulus_minNhits = cms.uint32(8),
    PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
    ECALIsolConeMetric = cms.string('DR'),
    ECALIsolConeSizeFormula = cms.string('0.50'),
    UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint = cms.bool(True),
    JetPtMin = cms.double(15.0),
    LeadChargedHadrCand_minPt = cms.double(5.0),
    ECALSignalConeMetric = cms.string('DR'),
    TrackLeadTrack_maxDZ = cms.double(0.2),
    HCALSignalConeSize_max = cms.double(0.6),
    HCALIsolConeMetric = cms.string('DR'),
    TrackerIsolConeSizeFormula = cms.string('0.50'),
    HCALSignalConeSize_min = cms.double(0.0),
    ECALSignalConeSize_max = cms.double(0.6),
    HCALSignalConeSizeFormula = cms.string('0.10'),
    HCALIsolConeSize_max = cms.double(0.6),
    ChargedHadrCand_minPt = cms.double(1.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    ECALIsolConeSize_min = cms.double(0.0),
    MatchingConeSize_max = cms.double(0.6),
    NeutrHadrCand_minPt = cms.double(1.0),
    HCALIsolConeSize_min = cms.double(0.0)
)

process.electronPixelSeeds = cms.EDProducer("ElectronPixelSeedProducer",
    endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower"),
    SeedConfiguration = cms.PSet(
        searchInTIDTEC = cms.bool(True),
        HighPtThreshold = cms.double(35.0),
        r2MinF = cms.double(-0.15),
        DeltaPhi1Low = cms.double(0.23),
        DeltaPhi1High = cms.double(0.08),
        ePhiMin1 = cms.double(-0.125),
        PhiMin2 = cms.double(-0.002),
        LowPtThreshold = cms.double(5.0),
        maxHOverE = cms.double(0.2),
        dynamicPhiRoad = cms.bool(True),
        ePhiMax1 = cms.double(0.075),
        DeltaPhi2 = cms.double(0.004),
        SizeWindowENeg = cms.double(0.675),
        rMaxI = cms.double(0.2),
        PhiMax2 = cms.double(0.002),
        r2MaxF = cms.double(0.15),
        pPhiMin1 = cms.double(-0.075),
        pPhiMax1 = cms.double(0.125),
        SCEtCut = cms.double(5.0),
        z2MaxB = cms.double(0.09),
        hcalRecHits = cms.InputTag("hbhereco"),
        z2MinB = cms.double(-0.09),
        rMinI = cms.double(-0.2)
    ),
    SeedAlgo = cms.string(''),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters")
)

process.kt4CaloJets = cms.EDProducer("KtJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("caloTowers"),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    JetPtMin = cms.double(1.0),
    UE_Subtraction = cms.string('no'),
    alias = cms.untracked.string('KT4CaloJet'),
    Strategy = cms.string('Best'),
    FJ_ktRParam = cms.double(0.4),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    GhostArea = cms.double(1.0),
    inputEMin = cms.double(0.0)
)

process.rpcRecHits = cms.EDProducer("RPCRecHitProducer",
    recAlgoConfig = cms.PSet(

    ),
    recAlgo = cms.string('RPCRecHitStandardAlgo'),
    rpcDigiLabel = cms.InputTag("muonRPCDigis")
)

process.genEventPdfInfo = cms.EDProducer("GenEventPdfInfoProducer",
    src = cms.InputTag("source")
)

process.combinedSecondaryVertexMVABJetTags = cms.EDProducer("JetTagProducer",
    ipTagInfos = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('combinedSecondaryVertexMVA'),
    svTagInfos = cms.InputTag("secondaryVertexTagInfos")
)

process.globalMuons = cms.EDProducer("GlobalMuonProducer",
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
    GLBTrajBuilderParameters = cms.PSet(
        Chi2ProbabilityCut = cms.double(30.0),
        Direction = cms.int32(0),
        Chi2CutCSC = cms.double(150.0),
        HitThreshold = cms.int32(1),
        MuonHitsOption = cms.int32(1),
        TrackRecHitBuilder = cms.string('WithTrackAngle'),
        GlobalMuonTrackMatcher = cms.PSet(
            MinP = cms.double(2.5),
            Chi2Cut = cms.double(50.0),
            MinPt = cms.double(1.0),
            DeltaDCut = cms.double(10.0),
            DeltaRCut = cms.double(0.2)
        ),
        Chi2CutRPC = cms.double(1.0),
        CSCRecSegmentLabel = cms.InputTag("cscSegments"),
        MuonTrackingRegionBuilder = cms.PSet(
            VertexCollection = cms.string('pixelVertices'),
            EtaR_UpperLimit_Par1 = cms.double(0.25),
            Eta_fixed = cms.double(0.2),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            Rescale_Dz = cms.double(3.0),
            Rescale_phi = cms.double(3.0),
            DeltaR = cms.double(0.2),
            DeltaZ_Region = cms.double(15.9),
            Rescale_eta = cms.double(3.0),
            PhiR_UpperLimit_Par2 = cms.double(0.2),
            Eta_min = cms.double(0.013),
            Phi_fixed = cms.double(0.2),
            EscapePt = cms.double(1.5),
            UseFixedRegion = cms.bool(False),
            PhiR_UpperLimit_Par1 = cms.double(0.6),
            EtaR_UpperLimit_Par2 = cms.double(0.15),
            Phi_min = cms.double(0.02),
            UseVertex = cms.bool(False)
        ),
        Chi2CutDT = cms.double(10.0),
        TrackTransformer = cms.PSet(
            Fitter = cms.string('KFFitterForRefitInsideOut'),
            TrackerRecHitBuilder = cms.string('WithTrackAngle'),
            Smoother = cms.string('KFSmootherForRefitInsideOut'),
            MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
            RefitDirection = cms.string('insideOut'),
            RefitRPCHits = cms.bool(True)
        ),
        StateOnTrackerBoundOutPropagator = cms.string('SmartPropagatorAnyRK'),
        RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
        PtCut = cms.double(1.0),
        TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
        DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
        KFFitter = cms.string('GlbMuKFFitter')
    ),
    TrackerCollectionLabel = cms.InputTag("generalTracks"),
    TrackLoaderParameters = cms.PSet(
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(True),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        VertexConstraint = cms.bool(False)
    ),
    MuonCollectionLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx")
)

process.globalSeedsFromTripletsWithVertices = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.muons = cms.EDProducer("MuonIdProducer",
    fillEnergy = cms.bool(True),
    maxAbsPullX = cms.double(4.0),
    maxAbsEta = cms.double(3.0),
    minPt = cms.double(1.5),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    ),
    inputCollectionTypes = cms.vstring('inner tracks', 
        'links', 
        'outer tracks'),
    addExtraSoftMuons = cms.bool(False),
    debugWithTruthMatching = cms.bool(False),
    TrackExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    ),
    MuonCaloCompatibility = cms.PSet(
        PionTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_pions_allPt_2_0_norm.root'),
        MuonTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_muons_allPt_2_0_norm.root')
    ),
    inputCollectionLabels = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("globalMuons"), cms.InputTag("standAloneMuons","UpdatedAtVtx")),
    fillCaloCompatibility = cms.bool(True),
    maxAbsPullY = cms.double(9999.0),
    CaloExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        NoiseTow_EE = cms.double(0.15),
        Threshold_HO = cms.double(0.5),
        DR_Veto_E = cms.double(0.07),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        Noise_EB = cms.double(0.025),
        Threshold_E = cms.double(0.2),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(False),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_H = cms.double(0.5),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    ),
    minP = cms.double(3.0),
    maxAbsDx = cms.double(3.0),
    maxAbsDy = cms.double(9999.0),
    fillIsolation = cms.bool(True),
    minNumberOfMatches = cms.int32(1),
    fillMatching = cms.bool(True)
)

process.softMuonBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softMuonTagInfos"),
    jetTagComputer = cms.string('softMuon')
)

process.ecalRecHit = cms.EDProducer("EcalRecHitProducer",
    EErechitCollection = cms.string('EcalRecHitsEE'),
    EEuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    EBuncalibRecHitCollection = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    ChannelStatusToBeExcluded = cms.vint32()
)

process.kt6CaloJets = cms.EDProducer("KtJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("caloTowers"),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    inputEtMin = cms.double(0.5),
    JetPtMin = cms.double(1.0),
    UE_Subtraction = cms.string('no'),
    alias = cms.untracked.string('KT6CaloJet'),
    Strategy = cms.string('Best'),
    FJ_ktRParam = cms.double(0.6),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    GhostArea = cms.double(1.0),
    inputEMin = cms.double(0.0)
)

process.islandSuperClusters = cms.EDProducer("SuperClusterProducer",
    barrelSuperclusterCollection = cms.string('islandBarrelSuperClusters'),
    endcapEtaSearchRoad = cms.double(0.14),
    barrelClusterCollection = cms.string('islandBarrelBasicClusters'),
    endcapClusterProducer = cms.string('islandBasicClusters'),
    barrelPhiSearchRoad = cms.double(0.8),
    endcapPhiSearchRoad = cms.double(0.6),
    VerbosityLevel = cms.string('ERROR'),
    seedTransverseEnergyThreshold = cms.double(1.0),
    endcapSuperclusterCollection = cms.string('islandEndcapSuperClusters'),
    barrelEtaSearchRoad = cms.double(0.06),
    doBarrel = cms.bool(True),
    doEndcaps = cms.bool(True),
    endcapClusterCollection = cms.string('islandEndcapBasicClusters'),
    barrelClusterProducer = cms.string('islandBasicClusters')
)

process.globalPixelSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.jetBProbabilityBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('jetBProbability')
)

process.egammaCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
    src = cms.InputTag("siStripElectrons"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('KFFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('egammaCTFWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('PropagatorWithMaterial')
)

process.trackCountingHighEffBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('trackCounting3D2nd')
)

process.iterativeCone5CaloJets = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("caloTowers"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    jetPtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)

process.GsfGlobalElectronTest = cms.EDProducer("GsfTrackProducer",
    src = cms.InputTag("CkfElectronCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    producer = cms.string(''),
    Fitter = cms.string('GsfElectronFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    Propagator = cms.string('fwdElectronPropagator')
)

process.ctfWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("ckfTrackCandidates"),
    clusterRemovalInfo = cms.InputTag(""),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.trackingtruthprod = cms.EDProducer("TrackingTruthProducer",
    discardOutVolume = cms.bool(False),
    DiscardHitsFromDeltas = cms.bool(True),
    simHitLabel = cms.string('g4SimHits'),
    volumeRadius = cms.double(1200.0),
    vertexDistanceCut = cms.double(0.003),
    HepMCDataLabels = cms.vstring('VtxSmeared', 
        'PythiaSource', 
        'source'),
    TrackerHitLabels = cms.vstring('TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof', 
        'TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof'),
    volumeZ = cms.double(3000.0)
)

process.metOpt = cms.EDProducer("METProducer",
    src = cms.InputTag("caloTowersOpt"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOpt'),
    noHF = cms.bool(False),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

process.DiJProd = cms.EDProducer("AlCaDiJetsProducer",
    hbheInput = cms.InputTag("hbhereco"),
    hfInput = cms.InputTag("hfreco"),
    hoInput = cms.InputTag("horeco"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    srcCalo = cms.VInputTag(cms.InputTag("iterativeCone7CaloJets")),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.iterativeCone5GenJets = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5GenJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)

process.MuonSeed = cms.EDProducer("MuonSeedProducer",
    maxDeltaEtaOverlap = cms.double(0.08),
    DebugMuonSeed = cms.bool(False),
    minimumSeedPt = cms.double(5.0),
    minCSCHitsPerSegment = cms.int32(4),
    maxDeltaPhiDT = cms.double(0.3),
    maxDeltaPhiOverlap = cms.double(0.25),
    minDTHitsPerSegment = cms.int32(2),
    maxPhiResolutionDT = cms.double(0.03),
    DTSegmentLabel = cms.InputTag("dt4DSegments"),
    SeedPtSystematics = cms.double(0.1),
    maximumSeedPt = cms.double(3000.0),
    defaultSeedPt = cms.double(25.0),
    CSCSegmentLabel = cms.InputTag("cscSegments"),
    maxEtaResolutionCSC = cms.double(0.06),
    EnableDTMeasurement = cms.bool(True),
    maxEtaResolutionDT = cms.double(0.02),
    maxDeltaEtaDT = cms.double(0.3),
    maxPhiResolutionCSC = cms.double(0.03),
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
    maxDeltaEtaCSC = cms.double(0.2),
    maxDeltaPhiCSC = cms.double(0.5),
    EnableCSCMeasurement = cms.bool(True)
)

process.kt4GenJets = cms.EDProducer("KtJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    JetPtMin = cms.double(1.0),
    UE_Subtraction = cms.string('no'),
    alias = cms.untracked.string('KT4GenJet'),
    FJ_ktRParam = cms.double(0.4),
    jetType = cms.untracked.string('GenJet'),
    Strategy = cms.string('Best'),
    GhostArea = cms.double(1.0),
    inputEMin = cms.double(0.0)
)

process.IsoProd = cms.EDProducer("AlCaIsoTracksProducer",
    hbheInput = cms.InputTag("hbhereco"),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    ),
    hoInput = cms.InputTag("horeco"),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    histoFlag = cms.untracked.int32(0),
    inputTrackLabel = cms.untracked.string('generalTracks')
)

process.hybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    eThreshA = cms.double(0.003),
    posCalc_x0 = cms.double(0.89),
    clustershapecollection = cms.string(''),
    shapeAssociation = cms.string('hybridShapeAssoc'),
    ewing = cms.double(1.0),
    HybridBarrelSeedThr = cms.double(1.0),
    dynamicPhiRoad = cms.bool(False),
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    step = cms.int32(17),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    posCalc_t0 = cms.double(7.4),
    debugLevel = cms.string('INFO'),
    dynamicEThresh = cms.bool(False),
    eseed = cms.double(0.35),
    superclusterCollection = cms.string(''),
    posCalc_logweight = cms.bool(True),
    ethresh = cms.double(0.1),
    eThreshB = cms.double(0.1),
    ecalhitproducer = cms.string('ecalRecHit')
)

process.dt1DRecHits = cms.EDProducer("DTRecHitProducer",
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
    dtDigiLabel = cms.InputTag("muonDTDigis"),
    recAlgo = cms.string('DTParametrizedDriftAlgo')
)

process.mergedtruth = cms.EDProducer("MergedTruthProducer")

process.softElectronBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softElectronTagInfos"),
    jetTagComputer = cms.string('softElectron')
)

process.particleFlowClusterPS = cms.EDProducer("PFClusterProducer",
    thresh_Seed_Endcap = cms.double(0.0005),
    verbose = cms.untracked.bool(False),
    showerSigma = cms.double(0.2),
    thresh_Seed_Barrel = cms.double(0.0005),
    depthCor_Mode = cms.int32(0),
    posCalcNCrystal = cms.int32(-1),
    depthCor_B_preshower = cms.double(4.0),
    nNeighbours = cms.int32(8),
    PFRecHits = cms.InputTag("particleFlowRecHitPS"),
    thresh_Barrel = cms.double(7e-06),
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    depthCor_A = cms.double(0.89),
    thresh_Endcap = cms.double(7e-06),
    posCalcP1 = cms.double(0.0)
)

process.calomuons = cms.EDProducer("CaloMuonProducer",
    inputMuons = cms.InputTag("muons"),
    inputTracks = cms.InputTag("generalTracks"),
    MuonCaloCompatibility = cms.PSet(
        PionTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_pions_allPt_2_0_norm.root'),
        MuonTemplateFileName = cms.FileInPath('RecoMuon/MuonIdentification/data/MuID_templates_muons_allPt_2_0_norm.root')
    ),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(9999.0),
        dREcal = cms.double(9999.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(0.05),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.2),
        useMuon = cms.bool(True),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    ),
    minCaloCompatibility = cms.double(0.6)
)

process.caloRecoTauTagInfoProducer = cms.EDProducer("CaloRecoTauTagInfoProducer",
    tkminTrackerHitsn = cms.int32(8),
    tkminPixelHitsn = cms.int32(2),
    ECALBasicClusterpropagTrack_matchingDRConeSize = cms.double(0.015),
    PVProducer = cms.string('offlinePrimaryVertices'),
    tkminPt = cms.double(1.0),
    ESRecHitsSource = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    UsePVconstraint = cms.bool(False),
    tkmaxChi2 = cms.double(100.0),
    EBRecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    ECALBasicClusterminE = cms.double(1.0),
    smearedPVsigmaZ = cms.double(0.005),
    tkPVmaxDZ = cms.double(0.2),
    ECALBasicClustersAroundCaloJet_DRConeSize = cms.double(0.5),
    tkmaxipt = cms.double(0.03),
    EERecHitsSource = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    CaloJetTracksAssociatorProducer = cms.string('ic5JetTracksAssociatorAtVertex')
)

process.caloRecoTauProducer = cms.EDProducer("CaloRecoTauProducer",
    LeadTrack_minPt = cms.double(5.0),
    PVProducer = cms.string('offlinePrimaryVertices'),
    ECALSignalConeSizeFormula = cms.string('0.15'),
    TrackerIsolConeMetric = cms.string('DR'),
    TrackerSignalConeMetric = cms.string('DR'),
    ECALSignalConeSize_min = cms.double(0.0),
    ECALRecHit_minEt = cms.double(0.5),
    MatchingConeMetric = cms.string('DR'),
    TrackerSignalConeSizeFormula = cms.string('0.07'),
    MatchingConeSizeFormula = cms.string('0.10'),
    TrackerIsolConeSize_min = cms.double(0.0),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.6),
    MatchingConeSize_min = cms.double(0.0),
    TrackerSignalConeSize_min = cms.double(0.0),
    ECALIsolConeSize_max = cms.double(0.6),
    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    ECALIsolConeMetric = cms.string('DR'),
    ECALIsolConeSizeFormula = cms.string('0.50'),
    JetPtMin = cms.double(15.0),
    ECALSignalConeMetric = cms.string('DR'),
    TrackLeadTrack_maxDZ = cms.double(0.2),
    Track_minPt = cms.double(1.0),
    TrackerIsolConeSizeFormula = cms.string('0.50'),
    ECALSignalConeSize_max = cms.double(0.6),
    ECALIsolConeSize_min = cms.double(0.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    CaloRecoTauTagInfoProducer = cms.InputTag("caloRecoTauTagInfoProducer"),
    MatchingConeSize_max = cms.double(0.6)
)

process.sisCone5CaloJets = cms.EDProducer("SISConeJetProducer",
    Active_Area_Repeats = cms.int32(0),
    src = cms.InputTag("caloTowers"),
    protojetPtMin = cms.double(0.0),
    verbose = cms.untracked.bool(False),
    Ghost_EtaMax = cms.double(0.0),
    JetPtMin = cms.double(1.0),
    jetPtMin = cms.double(0.0),
    coneRadius = cms.double(0.5),
    coneOverlapThreshold = cms.double(0.75),
    alias = cms.untracked.string('SISC5CaloJet'),
    inputEtMin = cms.double(0.5),
    caching = cms.bool(False),
    maxPasses = cms.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    UE_Subtraction = cms.string('no'),
    splitMergeScale = cms.string('pttilde'),
    inputEMin = cms.double(0.0),
    GhostArea = cms.double(1.0)
)

process.offlinePrimaryVerticesFromCTFTracks = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.02),
        minVertexFitProb = cms.double(0.01)
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(7),
        maxD0Significance = cms.double(5.0),
        minPt = cms.double(0.0),
        minPixelHits = cms.int32(2)
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(False),
    VtxFinderParameters = cms.PSet(
        minTrackCompatibilityToOtherVertex = cms.double(0.01),
        minTrackCompatibilityToMainVertex = cms.double(0.05),
        maxNbVertices = cms.int32(0)
    ),
    TkClusParameters = cms.PSet(
        zSeparation = cms.double(0.1)
    )
)

process.thWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("thTrackCandidates"),
    clusterRemovalInfo = cms.InputTag("thClusters"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.metNoHF = cms.EDProducer("METProducer",
    src = cms.InputTag("caloTowers"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.5),
    InputType = cms.string('CandidateCollection')
)

process.newSeedFromPairs = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('MixedLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalTrackingRegionWithVerticesProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            beamSpot = cms.InputTag("offlineBeamSpot"),
            useFixedError = cms.bool(True),
            nSigmaZ = cms.double(3.0),
            sigmaZVertex = cms.double(3.0),
            fixedError = cms.double(0.2),
            VertexCollection = cms.string('pixelVertices'),
            ptMin = cms.double(0.9),
            useFoundVertices = cms.bool(True),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.rsWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("rsTrackCandidates"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('RKFittingSmoother'),
    useHitsSplitting = cms.bool(False),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('rs'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.dt4DSegments = cms.EDProducer("DTRecSegment4DProducer",
    debug = cms.untracked.bool(False),
    Reco4DAlgoName = cms.string('DTCombinatorialPatternReco4D'),
    recHits2DLabel = cms.InputTag("dt2DSegments"),
    recHits1DLabel = cms.InputTag("dt1DRecHits"),
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
        Reco2DAlgoConfig = cms.PSet(
            segmCleanerMode = cms.int32(1),
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
            AlphaMaxPhi = cms.double(1.0),
            MaxAllowedHits = cms.uint32(50),
            nSharedHitsMax = cms.int32(2),
            AlphaMaxTheta = cms.double(0.1),
            debug = cms.untracked.bool(False),
            recAlgo = cms.string('DTParametrizedDriftAlgo'),
            nUnSharedHitsMin = cms.int32(2)
        ),
        debug = cms.untracked.bool(False),
        recAlgo = cms.string('DTParametrizedDriftAlgo'),
        nUnSharedHitsMin = cms.int32(2),
        AllDTRecHits = cms.bool(True)
    )
)

process.combinedSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    ipTagInfos = cms.InputTag("impactParameterTagInfos"),
    jetTagComputer = cms.string('combinedSecondaryVertex'),
    svTagInfos = cms.InputTag("secondaryVertexTagInfos")
)

process.ecalPreshowerRecHit = cms.EDProducer("ESRecHitProducer",
    ESrechitCollection = cms.string('EcalRecHitsES'),
    ESdigiCollection = cms.InputTag("ecalPreshowerDigis")
)

process.pfRecoTauProducer = cms.EDProducer("PFRecoTauProducer",
    LeadTrack_minPt = cms.double(5.0),
    PVProducer = cms.string('offlinePrimaryVertices'),
    ECALSignalConeSizeFormula = cms.string('0.15'),
    TrackerIsolConeMetric = cms.string('DR'),
    TrackerSignalConeMetric = cms.string('DR'),
    ECALSignalConeSize_min = cms.double(0.0),
    Track_minPt = cms.double(1.0),
    MatchingConeMetric = cms.string('DR'),
    TrackerSignalConeSizeFormula = cms.string('0.07'),
    MatchingConeSizeFormula = cms.string('0.1'),
    TrackerIsolConeSize_min = cms.double(0.0),
    GammaCand_minPt = cms.double(1.5),
    HCALSignalConeMetric = cms.string('DR'),
    ChargedHadrCandLeadChargedHadrCand_tksmaxDZ = cms.double(0.2),
    TrackerIsolConeSize_max = cms.double(0.6),
    TrackerSignalConeSize_max = cms.double(0.6),
    MatchingConeSize_min = cms.double(0.0),
    TrackerSignalConeSize_min = cms.double(0.0),
    ECALIsolConeSize_max = cms.double(0.6),
    HCALIsolConeSizeFormula = cms.string('0.50'),
    AreaMetric_recoElements_maxabsEta = cms.double(2.5),
    Track_IsolAnnulus_minNhits = cms.uint32(8),
    ChargedHadrCand_IsolAnnulus_minNhits = cms.uint32(8),
    PFTauTagInfoProducer = cms.InputTag("pfRecoTauTagInfoProducer"),
    ECALIsolConeMetric = cms.string('DR'),
    ECALIsolConeSizeFormula = cms.string('0.50'),
    UseChargedHadrCandLeadChargedHadrCand_tksDZconstraint = cms.bool(True),
    JetPtMin = cms.double(15.0),
    LeadChargedHadrCand_minPt = cms.double(5.0),
    ECALSignalConeMetric = cms.string('DR'),
    TrackLeadTrack_maxDZ = cms.double(0.2),
    HCALSignalConeSize_max = cms.double(0.6),
    HCALIsolConeMetric = cms.string('DR'),
    TrackerIsolConeSizeFormula = cms.string('0.50'),
    HCALSignalConeSize_min = cms.double(0.0),
    ECALSignalConeSize_max = cms.double(0.6),
    HCALSignalConeSizeFormula = cms.string('0.10'),
    HCALIsolConeSize_max = cms.double(0.6),
    ChargedHadrCand_minPt = cms.double(1.0),
    UseTrackLeadTrackDZconstraint = cms.bool(True),
    smearedPVsigmaY = cms.double(0.0015),
    smearedPVsigmaX = cms.double(0.0015),
    smearedPVsigmaZ = cms.double(0.005),
    ECALIsolConeSize_min = cms.double(0.0),
    MatchingConeSize_max = cms.double(0.6),
    NeutrHadrCand_minPt = cms.double(1.0),
    HCALIsolConeSize_min = cms.double(0.0)
)

process.newSeedFromTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.5),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.conversionTrackCandidates = cms.EDProducer("ConversionTrackCandidateProducer",
    scHybridBarrelProducer = cms.string('correctedHybridSuperClusters'),
    inOutTrackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    maxHOverE = cms.double(0.2),
    scHybridBarrelCollection = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    inOutTrackCandidateCollection = cms.string('inOutTracksFromConversions'),
    scIslandEndcapCollection = cms.string(''),
    minSCEt = cms.double(5.0),
    MeasurementTrackerName = cms.string(''),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('alongMomElePropagator'),
        propagatorOppositeTISE = cms.string('oppositeToMomElePropagator')
    ),
    InOutRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    bcEndcapCollection = cms.string('islandEndcapBasicClusters'),
    outInTrackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),
    bcBarrelCollection = cms.string('islandBarrelBasicClusters'),
    scIslandEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower'),
    bcProducer = cms.string('islandBasicClusters'),
    OutInRedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    outInTrackCandidateCollection = cms.string('outInTracksFromConversions'),
    hbheInstance = cms.string(''),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForConversions'),
    hOverEConeSize = cms.double(0.1)
)

process.elecpreid = cms.EDProducer("GoodSeedProducer",
    ProduceCkfPFTracks = cms.untracked.bool(True),
    MaxEOverP = cms.double(3.0),
    Smoother = cms.string('GsfTrajectorySmoother_forPreId'),
    UseQuality = cms.bool(True),
    PFPSClusterLabel = cms.InputTag("particleFlowClusterPS"),
    ThresholdFile = cms.string('RecoParticleFlow/PFTracking/data/Threshold.dat'),
    TMVAMethod = cms.string('BDT'),
    MaxEta = cms.double(2.4),
    EtaMap = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_eta.dat'),
    PhiMap = cms.string('RecoParticleFlow/PFBlockProducer/data/resmap_ECAL_phi.dat'),
    PreCkfLabel = cms.string('SeedsForCkf'),
    NHitsInSeed = cms.int32(3),
    Fitter = cms.string('GsfTrajectoryFitter_forPreId'),
    PreGsfLabel = cms.string('SeedsForGsf'),
    MinEOverP = cms.double(0.3),
    Weights = cms.string('RecoParticleFlow/PFTracking/data/BDT_weights.txt'),
    PFEcalClusterLabel = cms.InputTag("particleFlowClusterECAL"),
    PSThresholdFile = cms.string('RecoParticleFlow/PFTracking/data/PSThreshold.dat'),
    MinPt = cms.double(2.0),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks"), cms.InputTag("secStep"), cms.InputTag("thStep")),
    UseTMVA = cms.untracked.bool(False),
    TrackQuality = cms.string('highPurity'),
    MaxPt = cms.double(50.0),
    ClusterThreshold = cms.double(0.5)
)

process.simpleSecondaryVertexBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("secondaryVertexTagInfos"),
    jetTagComputer = cms.string('simpleSecondaryVertex')
)

process.pixelTracks = cms.EDProducer("PixelTrackProducer",
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
            originHalfLength = cms.double(15.9),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
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

process.csc2DRecHits = cms.EDProducer("CSCRecHitDProducer",
    CSCStripClusterSize = cms.untracked.int32(3),
    CSCStripPeakThreshold = cms.untracked.double(10.0),
    ConstSyst = cms.untracked.double(0.03),
    readBadChannels = cms.bool(False),
    NoiseLevel = cms.untracked.double(7.0),
    CSCStripxtalksOffset = cms.untracked.double(0.03),
    CSCstripWireDeltaTime = cms.untracked.int32(8),
    CSCUseCalibrations = cms.untracked.bool(True),
    XTasymmetry = cms.untracked.double(0.005),
    CSCStripDigiProducer = cms.string('muonCSCDigis'),
    CSCWireDigiProducer = cms.string('muonCSCDigis'),
    CSCDebug = cms.untracked.bool(False),
    CSCproduce1DHits = cms.untracked.bool(False),
    CSCWireClusterDeltaT = cms.untracked.int32(1),
    CSCStripClusterChargeCut = cms.untracked.double(25.0)
)

process.cscSegments = cms.EDProducer("CSCSegmentProducer",
    inputObjects = cms.InputTag("csc2DRecHits"),
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
                preClustering = cms.untracked.bool(False),
                minHitsPerSegment = cms.int32(3),
                dPhiFineMax = cms.double(0.025),
                chi2Max = cms.double(5000.0),
                dXclusBoxMax = cms.double(8.0),
                tanThetaMax = cms.double(1.2),
                tanPhiMax = cms.double(0.5),
                maxRatioResidualPrune = cms.double(3.0),
                minHitsForPreClustering = cms.int32(10),
                CSCSegmentDebug = cms.untracked.bool(False),
                dRPhiFineMax = cms.double(8.0),
                nHitsPerClusterIsShower = cms.int32(20),
                minLayersApart = cms.int32(2),
                Pruning = cms.untracked.bool(False),
                dYclusBoxMax = cms.double(8.0)
            ), 
                cms.PSet(
                    preClustering = cms.untracked.bool(False),
                    minHitsPerSegment = cms.int32(3),
                    dPhiFineMax = cms.double(0.025),
                    chi2Max = cms.double(5000.0),
                    dXclusBoxMax = cms.double(8.0),
                    tanThetaMax = cms.double(2.0),
                    tanPhiMax = cms.double(0.8),
                    maxRatioResidualPrune = cms.double(3.0),
                    minHitsForPreClustering = cms.int32(10),
                    CSCSegmentDebug = cms.untracked.bool(False),
                    dRPhiFineMax = cms.double(12.0),
                    nHitsPerClusterIsShower = cms.int32(20),
                    minLayersApart = cms.int32(2),
                    Pruning = cms.untracked.bool(False),
                    dYclusBoxMax = cms.double(12.0)
                ), 
                cms.PSet(
                    preClustering = cms.untracked.bool(False),
                    minHitsPerSegment = cms.int32(3),
                    dPhiFineMax = cms.double(0.025),
                    chi2Max = cms.double(5000.0),
                    dXclusBoxMax = cms.double(8.0),
                    tanThetaMax = cms.double(1.2),
                    tanPhiMax = cms.double(0.5),
                    maxRatioResidualPrune = cms.double(3.0),
                    minHitsForPreClustering = cms.int32(30),
                    CSCSegmentDebug = cms.untracked.bool(False),
                    dRPhiFineMax = cms.double(8.0),
                    nHitsPerClusterIsShower = cms.int32(20),
                    minLayersApart = cms.int32(2),
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
                curvePenaltyThreshold = cms.untracked.double(0.85),
                minHitsPerSegment = cms.untracked.int32(3),
                yweightPenaltyThreshold = cms.untracked.double(1.0),
                curvePenalty = cms.untracked.double(2.0),
                dXclusBoxMax = cms.untracked.double(4.0),
                BrutePruning = cms.untracked.bool(False),
                yweightPenalty = cms.untracked.double(1.5),
                hitDropLimit5Hits = cms.untracked.double(0.8),
                preClustering = cms.untracked.bool(True),
                hitDropLimit4Hits = cms.untracked.double(0.6),
                hitDropLimit6Hits = cms.untracked.double(0.3333),
                maxRecHitsInCluster = cms.untracked.int32(20),
                CSCDebug = cms.untracked.bool(False),
                onlyBestSegment = cms.untracked.bool(False),
                Pruning = cms.untracked.bool(False),
                dYclusBoxMax = cms.untracked.double(8.0)
            ), 
                cms.PSet(
                    curvePenaltyThreshold = cms.untracked.double(0.85),
                    minHitsPerSegment = cms.untracked.int32(3),
                    yweightPenaltyThreshold = cms.untracked.double(1.0),
                    curvePenalty = cms.untracked.double(2.0),
                    dXclusBoxMax = cms.untracked.double(4.0),
                    BrutePruning = cms.untracked.bool(False),
                    yweightPenalty = cms.untracked.double(1.5),
                    hitDropLimit5Hits = cms.untracked.double(0.8),
                    preClustering = cms.untracked.bool(True),
                    hitDropLimit4Hits = cms.untracked.double(0.6),
                    hitDropLimit6Hits = cms.untracked.double(0.3333),
                    maxRecHitsInCluster = cms.untracked.int32(24),
                    CSCDebug = cms.untracked.bool(False),
                    onlyBestSegment = cms.untracked.bool(False),
                    Pruning = cms.untracked.bool(False),
                    dYclusBoxMax = cms.untracked.double(8.0)
                ))
        ))
)

process.muIsoDepositCalByAssociatorTowers = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("muons"),
        MultipleDepositsFlag = cms.bool(True),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        NoiseTow_EE = cms.double(0.15),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        Threshold_HO = cms.double(0.5),
        Noise_EB = cms.double(0.025),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_E = cms.double(0.2),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(False),
        Threshold_H = cms.double(0.5),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        DR_Veto_E = cms.double(0.07),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    )
)

process.dynamicHybridSuperClusters = cms.EDProducer("HybridClusterProducer",
    eThreshA = cms.double(0.003),
    posCalc_x0 = cms.double(0.89),
    clustershapecollection = cms.string(''),
    shapeAssociation = cms.string('dynamicHybridShapeAssoc'),
    ewing = cms.double(0.0),
    HybridBarrelSeedThr = cms.double(1.0),
    dynamicPhiRoad = cms.bool(True),
    basicclusterCollection = cms.string(''),
    posCalc_w0 = cms.double(4.2),
    posCalc_logweight = cms.bool(True),
    step = cms.int32(17),
    ecalhitcollection = cms.string('EcalRecHitsEB'),
    posCalc_t0 = cms.double(7.4),
    debugLevel = cms.string('INFO'),
    dynamicEThresh = cms.bool(True),
    eseed = cms.double(0.35),
    superclusterCollection = cms.string(''),
    ecalhitproducer = cms.string('ecalRecHit'),
    ethresh = cms.double(0.1),
    eThreshB = cms.double(0.1),
    bremRecoveryPset = cms.PSet(
        barrel = cms.PSet(
            cryVec = cms.vint32(17, 15, 13, 12, 11, 
                10, 9, 8, 7, 6),
            cryMin = cms.int32(5),
            etVec = cms.vdouble(5.0, 10.0, 15.0, 20.0, 30.0, 
                40.0, 45.0, 135.0, 195.0, 225.0)
        ),
        endcap = cms.PSet(
            a = cms.double(47.85),
            c = cms.double(0.1201),
            b = cms.double(108.8)
        )
    )
)

process.metOptNoHF = cms.EDProducer("METProducer",
    src = cms.InputTag("caloTowersOpt"),
    METType = cms.string('CaloMET'),
    alias = cms.string('RawCaloMETOptNoHF'),
    noHF = cms.bool(True),
    globalThreshold = cms.double(0.0),
    InputType = cms.string('CandidateCollection')
)

process.secTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('SecLayerTriplets'),
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
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(22.7),
            originZPos = cms.double(0.0),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.3),
            originXPos = cms.double(0.0),
            originRadius = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

process.pfNuclear = cms.EDProducer("PFNuclearProducer",
    likelihoodCut = cms.double(0.1),
    nuclearColList = cms.VInputTag(cms.InputTag("firstnuclearInteractionMaker"), cms.InputTag("secondnuclearInteractionMaker"), cms.InputTag("thirdnuclearInteractionMaker"), cms.InputTag("fourthnuclearInteractionMaker"))
)

process.softMuonNoIPBJetTags = cms.EDProducer("JetTagProducer",
    tagInfo = cms.InputTag("softMuonTagInfos"),
    jetTagComputer = cms.string('softMuonNoIP')
)

process.muParamGlobalIsoDepositCalHcal = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('track'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        DR_Veto_H = cms.double(0.1),
        Vertex_Constraint_Z = cms.bool(False),
        Threshold_H = cms.double(0.5),
        ComponentName = cms.string('CaloExtractor'),
        Threshold_E = cms.double(0.2),
        DR_Max = cms.double(1.0),
        DR_Veto_E = cms.double(0.07),
        Weight_E = cms.double(0.0),
        Vertex_Constraint_XY = cms.bool(False),
        DepositLabel = cms.untracked.string('EcalPlusHcal'),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        Weight_H = cms.double(1.0)
    )
)

process.muParamGlobalIsoDepositJets = cms.EDProducer("MuIsoDepositProducer",
    IOPSet = cms.PSet(
        ExtractForCandidate = cms.bool(False),
        inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
        MultipleDepositsFlag = cms.bool(False),
        MuonTrackRefType = cms.string('bestGlbTrkSta'),
        InputType = cms.string('MuonCollection')
    ),
    ExtractorPSet = cms.PSet(
        PrintTimeReport = cms.untracked.bool(False),
        ExcludeMuonVeto = cms.bool(True),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(0.5),
            dREcal = cms.double(0.5),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(0.5),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(0.5),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        ComponentName = cms.string('JetExtractor'),
        DR_Max = cms.double(1.0),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),
        DR_Veto = cms.double(0.1),
        Threshold = cms.double(5.0)
    )
)

process.secWithMaterialTracks = cms.EDProducer("TrackProducer",
    src = cms.InputTag("secTrackCandidates"),
    clusterRemovalInfo = cms.InputTag("secClusters"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

process.standAloneMuons = cms.EDProducer("StandAloneMuonProducer",
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
    InputObjects = cms.InputTag("MuonSeed"),
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
    ),
    STATrajBuilderParameters = cms.PSet(
        SeedPropagator = cms.string('SteppingHelixPropagatorAny'),
        NavigationType = cms.string('Standard'),
        SmootherParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            MaxChi2 = cms.double(25.0),
            Propagator = cms.string('SteppingHelixPropagatorAlong'),
            ErrorRescalingFactor = cms.double(10.0)
        ),
        SeedPosition = cms.string('in'),
        BWFilterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            FitDirection = cms.string('outsideIn'),
            MaxChi2 = cms.double(25.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(25.0),
                Granularity = cms.int32(2),
                RescaleErrorFactor = cms.double(100.0),
                RescaleError = cms.bool(False)
            ),
            EnableRPCMeasurement = cms.bool(True),
            BWSeedType = cms.string('fromGenerator'),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        RefitterParameters = cms.PSet(
            NumberOfSigma = cms.double(3.0),
            FitDirection = cms.string('insideOut'),
            MaxChi2 = cms.double(1000.0),
            MuonTrajectoryUpdatorParameters = cms.PSet(
                MaxChi2 = cms.double(1000.0),
                Granularity = cms.int32(0),
                RescaleErrorFactor = cms.double(100.0),
                RescaleError = cms.bool(False)
            ),
            EnableRPCMeasurement = cms.bool(True),
            CSCRecSegmentLabel = cms.InputTag("cscSegments"),
            EnableDTMeasurement = cms.bool(True),
            RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
            DTRecSegmentLabel = cms.InputTag("dt4DSegments"),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            EnableCSCMeasurement = cms.bool(True)
        ),
        DoSmoothing = cms.bool(False),
        DoBackwardRefit = cms.bool(True)
    )
)

process.preshowerClusterShape = cms.EDProducer("PreshowerClusterShapeProducer",
    preshStripEnergyCut = cms.double(0.0),
    preshRecHitProducer = cms.string('ecalPreshowerRecHit'),
    preshPi0Nstrip = cms.int32(5),
    endcapSClusterProducer = cms.string('islandSuperClusters'),
    PreshowerClusterShapeCollectionX = cms.string('preshowerXClustersShape'),
    PreshowerClusterShapeCollectionY = cms.string('preshowerYClustersShape'),
    endcapSClusterCollection = cms.string('islandEndcapSuperClusters'),
    debugLevel = cms.string('INFO'),
    preshRecHitCollection = cms.string('EcalRecHitsES')
)

process.particleFlowClusterHCAL = cms.EDProducer("PFClusterProducer",
    thresh_Seed_Endcap = cms.double(1.4),
    verbose = cms.untracked.bool(False),
    showerSigma = cms.double(10.0),
    thresh_Seed_Barrel = cms.double(1.4),
    depthCor_Mode = cms.int32(0),
    posCalcNCrystal = cms.int32(5),
    depthCor_B_preshower = cms.double(4.0),
    nNeighbours = cms.int32(4),
    PFRecHits = cms.InputTag("particleFlowRecHitHCAL"),
    thresh_Barrel = cms.double(0.8),
    depthCor_A_preshower = cms.double(0.89),
    depthCor_B = cms.double(7.4),
    depthCor_A = cms.double(0.89),
    thresh_Endcap = cms.double(0.8),
    posCalcP1 = cms.double(1.0)
)

process.horeco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HO'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True)
)

process.firstfilter = cms.EDFilter("QualityFilter",
    TrackQuality = cms.string('highPurity'),
    recTracks = cms.InputTag("generalTracks")
)

process.gtDigis = cms.EDFilter("L1GlobalTriggerRawToDigi",
    DaqGtFedId = cms.untracked.int32(813),
    DaqGtInputTag = cms.InputTag("rawDataCollector"),
    UnpackBxInEvent = cms.int32(-1),
    ActiveBoardsMask = cms.uint32(65535)
)

process.secClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

process.thTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('thPLSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('thCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.ALCARECOMuAlZMuMuHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonIso', 
        'HLT1MuonNonIso'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.secTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('secTriplets'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('secCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.correctedIslandBarrelSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('islandSuperClusters'),
    applyOldCorrection = cms.bool(True),
    applyEnergyCorrection = cms.bool(True),
    rawSuperClusterCollection = cms.string('islandBarrelSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.string('ecalRecHit')
)

process.newTrackCandidateMaker = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(True),
    doSeedingRegionRebuilding = cms.bool(True),
    SeedProducer = cms.string('newCombinedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('newTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.hbhereco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HBHE'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True)
)

process.ALCARECOTkAlZMuMu = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(True),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(True),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(True),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(110.0),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(70.0),
        applyChargeFilter = cms.bool(True),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(15.0),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.particleFlowJetCandidates = cms.EDFilter("PFJetCandidateCreator",
    src = cms.InputTag("particleFlow"),
    verbose = cms.untracked.bool(True)
)

process.siStripMatchedRecHits = cms.EDFilter("SiStripRecHitConverter",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Regional = cms.bool(False),
    stereoRecHits = cms.string('stereoRecHit'),
    Matcher = cms.string('StandardMatcher'),
    matchedRecHits = cms.string('matchedRecHit'),
    LazyGetterProducer = cms.string('SiStripRawToClustersFacility'),
    ClusterProducer = cms.string('siStripClusters'),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit')
)

process.withTightQuality = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("withLooseQuality"),
    keepAllTracks = cms.bool(True),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3),
    vtxChi2Prob = cms.double(0.01),
    copyTrajectories = cms.untracked.bool(True),
    vertices = cms.InputTag("pixelVertices"),
    qualityBit = cms.string('tight'),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True),
    minNumberLayers = cms.uint32(0),
    chi2n_par = cms.double(0.9),
    d0_par2 = cms.vdouble(0.4, 4.0),
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0)
)

process.correctedHybridSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('Hybrid'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('hybridSuperClusters'),
    applyEnergyCorrection = cms.bool(True),
    rawSuperClusterCollection = cms.string(''),
    VerbosityLevel = cms.string('ERROR'),
    hyb_fCorrPset = cms.PSet(
        brLinearThr = cms.double(12.0),
        fBremVec = cms.vdouble(-0.01258, 0.03154, 0.9888, -0.0007973, 1.59),
        fEtEtaVec = cms.vdouble(1.0, -0.8206, 3.16, 0.8637, 44.88, 
            2.292, 1.023, 8.0)
    ),
    recHitProducer = cms.string('ecalRecHit')
)

process.genParticlesForJets = cms.EDFilter("InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    ignoreParticleIDs = cms.vuint32(1000022, 2000012, 2000014, 2000016, 1000039, 
        5000039, 4000012, 9900012, 9900014, 9900016, 
        39),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(True),
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    tausAsJets = cms.bool(False)
)

process.iterativeCone5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.ecalDigis = cms.EDFilter("EcalRawToDigiDev",
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
    FedLabel = cms.untracked.string('listfeds'),
    srpUnpacking = cms.untracked.bool(True),
    syncCheck = cms.untracked.bool(False),
    headerUnpacking = cms.untracked.bool(True),
    feUnpacking = cms.untracked.bool(True),
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
    FEDs = cms.untracked.vint32(),
    DoRegional = cms.untracked.bool(False),
    memUnpacking = cms.untracked.bool(True)
)

process.secStep = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter2'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.4),
    recTracks = cms.InputTag("secWithMaterialTracks"),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(130.0),
    TrackQuality = cms.string('highPurity'),
    VertexCut = cms.bool(True)
)

process.pfRecoTauDiscriminationByIsolation = cms.EDFilter("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    PFTauProducer = cms.string('pfRecoTauProducer'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    TrackerIsolAnnulus_Candsmaxn = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn = cms.int32(0)
)

process.thPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    src = cms.InputTag("thClusters"),
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)
)

process.gsfElCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string('SeedsForGsf'),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('elecpreid'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfElectronTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.caloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMaker"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

process.roadSearchClouds = cms.EDFilter("RoadSearchCloudMaker",
    MinimalFractionOfUsedLayersPerCloud = cms.double(0.5),
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    StraightLineNoBeamSpotCloud = cms.bool(False),
    UsePixelsinRS = cms.bool(True),
    SeedProducer = cms.InputTag("roadSearchSeeds"),
    DoCloudCleaning = cms.bool(True),
    IncreaseMaxNumberOfConsecutiveMissedLayersPerCloud = cms.uint32(4),
    rphiStripRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    UseStereoRecHits = cms.bool(False),
    ZPhiRoadSize = cms.double(0.06),
    MaximalFractionOfConsecutiveMissedLayersPerCloud = cms.double(0.15),
    stereoStripRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    MaximalFractionOfMissedLayersPerCloud = cms.double(0.3),
    scalefactorRoadSeedWindow = cms.double(1.5),
    MaxDetHitsInCloudPerDetId = cms.uint32(32),
    IncreaseMaxNumberOfMissedLayersPerCloud = cms.uint32(3),
    RoadsLabel = cms.string(''),
    MaxRecHitsInCloud = cms.int32(100),
    UseRphiRecHits = cms.bool(False),
    MergingFraction = cms.double(0.8),
    RPhiRoadSize = cms.double(0.02),
    matchedStripRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    MinimumHalfRoad = cms.double(0.55)
)

process.roadSearchSeeds = cms.EDFilter("RoadSearchSeedFinder",
    OuterSeedRecHitAccessMode = cms.string('RPHI'),
    pixelRecHits = cms.InputTag("siPixelRecHits"),
    MaximalEndcapImpactParameter = cms.double(1.2),
    MergeSeedsCenterCut_C = cms.double(0.4),
    MergeSeedsCenterCut_B = cms.double(0.25),
    MergeSeedsCenterCut_A = cms.double(0.05),
    MergeSeedsDifferentHitsCut = cms.uint32(1),
    rphiStripRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    MaximalBarrelImpactParameter = cms.double(0.2),
    stereoStripRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    RoadsLabel = cms.string(''),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    OuterSeedRecHitAccessUseStereo = cms.bool(False),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    MinimalReconstructedTransverseMomentum = cms.double(1.5),
    PhiRangeForDetIdLookupInRings = cms.double(0.5),
    Mode = cms.string('STANDARD'),
    CosmicTracking = cms.bool(False),
    MergeSeedsRadiusCut_A = cms.double(0.05),
    InnerSeedRecHitAccessMode = cms.string('RPHI'),
    InnerSeedRecHitAccessUseStereo = cms.bool(False),
    OuterSeedRecHitAccessUseRPhi = cms.bool(False),
    MergeSeedsRadiusCut_B = cms.double(0.25),
    MergeSeedsRadiusCut_C = cms.double(0.4),
    matchedStripRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    InnerSeedRecHitAccessUseRPhi = cms.bool(False)
)

process.generalTracks = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("withTightQuality"),
    keepAllTracks = cms.bool(True),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3),
    vtxChi2Prob = cms.double(0.01),
    copyTrajectories = cms.untracked.bool(True),
    vertices = cms.InputTag("pixelVertices"),
    qualityBit = cms.string('highPurity'),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True),
    minNumberLayers = cms.uint32(5),
    chi2n_par = cms.double(0.9),
    d0_par2 = cms.vdouble(0.4, 4.0),
    d0_par1 = cms.vdouble(0.3, 4.0),
    dz_par1 = cms.vdouble(0.35, 4.0),
    dz_par2 = cms.vdouble(0.4, 4.0)
)

process.ALCARECOTkAlUpsilonMuMu = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(True),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(True),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(9.8),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(9.25),
        applyChargeFilter = cms.bool(True),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(0.8),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.isoMuonHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonIso'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.ckfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('globalMixedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('GroupedCkfTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.kt4JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("kt4CaloJets"),
    jet2TracksAtCALO = cms.InputTag("kt4JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("kt4JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

process.siStripClusters = cms.EDFilter("SiStripClusterizer",
    MaxHolesInCluster = cms.int32(0),
    ChannelThreshold = cms.double(2.0),
    DigiProducersList = cms.VPSet(cms.PSet(
        DigiLabel = cms.string('ZeroSuppressed'),
        DigiProducer = cms.string('siStripDigis')
    ), 
        cms.PSet(
            DigiLabel = cms.string('VirginRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ProcessedRaw'),
            DigiProducer = cms.string('siStripZeroSuppression')
        ), 
        cms.PSet(
            DigiLabel = cms.string('ScopeMode'),
            DigiProducer = cms.string('siStripZeroSuppression')
        )),
    ClusterMode = cms.string('ThreeThresholdClusterizer'),
    SeedThreshold = cms.double(3.0),
    SiStripQualityLabel = cms.string(''),
    ClusterThreshold = cms.double(5.0)
)

process.ALCARECOTkAlUpsilonMuMuHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2MuonUpsilon'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.muonRPCDigis = cms.EDFilter("RPCUnpackingModule",
    InputLabel = cms.untracked.InputTag("rawDataCollector")
)

process.standAloneMuonsMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("trackingtruthprod"),
    tracks = cms.InputTag("standAloneMuons"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)

process.egammaCkfTrackCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('electronPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('TrajectoryBuilderForPixelMatchGsfElectrons'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.ALCARECOMuAlZMuMu = cms.EDFilter("AlignmentMuonSelectorModule",
    chi2nMaxSA = cms.double(9999999.0),
    nHitMaxTO = cms.double(9999999.0),
    nHitMaxGB = cms.double(9999999.0),
    applyMultiplicityFilter = cms.bool(False),
    etaMin = cms.double(-2.4),
    minMassPair = cms.double(89.0),
    etaMax = cms.double(2.4),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    ptMin = cms.double(10.0),
    minMultiplicity = cms.int32(1),
    applyNHighestPt = cms.bool(False),
    nHitMaxSA = cms.double(9999999.0),
    ptMax = cms.double(9999.0),
    nHitMinSA = cms.double(0.0),
    chi2nMaxTO = cms.double(9999999.0),
    chi2nMaxGB = cms.double(9999999.0),
    nHighestPt = cms.int32(2),
    applyMassPairFilter = cms.bool(False),
    src = cms.InputTag("muons"),
    nHitMinGB = cms.double(0.0),
    filter = cms.bool(True),
    maxMassPair = cms.double(90.0),
    nHitMinTO = cms.double(0.0),
    applyBasicCuts = cms.bool(True)
)

process.towerMakerPF = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(-1000.0),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(True),
    EESumThreshold = cms.double(-1000.0),
    HOThreshold = cms.double(999999.0),
    HBThreshold = cms.double(0.0),
    EBThreshold = cms.double(999999.0),
    HcalThreshold = cms.double(-1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(False),
    HF1Weight = cms.double(1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(999999.0),
    EEThreshold = cms.double(999999.0),
    HESThreshold = cms.double(0.0),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(999999.0),
    HEDThreshold = cms.double(0.0),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

process.ecalpi0CalibHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCaEcalPi0'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.VtxSmeared = cms.EDFilter("BetafuncEvtVtxGenerator",
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Emittance = cms.double(5.03e-08),
    SigmaZ = cms.double(5.3),
    Y0 = cms.double(0.0),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.0322),
    Z0 = cms.double(0.0)
)

process.muonCSCDigis = cms.EDFilter("CSCDCCUnpacker",
    PrintEventNumber = cms.untracked.bool(False),
    UseExaminer = cms.untracked.bool(False),
    ErrorMask = cms.untracked.uint32(3754946559),
    InputObjects = cms.InputTag("rawDataCollector"),
    ExaminerMask = cms.untracked.uint32(374076406),
    UnpackStatusDigis = cms.untracked.bool(False),
    isMTCCData = cms.untracked.bool(False),
    Debug = cms.untracked.bool(False)
)

process.l1MuonHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonLevel1'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)

process.particleFlowRecHitPS = cms.EDFilter("PFRecHitProducerPS",
    thresh_Barrel = cms.double(7e-06),
    ecalRecHitsES = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    thresh_Endcap = cms.double(7e-06),
    verbose = cms.untracked.bool(False)
)

process.ALCARECOSiStripCalMinBias = cms.EDFilter("CalibrationTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    ptMin = cms.double(0.8),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(False),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.thStripRecHits = cms.EDFilter("SiStripRecHitConverter",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Regional = cms.bool(False),
    stereoRecHits = cms.string('stereoRecHit'),
    Matcher = cms.string('StandardMatcher'),
    matchedRecHits = cms.string('matchedRecHit'),
    LazyGetterProducer = cms.string('SiStripRawToClustersFacility'),
    ClusterProducer = cms.string('thClusters'),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit')
)

process.ALCARECOTkAlCosmicsCTF = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-99.0),
    etaMax = cms.double(99.0),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(False),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(0.0),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(5.0),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(10.0),
    ptMax = cms.double(99999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(True),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(1),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("ctfWithMaterialTracksP5"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.secStripRecHits = cms.EDFilter("SiStripRecHitConverter",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    Regional = cms.bool(False),
    stereoRecHits = cms.string('stereoRecHit'),
    Matcher = cms.string('StandardMatcher'),
    matchedRecHits = cms.string('matchedRecHit'),
    LazyGetterProducer = cms.string('SiStripRawToClustersFacility'),
    ClusterProducer = cms.string('secClusters'),
    VerbosityLevel = cms.untracked.int32(1),
    rphiRecHits = cms.string('rphiRecHit')
)

process.CkfElectronCandidates = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool(False),
    doSeedingRegionRebuilding = cms.bool(False),
    SeedProducer = cms.string('globalMixedSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('CkfElectronTrajectoryBuilder'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

process.gctDigis = cms.EDFilter("GctRawToDigi",
    unpackEm = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    inputLabel = cms.InputTag("rawDataCollector"),
    unpackFibres = cms.untracked.bool(False),
    grenCompatibilityMode = cms.bool(False),
    gctFedId = cms.int32(745),
    unpackInternEm = cms.untracked.bool(False),
    unpackJets = cms.untracked.bool(True),
    unpackRct = cms.untracked.bool(True),
    hltMode = cms.bool(False),
    unpackEtSums = cms.untracked.bool(True)
)

process.trackMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("trackingtruthprod"),
    tracks = cms.InputTag("generalTracks"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)

process.dijetsHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2jetAve30', 
        'HLT2jetAve60'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.towerMaker = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    AllowMissingInputs = cms.untracked.bool(False),
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
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(1.4),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(1.2),
    HEDThreshold = cms.double(1.4),
    EcutTower = cms.double(-1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0)
)

process.withLooseQuality = cms.EDFilter("AnalyticalTrackSelector",
    src = cms.InputTag("preFilterCmsTracks"),
    keepAllTracks = cms.bool(False),
    beamspot = cms.InputTag("offlineBeamSpot"),
    vtxTracks = cms.uint32(3),
    vtxChi2Prob = cms.double(0.01),
    copyTrajectories = cms.untracked.bool(True),
    vertices = cms.InputTag("pixelVertices"),
    qualityBit = cms.string('loose'),
    vtxNumber = cms.int32(-1),
    copyExtras = cms.untracked.bool(True),
    minNumberLayers = cms.uint32(0),
    chi2n_par = cms.double(2.0),
    d0_par2 = cms.vdouble(0.55, 4.0),
    d0_par1 = cms.vdouble(0.55, 4.0),
    dz_par1 = cms.vdouble(0.65, 4.0),
    dz_par2 = cms.vdouble(0.45, 4.0)
)

process.particleFlowRecHitECAL = cms.EDFilter("PFRecHitProducerECAL",
    ecalRecHitsEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    thresh_Barrel = cms.double(0.08),
    verbose = cms.untracked.bool(False),
    thresh_Endcap = cms.double(0.3)
)

process.MEtoEDMConverter = cms.EDFilter("MEtoEDMConverter",
    Verbosity = cms.untracked.int32(0),
    Frequency = cms.untracked.int32(50),
    Name = cms.untracked.string('MEtoEDMConverter')
)

process.ecalPreshowerDigis = cms.EDFilter("ESRawToDigi",
    debugMode = cms.untracked.bool(False),
    InstanceES = cms.string(''),
    ESdigiCollection = cms.string(''),
    Label = cms.string('rawDataCollector')
)

process.correctedFixedMatrixSuperClustersWithPreshower = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('FixedMatrix'),
    recHitCollection = cms.string('EcalRecHitsEE'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('fixedMatrixSuperClustersWithPreshower'),
    applyEnergyCorrection = cms.bool(True),
    fix_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0, 0.0, 0.0),
        fEtEtaVec = cms.vdouble(0.0, 0.0, 0.0)
    ),
    rawSuperClusterCollection = cms.string(''),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.string('ecalRecHit')
)

process.ALCARECOTkAlJpsiMuMu = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(True),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(True),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(3.2),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(3.0),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(0.8),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.kt4JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("kt4CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.siStripDigis = cms.EDFilter("SiStripRawToDigiModule",
    ProductLabel = cms.untracked.string('rawDataCollector'),
    AppendedBytes = cms.untracked.int32(0),
    UseFedKey = cms.untracked.bool(False),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.untracked.int32(0),
    CreateDigis = cms.untracked.bool(True)
)

process.ALCARECOTkAlJpsiMuMuHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2MuonJPsi', 
        'HLTBJPsiMuMu'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.thStep = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter3'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.1),
    recTracks = cms.InputTag("thWithMaterialTracks"),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(250000.0),
    TrackQuality = cms.string('highPurity'),
    VertexCut = cms.bool(True)
)

process.ALCARECOTkAlMuonIsolated = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(True),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(True),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(0.0),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(11.0),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.softElectronTagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    leptons = cms.InputTag("btagSoftElectrons"),
    leptonQualityCut = cms.double(0.0),
    jets = cms.InputTag("iterativeCone5CaloJets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(9999.0)
)

process.globalMuonsMCMatch = cms.EDFilter("MCTrackMatcher",
    trackingParticles = cms.InputTag("trackingtruthprod"),
    tracks = cms.InputTag("globalMuons"),
    genParticles = cms.InputTag("genParticles"),
    associator = cms.string('TrackAssociatorByHits')
)

process.ALCARECOTkAlMinBias = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-3.5),
    etaMax = cms.double(3.5),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(False),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(0.0),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(1.5),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(0.0),
    ptMax = cms.double(999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(False),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(2),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("generalTracks"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.gammajetHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1Photon', 
        'HLT1PhotonL1Isolated'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.ALCARECOTkAlMuonIsolatedHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonIso'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.softMuonTagInfos = cms.EDFilter("SoftLepton",
    refineJetAxis = cms.uint32(0),
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    leptons = cms.InputTag("globalMuons"),
    leptonQualityCut = cms.double(0.5),
    jets = cms.InputTag("iterativeCone5CaloJets"),
    leptonDeltaRCut = cms.double(0.4),
    leptonChi2Cut = cms.double(9999.0)
)

process.hcalminbiasHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCaHcalPhiSym'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.siPixelDigis = cms.EDFilter("SiPixelRawToDigi",
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.untracked.bool(False),
    InputLabel = cms.untracked.string('rawDataCollector'),
    CheckPixelOrder = cms.untracked.bool(False)
)

process.correctedDynamicHybridSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.03),
    superClusterAlgo = cms.string('DynamicHybrid'),
    recHitCollection = cms.string('EcalRecHitsEB'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('dynamicHybridSuperClusters'),
    applyEnergyCorrection = cms.bool(True),
    rawSuperClusterCollection = cms.string(''),
    VerbosityLevel = cms.string('ERROR'),
    dyn_fCorrPset = cms.PSet(
        brLinearThr = cms.double(12.0),
        fBremVec = cms.vdouble(-0.01762, 0.04224, 0.9793, 0.0008075, 1.774),
        fEtEtaVec = cms.vdouble(0.9929, -0.01751, -4.636, 5.945, 737.9, 
            4.057, 1.023, 8.0)
    ),
    recHitProducer = cms.string('ecalRecHit')
)

process.thClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secClusters"),
    trajectories = cms.InputTag("secStep"),
    pixelClusters = cms.InputTag("secClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("secClusters")
)

process.correctedIslandEndcapSuperClusters = cms.EDFilter("EgammaSCCorrectionMaker",
    corectedSuperClusterCollection = cms.string(''),
    sigmaElectronicNoise = cms.double(0.15),
    superClusterAlgo = cms.string('Island'),
    recHitCollection = cms.string('EcalRecHitsEE'),
    etThresh = cms.double(0.0),
    rawSuperClusterProducer = cms.string('islandSuperClusters'),
    applyEnergyCorrection = cms.bool(True),
    rawSuperClusterCollection = cms.string('islandEndcapSuperClusters'),
    isl_fCorrPset = cms.PSet(
        brLinearThr = cms.double(0.0),
        fBremVec = cms.vdouble(0.0),
        fEtEtaVec = cms.vdouble(0.0)
    ),
    VerbosityLevel = cms.string('ERROR'),
    recHitProducer = cms.string('ecalRecHit')
)

process.hcalDigis = cms.EDFilter("HcalRawToDigi",
    UnpackZDC = cms.untracked.bool(True),
    FilterDataQuality = cms.bool(True),
    ExceptionEmptyData = cms.untracked.bool(False),
    InputLabel = cms.InputTag("rawDataCollector"),
    ComplainEmptyData = cms.untracked.bool(False),
    UnpackCalib = cms.untracked.bool(True),
    lastSample = cms.int32(9),
    firstSample = cms.int32(0)
)

process.pfRecoTauDiscriminationHighEfficiency = cms.EDFilter("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    PFTauProducer = cms.string('pfRecoTauProducerHighEfficiency'),
    ManipulateTracks_insteadofChargedHadrCands = cms.bool(False),
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    TrackerIsolAnnulus_Candsmaxn = cms.int32(0),
    ECALIsolAnnulus_Candsmaxn = cms.int32(0)
)

process.electronFilter = cms.EDFilter("EtaPtMinPixelMatchGsfElectronFullCloneSelector",
    filter = cms.bool(True),
    src = cms.InputTag("pixelMatchGsfElectrons"),
    etaMin = cms.double(-2.7),
    etaMax = cms.double(2.7),
    ptMin = cms.double(5.0)
)

process.newCombinedSeeds = cms.EDFilter("SeedCombiner",
    TripletCollection = cms.untracked.string('newSeedFromTriplets'),
    PairCollection = cms.untracked.string('newSeedFromPairs')
)

process.sisCone5JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    jets = cms.InputTag("sisCone5CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.ALCARECOSiStripCalMinBiasHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLTMinBias', 
        'HLTMinBiasPixel'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.rsTrackCandidates = cms.EDFilter("RoadSearchTrackCandidateMaker",
    NumHitCut = cms.int32(5),
    InitialVertexErrorXY = cms.double(0.2),
    HitChi2Cut = cms.double(30.0),
    StraightLineNoBeamSpotCloud = cms.bool(False),
    MeasurementTrackerName = cms.string(''),
    MinimumChunkLength = cms.int32(7),
    TTRHBuilder = cms.string('WithTrackAngle'),
    CosmicTrackMerging = cms.bool(False),
    nFoundMin = cms.int32(4),
    CloudProducer = cms.InputTag("roadSearchClouds")
)

process.ALCARECOTkAlCosmicsCosmicTF = cms.EDFilter("AlignmentTrackSelectorModule",
    minHitChargeStrip = cms.double(20.0),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    applyMultiplicityFilter = cms.bool(False),
    matchedrecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    etaMin = cms.double(-99.0),
    etaMax = cms.double(99.0),
    minHitIsolation = cms.double(0.01),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    GlobalSelector = cms.PSet(
        applyIsolationtest = cms.bool(False),
        muonSource = cms.InputTag("muons"),
        minIsolatedCount = cms.int32(0),
        jetIsoSource = cms.InputTag("kt6CaloJets"),
        minJetDeltaR = cms.double(0.2),
        minJetPt = cms.double(40.0),
        maxJetCount = cms.int32(3),
        applyGlobalMuonFilter = cms.bool(False),
        maxTrackDeltaR = cms.double(0.001),
        jetCountSource = cms.InputTag("kt6CaloJets"),
        maxJetPt = cms.double(40.0),
        applyJetCountFilter = cms.bool(False),
        minGlobalMuonCount = cms.int32(1)
    ),
    TwoBodyDecaySelector = cms.PSet(
        applyMassrangeFilter = cms.bool(False),
        daughterMass = cms.double(0.105),
        useUnsignedCharge = cms.bool(True),
        missingETSource = cms.InputTag("met"),
        maxXMass = cms.double(15000.0),
        charge = cms.int32(0),
        acoplanarDistance = cms.double(1.0),
        minXMass = cms.double(0.0),
        applyChargeFilter = cms.bool(False),
        applyAcoplanarityFilter = cms.bool(False),
        applyMissingETFilter = cms.bool(False)
    ),
    ptMin = cms.double(5.0),
    minMultiplicity = cms.int32(1),
    nHitMin = cms.double(14.0),
    ptMax = cms.double(99999.0),
    nHitMax = cms.double(999.0),
    applyNHighestPt = cms.bool(True),
    applyChargeCheck = cms.bool(False),
    minHitsPerSubDet = cms.PSet(
        inTEC = cms.int32(0),
        inTOB = cms.int32(0),
        inFPIX = cms.int32(0),
        inTID = cms.int32(0),
        inBPIX = cms.int32(0),
        inTIB = cms.int32(0)
    ),
    nHighestPt = cms.int32(1),
    nHitMin2D = cms.uint32(0),
    src = cms.InputTag("cosmictrackfinderP5"),
    applyIsolationCut = cms.bool(False),
    multiplicityOnInput = cms.bool(False),
    filter = cms.bool(True),
    maxMultiplicity = cms.int32(999999),
    seedOnlyFrom = cms.int32(0),
    chi2nMax = cms.double(999999.0),
    applyBasicCuts = cms.bool(True)
)

process.ALCARECOMuAlOverlapsHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonIso', 
        'HLT1MuonNonIso'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.isoHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCaIsoTrack'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.particleFlowRecHitHCAL = cms.EDFilter("PFRecHitProducerHCAL",
    hcalRecHitsHBHE = cms.InputTag(""),
    thresh_Barrel = cms.double(0.8),
    verbose = cms.untracked.bool(False),
    thresh_Endcap = cms.double(0.8),
    caloTowers = cms.InputTag("towerMakerPF")
)

process.ckfInOutTracksFromConversions = cms.EDFilter("TrackProducerWithSCAssociation",
    src = cms.InputTag("conversionTrackCandidates","inOutTracksFromConversions"),
    recoTrackSCAssociationCollection = cms.string('inOutTrackSCAssociationCollection'),
    producer = cms.string('conversionTrackCandidates'),
    Fitter = cms.string('KFFittingSmootherForInOut'),
    useHitsSplitting = cms.bool(False),
    trackCandidateSCAssociationCollection = cms.string('inOutTrackCandidateSCAssociationCollection'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    ComponentName = cms.string('ckfInOutTracksFromConversions'),
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot")
)

process.caloRecoTauDiscriminationByIsolation = cms.EDFilter("CaloRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    CaloTauProducer = cms.string('caloRecoTauProducer'),
    TrackerIsolAnnulus_Tracksmaxn = cms.int32(0)
)

process.muonDTDigis = cms.EDFilter("DTUnpackingModule",
    dataType = cms.string('DDU'),
    fedColl = cms.untracked.string('rawDataCollector'),
    fedbyType = cms.untracked.bool(False),
    useStandardFEDid = cms.untracked.bool(True),
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
    )
)

process.ewkHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1Electron', 
        'HLT2Electron', 
        'HLT1ElectronRelaxed', 
        'HLT2ElectronRelaxed'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.ALCARECOMuAlOverlapsMuonSelector = cms.EDFilter("AlignmentMuonSelectorModule",
    chi2nMaxSA = cms.double(9999999.0),
    nHitMaxTO = cms.double(9999999.0),
    nHitMaxGB = cms.double(9999999.0),
    applyMultiplicityFilter = cms.bool(False),
    etaMin = cms.double(-2.4),
    minMassPair = cms.double(89.0),
    etaMax = cms.double(2.4),
    phiMax = cms.double(3.1416),
    phiMin = cms.double(-3.1416),
    ptMin = cms.double(3.0),
    minMultiplicity = cms.int32(1),
    applyNHighestPt = cms.bool(False),
    nHitMaxSA = cms.double(9999999.0),
    ptMax = cms.double(9999.0),
    nHitMinSA = cms.double(0.0),
    chi2nMaxTO = cms.double(9999999.0),
    chi2nMaxGB = cms.double(9999999.0),
    nHighestPt = cms.int32(2),
    applyMassPairFilter = cms.bool(False),
    src = cms.InputTag("muons"),
    nHitMinGB = cms.double(0.0),
    filter = cms.bool(True),
    maxMassPair = cms.double(90.0),
    nHitMinTO = cms.double(0.0),
    applyBasicCuts = cms.bool(True)
)

process.sisCone5JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("sisCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("sisCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("sisCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

process.hfreco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('HF'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False)
)

process.caloTowersOpt = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("calotoweroptmaker"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

process.genCandidatesForMET = cms.EDFilter("GenJetParticleRefSelector",
    includeList = cms.vstring(),
    src = cms.InputTag("genParticles"),
    stableOnly = cms.bool(True),
    verbose = cms.untracked.bool(True),
    excludeList = cms.vstring('nu_e', 
        'nu_mu', 
        'nu_tau', 
        'mu-', 
        '~chi_10', 
        '~nu_eR', 
        '~nu_muR', 
        '~nu_tauR', 
        'Graviton', 
        '~Gravitino', 
        'nu_Re', 
        'nu_Rmu', 
        'nu_Rtau', 
        'nu*_e0', 
        'Graviton*')
)

process.iterativeCone5JetExtender = cms.EDFilter("JetExtender",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    jet2TracksAtCALO = cms.InputTag("iterativeCone5JetTracksAssociatorAtCaloFace"),
    jet2TracksAtVX = cms.InputTag("iterativeCone5JetTracksAssociatorAtVertex"),
    coneSize = cms.double(0.5)
)

process.siStripZeroSuppression = cms.EDFilter("SiStripZeroSuppression",
    RawDigiProducersList = cms.VPSet(cms.PSet(
        RawDigiProducer = cms.string('siStripDigis'),
        RawDigiLabel = cms.string('VirginRaw')
    ), 
        cms.PSet(
            RawDigiProducer = cms.string('siStripDigis'),
            RawDigiLabel = cms.string('ProcessedRaw')
        ), 
        cms.PSet(
            RawDigiProducer = cms.string('siStripDigis'),
            RawDigiLabel = cms.string('ScopeMode')
        )),
    FEDalgorithm = cms.uint32(4),
    ZeroSuppressionMode = cms.string('SiStripFedZeroSuppression'),
    CutToAvoidSignal = cms.double(3.0),
    CommonModeNoiseSubtractionMode = cms.string('Median')
)

process.allTrackMCMatch = cms.EDFilter("GenParticleMatchMerger",
    src = cms.VInputTag(cms.InputTag("trackMCMatch"), cms.InputTag("standAloneMuonsMCMatch"), cms.InputTag("globalMuonsMCMatch"))
)

process.sisCone5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("sisCone5CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.ALCARECOTkAlMinBiasHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLTMinBias', 
        'HLTMinBiasPixel'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.caloTowersPF = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("towerMakerPF"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

process.kt4JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    jets = cms.InputTag("kt4CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.globalCombinedSeeds = cms.EDFilter("SeedCombiner",
    TripletCollection = cms.untracked.string('globalSeedsFromTripletsWithVertices'),
    PairCollection = cms.untracked.string('globalSeedsFromPairsWithVertices')
)

process.dttfDigis = cms.EDFilter("DTTFFEDReader",
    DTTF_FED_Source = cms.InputTag("rawDataCollector")
)

process.siPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    src = cms.InputTag("siPixelClusters"),
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)
)

process.ic5PFJetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("iterativeCone5PFJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.calotoweroptmaker = cms.EDFilter("CaloTowersCreator",
    EBSumThreshold = cms.double(0.2),
    EBWeight = cms.double(1.0),
    hfInput = cms.InputTag("hfreco"),
    EESumThreshold = cms.double(0.45),
    HOThreshold = cms.double(0.5),
    HBGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HBThreshold = cms.double(0.5),
    EBThreshold = cms.double(0.09),
    EEWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HcalThreshold = cms.double(-1000.0),
    HF2Weights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    EEGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeight = cms.double(1.0),
    EEWeight = cms.double(1.0),
    UseHO = cms.bool(True),
    HBWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HF1Weight = cms.double(1.0),
    HF2Grid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HEDWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HF1Grid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HOWeight = cms.double(1.0),
    HESWeight = cms.double(1.0),
    hbheInput = cms.InputTag("hbhereco"),
    HF2Weight = cms.double(1.0),
    HF2Threshold = cms.double(1.8),
    EEThreshold = cms.double(0.45),
    HESThreshold = cms.double(0.7),
    HF1Weights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    hoInput = cms.InputTag("horeco"),
    HF1Threshold = cms.double(1.2),
    HESGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    HESWeights = cms.untracked.vdouble(1.0, 1.0, 1.0, 1.0, 1.0),
    HEDThreshold = cms.double(0.5),
    EcutTower = cms.double(-1000.0),
    HEDGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE")),
    HBWeight = cms.double(1.0),
    HOGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0),
    EBGrid = cms.untracked.vdouble(-1.0, 1.0, 10.0, 100.0, 1000.0)
)

process.csctfDigis = cms.EDFilter("CSCTFUnpacker",
    mappingFile = cms.string(''),
    producer = cms.untracked.InputTag("rawDataCollector"),
    MaxBX = cms.int32(9),
    slot2sector = cms.vint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0),
    swapME1strips = cms.bool(True),
    MinBX = cms.int32(3)
)

process.ALCARECOTkAlZMuMuHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT2MuonZ'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.ckfOutInTracksFromConversions = cms.EDFilter("TrackProducerWithSCAssociation",
    src = cms.InputTag("conversionTrackCandidates","outInTracksFromConversions"),
    recoTrackSCAssociationCollection = cms.string('outInTrackSCAssociationCollection'),
    producer = cms.string('conversionTrackCandidates'),
    Fitter = cms.string('KFFitterForOutIn'),
    useHitsSplitting = cms.bool(False),
    trackCandidateSCAssociationCollection = cms.string('outInTrackCandidateSCAssociationCollection'),
    TrajectoryInEvent = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    AlgorithmName = cms.string('undefAlgorithm'),
    ComponentName = cms.string('ckfOutInTracksFromConversions'),
    Propagator = cms.string('alongMomElePropagator'),
    beamSpot = cms.InputTag("offlineBeamSpot")
)

process.siStripElectronToTrackAssociator = cms.EDFilter("SiStripElectronAssociator",
    siStripElectronCollection = cms.string(''),
    trackCollection = cms.string(''),
    electronsLabel = cms.string('siStripElectrons'),
    siStripElectronProducer = cms.string('siStripElectrons'),
    trackProducer = cms.string('egammaCTFFinalFitWithMaterial')
)

process.ic5JetTracksAssociatorAtVertex = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.iterativeCone5JetTracksAssociatorAtCaloFace = cms.EDFilter("JetTracksAssociatorAtCaloFace",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)

process.coneIsolationTauJetTags = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(0.0),
    MaximumTransverseImpactParameter = cms.double(0.03),
    VariableConeParameter = cms.double(3.5),
    useVertex = cms.bool(True),
    MinimumNumberOfHits = cms.int32(8),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("ic5JetTracksAssociatorAtVertex"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.07),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(False),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
    IsolationCone = cms.double(0.45),
    MinimumTransverseMomentumLeadingTrack = cms.double(6.0),
    MaximumChiSquared = cms.double(100.0)
)

process.ALCARECOMuAlOverlaps = cms.EDFilter("AlignmentCSCOverlapSelectorModule",
    filter = cms.bool(True),
    src = cms.InputTag("ALCARECOMuAlOverlapsMuonSelector","GlobalMuon"),
    minHitsPerChamber = cms.uint32(4),
    station = cms.int32(0)
)

process.secPixelRecHits = cms.EDFilter("SiPixelRecHitConverter",
    eff_charge_cut_lowY = cms.untracked.double(0.0),
    src = cms.InputTag("secClusters"),
    eff_charge_cut_lowX = cms.untracked.double(0.0),
    eff_charge_cut_highX = cms.untracked.double(1.0),
    eff_charge_cut_highY = cms.untracked.double(1.0),
    size_cutY = cms.untracked.double(3.0),
    size_cutX = cms.untracked.double(3.0),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
    TanLorentzAnglePerTesla = cms.double(0.106),
    Alpha2Order = cms.bool(True),
    speed = cms.int32(0)
)

process.ecalphiSymHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCaEcalPhiSym'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)

process.globalrechitsanalyze = cms.EDAnalyzer("GlobalRecHitsAnalyzer",
    MuDTSrc = cms.InputTag("dt1DRecHits"),
    SiPxlSrc = cms.InputTag("siPixelRecHits"),
    VtxUnit = cms.untracked.int32(1),
    associateRecoTracks = cms.bool(False),
    MuDTSimSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof', 
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    ECalEESrc = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    MuRPCSimSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    SiStripSrc = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalESSrc = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    ECalUncalEESrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEE"),
    Name = cms.untracked.string('GlobalRecHitsAnalyzer'),
    Verbosity = cms.untracked.int32(0),
    associateStrip = cms.bool(True),
    MuRPCSrc = cms.InputTag("rpcRecHits"),
    ECalUncalEBSrc = cms.InputTag("ecalWeightUncalibRecHit","EcalUncalibRecHitsEB"),
    MuCSCSrc = cms.InputTag("csc2DRecHits"),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)

process.globaldigisanalyze = cms.EDAnalyzer("GlobalDigisAnalyzer",
    MuCSCStripSrc = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
    MuDTSrc = cms.InputTag("muonDTDigis"),
    Name = cms.untracked.string('GlobalDigisAnalyzer'),
    MuCSCWireSrc = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
    Verbosity = cms.untracked.int32(0),
    ECalEESrc = cms.InputTag("ecalDigis","eeDigis"),
    SiStripSrc = cms.InputTag("siStripDigis","ZeroSuppressed"),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    SiPxlSrc = cms.InputTag("siPixelDigis"),
    Frequency = cms.untracked.int32(50),
    MuRPCSrc = cms.InputTag("muonRPCDigis"),
    ECalEBSrc = cms.InputTag("ecalDigis","ebDigis"),
    ECalESSrc = cms.InputTag("ecalPreshowerDigis"),
    VtxUnit = cms.untracked.int32(1),
    HCalDigi = cms.InputTag("hcalUnsuppressedDigis")
)

process.globalhitsanalyze = cms.EDAnalyzer("GlobalHitsAnalyzer",
    MuonRpcSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    PxlBrlHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelHighTof"),
    SiTOBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"),
    SiTECHighSrc = cms.InputTag("g4SimHits","TrackerHitsTECHighTof"),
    PxlFwdHighSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapHighTof"),
    HCalSrc = cms.InputTag("g4SimHits","HcalHits"),
    ECalEESrc = cms.InputTag("g4SimHits","EcalHitsEE"),
    SiTIBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"),
    SiTECLowSrc = cms.InputTag("g4SimHits","TrackerHitsTECLowTof"),
    MuonCscSrc = cms.InputTag("g4SimHits","MuonCSCHits"),
    SiTIDHighSrc = cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"),
    Name = cms.untracked.string('GlobalHitsAnalyzer'),
    Verbosity = cms.untracked.int32(0),
    PxlFwdLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"),
    PxlBrlLowSrc = cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
    SiTIBLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"),
    SiTOBHighSrc = cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"),
    VtxUnit = cms.untracked.int32(1),
    ECalESSrc = cms.InputTag("g4SimHits","EcalHitsES"),
    SiTIDLowSrc = cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"),
    MuonDtSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    Frequency = cms.untracked.int32(50),
    ECalEBSrc = cms.InputTag("g4SimHits","EcalHitsEB")
)

process.out_step = cms.OutputModule("PoolOutputModule",
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep *_g4SimHits_*_*', 'keep edmHepMCProduct_source_*_*', 'keep recoGenJets_*_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep recoGenMETs_*_*_*', 'keep *_siPixelDigis_*_*', 'keep *_siStripDigis_*_*', 'keep *_trackMCMatch_*_*', 'keep *_muonCSCDigis_*_*', 'keep *_muonDTDigis_*_*', 'keep *_muonRPCDigis_*_*', 'keep *_ecalDigis_*_*', 'keep *_ecalPreshowerDigis_*_*', 'keep *_ecalTriggerPrimitiveDigis_*_*', 'keep *_hcalDigis_*_*', 'keep *_hcalTriggerPrimitiveDigis_*_*', 'keep *_cscTriggerPrimitiveDigis_*_*', 'keep *_dtTriggerPrimitiveDigis_*_*', 'keep *_rpcTriggerDigis_*_*', 'keep *_rctDigis_*_*', 'keep *_csctfDigis_*_*', 'keep *_dttfDigis_*_*', 'keep *_gctDigis_*_*', 'keep *_gmtDigis_*_*', 'keep *_gtDigis_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep *_MEtoEDMConverter_*_*')),
    fileName = cms.untracked.string('SingleMuPt1.cfi__RAW2DIGI,RECO.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring(),
    debugs = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    suppressDebug = cms.untracked.vstring(),
    cout = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    warnings = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    default = cms.untracked.PSet(

    ),
    errors = cms.untracked.PSet(
        placeholder = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noTimeStamps = cms.untracked.bool(False),
        FwkReport = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkSummary = cms.untracked.PSet(
            reportEvery = cms.untracked.int32(1),
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO')
    ),
    FrameworkJobReport = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    suppressWarning = cms.untracked.vstring(),
    statistics = cms.untracked.vstring('cerr'),
    infos = cms.untracked.PSet(
        Root_NoDictionary = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        placeholder = cms.untracked.bool(True)
    ),
    destinations = cms.untracked.vstring('warnings', 
        'errors', 
        'infos', 
        'debugs', 
        'cout', 
        'cerr'),
    debugModules = cms.untracked.vstring(),
    categories = cms.untracked.vstring('FwkJob', 
        'FwkReport', 
        'FwkSummary', 
        'Root_NoDictionary'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)

process.DQMStore = cms.Service("DQMStore")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        ecalUnsuppressedDigis = cms.untracked.uint32(1234567),
        muonCSCDigis = cms.untracked.uint32(11223344),
        mix = cms.untracked.uint32(12345),
        hcalUnsuppressedDigis = cms.untracked.uint32(11223344),
        VtxSmeared = cms.untracked.uint32(98765432),
        siPixelDigis = cms.untracked.uint32(1234567),
        muonDTDigis = cms.untracked.uint32(1234567),
        siStripDigis = cms.untracked.uint32(1234567),
        muonRPCDigis = cms.untracked.uint32(1234567)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)

process.Chi2MeasurementEstimator = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0)
)

process.CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL', 
        'ZDC', 
        'EcalBarrel', 
        'EcalEndcap', 
        'EcalPreshower', 
        'TOWER')
)

process.softMuon = cms.ESProducer("MuonTaggerESProducer")

process.KFSmootherForRefitInsideOut = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

process.SmartPropagatorAnyRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyRK'),
    TrackerPropagator = cms.string('RKTrackerPropagator'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

process.TrajectoryFilterForConversions = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minHitsMinPt = cms.int32(-1),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        minimumNumberOfHits = cms.int32(3),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(0.9)
    ),
    ComponentName = cms.string('TrajectoryFilterForConversions')
)

process.GsfTrajectorySmoother_forPreId = cms.ESProducer("GsfTrajectorySmootherESProducer",
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    ComponentName = cms.string('GsfTrajectorySmoother_forPreId'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator')
)

process.KFTrajectoryFitterForInOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForInOut'),
    Estimator = cms.string('Chi2ForInOut'),
    Propagator = cms.string('alongMomElePropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.TrackAssociatorByChi2ESProducer = cms.ESProducer("TrackAssociatorByChi2ESProducer",
    chi2cut = cms.double(25.0),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    onlyDiagonal = cms.bool(False)
)

process.ttrhbwor = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('WithoutRefit'),
    PixelCPE = cms.string('Fake'),
    Matcher = cms.string('Fake')
)

process.secCkfTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minPt = cms.double(0.3),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        nSigmaMinPt = cms.double(5.0),
        minimumNumberOfHits = cms.int32(3)
    ),
    ComponentName = cms.string('secCkfTrajectoryFilter')
)

process.CloseComponentsMerger5D = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger5D'),
    MaxComponents = cms.int32(12),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D')
)

process.CkfTrajectoryBuilder = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('CkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    lostHitPenalty = cms.double(30.0)
)

process.l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
    TriggerMenuLuminosity = cms.string('lumi1x1032'),
    DefXmlFile = cms.string('L1Menu2007.xml'),
    VmeXmlFile = cms.string('')
)

process.EcalEndcapGeometryEP = cms.ESProducer("EcalEndcapGeometryEP")

process.L1MuTriggerScales = cms.ESProducer("L1MuTriggerScalesProducer")

process.hcalDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('HcalDetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(70),
    nPhi = cms.int32(72)
)

process.CloseComponentsMerger_forPreId = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger_forPreId'),
    MaxComponents = cms.int32(4),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D')
)

process.alongMomElePropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('alongMomElePropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)

process.HcalHardcodeGeometryEP = cms.ESProducer("HcalHardcodeGeometryEP")

process.TrackerDigiGeometryESModule = cms.ESProducer("TrackerDigiGeometryESModule",
    fromDDD = cms.bool(True),
    applyAlignment = cms.untracked.bool(True)
)

process.SmartPropagatorRK = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorRK'),
    TrackerPropagator = cms.string('RKTrackerPropagator'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

process.SmartPropagator = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagator'),
    TrackerPropagator = cms.string('PropagatorWithMaterial'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAlong'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

process.RKTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('RKFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.CaloTowerConstituentsMapBuilder = cms.ESProducer("CaloTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/CaloTopology/data/CaloTowerEEGeometric.map.gz')
)

process.oppositeToMomElePropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('oppositeToMomElePropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False)
)

process.ecalDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('EcalDetIdAssociator'),
    etaBinSize = cms.double(0.02),
    nEta = cms.int32(300),
    nPhi = cms.int32(360)
)

process.muonDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('MuonDetIdAssociator'),
    etaBinSize = cms.double(0.125),
    nEta = cms.int32(48),
    nPhi = cms.int32(48)
)

process.EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP")

process.SiStripGainESProducer = cms.ESProducer("SiStripGainESProducer",
    printDebug = cms.untracked.bool(False),
    NormalizationFactor = cms.double(1.0),
    AutomaticNormalization = cms.bool(False)
)

process.bwdGsfElectronPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('bwdGsfElectronPropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False)
)

process.TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)

process.TTRHBuilderAngleAndTemplate = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithAngleAndTemplate'),
    PixelCPE = cms.string('PixelCPETemplateReco'),
    Matcher = cms.string('StandardMatcher')
)

process.EcalTrigTowerConstituentsMapBuilder = cms.ESProducer("EcalTrigTowerConstituentsMapBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EndCap_TTMap.txt')
)

process.myTTRHBuilderWithoutAngle4MixedTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.MuonDetLayerGeometryESProducer = cms.ESProducer("MuonDetLayerGeometryESProducer")

process.OppositeMaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('PropagatorWithMaterialOpposite'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False)
)

process.l1GtParameters = cms.ESProducer("L1GtParametersTrivialProducer",
    DaqActiveBoards = cms.uint32(65535),
    TotalBxInEvent = cms.int32(3),
    EvmActiveBoards = cms.uint32(65535)
)

process.l1GtTriggerMaskTechTrig = cms.ESProducer("L1GtTriggerMaskTechTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0)
)

process.myTTRHBuilderWithoutAngle = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('PixelTTRHBuilderWithoutAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.GsfTrajectoryFitter = cms.ESProducer("GsfTrajectoryFitterESProducer",
    Merger = cms.string('CloseComponentsMerger5D'),
    ComponentName = cms.string('GsfTrajectoryFitter'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator')
)

process.trackCounting3D3rd = cms.ESProducer("TrackCountingESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(3)
)

process.thCkfTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minPt = cms.double(0.3),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        nSigmaMinPt = cms.double(5.0),
        minimumNumberOfHits = cms.int32(5)
    ),
    ComponentName = cms.string('thCkfTrajectoryFilter')
)

process.l1GtTriggerMaskVetoAlgoTrig = cms.ESProducer("L1GtTriggerMaskVetoAlgoTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0)
)

process.caloDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('CaloDetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(70),
    nPhi = cms.int32(72)
)

process.KFTrajectoryFitterForOutIn = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForOutIn'),
    Estimator = cms.string('Chi2ForOutIn'),
    Propagator = cms.string('alongMomElePropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.KFSmootherForMuonTrackLoader = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(10.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForMuonTrackLoader'),
    Estimator = cms.string('Chi2EstimatorForMuonTrackLoader'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagatorAnyOpposite')
)

process.ElectronMaterialEffects = cms.ESProducer("GsfMaterialEffectsESProducer",
    BetheHeitlerParametrization = cms.string('BetheHeitler_cdfmom_nC6_O5.par'),
    EnergyLossUpdator = cms.string('GsfBetheHeitlerUpdator'),
    ComponentName = cms.string('ElectronMaterialEffects'),
    MultipleScatteringUpdator = cms.string('MultipleScatteringUpdator'),
    Mass = cms.double(0.000511),
    BetheHeitlerCorrection = cms.int32(2)
)

process.KFFittingSmootherForInOut = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('KFFitterForInOut'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('KFSmootherForInOut'),
    ComponentName = cms.string('KFFittingSmootherForInOut'),
    RejectTracks = cms.bool(True)
)

process.SmartPropagatorAnyOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAnyOpposite'),
    TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    Epsilon = cms.double(5.0)
)

process.seclayerPairs = cms.ESProducer("MixedLayerPairsESProducer",
    TIB3 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    TIB2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    ComponentName = cms.string('SecLayerPairs'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
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
        'TEC2_neg+TEC3_neg', 
        'TIB1+TIB2+TIB3'),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('secPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('secPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    )
)

process.KFUpdatorESProducer = cms.ESProducer("KFUpdatorESProducer",
    ComponentName = cms.string('KFUpdator')
)

process.newTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('newTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('newTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    minNrOfHitsForRebuild = cms.int32(5)
)

process.KFTrajectorySmootherForInOut = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForInOut'),
    Estimator = cms.string('Chi2ForInOut'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('oppositeToMomElePropagator')
)

process.MaterialPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('PropagatorWithMaterial'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)

process.l1GtTriggerMaskVetoTechTrig = cms.ESProducer("L1GtTriggerMaskVetoTechTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0)
)

process.rings = cms.ESProducer("RingMakerESProducer",
    DumpDetIds = cms.untracked.bool(False),
    ComponentName = cms.string(''),
    RingAsciiFileName = cms.untracked.string('rings.dat'),
    DetIdsDumpFileName = cms.untracked.string('tracker_detids.dat'),
    WriteOutRingsToAsciiFile = cms.untracked.bool(False),
    Configuration = cms.untracked.string('FULL')
)

process.jetProbability = cms.ESProducer("JetProbabilityESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(0.3),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    trackIpSign = cms.int32(1),
    minimumProbability = cms.double(0.005),
    maximumDecayLength = cms.double(5.0)
)

process.EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EEMap.txt')
)

process.bwdAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('bwdAnalyticalPropagator'),
    PropagationDirection = cms.string('oppositeToMomentum')
)

process.Chi2MeasurementEstimatorForInOut = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2ForInOut'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100.0)
)

process.RungeKuttaTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RungeKuttaTrackerPropagator'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(True)
)

process.Chi2MeasurementEstimatorForOutIn = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2ForOutIn'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(500.0)
)

process.Chi2EstimatorForMuRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForMuRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

process.TrajectoryBuilderForPixelMatchGsfElectrons = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('fwdGsfElectronPropagator'),
    trajectoryFilterName = cms.string('TrajectoryFilterForPixelMatchGsfElectrons'),
    maxCand = cms.int32(3),
    ComponentName = cms.string('TrajectoryBuilderForPixelMatchGsfElectrons'),
    intermediateCleaning = cms.bool(False),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('gsfElectronChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('bwdGsfElectronPropagator'),
    lostHitPenalty = cms.double(30.0)
)

process.KFFitterForRefitInsideOut = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitInsideOut'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAny'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.KFTrajectoryFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitter'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('PropagatorWithMaterial'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.RKFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('RKSmoother'),
    ComponentName = cms.string('RKFittingSmoother'),
    RejectTracks = cms.bool(True)
)

process.bwdElectronPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('bwdElectronPropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('oppositeToMomentum'),
    useRungeKutta = cms.bool(False)
)

process.SteppingHelixPropagatorAlong = cms.ESProducer("SteppingHelixPropagatorESProducer",
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

process.GroupedCkfTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('GroupedCkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string(''),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    minNrOfHitsForRebuild = cms.int32(5)
)

process.thCkfTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('thCkfTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('thCkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string('thMeasurementTracker'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    minNrOfHitsForRebuild = cms.int32(5)
)

process.fwdGsfElectronPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('fwdGsfElectronPropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)

process.L1MuGMTParameters = cms.ESProducer("L1MuGMTParametersProducer",
    MergeMethodSRKFwd = cms.string('takeCSC'),
    HaloOverwritesMatchedFwd = cms.bool(True),
    PhiWeight_barrel = cms.double(1.0),
    MergeMethodISOSpecialUseANDBrl = cms.bool(True),
    HaloOverwritesMatchedBrl = cms.bool(True),
    IsolationCellSizeEta = cms.int32(2),
    MergeMethodEtaFwd = cms.string('Special'),
    EtaPhiThreshold_COU = cms.double(0.127),
    MergeMethodISOBrl = cms.string('Special'),
    EtaWeight_barrel = cms.double(0.028),
    MergeMethodMIPBrl = cms.string('Special'),
    EtaPhiThreshold_barrel = cms.double(0.062),
    IsolationCellSizePhi = cms.int32(2),
    PhiWeight_endcap = cms.double(1.0),
    MergeMethodPtBrl = cms.string('byMinPt'),
    CaloTrigger = cms.bool(True),
    MergeMethodPtFwd = cms.string('byMinPt'),
    PropagatePhi = cms.bool(False),
    MergeMethodChargeFwd = cms.string('takeCSC'),
    MergeMethodEtaBrl = cms.string('Special'),
    CDLConfigWordfRPCDT = cms.uint32(1),
    CDLConfigWordDTCSC = cms.uint32(2),
    EtaWeight_endcap = cms.double(0.13),
    DoOvlRpcAnd = cms.bool(False),
    EtaWeight_COU = cms.double(0.316),
    MergeMethodISOSpecialUseANDFwd = cms.bool(True),
    MergeMethodMIPFwd = cms.string('Special'),
    MergeMethodPhiBrl = cms.string('takeDT'),
    EtaPhiThreshold_endcap = cms.double(0.062),
    CDLConfigWordCSCDT = cms.uint32(3),
    MergeMethodMIPSpecialUseANDFwd = cms.bool(False),
    MergeMethodPhiFwd = cms.string('takeCSC'),
    MergeMethodISOFwd = cms.string('Special'),
    PhiWeight_COU = cms.double(1.0),
    CDLConfigWordbRPCCSC = cms.uint32(16),
    MergeMethodChargeBrl = cms.string('takeDT'),
    SortRankOffsetBrl = cms.uint32(10),
    MergeMethodSRKBrl = cms.string('takeDT'),
    MergeMethodMIPSpecialUseANDBrl = cms.bool(False),
    SortRankOffsetFwd = cms.uint32(10)
)

process.impactParameterMVAComputer = cms.ESProducer("GenericMVAJetTagESProducer",
    useCategories = cms.bool(False),
    calibrationRecord = cms.string('ImpactParameterMVA')
)

process.PixelCPEParmErrorESProducer = cms.ESProducer("PixelCPEParmErrorESProducer",
    UseNewParametrization = cms.bool(True),
    ComponentName = cms.string('PixelCPEfromTrackAngle'),
    UseSigma = cms.bool(True),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

process.jetBProbability = cms.ESProducer("JetBProbabilityESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    trackIpSign = cms.int32(1),
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    maximumDecayLength = cms.double(5.0)
)

process.MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")

process.chi2CutForConversionTrajectoryBuilder = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('eleLooseChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

process.TrackAssociatorByHitsESProducer = cms.ESProducer("TrackAssociatorByHitsESProducer",
    associateRecoTracks = cms.bool(True),
    UseGrouped = cms.bool(True),
    associatePixel = cms.bool(True),
    ROUList = cms.vstring('TrackerHitsTIBLowTof', 
        'TrackerHitsTIBHighTof', 
        'TrackerHitsTIDLowTof', 
        'TrackerHitsTIDHighTof', 
        'TrackerHitsTOBLowTof', 
        'TrackerHitsTOBHighTof', 
        'TrackerHitsTECLowTof', 
        'TrackerHitsTECHighTof', 
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    UseSplitting = cms.bool(True),
    UsePixels = cms.bool(True),
    ThreeHitTracksAreSpecial = cms.bool(True),
    AbsoluteNumberOfHits = cms.bool(False),
    associateStrip = cms.bool(True),
    MinHitCut = cms.double(0.5),
    SimToRecoDenominator = cms.string('sim')
)

process.l1GtBoardMaps = cms.ESProducer("L1GtBoardMapsTrivialProducer",
    ActiveBoardsDaqRecord = cms.vint32(-1, 0, 1, 2, 3, 
        4, 5, 6, 7, 8, 
        -1, -1),
    BoardSlotMap = cms.vint32(17, 10, 9, 13, 14, 
        15, 19, 20, 21, 18, 
        7, 16),
    BoardPositionDaqRecord = cms.vint32(1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 
        -1, -1),
    BoardPositionEvmRecord = cms.vint32(1, 3, -1, -1, -1, 
        -1, -1, -1, -1, -1, 
        2, -1),
    BoardList = cms.vstring('GTFE', 
        'FDL', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'PSB', 
        'GMT', 
        'TCS', 
        'TIM'),
    BoardIndex = cms.vint32(0, 0, 0, 1, 2, 
        3, 4, 5, 6, 0, 
        0, 0),
    BoardHexNameMap = cms.vint32(0, 253, 187, 187, 187, 
        187, 187, 187, 187, 221, 
        204, 173),
    ActiveBoardsEvmRecord = cms.vint32(-1, 1, -1, -1, -1, 
        -1, -1, -1, -1, -1, 
        0, -1),
    CableToPsbMap = cms.vint32(0, 0, 0, 0, 1, 
        1, 1, 1, 2, 2, 
        2, 2, 3, 3, 3, 
        3, 4, 4, 4, 4, 
        5, 5, 5, 5, 6, 
        6, 6, 6),
    CableList = cms.vstring('Free', 
        'Free', 
        'Free', 
        'TechTr', 
        'IsoEGQ', 
        'NoIsoEGQ', 
        'CenJetQ', 
        'ForJetQ', 
        'TauJetQ', 
        'ESumsQ', 
        'JetCountsQ', 
        'Free', 
        'Free', 
        'Free', 
        'Free', 
        'Free', 
        'MQF4', 
        'MQF3', 
        'MQB2', 
        'MQB1', 
        'MQF8', 
        'MQF7', 
        'MQB6', 
        'MQB5', 
        'MQF12', 
        'MQF11', 
        'MQB10', 
        'MQB9')
)

process.PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",
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

process.KFFitterForRefitOutsideIn = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('KFFitterForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Propagator = cms.string('SmartPropagatorAnyOpposite'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.Chi2EstimatorForMuonTrackLoader = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForMuonTrackLoader'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

process.fwdElectronPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('fwdElectronPropagator'),
    Mass = cms.double(0.000511),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(False)
)

process.myTTRHBuilderWithoutAngle4MixedPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.pixellayertriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('PixelLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    )
)

process.SteppingHelixPropagatorOpposite = cms.ESProducer("SteppingHelixPropagatorESProducer",
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

process.DTGeometryESModule = cms.ESProducer("DTGeometryESModule",
    applyAlignment = cms.untracked.bool(True)
)

process.VolumeBasedMagneticFieldESProducer = cms.ESProducer("VolumeBasedMagneticFieldESProducer",
    timerOn = cms.untracked.bool(False),
    useParametrizedTrackerField = cms.bool(False),
    findVolumeTolerance = cms.double(0.0),
    label = cms.untracked.string(''),
    version = cms.string('grid_85l_030919'),
    debugBuilder = cms.untracked.bool(False),
    cacheLastVolume = cms.untracked.bool(True)
)

process.KFFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('KFFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('KFSmoother'),
    ComponentName = cms.string('KFFittingSmoother'),
    RejectTracks = cms.bool(True)
)

process.myTTRHBuilderWithoutAngle4PixelPairs = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.CaloTopologyBuilder = cms.ESProducer("CaloTopologyBuilder")

process.roads = cms.ESProducer("RoadMapMakerESProducer",
    GeometryStructure = cms.string('FullDetector'),
    ComponentName = cms.string(''),
    RingsLabel = cms.string(''),
    WriteOutRoadMapToAsciiFile = cms.untracked.bool(False),
    SeedingType = cms.string('FourRingSeeds'),
    RoadMapAsciiFile = cms.untracked.string('roads.dat')
)

process.GsfTrajectoryFitter_forPreId = cms.ESProducer("GsfTrajectoryFitterESProducer",
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    ComponentName = cms.string('GsfTrajectoryFitter_forPreId'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator')
)

process.compositeTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('compositeTrajectoryFilter'),
    filterNames = cms.vstring()
)

process.PixelCPEInitialESProducer = cms.ESProducer("PixelCPEInitialESProducer",
    ComponentName = cms.string('PixelCPEInitial'),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

process.SiStripRecHitMatcherESProducer = cms.ESProducer("SiStripRecHitMatcherESProducer",
    ComponentName = cms.string('StandardMatcher'),
    NSigmaInside = cms.double(3.0)
)

process.pixellayerpairs = cms.ESProducer("PixelLayerPairsESProducer",
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
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs')
    )
)

process.seclayertriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('SecLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('secPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('secPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets')
    )
)

process.l1GtStableParameters = cms.ESProducer("L1GtStableParametersTrivialProducer",
    NumberL1IsoEG = cms.uint32(4),
    NumberL1JetCounts = cms.uint32(12),
    NumberTechnicalTriggers = cms.uint32(64),
    NumberL1NoIsoEG = cms.uint32(4),
    IfCaloEtaNumberBits = cms.uint32(4),
    NumberL1CenJet = cms.uint32(4),
    NumberL1TauJet = cms.uint32(4),
    NumberL1Mu = cms.uint32(4),
    NumberConditionChips = cms.uint32(2),
    IfMuEtaNumberBits = cms.uint32(6),
    NumberPsbBoards = cms.int32(7),
    NumberPhysTriggers = cms.uint32(128),
    PinsOnConditionChip = cms.uint32(96),
    UnitLength = cms.int32(8),
    OrderConditionChip = cms.vint32(2, 1),
    NumberPhysTriggersExtended = cms.uint32(64),
    WordLength = cms.int32(64),
    NumberL1ForJet = cms.uint32(4)
)

process.l1GtPrescaleFactorsTechTrig = cms.ESProducer("L1GtPrescaleFactorsTechTrigTrivialProducer",
    PrescaleFactors = cms.vint32(1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1)
)

process.CSCGeometryESModule = cms.ESProducer("CSCGeometryESModule",
    debugV = cms.untracked.bool(False),
    useGangedStripsInME1a = cms.bool(True),
    useOnlyWiresInME1a = cms.bool(False),
    useRealWireGeometry = cms.bool(True),
    useCentreTIOffsets = cms.bool(False),
    applyAlignment = cms.untracked.bool(True)
)

process.MuonTransientTrackingRecHitBuilderESProducer = cms.ESProducer("MuonTransientTrackingRecHitBuilderESProducer",
    ComponentName = cms.string('MuonRecHitBuilder')
)

process.KFSmootherForRefitOutsideIn = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmootherForRefitOutsideIn'),
    Estimator = cms.string('Chi2EstimatorForRefit'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('SmartPropagator')
)

process.navigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('SimpleNavigationSchool')
)

process.mixedlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
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
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    )
)

process.GsfElectronFittingSmoother = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('GsfTrajectoryFitter'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('GsfTrajectorySmoother'),
    ComponentName = cms.string('GsfElectronFittingSmoother'),
    RejectTracks = cms.bool(True)
)

process.secCkfTrajectoryBuilder = cms.ESProducer("GroupedCkfTrajectoryBuilderESProducer",
    bestHitOnly = cms.bool(True),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilterName = cms.string('secCkfTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('secCkfTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    lostHitPenalty = cms.double(30.0),
    MeasurementTrackerName = cms.string('secMeasurementTracker'),
    lockHits = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    foundHitBonus = cms.double(5.0),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    requireSeedHitsInRebuild = cms.bool(True),
    estimator = cms.string('Chi2'),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    minNrOfHitsForRebuild = cms.int32(5)
)

process.RPCGeometryESModule = cms.ESProducer("RPCGeometryESModule",
    compatibiltyWith11 = cms.untracked.bool(True)
)

process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.SmootherRK = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('SmootherRK'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.SmartPropagatorAny = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorAny'),
    TrackerPropagator = cms.string('PropagatorWithMaterial'),
    MuonPropagator = cms.string('SteppingHelixPropagatorAny'),
    PropagationDirection = cms.string('alongMomentum'),
    Epsilon = cms.double(5.0)
)

process.CaloTowerHardcodeGeometryEP = cms.ESProducer("CaloTowerHardcodeGeometryEP")

process.GsfTrajectorySmoother = cms.ESProducer("GsfTrajectorySmootherESProducer",
    Merger = cms.string('CloseComponentsMerger5D'),
    ComponentName = cms.string('GsfTrajectorySmoother'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator')
)

process.gsfElectronChi2 = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('gsfElectronChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

process.ZdcHardcodeGeometryEP = cms.ESProducer("ZdcHardcodeGeometryEP")

process.FittingSmootherWithOutlierRejection = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(20.0),
    Fitter = cms.string('RKFitter'),
    MinNumberOfHits = cms.int32(3),
    Smoother = cms.string('RKSmoother'),
    ComponentName = cms.string('FittingSmootherWithOutlierRejection'),
    RejectTracks = cms.bool(True)
)

process.l1GtPrescaleFactorsAlgoTrig = cms.ESProducer("L1GtPrescaleFactorsAlgoTrigTrivialProducer",
    PrescaleFactors = cms.vint32(4000, 2000, 1, 1, 1, 
        1, 1, 10000, 1000, 100, 
        1, 1, 1, 1, 10000, 
        1000, 100, 100, 1, 1, 
        1, 100000, 100000, 10000, 10000, 
        100, 1, 1, 1, 100000, 
        100000, 10000, 1, 1000, 1000, 
        1, 1, 1000, 100, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1000, 100, 100, 1, 
        1, 1, 1, 20, 1, 
        1, 1, 1, 1, 20, 
        1, 1, 1, 1, 1, 
        20, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 100, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 3000000, 
        3000000, 10000, 5000, 100000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1)
)

process.myTTRHBuilderWithoutAngle4PixelTriplets = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('Fake'),
    ComponentName = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.SteppingHelixPropagatorAny = cms.ESProducer("SteppingHelixPropagatorESProducer",
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

process.combinedSecondaryVertex = cms.ESProducer("CombinedSecondaryVertexESProducer",
    useTrackWeights = cms.bool(True),
    useCategories = cms.bool(True),
    pseudoMultiplicityMin = cms.uint32(2),
    correctVertexMass = cms.bool(True),
    trackPseudoSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    categoryVariableName = cms.string('vertexCategory'),
    calibrationRecords = cms.vstring('CombinedSVRecoVertex', 
        'CombinedSVPseudoVertex', 
        'CombinedSVNoVertex'),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    charmCut = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackMultiplicityMin = cms.uint32(3),
    trackSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    trackSort = cms.string('sip2dSig')
)

process.softMuonNoIP = cms.ESProducer("MuonTaggerNoIPESProducer")

process.cosmicsNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('CosmicNavigationSchool')
)

process.hoDetIdAssociator = cms.ESProducer("DetIdAssociatorESProducer",
    ComponentName = cms.string('HODetIdAssociator'),
    etaBinSize = cms.double(0.087),
    nEta = cms.int32(30),
    nPhi = cms.int32(72)
)

process.trackCounting3D2nd = cms.ESProducer("TrackCountingESProducer",
    impactParamterType = cms.int32(0),
    deltaR = cms.double(-1.0),
    maximumDistanceToJetAxis = cms.double(0.07),
    trackQualityClass = cms.string('any'),
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(2)
)

process.Chi2EstimatorForRefit = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('Chi2EstimatorForRefit'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100000.0)
)

process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerRecoGeometryESProducer")

process.StripCPEfromTrackAngleESProducer = cms.ESProducer("StripCPEfromTrackAngleESProducer",
    ComponentName = cms.string('StripCPEfromTrackAngle')
)

process.l1GtFactors = cms.ESProducer("L1GtFactorsTrivialProducer",
    PrescaleFactors = cms.vint32(4000, 2000, 1, 1, 1, 
        1, 1, 10000, 1000, 100, 
        1, 1, 1, 1, 10000, 
        1000, 100, 100, 1, 1, 
        1, 100000, 100000, 10000, 10000, 
        100, 1, 1, 1, 100000, 
        100000, 10000, 1, 1000, 1000, 
        1, 1, 1000, 100, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1000, 100, 100, 1, 
        1, 1, 1, 20, 1, 
        1, 1, 1, 1, 20, 
        1, 1, 1, 1, 1, 
        20, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 100, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 10000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 3000000, 
        3000000, 10000, 5000, 100000, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1),
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 1, 0, 1, 
        0, 0, 0, 0, 1, 
        1, 1, 0, 0, 1, 
        0, 0, 0, 0, 1, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 1, 0, 
        0, 0, 0, 0, 0, 
        1, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        1, 0, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        1, 0, 1, 1, 1, 
        0, 0, 1, 0, 1, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 0, 0, 
        0, 0, 0, 1, 1, 
        1, 0, 1, 1, 1, 
        1, 0, 1, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0)
)

process.GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

process.RKTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('RKSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)

process.thlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    TIB2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("thStripRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("thStripRecHits","rphiRecHit")
    ),
    ComponentName = cms.string('ThLayerPairs'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("thStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
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
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('thPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('thPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs')
    )
)

process.EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService")

process.ElectronMaterialEffects_forPreId = cms.ESProducer("GsfMaterialEffectsESProducer",
    BetheHeitlerParametrization = cms.string('BetheHeitler_cdfmom_nC3_O5.par'),
    EnergyLossUpdator = cms.string('GsfBetheHeitlerUpdator'),
    ComponentName = cms.string('ElectronMaterialEffects_forPreId'),
    MultipleScatteringUpdator = cms.string('MultipleScatteringUpdator'),
    Mass = cms.double(0.000511),
    BetheHeitlerCorrection = cms.int32(2)
)

process.L1GctConfigProducers = cms.ESProducer("L1GctConfigProducers",
    JetFinderCentralJetSeed = cms.uint32(1),
    CalibrationStyle = cms.string('ORCAStyle'),
    L1CaloHtScaleLsbInGeV = cms.double(1.0),
    PowerSeriesCoefficients = cms.PSet(
        tauJetCalib0 = cms.vdouble(),
        nonTauJetCalib0 = cms.vdouble(),
        nonTauJetCalib5 = cms.vdouble(),
        nonTauJetCalib6 = cms.vdouble(),
        nonTauJetCalib7 = cms.vdouble(),
        nonTauJetCalib2 = cms.vdouble(),
        nonTauJetCalib8 = cms.vdouble(),
        nonTauJetCalib9 = cms.vdouble(),
        nonTauJetCalib4 = cms.vdouble(),
        nonTauJetCalib3 = cms.vdouble(),
        tauJetCalib4 = cms.vdouble(),
        tauJetCalib5 = cms.vdouble(),
        tauJetCalib6 = cms.vdouble(),
        nonTauJetCalib1 = cms.vdouble(),
        nonTauJetCalib10 = cms.vdouble(),
        tauJetCalib1 = cms.vdouble(),
        tauJetCalib2 = cms.vdouble(),
        tauJetCalib3 = cms.vdouble()
    ),
    JetFinderForwardJetSeed = cms.uint32(1),
    PiecewiseCubicCoefficients = cms.PSet(
        tauJetCalib0 = cms.vdouble(500.0, 100.0, 17.7409, 0.351901, -0.000701462, 
            5.77204e-07, 5.0, 0.720604, 1.25179, -0.0150777, 
            7.13711e-05),
        nonTauJetCalib0 = cms.vdouble(500.0, 100.0, 17.7409, 0.351901, -0.000701462, 
            5.77204e-07, 5.0, 0.720604, 1.25179, -0.0150777, 
            7.13711e-05),
        nonTauJetCalib5 = cms.vdouble(500.0, 100.0, 29.5396, 0.001137, -0.000145232, 
            6.91445e-08, 5.0, 4.16752, 1.08477, -0.016134, 
            7.69652e-05),
        nonTauJetCalib6 = cms.vdouble(500.0, 100.0, 30.1405, -0.14281, 0.000555849, 
            -7.52446e-07, 5.0, 4.79283, 0.672125, -0.00879174, 
            3.65776e-05),
        nonTauJetCalib7 = cms.vdouble(300.0, 80.0, 30.2715, -0.539688, 0.00499898, 
            -1.2204e-05, 5.0, 1.97284, 0.0610729, 0.00671548, 
            -7.22583e-05),
        nonTauJetCalib2 = cms.vdouble(500.0, 100.0, 24.3454, 0.257989, -0.000450184, 
            3.09951e-07, 5.0, 2.1034, 1.32441, -0.0173659, 
            8.50669e-05),
        nonTauJetCalib8 = cms.vdouble(250.0, 150.0, 1.38861, 0.0362661, 0.0, 
            0.0, 5.0, 1.87993, 0.0329907, 0.0, 
            0.0),
        nonTauJetCalib9 = cms.vdouble(200.0, 80.0, 35.0095, -0.669677, 0.00208498, 
            -1.50554e-06, 5.0, 3.16074, -0.114404, 0.0, 
            0.0),
        nonTauJetCalib4 = cms.vdouble(500.0, 100.0, 26.6384, 0.0567369, -0.000416292, 
            2.60929e-07, 5.0, 2.63299, 1.16558, -0.0170351, 
            7.95703e-05),
        nonTauJetCalib3 = cms.vdouble(500.0, 100.0, 27.7822, 0.155986, -0.000266441, 
            6.69814e-08, 5.0, 2.64613, 1.30745, -0.0180964, 
            8.83567e-05),
        tauJetCalib4 = cms.vdouble(500.0, 100.0, 26.6384, 0.0567369, -0.000416292, 
            2.60929e-07, 5.0, 2.63299, 1.16558, -0.0170351, 
            7.95703e-05),
        tauJetCalib5 = cms.vdouble(500.0, 100.0, 29.5396, 0.001137, -0.000145232, 
            6.91445e-08, 5.0, 4.16752, 1.08477, -0.016134, 
            7.69652e-05),
        tauJetCalib6 = cms.vdouble(500.0, 100.0, 30.1405, -0.14281, 0.000555849, 
            -7.52446e-07, 5.0, 4.79283, 0.672125, -0.00879174, 
            3.65776e-05),
        nonTauJetCalib1 = cms.vdouble(500.0, 100.0, 20.0549, 0.321867, -0.00064901, 
            5.50042e-07, 5.0, 1.30465, 1.2774, -0.0159193, 
            7.64496e-05),
        nonTauJetCalib10 = cms.vdouble(150.0, 80.0, 1.70475, -0.142171, 0.00104963, 
            -1.62214e-05, 5.0, 1.70475, -0.142171, 0.00104963, 
            -1.62214e-05),
        tauJetCalib1 = cms.vdouble(500.0, 100.0, 20.0549, 0.321867, -0.00064901, 
            5.50042e-07, 5.0, 1.30465, 1.2774, -0.0159193, 
            7.64496e-05),
        tauJetCalib2 = cms.vdouble(500.0, 100.0, 24.3454, 0.257989, -0.000450184, 
            3.09951e-07, 5.0, 2.1034, 1.32441, -0.0173659, 
            8.50669e-05),
        tauJetCalib3 = cms.vdouble(500.0, 100.0, 27.7822, 0.155986, -0.000266441, 
            6.69814e-08, 5.0, 2.64613, 1.30745, -0.0180964, 
            8.83567e-05)
    ),
    OrcaStyleCoefficients = cms.PSet(
        tauJetCalib0 = cms.vdouble(47.4, -20.7, 0.7922, 9.53e-05),
        nonTauJetCalib0 = cms.vdouble(47.4, -20.7, 0.7922, 9.53e-05),
        nonTauJetCalib5 = cms.vdouble(42.0, -23.9, 0.7945, 0.0001458),
        nonTauJetCalib6 = cms.vdouble(33.8, -22.1, 0.8202, 0.0001403),
        nonTauJetCalib7 = cms.vdouble(17.1, -6.6, 0.6958, 6.88e-05),
        nonTauJetCalib2 = cms.vdouble(47.1, -22.2, 0.7645, 0.0001209),
        nonTauJetCalib8 = cms.vdouble(13.1, -4.5, 0.7071, 7.26e-05),
        nonTauJetCalib9 = cms.vdouble(12.4, -3.8, 0.6558, 0.000489),
        nonTauJetCalib4 = cms.vdouble(48.2, -24.5, 0.7706, 0.000128),
        nonTauJetCalib3 = cms.vdouble(49.3, -22.9, 0.7331, 0.0001221),
        tauJetCalib4 = cms.vdouble(48.2, -24.5, 0.7706, 0.000128),
        tauJetCalib5 = cms.vdouble(42.0, -23.9, 0.7945, 0.0001458),
        tauJetCalib6 = cms.vdouble(33.8, -22.1, 0.8202, 0.0001403),
        nonTauJetCalib1 = cms.vdouble(49.4, -22.5, 0.7867, 9.6e-05),
        nonTauJetCalib10 = cms.vdouble(9.3, 1.3, 0.2719, 0.003418),
        tauJetCalib1 = cms.vdouble(49.4, -22.5, 0.7867, 9.6e-05),
        tauJetCalib2 = cms.vdouble(47.1, -22.2, 0.7645, 0.0001209),
        tauJetCalib3 = cms.vdouble(49.3, -22.9, 0.7331, 0.0001221)
    ),
    jetCounterSetup = cms.PSet(
        jetCountersNegativeWheel = cms.VPSet(cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_1')
        ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_1', 
                    'JC_centralEta_6')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_11')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_11', 
                    'JC_centralEta_6')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_19')
            )),
        jetCountersPositiveWheel = cms.VPSet(cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_1')
        ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_1', 
                    'JC_centralEta_6')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_11')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_11', 
                    'JC_centralEta_6')
            ), 
            cms.PSet(
                cutDescriptionList = cms.vstring('JC_minRank_19')
            ))
    ),
    L1CaloJetZeroSuppressionThresholdInGeV = cms.double(5.0)
)

process.EcalTrigPrimESProducer = cms.ESProducer("EcalTrigPrimESProducer",
    DatabaseFileEE = cms.untracked.string('TPG_EE.txt'),
    DatabaseFileEB = cms.untracked.string('TPG_EB.txt')
)

process.MeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(False),
    OnDemand = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(False),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string(''),
    stripClusterProducer = cms.string('siStripClusters'),
    Regional = cms.bool(False),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    pixelClusterProducer = cms.string('siPixelClusters'),
    stripLazyGetterProducer = cms.string(''),
    UseStripModuleQualityDB = cms.bool(False),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)

process.SmartPropagatorOpposite = cms.ESProducer("SmartPropagatorESProducer",
    ComponentName = cms.string('SmartPropagatorOpposite'),
    TrackerPropagator = cms.string('PropagatorWithMaterialOpposite'),
    MuonPropagator = cms.string('SteppingHelixPropagatorOpposite'),
    PropagationDirection = cms.string('oppositeToMomentum'),
    Epsilon = cms.double(5.0)
)

process.TrajectoryFilterForPixelMatchGsfElectrons = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minHitsMinPt = cms.int32(-1),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        minimumNumberOfHits = cms.int32(5),
        nSigmaMinPt = cms.double(5.0),
        minPt = cms.double(3.0)
    ),
    ComponentName = cms.string('TrajectoryFilterForPixelMatchGsfElectrons')
)

process.softElectron = cms.ESProducer("ElectronTaggerESProducer")

process.RKTrackerPropagator = cms.ESProducer("PropagatorWithMaterialESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('RKTrackerPropagator'),
    Mass = cms.double(0.105),
    PropagationDirection = cms.string('alongMomentum'),
    useRungeKutta = cms.bool(True)
)

process.mixedlayertriplets = cms.ESProducer("MixedLayerTripletsESProducer",
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
        hitErrorRZ = cms.double(0.0036),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets')
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        hitErrorRZ = cms.double(0.006),
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets')
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

process.l1GtTriggerMaskAlgoTrig = cms.ESProducer("L1GtTriggerMaskAlgoTrigTrivialProducer",
    TriggerMask = cms.vuint32(0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 255, 0, 255, 
        0, 0, 0, 0, 255, 
        255, 255, 0, 0, 255, 
        0, 0, 0, 0, 255, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 255, 0, 
        0, 0, 0, 0, 0, 
        255, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        255, 0, 255, 255, 255, 
        255, 255, 255, 255, 255, 
        255, 0, 255, 255, 255, 
        0, 0, 255, 0, 255, 
        255, 255, 255, 255, 255, 
        255, 255, 255, 0, 0, 
        0, 0, 0, 255, 255, 
        255, 0, 255, 255, 255, 
        255, 0, 255, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 0, 0)
)

process.GlbMuKFFitter = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('GlbMuKFFitter'),
    Estimator = cms.string('Chi2EstimatorForMuRefit'),
    Propagator = cms.string('SmartPropagatorAnyRK'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.ckfBaseTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minPt = cms.double(0.9),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        nSigmaMinPt = cms.double(5.0),
        minimumNumberOfHits = cms.int32(5)
    ),
    ComponentName = cms.string('ckfBaseTrajectoryFilter')
)

process.templates = cms.ESProducer("PixelCPETemplateRecoESProducer",
    ComponentName = cms.string('PixelCPETemplateReco'),
    TanLorentzAnglePerTesla = cms.double(0.106),
    speed = cms.int32(0),
    PixelErrorParametrization = cms.string('NOTcmsim'),
    Alpha2Order = cms.bool(True)
)

process.secMeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(False),
    OnDemand = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(False),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string('secMeasurementTracker'),
    stripClusterProducer = cms.string('secClusters'),
    Regional = cms.bool(False),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    pixelClusterProducer = cms.string('secClusters'),
    stripLazyGetterProducer = cms.string(''),
    UseStripModuleQualityDB = cms.bool(False),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

process.trajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('TrajectoryCleanerBySharedHits')
)

process.fwdAnalyticalPropagator = cms.ESProducer("AnalyticalPropagatorESProducer",
    MaxDPhi = cms.double(1.6),
    ComponentName = cms.string('fwdAnalyticalPropagator'),
    PropagationDirection = cms.string('alongMomentum')
)

process.StripCPEESProducer = cms.ESProducer("StripCPEESProducer",
    ComponentName = cms.string('SimpleStripCPE')
)

process.CkfElectronTrajectoryBuilder = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('fwdElectronPropagator'),
    trajectoryFilterName = cms.string('ckfBaseTrajectoryFilter'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('CkfElectronTrajectoryBuilder'),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('electronChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('bwdElectronPropagator'),
    lostHitPenalty = cms.double(30.0)
)

process.simpleSecondaryVertex = cms.ESProducer("SimpleSecondaryVertexESProducer",
    use3d = cms.bool(True),
    unBoost = cms.bool(False),
    useSignificance = cms.bool(True)
)

process.newTrajectoryFilter = cms.ESProducer("TrajectoryFilterESProducer",
    filterPset = cms.PSet(
        chargeSignificance = cms.double(-1.0),
        minPt = cms.double(0.3),
        minHitsMinPt = cms.int32(3),
        ComponentType = cms.string('CkfBaseTrajectoryFilter'),
        maxLostHits = cms.int32(1),
        maxNumberOfHits = cms.int32(-1),
        maxConsecLostHits = cms.int32(1),
        nSigmaMinPt = cms.double(5.0),
        minimumNumberOfHits = cms.int32(3)
    ),
    ComponentName = cms.string('newTrajectoryFilter')
)

process.KullbackLeiblerDistance5D = cms.ESProducer("DistanceBetweenComponentsESProducer5D",
    ComponentName = cms.string('KullbackLeiblerDistance5D'),
    DistanceMeasure = cms.string('KullbackLeibler')
)

process.HcalTopologyIdealEP = cms.ESProducer("HcalTopologyIdealEP")

process.FittingSmootherRK = cms.ESProducer("KFFittingSmootherESProducer",
    EstimateCut = cms.double(-1.0),
    Fitter = cms.string('FitterRK'),
    MinNumberOfHits = cms.int32(5),
    Smoother = cms.string('SmootherRK'),
    ComponentName = cms.string('FittingSmootherRK'),
    RejectTracks = cms.bool(True)
)

process.ttrhbwr = cms.ESProducer("TkTransientTrackingRecHitBuilderESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    ComponentName = cms.string('WithTrackAngle'),
    PixelCPE = cms.string('PixelCPEfromTrackAngle'),
    Matcher = cms.string('StandardMatcher')
)

process.L1MuGMTScales = cms.ESProducer("L1MuGMTScalesProducer")

process.thMeasurementTracker = cms.ESProducer("MeasurementTrackerESProducer",
    StripCPE = cms.string('StripCPEfromTrackAngle'),
    UseStripStripQualityDB = cms.bool(False),
    OnDemand = cms.bool(False),
    UseStripAPVFiberQualityDB = cms.bool(False),
    DebugStripModuleQualityDB = cms.untracked.bool(False),
    ComponentName = cms.string('thMeasurementTracker'),
    stripClusterProducer = cms.string('thClusters'),
    Regional = cms.bool(False),
    DebugStripAPVFiberQualityDB = cms.untracked.bool(False),
    HitMatcher = cms.string('StandardMatcher'),
    DebugStripStripQualityDB = cms.untracked.bool(False),
    pixelClusterProducer = cms.string('thClusters'),
    stripLazyGetterProducer = cms.string(''),
    UseStripModuleQualityDB = cms.bool(False),
    PixelCPE = cms.string('PixelCPEfromTrackAngle')
)

process.TrackerGeometricDetESModule = cms.ESProducer("TrackerGeometricDetESModule",
    fromDDD = cms.bool(True)
)

process.FitterRK = cms.ESProducer("KFTrajectoryFitterESProducer",
    ComponentName = cms.string('FitterRK'),
    Estimator = cms.string('Chi2'),
    Propagator = cms.string('RungeKuttaTrackerPropagator'),
    Updator = cms.string('KFUpdator'),
    minHits = cms.int32(3)
)

process.combinedSecondaryVertexMVA = cms.ESProducer("CombinedSecondaryVertexESProducer",
    useTrackWeights = cms.bool(True),
    useCategories = cms.bool(True),
    pseudoMultiplicityMin = cms.uint32(2),
    correctVertexMass = cms.bool(True),
    trackPseudoSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    categoryVariableName = cms.string('vertexCategory'),
    calibrationRecords = cms.vstring('CombinedSVMVARecoVertex', 
        'CombinedSVMVAPseudoVertex', 
        'CombinedSVMVANoVertex'),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    charmCut = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackMultiplicityMin = cms.uint32(3),
    trackSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    trackSort = cms.string('sip2dSig')
)

process.electronChi2 = cms.ESProducer("Chi2MeasurementEstimatorESProducer",
    ComponentName = cms.string('electronChi2'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(100.0)
)

process.EcalPreshowerGeometryEP = cms.ESProducer("EcalPreshowerGeometryEP")

process.beamHaloNavigationSchoolESProducer = cms.ESProducer("NavigationSchoolESProducer",
    ComponentName = cms.string('BeamHaloNavigationSchool')
)

process.TrajectoryBuilderForConversions = cms.ESProducer("CkfTrajectoryBuilderESProducer",
    propagatorAlong = cms.string('alongMomElePropagator'),
    trajectoryFilterName = cms.string('TrajectoryFilterForConversions'),
    maxCand = cms.int32(5),
    ComponentName = cms.string('TrajectoryBuilderForConversions'),
    intermediateCleaning = cms.bool(True),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('eleLooseChi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    propagatorOpposite = cms.string('oppositeToMomElePropagator'),
    lostHitPenalty = cms.double(30.0)
)

process.KFTrajectorySmoother = cms.ESProducer("KFTrajectorySmootherESProducer",
    errorRescaling = cms.double(100.0),
    minHits = cms.int32(3),
    ComponentName = cms.string('KFSmoother'),
    Estimator = cms.string('Chi2'),
    Updator = cms.string('KFUpdator'),
    Propagator = cms.string('PropagatorWithMaterial')
)

process.L1GtBoardMapsRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtBoardMapsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLinearizationConstRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.SiStripPedestalsFakeESSource = cms.ESSource("SiStripPedestalsFakeESSource",
    printDebug = cms.untracked.bool(False),
    HighThValue = cms.double(5.0),
    PedestalsValue = cms.uint32(30),
    file = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    LowThValue = cms.double(2.0)
)

process.tpparams11 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainTowerEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams10 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainStripEERcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.l1GctParamsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetFinderParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtPrescaleFactorsRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.l1GctJcPosParsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCounterPositiveEtaRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMother.xml', 
        'Geometry/CMSCommonData/data/cmsTracker.xml', 
        'Geometry/CMSCommonData/data/caloBase.xml', 
        'Geometry/CMSCommonData/data/cmsCalo.xml', 
        'Geometry/CMSCommonData/data/muonBase.xml', 
        'Geometry/CMSCommonData/data/cmsMuon.xml', 
        'Geometry/CMSCommonData/data/mgnt.xml', 
        'Geometry/CMSCommonData/data/beampipe.xml', 
        'Geometry/CMSCommonData/data/cmsBeam.xml', 
        'Geometry/CMSCommonData/data/muonMB.xml', 
        'Geometry/CMSCommonData/data/muonMagnet.xml', 
        'Geometry/TrackerCommonData/data/pixfwdMaterials.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCommon.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x2.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq1x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x3.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x4.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPlaq2x5.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanelBase.xml', 
        'Geometry/TrackerCommonData/data/pixfwdPanel.xml', 
        'Geometry/TrackerCommonData/data/pixfwdBlade.xml', 
        'Geometry/TrackerCommonData/data/pixfwdNipple.xml', 
        'Geometry/TrackerCommonData/data/pixfwdDisk.xml', 
        'Geometry/TrackerCommonData/data/pixfwdCylinder.xml', 
        'Geometry/TrackerCommonData/data/pixfwd.xml', 
        'Geometry/TrackerCommonData/data/pixbarmaterial.xml', 
        'Geometry/TrackerCommonData/data/pixbarladder.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderfull.xml', 
        'Geometry/TrackerCommonData/data/pixbarladderhalf.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer0.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer1.xml', 
        'Geometry/TrackerCommonData/data/pixbarlayer2.xml', 
        'Geometry/TrackerCommonData/data/pixbar.xml', 
        'Geometry/TrackerCommonData/data/tibtidcommonmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmaterial.xml', 
        'Geometry/TrackerCommonData/data/tibmodpar.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0a.xml', 
        'Geometry/TrackerCommonData/data/tibmodule0b.xml', 
        'Geometry/TrackerCommonData/data/tibmodule2.xml', 
        'Geometry/TrackerCommonData/data/tibstringpar.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring0lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring0ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring0.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring1lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring1ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring1.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring2lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring2ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring2.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ll.xml', 
        'Geometry/TrackerCommonData/data/tibstring3lr.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ul.xml', 
        'Geometry/TrackerCommonData/data/tibstring3ur.xml', 
        'Geometry/TrackerCommonData/data/tibstring3.xml', 
        'Geometry/TrackerCommonData/data/tiblayerpar.xml', 
        'Geometry/TrackerCommonData/data/tiblayer0.xml', 
        'Geometry/TrackerCommonData/data/tiblayer1.xml', 
        'Geometry/TrackerCommonData/data/tiblayer2.xml', 
        'Geometry/TrackerCommonData/data/tiblayer3.xml', 
        'Geometry/TrackerCommonData/data/tib.xml', 
        'Geometry/TrackerCommonData/data/tidmaterial.xml', 
        'Geometry/TrackerCommonData/data/tidmodpar.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule0l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tidmodule1l.xml', 
        'Geometry/TrackerCommonData/data/tidmodule2.xml', 
        'Geometry/TrackerCommonData/data/tidringpar.xml', 
        'Geometry/TrackerCommonData/data/tidring0.xml', 
        'Geometry/TrackerCommonData/data/tidring0f.xml', 
        'Geometry/TrackerCommonData/data/tidring0b.xml', 
        'Geometry/TrackerCommonData/data/tidring1.xml', 
        'Geometry/TrackerCommonData/data/tidring1f.xml', 
        'Geometry/TrackerCommonData/data/tidring1b.xml', 
        'Geometry/TrackerCommonData/data/tidring2.xml', 
        'Geometry/TrackerCommonData/data/tid.xml', 
        'Geometry/TrackerCommonData/data/tidf.xml', 
        'Geometry/TrackerCommonData/data/tidb.xml', 
        'Geometry/TrackerCommonData/data/tibtidservices.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesf.xml', 
        'Geometry/TrackerCommonData/data/tibtidservicesb.xml', 
        'Geometry/TrackerCommonData/data/tobmaterial.xml', 
        'Geometry/TrackerCommonData/data/tobmodpar.xml', 
        'Geometry/TrackerCommonData/data/tobmodule0.xml', 
        'Geometry/TrackerCommonData/data/tobmodule2.xml', 
        'Geometry/TrackerCommonData/data/tobmodule4.xml', 
        'Geometry/TrackerCommonData/data/tobrodpar.xml', 
        'Geometry/TrackerCommonData/data/tobrod0c.xml', 
        'Geometry/TrackerCommonData/data/tobrod0l.xml', 
        'Geometry/TrackerCommonData/data/tobrod0h.xml', 
        'Geometry/TrackerCommonData/data/tobrod0.xml', 
        'Geometry/TrackerCommonData/data/tobrod1l.xml', 
        'Geometry/TrackerCommonData/data/tobrod1h.xml', 
        'Geometry/TrackerCommonData/data/tobrod1.xml', 
        'Geometry/TrackerCommonData/data/tobrod2c.xml', 
        'Geometry/TrackerCommonData/data/tobrod2l.xml', 
        'Geometry/TrackerCommonData/data/tobrod2h.xml', 
        'Geometry/TrackerCommonData/data/tobrod2.xml', 
        'Geometry/TrackerCommonData/data/tobrod3l.xml', 
        'Geometry/TrackerCommonData/data/tobrod3h.xml', 
        'Geometry/TrackerCommonData/data/tobrod3.xml', 
        'Geometry/TrackerCommonData/data/tobrod4c.xml', 
        'Geometry/TrackerCommonData/data/tobrod4l.xml', 
        'Geometry/TrackerCommonData/data/tobrod4h.xml', 
        'Geometry/TrackerCommonData/data/tobrod4.xml', 
        'Geometry/TrackerCommonData/data/tobrod5l.xml', 
        'Geometry/TrackerCommonData/data/tobrod5h.xml', 
        'Geometry/TrackerCommonData/data/tobrod5.xml', 
        'Geometry/TrackerCommonData/data/tob.xml', 
        'Geometry/TrackerCommonData/data/tecmaterial.xml', 
        'Geometry/TrackerCommonData/data/tecmodpar.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule0s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule1s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule2.xml', 
        'Geometry/TrackerCommonData/data/tecmodule3.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4r.xml', 
        'Geometry/TrackerCommonData/data/tecmodule4s.xml', 
        'Geometry/TrackerCommonData/data/tecmodule5.xml', 
        'Geometry/TrackerCommonData/data/tecmodule6.xml', 
        'Geometry/TrackerCommonData/data/tecpetpar.xml', 
        'Geometry/TrackerCommonData/data/tecring0.xml', 
        'Geometry/TrackerCommonData/data/tecring1.xml', 
        'Geometry/TrackerCommonData/data/tecring2.xml', 
        'Geometry/TrackerCommonData/data/tecring3.xml', 
        'Geometry/TrackerCommonData/data/tecring4.xml', 
        'Geometry/TrackerCommonData/data/tecring5.xml', 
        'Geometry/TrackerCommonData/data/tecring6.xml', 
        'Geometry/TrackerCommonData/data/tecring0f.xml', 
        'Geometry/TrackerCommonData/data/tecring1f.xml', 
        'Geometry/TrackerCommonData/data/tecring2f.xml', 
        'Geometry/TrackerCommonData/data/tecring3f.xml', 
        'Geometry/TrackerCommonData/data/tecring4f.xml', 
        'Geometry/TrackerCommonData/data/tecring5f.xml', 
        'Geometry/TrackerCommonData/data/tecring6f.xml', 
        'Geometry/TrackerCommonData/data/tecring0b.xml', 
        'Geometry/TrackerCommonData/data/tecring1b.xml', 
        'Geometry/TrackerCommonData/data/tecring2b.xml', 
        'Geometry/TrackerCommonData/data/tecring3b.xml', 
        'Geometry/TrackerCommonData/data/tecring4b.xml', 
        'Geometry/TrackerCommonData/data/tecring5b.xml', 
        'Geometry/TrackerCommonData/data/tecring6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetalf.xml', 
        'Geometry/TrackerCommonData/data/tecpetalb.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal0b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal3b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal6b.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8f.xml', 
        'Geometry/TrackerCommonData/data/tecpetal8b.xml', 
        'Geometry/TrackerCommonData/data/tecwheel.xml', 
        'Geometry/TrackerCommonData/data/tecwheela.xml', 
        'Geometry/TrackerCommonData/data/tecwheelb.xml', 
        'Geometry/TrackerCommonData/data/tecwheelc.xml', 
        'Geometry/TrackerCommonData/data/tecwheeld.xml', 
        'Geometry/TrackerCommonData/data/tecwheel6.xml', 
        'Geometry/TrackerCommonData/data/tecservices.xml', 
        'Geometry/TrackerCommonData/data/tecbackplate.xml', 
        'Geometry/TrackerCommonData/data/tec.xml', 
        'Geometry/TrackerCommonData/data/trackermaterial.xml', 
        'Geometry/TrackerCommonData/data/tracker.xml', 
        'Geometry/TrackerCommonData/data/trackerpixbar.xml', 
        'Geometry/TrackerCommonData/data/trackerpixfwd.xml', 
        'Geometry/TrackerCommonData/data/trackertibtidservices.xml', 
        'Geometry/TrackerCommonData/data/trackertib.xml', 
        'Geometry/TrackerCommonData/data/trackertid.xml', 
        'Geometry/TrackerCommonData/data/trackertob.xml', 
        'Geometry/TrackerCommonData/data/trackertec.xml', 
        'Geometry/TrackerCommonData/data/trackerbulkhead.xml', 
        'Geometry/TrackerCommonData/data/trackerother.xml', 
        'Geometry/EcalCommonData/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/EcalCommonData/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalCommonData/data/eecon.xml', 
        'Geometry/EcalCommonData/data/eefixed.xml', 
        'Geometry/EcalCommonData/data/eehier.xml', 
        'Geometry/EcalCommonData/data/eealgo.xml', 
        'Geometry/EcalCommonData/data/escon.xml', 
        'Geometry/EcalCommonData/data/esalgo.xml', 
        'Geometry/EcalCommonData/data/eeF.xml', 
        'Geometry/EcalCommonData/data/eeB.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'Geometry/HcalCommonData/data/hcalalgo.xml', 
        'Geometry/HcalCommonData/data/hcalbarrelalgo.xml', 
        'Geometry/HcalCommonData/data/hcalendcapalgo.xml', 
        'Geometry/HcalCommonData/data/hcalouteralgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardalgo.xml', 
        'Geometry/HcalCommonData/data/hcalforwardfibre.xml', 
        'Geometry/HcalCommonData/data/hcalforwardmaterial.xml', 
        'Geometry/MuonCommonData/data/mbCommon.xml', 
        'Geometry/MuonCommonData/data/mb1.xml', 
        'Geometry/MuonCommonData/data/mb2.xml', 
        'Geometry/MuonCommonData/data/mb3.xml', 
        'Geometry/MuonCommonData/data/mb4.xml', 
        'Geometry/MuonCommonData/data/muonYoke.xml', 
        'Geometry/MuonCommonData/data/mf.xml', 
        'Geometry/ForwardCommonData/data/forward.xml', 
        'Geometry/ForwardCommonData/data/forwardshield.xml', 
        'Geometry/ForwardCommonData/data/brmrotations.xml', 
        'Geometry/ForwardCommonData/data/brm.xml', 
        'Geometry/ForwardCommonData/data/totemMaterials.xml', 
        'Geometry/ForwardCommonData/data/totemRotations.xml', 
        'Geometry/ForwardCommonData/data/totemt1.xml', 
        'Geometry/ForwardCommonData/data/totemt2.xml', 
        'Geometry/ForwardCommonData/data/ionpump.xml', 
        'Geometry/MuonCommonData/data/muonNumbering.xml', 
        'Geometry/TrackerCommonData/data/trackerStructureTopology.xml', 
        'Geometry/TrackerSimData/data/trackersens.xml', 
        'Geometry/TrackerRecoData/data/trackerRecoMaterial.xml', 
        'Geometry/EcalSimData/data/ecalsens.xml', 
        'Geometry/HcalCommonData/data/hcalsens.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'Geometry/MuonSimData/data/muonSens.xml', 
        'Geometry/DTGeometryBuilder/data/dtSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecsFilter.xml', 
        'Geometry/CSCGeometryBuilder/data/cscSpecs.xml', 
        'Geometry/RPCGeometryBuilder/data/RPCSpecs.xml', 
        'Geometry/ForwardCommonData/data/brmsens.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'Geometry/CMSCommonData/data/MagneticFieldVolumes.xml', 
        'Geometry/HcalSimData/data/HcalProdCuts.xml', 
        'Geometry/EcalSimData/data/EcalProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCuts.xml', 
        'Geometry/TrackerSimData/data/trackerProdCutsBEAM.xml', 
        'Geometry/MuonSimData/data/muonProdCuts.xml', 
        'Geometry/ForwardSimData/data/TotemProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('cms:OCMS')
)

process.tpparams9 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams8 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGFineGrainEBIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.eegeom = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalMappingRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.l1GctConfigRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCalibFunRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtTriggerMaskVetoAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskVetoAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams3 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGSlidingWindowRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams2 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGPedestalsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtStableParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtStableParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams7 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams6 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGLutGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams5 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightGroupRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.tpparams4 = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalTPGWeightIdMapRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths', 
        'channelQuality', 
        'ZSThresholds')
)

process.L1MuGMTParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuGMTParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1MuTriggerScalesRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuTriggerScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.magfield = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/normal/cmsextent.xml', 
        'Geometry/CMSCommonData/data/cms.xml', 
        'Geometry/CMSCommonData/data/cmsMagneticField.xml', 
        'Geometry/CMSCommonData/data/MagneticFieldVolumes.xml'),
    rootNodeName = cms.string('MagneticFieldVolumes:MAGF')
)

process.L1GtTriggerMaskVetoTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskVetoTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtTriggerMaskAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtParametersRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtParametersRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtTriggerMaskTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.SiStripQualityFakeESSource = cms.ESSource("SiStripQualityFakeESSource")

process.HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)

process.L1MuGMTScalesRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1MuGMTScalesRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtPrescaleFactorsTechTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsTechTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.GlobalTag = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    connect = cms.string('frontier://FrontierInt/CMS_COND_20X_GLOBALTAG'),
    globaltag = cms.untracked.string('STARTUP::All')
)

process.BTagRecord = cms.ESSource("EmptyESSource",
    recordName = cms.string('JetTagComputerRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtTriggerMaskRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtTriggerMaskRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.l1GctJcNegParsRecords = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GctJetCounterNegativeEtaRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.L1GtPrescaleFactorsAlgoTrigRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1GtPrescaleFactorsAlgoTrigRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.prefer("magfield")
process.topDiLepton2ElectronEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topDiLepton2ElectronPath')
    )
)
process.topSemiLepElectronPlus2JetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepElectronPlus2JetsPath')
    )
)
process.RecoParticleFlowFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFClusters_*_*_*', 
        'keep recoPFBlocks_*_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep *_secStep_*_*', 
        'keep *_thStep_*_*')
)
process.CSCCommonTrigger = cms.PSet(
    MaxBX = cms.int32(9),
    MinBX = cms.int32(3)
)
process.MIsoDepositParamGlobalViewMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
process.MIsoTrackAssociatorDefault = cms.PSet(
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(1.0),
        dREcal = cms.double(1.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcalPreselection = cms.double(1.0),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(1.0),
        useMuon = cms.bool(False),
        useCalo = cms.bool(True),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    )
)
process.FEVTSIMEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep *_g4SimHits_*_*', 'keep edmHepMCProduct_source_*_*', 'keep PixelDigiSimLinkedmDetSetVector_siPixelDigis_*_*', 'keep StripDigiSimLinkedmDetSetVector_siStripDigis_*_*', 'keep *_trackMCMatch_*_*', 'keep StripDigiSimLinkedmDetSetVector_muonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_*_*', 'keep EBSrFlagsSorted_*_*_*', 'keep EESrFlagsSorted_*_*_*', 'keep recoGenJets_*_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep recoGenMETs_*_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep *_MEtoEDMConverter_*_*'))
)
process.RecoLocalMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
process.RecoTauTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)
process.higgsToInvisibleEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HToInvisFilterPath')
    )
)
process.RecoMuonIsolationFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_muGlobalIsoDepositCtfTk_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muGlobalIsoDepositJets_*_*')
)
process.MIdIsoExtractorPSetBlock = cms.PSet(
    CaloExtractorPSet = cms.PSet(
        Noise_HE = cms.double(0.2),
        NoiseTow_EB = cms.double(0.04),
        Noise_EE = cms.double(0.1),
        PrintTimeReport = cms.untracked.bool(False),
        TrackAssociatorParameters = cms.PSet(
            muonMaxDistanceSigmaX = cms.double(0.0),
            muonMaxDistanceSigmaY = cms.double(0.0),
            CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
            dRHcal = cms.double(1.0),
            dREcal = cms.double(1.0),
            CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
            useEcal = cms.bool(False),
            dREcalPreselection = cms.double(1.0),
            HORecHitCollectionLabel = cms.InputTag("horeco"),
            dRMuon = cms.double(9999.0),
            crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
            muonMaxDistanceX = cms.double(5.0),
            muonMaxDistanceY = cms.double(5.0),
            useHO = cms.bool(False),
            accountForTrajectoryChangeCalo = cms.bool(False),
            DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
            EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
            dRHcalPreselection = cms.double(1.0),
            useMuon = cms.bool(False),
            useCalo = cms.bool(True),
            EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
            dRMuonPreselection = cms.double(0.2),
            truthMatch = cms.bool(False),
            HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
            useHcal = cms.bool(False)
        ),
        NoiseTow_EE = cms.double(0.15),
        Threshold_HO = cms.double(0.5),
        DR_Veto_E = cms.double(0.07),
        Noise_HO = cms.double(0.2),
        DR_Max = cms.double(1.0),
        Noise_EB = cms.double(0.025),
        Threshold_E = cms.double(0.2),
        Noise_HB = cms.double(0.2),
        UseRecHitsFlag = cms.bool(False),
        PropagatorName = cms.string('SteppingHelixPropagatorAny'),
        Threshold_H = cms.double(0.5),
        DR_Veto_H = cms.double(0.1),
        DepositLabel = cms.untracked.string('Cal'),
        ComponentName = cms.string('CaloExtractorByAssociator'),
        DR_Veto_HO = cms.double(0.1),
        DepositInstanceLabels = cms.vstring('ecal', 
            'hcal', 
            'ho')
    ),
    TrackExtractorPSet = cms.PSet(
        Diff_z = cms.double(0.2),
        inputTrackCollection = cms.InputTag("generalTracks"),
        BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
        ComponentName = cms.string('TrackExtractor'),
        DR_Max = cms.double(1.0),
        Diff_r = cms.double(0.1),
        Chi2Prob_Min = cms.double(-1.0),
        DR_Veto = cms.double(0.01),
        NHits_Min = cms.uint32(0),
        Chi2Ndof_Max = cms.double(1e+64),
        Pt_Min = cms.double(-1.0),
        DepositLabel = cms.untracked.string(''),
        BeamlineOption = cms.string('BeamSpotFromEvent')
    )
)
process.topFullyHadronicEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.MuonTrackingRegionCommon = cms.PSet(
    MuonTrackingRegionBuilder = cms.PSet(
        VertexCollection = cms.string('pixelVertices'),
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        Eta_fixed = cms.double(0.2),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        Rescale_Dz = cms.double(3.0),
        Rescale_phi = cms.double(3.0),
        DeltaR = cms.double(0.2),
        DeltaZ_Region = cms.double(15.9),
        Rescale_eta = cms.double(3.0),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        Eta_min = cms.double(0.013),
        Phi_fixed = cms.double(0.2),
        EscapePt = cms.double(1.5),
        UseFixedRegion = cms.bool(False),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phi_min = cms.double(0.02),
        UseVertex = cms.bool(False)
    )
)
process.DF_ME1A = cms.PSet(
    tanThetaMax = cms.double(1.2),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    tanPhiMax = cms.double(0.5),
    dXclusBoxMax = cms.double(8.0),
    preClustering = cms.untracked.bool(False),
    chi2Max = cms.double(5000.0),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(30),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(8.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(8.0)
)
process.L1TriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*')
)
process.SimMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep StripDigiSimLinkedmDetSetVector_muonCSCDigis_*_*', 
        'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_*_*')
)
process.RecoLocalCaloRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*')
)
process.RecoGenMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)
process.HLTriggerFEVT = cms.PSet(
    outputCommands = (cms.untracked.vstring('keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*')+cms.untracked.vstring('keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*'))
)
process.HLTMuJetsAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltMCJetCorJetIcone5"), cms.InputTag("hltIterativeCone5CaloJets"), cms.InputTag("hltMCJetCorJetIcone5Regional"), cms.InputTag("hltIterativeCone5CaloJetsRegional")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltMuJetsL3IsoFiltered"), cms.InputTag("hltMuJetsHLT1jet40"), cms.InputTag("hltMuNoL2IsoJetsL3IsoFiltered"), cms.InputTag("hltMuNoL2IsoJetsHLT1jet40"), cms.InputTag("hltMuNoIsoJetsL3PreFiltered"), 
        cms.InputTag("hltMuNoIsoJetsHLT1jet50")),
    outputCommands = cms.untracked.vstring()
)
process.SimTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_trackMCMatch_*_*')
)
process.zToEEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_zToEE_*_*', 
        'keep *_zToEEOneTrack_*_*', 
        'keep *_zToEEOneSuperCluster_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 
        'keep *_zToEEGenParticlesMatch_*_*', 
        'keep *_zToEEOneTrackGenParticlesMatch_*_*', 
        'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*')
)
process.zToTauTauETauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToTauTauETauHLTPath')
    )
)
process.HLTAlcaRecoEcalPi0StreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*_pi0EcalRecHitsEB_*')
)
process.RecoGenJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*')
)
process.topSemiLepMuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTAlcaRecoEcalPi0StreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)
process.FlatVtxSmearingParameters = cms.PSet(
    MaxZ = cms.double(5.3),
    MaxX = cms.double(0.0015),
    MaxY = cms.double(0.0015),
    MinX = cms.double(-0.0015),
    MinY = cms.double(-0.0015),
    MinZ = cms.double(-5.3)
)
process.MIsoCaloExtractorByAssociatorTowersBlock = cms.PSet(
    Noise_HE = cms.double(0.2),
    DR_Veto_H = cms.double(0.1),
    UseRecHitsFlag = cms.bool(False),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(1.0),
        dREcal = cms.double(1.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcalPreselection = cms.double(1.0),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(1.0),
        useMuon = cms.bool(False),
        useCalo = cms.bool(True),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    ),
    NoiseTow_EE = cms.double(0.15),
    Threshold_HO = cms.double(0.5),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    Noise_HO = cms.double(0.2),
    Noise_EE = cms.double(0.1),
    Noise_EB = cms.double(0.025),
    DR_Veto_HO = cms.double(0.1),
    Noise_HB = cms.double(0.2),
    PrintTimeReport = cms.untracked.bool(False),
    NoiseTow_EB = cms.double(0.04),
    Threshold_H = cms.double(0.5),
    DR_Veto_E = cms.double(0.07),
    DepositLabel = cms.untracked.string('Cal'),
    ComponentName = cms.string('CaloExtractorByAssociator'),
    Threshold_E = cms.double(0.2),
    DepositInstanceLabels = cms.vstring('ecal', 
        'hcal', 
        'ho')
)
process.HLTXchannelRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 
        'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 
        'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 
        'keep *_hltConeIsolationL25ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 
        'keep *_hltConeIsolationL3ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltMCJetCorJetIcone5*_*_*', 
        'keep *_hltIterativeCone5CaloJets*_*_*')
)
process.HLTMuonPlusBLifetimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.CSCSegAlgoST = cms.PSet(
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
    algo_psets = cms.VPSet(cms.PSet(
        curvePenaltyThreshold = cms.untracked.double(0.85),
        minHitsPerSegment = cms.untracked.int32(3),
        yweightPenaltyThreshold = cms.untracked.double(1.0),
        curvePenalty = cms.untracked.double(2.0),
        dXclusBoxMax = cms.untracked.double(4.0),
        BrutePruning = cms.untracked.bool(False),
        yweightPenalty = cms.untracked.double(1.5),
        hitDropLimit5Hits = cms.untracked.double(0.8),
        preClustering = cms.untracked.bool(True),
        hitDropLimit4Hits = cms.untracked.double(0.6),
        hitDropLimit6Hits = cms.untracked.double(0.3333),
        maxRecHitsInCluster = cms.untracked.int32(20),
        CSCDebug = cms.untracked.bool(False),
        onlyBestSegment = cms.untracked.bool(False),
        Pruning = cms.untracked.bool(False),
        dYclusBoxMax = cms.untracked.double(8.0)
    ), 
        cms.PSet(
            curvePenaltyThreshold = cms.untracked.double(0.85),
            minHitsPerSegment = cms.untracked.int32(3),
            yweightPenaltyThreshold = cms.untracked.double(1.0),
            curvePenalty = cms.untracked.double(2.0),
            dXclusBoxMax = cms.untracked.double(4.0),
            BrutePruning = cms.untracked.bool(False),
            yweightPenalty = cms.untracked.double(1.5),
            hitDropLimit5Hits = cms.untracked.double(0.8),
            preClustering = cms.untracked.bool(True),
            hitDropLimit4Hits = cms.untracked.double(0.6),
            hitDropLimit6Hits = cms.untracked.double(0.3333),
            maxRecHitsInCluster = cms.untracked.int32(24),
            CSCDebug = cms.untracked.bool(False),
            onlyBestSegment = cms.untracked.bool(False),
            Pruning = cms.untracked.bool(False),
            dYclusBoxMax = cms.untracked.double(8.0)
        )),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
        1, 1, 1, 1)
)
process.CSCSegAlgoSK = cms.PSet(
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
        )),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
        1, 1, 1, 1)
)
process.MIsoTrackExtractorCtfBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("generalTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)
process.MinHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinHitsTrajectoryFilter'),
    minimumNumberOfHits = cms.int32(5)
)
process.jcSetup1 = cms.PSet(
    jetCountersNegativeWheel = cms.VPSet(cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1')
    ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_1', 
                'JC_centralEta_6')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_11')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_11', 
                'JC_centralEta_6')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_19')
        )),
    jetCountersPositiveWheel = cms.VPSet(cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1')
    ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_1', 
                'JC_centralEta_6')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_11')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_11', 
                'JC_centralEta_6')
        ), 
        cms.PSet(
            cutDescriptionList = cms.vstring('JC_minRank_19')
        ))
)
process.JetTagSoftMuonHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*')
)
process.SimG4CoreAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.zToTauTauETauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*')
)
process.FastjetNoPU = cms.PSet(
    Active_Area_Repeats = cms.int32(0),
    UE_Subtraction = cms.string('no'),
    Ghost_EtaMax = cms.double(0.0),
    GhostArea = cms.double(1.0)
)
process.zToMuMuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuPath', 
            'zToMuMuOneTrackPath', 
            'zToMuMuOneStandAloneMuonTrackPath')
    )
)
process.SingleTauMETHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauMETJets_*_*', 
        'keep *_hltL2SingleTauMETIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltAssociatorL25SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL25SingleTauMET*_*_*', 
        'keep *_hltIsolatedL25SingleTauMET*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 
        'keep *_hltAssociatorL3SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL3SingleTauMET*_*_*', 
        'keep *_hltIsolatedL3SingleTauMET*_*_*')
)
process.topSemiLepElectronPlus1JetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepElectronPlus1JetPath')
    )
)
process.SimMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.DF_ME1234_1 = cms.PSet(
    tanThetaMax = cms.double(1.2),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    tanPhiMax = cms.double(0.5),
    dXclusBoxMax = cms.double(8.0),
    preClustering = cms.untracked.bool(False),
    chi2Max = cms.double(5000.0),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(10),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(8.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(8.0)
)
process.j2tParametersCALO = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)
process.ST_ME1A = cms.PSet(
    preClustering = cms.untracked.bool(True),
    minHitsPerSegment = cms.untracked.int32(3),
    yweightPenaltyThreshold = cms.untracked.double(1.0),
    curvePenalty = cms.untracked.double(2.0),
    dXclusBoxMax = cms.untracked.double(4.0),
    BrutePruning = cms.untracked.bool(False),
    yweightPenalty = cms.untracked.double(1.5),
    hitDropLimit5Hits = cms.untracked.double(0.8),
    curvePenaltyThreshold = cms.untracked.double(0.85),
    hitDropLimit4Hits = cms.untracked.double(0.6),
    Pruning = cms.untracked.bool(False),
    maxRecHitsInCluster = cms.untracked.int32(24),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.untracked.bool(False),
    hitDropLimit6Hits = cms.untracked.double(0.3333),
    dYclusBoxMax = cms.untracked.double(8.0)
)
process.SimGeneralFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 
        'drop *_electrontruth_*_*', 
        'keep *_mergedtruth_*_*')
)
process.HLTElectronMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.SimCalorimetryFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep EBSrFlagsSorted_*_*_*', 
        'keep EESrFlagsSorted_*_*_*')
)
process.HLTBTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltCaloTowersTau*_*_*', 
        'keep *_hltTowerMakerForAll_*_*', 
        'keep *_hltTowerMakerForTaus_*_*', 
        'keep *_hltSiPixelRecHits_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltSiStripRecHits_*_*', 
        'keep *_hltSiStripMatchedRecHits_*_*', 
        'keep *_hltIcone5Tau1*_*_*', 
        'keep *_hltIcone5Tau2*_*_*', 
        'keep *_hltIcone5Tau3*_*_*', 
        'keep *_hltIcone5Tau4*_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 
        'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumu_*_*', 
        'keep *_hltCtfWithMaterialTracksMumu_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumuk_*_*', 
        'keep *_hltCtfWithMaterialTracksMumuk_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukAllConeTracks_*_*', 
        'keep *_hltL2SingleTauJets_*_*', 
        'keep *_hltL2SingleTauIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau*_*_*', 
        'keep *_hltConeIsolationL25SingleTau*_*_*', 
        'keep *_hltIsolatedL25SingleTau*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 
        'keep *_hltAssociatorL3SingleTau*_*_*', 
        'keep *_hltConeIsolationL3SingleTau*_*_*', 
        'keep *_hltIsolatedL3SingleTau*_*_*', 
        'keep *_hltL2SingleTauMETJets_*_*', 
        'keep *_hltL2SingleTauMETIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltAssociatorL25SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL25SingleTauMET*_*_*', 
        'keep *_hltIsolatedL25SingleTauMET*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 
        'keep *_hltAssociatorL3SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL3SingleTauMET*_*_*', 
        'keep *_hltIsolatedL3SingleTauMET*_*_*', 
        'keep *_hltL2DoubleTauJets_*_*', 
        'keep *_hltL2DoubleTauIsolation*_*_*', 
        'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 
        'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 
        'keep *_hltIsolatedL25PixelTau*_*_*')
)
process.heavyChHiggsToTauNuEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('heavyChHiggsToTauNuFilterPath')
    )
)
process.ST_ME1234 = cms.PSet(
    preClustering = cms.untracked.bool(True),
    minHitsPerSegment = cms.untracked.int32(3),
    yweightPenaltyThreshold = cms.untracked.double(1.0),
    curvePenalty = cms.untracked.double(2.0),
    dXclusBoxMax = cms.untracked.double(4.0),
    BrutePruning = cms.untracked.bool(False),
    yweightPenalty = cms.untracked.double(1.5),
    hitDropLimit5Hits = cms.untracked.double(0.8),
    curvePenaltyThreshold = cms.untracked.double(0.85),
    hitDropLimit4Hits = cms.untracked.double(0.6),
    Pruning = cms.untracked.bool(False),
    maxRecHitsInCluster = cms.untracked.int32(20),
    CSCDebug = cms.untracked.bool(False),
    onlyBestSegment = cms.untracked.bool(False),
    hitDropLimit6Hits = cms.untracked.double(0.3333),
    dYclusBoxMax = cms.untracked.double(8.0)
)
process.RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.L1TriggerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*')
)
process.RecoJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetExtender_*_*')
)
process.RecoMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
process.HLTHcalIsolatedTrackRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*')
)
process.ChargeSignificanceTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ChargeSignificanceTrajectoryFilter'),
    chargeSignificance = cms.double(-1.0)
)
process.DisplacedMumuHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMumuPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumu_*_*', 
        'keep *_hltCtfWithMaterialTracksMumu_*_*', 
        'keep *_hltMuTracks_*_*')
)
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('SingleMuPt1.cfi energy: nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.HLTSpecialFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 
        'keep *_*_pi0EcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
process.HLTSpecialRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 
        'keep *_*_pi0EcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
process.DTCombinatorialPatternReco4DAlgo_ParamDrift = cms.PSet(
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
        Reco2DAlgoConfig = cms.PSet(
            segmCleanerMode = cms.int32(1),
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
            AlphaMaxPhi = cms.double(1.0),
            MaxAllowedHits = cms.uint32(50),
            nSharedHitsMax = cms.int32(2),
            AlphaMaxTheta = cms.double(0.1),
            debug = cms.untracked.bool(False),
            recAlgo = cms.string('DTParametrizedDriftAlgo'),
            nUnSharedHitsMin = cms.int32(2)
        ),
        debug = cms.untracked.bool(False),
        recAlgo = cms.string('DTParametrizedDriftAlgo'),
        nUnSharedHitsMin = cms.int32(2),
        AllDTRecHits = cms.bool(True)
    )
)
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)
process.RecoTauTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)
process.higgsToTauTauLeptonTauEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('higgsToTauTauLeptonTauPath')
    )
)
process.MIsoJetExtractorBlock = cms.PSet(
    PrintTimeReport = cms.untracked.bool(False),
    ExcludeMuonVeto = cms.bool(True),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(0.5),
        dREcal = cms.double(0.5),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcalPreselection = cms.double(0.5),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.5),
        useMuon = cms.bool(False),
        useCalo = cms.bool(True),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    ),
    ComponentName = cms.string('JetExtractor'),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    JetCollectionLabel = cms.InputTag("sisCone5CaloJets"),
    DR_Veto = cms.double(0.1),
    Threshold = cms.double(5.0)
)
process.TriggerSummaryAOD = cms.PSet(
    collections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"), 
        cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"), cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"), cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates"), 
        cms.InputTag("hltMCJetCorJetIcone5"), cms.InputTag("hltIterativeCone5CaloJets"), cms.InputTag("hltMCJetCorJetIcone5Regional"), cms.InputTag("hltIterativeCone5CaloJetsRegional"), cms.InputTag("hltMet"), 
        cms.InputTag("hltHtMet"), cms.InputTag("hltHtMetIC5"), cms.InputTag("hltIsolatedL3SingleTau"), cms.InputTag("hltIsolatedL3SingleTauMET"), cms.InputTag("hltIsolatedL25PixelTau"), 
        cms.InputTag("hltIsolatedL3SingleTauRelaxed"), cms.InputTag("hltIsolatedL3SingleTauMETRelaxed"), cms.InputTag("hltIsolatedL25PixelTauRelaxed"), cms.InputTag("hltBLifetimeL3Jets"), cms.InputTag("hltBSoftmuonL25Jets"), 
        cms.InputTag("hltMuTracks"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMumukAllConeTracks"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"), 
        cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"), cms.InputTag("hltIsolatedTauJetsSelectorL3ElectronTau"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("hltBLifetimeL3BJetTags"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBSoftmuonL3BJetTags"), cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), 
        cms.InputTag("hltPixelMatchElectronsL1IsoForHLT"), cms.InputTag("hltPixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("hltBLifetimeL3BJetTags"), cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), 
        cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("hltMCJetCorJetIcone5"), cms.InputTag("hltIterativeCone5CaloJets"), cms.InputTag("hltMCJetCorJetIcone5Regional"), cms.InputTag("hltIterativeCone5CaloJetsRegional"), cms.InputTag("hltIsolPixelTrackProd")),
    filters = cms.VInputTag(cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"), cms.InputTag("hltL1IsoDoubleExclPhotonTrackIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonPrescaledTrackIsolFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleExclElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronZeePMMassFilter"), cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt5PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoNoTrkIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter"), cms.InputTag("hltSingleMuIsoL3IsoFiltered"), cms.InputTag("hltSingleMuNoIsoL3PreFiltered"), cms.InputTag("hltDiMuonIsoL3IsoFiltered"), 
        cms.InputTag("hltDiMuonNoIsoL3PreFiltered"), cms.InputTag("hltZMML3Filtered"), cms.InputTag("hltJpsiMML3Filtered"), cms.InputTag("hltUpsilonMML3Filtered"), cms.InputTag("hltMultiMuonNoIsoL3PreFiltered"), 
        cms.InputTag("hltSameSignMuL3IsoFiltered"), cms.InputTag("hltExclDiMuonIsoL3IsoFiltered"), cms.InputTag("hltSingleMuPrescale3L3PreFiltered"), cms.InputTag("hltSingleMuPrescale5L3PreFiltered"), cms.InputTag("hltSingleMuPrescale710L3PreFiltered"), 
        cms.InputTag("hltSingleMuPrescale77L3PreFiltered"), cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm"), 
        cms.InputTag("hltSingleMuStartupL2PreFiltered"), cms.InputTag("hlt1jet200"), cms.InputTag("hlt1jet150"), cms.InputTag("hlt1jet110"), cms.InputTag("hlt1jet60"), 
        cms.InputTag("hlt1jet30"), cms.InputTag("hlt2jet150"), cms.InputTag("hlt3jet85"), cms.InputTag("hlt4jet60"), cms.InputTag("hlt1MET65"), 
        cms.InputTag("hlt1MET55"), cms.InputTag("hlt1MET30"), cms.InputTag("hlt1MET20"), cms.InputTag("hlt2jetAco"), cms.InputTag("hlt1jet180"), 
        cms.InputTag("hlt2jet125"), cms.InputTag("hlt3jet60"), cms.InputTag("hlt4jet35"), cms.InputTag("hlt1jet1METAco"), cms.InputTag("hlt2jetvbf"), 
        cms.InputTag("hltnv"), cms.InputTag("hltPhi2metAco"), cms.InputTag("hltPhiJet1metAco"), cms.InputTag("hltPhiJet2metAco"), cms.InputTag("hltPhiJet1Jet2Aco"), 
        cms.InputTag("hlt1SumET120"), cms.InputTag("hlt1HT400"), cms.InputTag("hlt1HT350"), cms.InputTag("hltRapGap"), cms.InputTag("hltdijetave110"), 
        cms.InputTag("hltdijetave150"), cms.InputTag("hltdijetave200"), cms.InputTag("hltdijetave30"), cms.InputTag("hltdijetave60"), cms.InputTag("hltFilterL3SingleTau"), 
        cms.InputTag("hltFilterL3SingleTauMET"), cms.InputTag("hltFilterL25PixelTau"), cms.InputTag("hltFilterL3SingleTauRelaxed"), cms.InputTag("hltFilterL3SingleTauMETRelaxed"), cms.InputTag("hltFilterL25PixelTauRelaxed"), 
        cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltBSoftmuonByDRL3filter"), cms.InputTag("hltDisplacedJpsitoMumuFilter"), cms.InputTag("hltDisplacedJpsitoMumuFilterRelaxed"), 
        cms.InputTag("hltmmkFilter"), cms.InputTag("hltMuonTauIsoL3IsoFiltered"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau"), cms.InputTag("hltElectronTrackIsolFilterElectronTau"), cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau"), 
        cms.InputTag("hltFilterIsolatedTauJetsL3ElectronTau"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsElectronTau"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltMuBIsoL3IsoFiltered"), cms.InputTag("hltBSoftmuonL3filter"), 
        cms.InputTag("hltMuBIsoL3IsoFiltered"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltElBElectronTrackIsolFilter"), cms.InputTag("hltemuL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltemuNonIsoL1IsoTrackIsolFilter"), 
        cms.InputTag("hltEMuL3MuonIsoFilter"), cms.InputTag("hltNonIsoEMuL3MuonPreFilter"), cms.InputTag("hltMuJetsL3IsoFiltered"), cms.InputTag("hltMuJetsHLT1jet40"), cms.InputTag("hltMuNoL2IsoJetsL3IsoFiltered"), 
        cms.InputTag("hltMuNoL2IsoJetsHLT1jet40"), cms.InputTag("hltMuNoIsoJetsL3PreFiltered"), cms.InputTag("hltMuNoIsoJetsHLT1jet50"), cms.InputTag("isolPixelTrackFilter"))
)
process.MIsoTrackAssociatorTowers = cms.PSet(
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(1.0),
        dREcal = cms.double(1.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcalPreselection = cms.double(1.0),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(1.0),
        useMuon = cms.bool(False),
        useCalo = cms.bool(True),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    )
)
process.HLTHcalIsolatedTrackAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltIsolPixelTrackProd")),
    triggerFilters = cms.VInputTag(cms.InputTag("isolPixelTrackFilter")),
    outputCommands = cms.untracked.vstring()
)
process.SimCalorimetryAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.BeamSpotRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)
process.RecoBTagRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*')
)
process.HLTJetMETAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltMCJetCorJetIcone5"), cms.InputTag("hltIterativeCone5CaloJets"), cms.InputTag("hltMCJetCorJetIcone5Regional"), cms.InputTag("hltIterativeCone5CaloJetsRegional"), cms.InputTag("hltMet"), 
        cms.InputTag("hltHtMet"), cms.InputTag("hltHtMetIC5")),
    triggerFilters = cms.VInputTag(cms.InputTag("hlt1jet200"), cms.InputTag("hlt1jet150"), cms.InputTag("hlt1jet110"), cms.InputTag("hlt1jet60"), cms.InputTag("hlt1jet30"), 
        cms.InputTag("hlt2jet150"), cms.InputTag("hlt3jet85"), cms.InputTag("hlt4jet60"), cms.InputTag("hlt1MET65"), cms.InputTag("hlt1MET55"), 
        cms.InputTag("hlt1MET30"), cms.InputTag("hlt1MET20"), cms.InputTag("hlt2jetAco"), cms.InputTag("hlt1jet180"), cms.InputTag("hlt2jet125"), 
        cms.InputTag("hlt3jet60"), cms.InputTag("hlt4jet35"), cms.InputTag("hlt1jet1METAco"), cms.InputTag("hlt2jetvbf"), cms.InputTag("hltnv"), 
        cms.InputTag("hltPhi2metAco"), cms.InputTag("hltPhiJet1metAco"), cms.InputTag("hltPhiJet2metAco"), cms.InputTag("hltPhiJet1Jet2Aco"), cms.InputTag("hlt1SumET120"), 
        cms.InputTag("hlt1HT400"), cms.InputTag("hlt1HT350"), cms.InputTag("hltRapGap"), cms.InputTag("hltdijetave110"), cms.InputTag("hltdijetave150"), 
        cms.InputTag("hltdijetave200"), cms.InputTag("hltdijetave30"), cms.InputTag("hltdijetave60")),
    outputCommands = cms.untracked.vstring()
)
process.MaxLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxLostHitsTrajectoryFilter'),
    maxLostHits = cms.int32(1)
)
process.RecoEgammaFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelMatchGsfElectrons_*_*', 
        'keep *_pixelMatchGsfFit_*_*', 
        'keep *_electronPixelSeeds_*_*', 
        'keep *_conversions_*_*', 
        'keep *_photons_*_*', 
        'keep *_ckfOutInTracksFromConversions_*_*', 
        'keep *_ckfInOutTracksFromConversions_*_*')
)
process.HLTBTauAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltIsolatedL3SingleTau"), cms.InputTag("hltIsolatedL3SingleTauMET"), cms.InputTag("hltIsolatedL25PixelTau"), cms.InputTag("hltIsolatedL3SingleTauRelaxed"), cms.InputTag("hltIsolatedL3SingleTauMETRelaxed"), 
        cms.InputTag("hltIsolatedL25PixelTauRelaxed"), cms.InputTag("hltBLifetimeL3Jets"), cms.InputTag("hltBSoftmuonL25Jets"), cms.InputTag("hltMuTracks"), cms.InputTag("hltMuTracks"), 
        cms.InputTag("hltMumukAllConeTracks")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltFilterL3SingleTau"), cms.InputTag("hltFilterL3SingleTauMET"), cms.InputTag("hltFilterL25PixelTau"), cms.InputTag("hltFilterL3SingleTauRelaxed"), cms.InputTag("hltFilterL3SingleTauMETRelaxed"), 
        cms.InputTag("hltFilterL25PixelTauRelaxed"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltBSoftmuonByDRL3filter"), cms.InputTag("hltDisplacedJpsitoMumuFilter"), 
        cms.InputTag("hltDisplacedJpsitoMumuFilterRelaxed"), cms.InputTag("hltmmkFilter")),
    outputCommands = cms.untracked.vstring()
)
process.HLTElectronTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 
        'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 
        'keep *_hltConeIsolationL25ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 
        'keep *_hltConeIsolationL3ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*')
)
process.topSemiLepMuonPlus2JetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepMuonPlus2JetsPath')
    )
)
process.BeamSpotAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)
process.HLTXchannelAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau"), cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"), 
        cms.InputTag("hltIsolatedTauJetsSelectorL3ElectronTau"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBLifetimeL3BJetTags"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBSoftmuonL3BJetTags"), 
        cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1IsoForHLT"), cms.InputTag("hltPixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("hltBLifetimeL3BJetTags"), 
        cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("hltL2MuonCandidates"), cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltMCJetCorJetIcone5"), cms.InputTag("hltIterativeCone5CaloJets"), cms.InputTag("hltMCJetCorJetIcone5Regional"), 
        cms.InputTag("hltIterativeCone5CaloJetsRegional")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltMuonTauIsoL3IsoFiltered"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau"), cms.InputTag("hltElectronTrackIsolFilterElectronTau"), cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau"), cms.InputTag("hltFilterIsolatedTauJetsL3ElectronTau"), 
        cms.InputTag("hltFilterPixelTrackIsolatedTauJetsElectronTau"), cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltMuBIsoL3IsoFiltered"), cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltMuBIsoL3IsoFiltered"), 
        cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltElBElectronTrackIsolFilter"), cms.InputTag("hltemuL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltemuNonIsoL1IsoTrackIsolFilter"), cms.InputTag("hltEMuL3MuonIsoFilter"), 
        cms.InputTag("hltNonIsoEMuL3MuonPreFilter"), cms.InputTag("hltMuJetsL3IsoFiltered"), cms.InputTag("hltMuJetsHLT1jet40"), cms.InputTag("hltMuNoL2IsoJetsL3IsoFiltered"), cms.InputTag("hltMuNoL2IsoJetsHLT1jet40"), 
        cms.InputTag("hltMuNoIsoJetsL3PreFiltered"), cms.InputTag("hltMuNoIsoJetsHLT1jet50")),
    outputCommands = cms.untracked.vstring()
)
process.HLTMuonTauAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorMuonTau")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltMuonTauIsoL3IsoFiltered"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsMuonTau")),
    outputCommands = cms.untracked.vstring()
)
process.JetTagLifetimeHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*')
)
process.ecalLocalRecoRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*')
)
process.HLTHcalIsolatedTrackFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelTracks_*_*', 
        'keep *_hltIsolPixelTrackProd_*_*', 
        'keep *_hltL1sIsoTrack_*_*', 
        'keep *_hltGtDigis_*_*', 
        'keep l1extraL1JetParticles_hltL1extraParticles_*_*')
)
process.MIsoDepositViewMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("muons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
process.topFullyHadronicEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicJetsPath', 
            'topFullyHadronicBJetsPath')
    )
)
process.RecoEcalRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*')
)
process.MuonUpdatorAtVertex = cms.PSet(
    MuonUpdatorAtVertexParameters = cms.PSet(
        MaxChi2 = cms.double(1000000.0),
        BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
        Propagator = cms.string('SteppingHelixPropagatorOpposite'),
        BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
    )
)
process.SimG4CoreRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmHepMCProduct_source_*_*', 
        'keep SimTracks_g4SimHits_*_*', 
        'keep SimVertexs_g4SimHits_*_*')
)
process.HLTAlcaRecoHcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
process.MIsoCaloExtractorHLTBlock = cms.PSet(
    DR_Veto_H = cms.double(0.1),
    Vertex_Constraint_Z = cms.bool(False),
    Threshold_H = cms.double(0.5),
    ComponentName = cms.string('CaloExtractor'),
    Threshold_E = cms.double(0.2),
    DR_Max = cms.double(1.0),
    DR_Veto_E = cms.double(0.07),
    Weight_E = cms.double(1.5),
    Vertex_Constraint_XY = cms.bool(False),
    DepositLabel = cms.untracked.string('EcalPlusHcal'),
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    Weight_H = cms.double(1.0)
)
process.RecoMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_muGlobalIsoDepositCtfTk_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muGlobalIsoDepositJets_*_*')
)
process.RecoJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloJets_*_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
        'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_kt4JetExtender_*_*')
)
process.SimMuonFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muonCSCDigis_*_*', 
        'keep *_muonDTDigis_*_*', 
        'keep *_muonRPCDigis_*_*')
)
process.BeamSpotFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_offlineBeamSpot_*_*')
)
process.HLTMuonTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 
        'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*')
)
process.DF_ME1234_2 = cms.PSet(
    tanThetaMax = cms.double(2.0),
    minHitsPerSegment = cms.int32(3),
    dPhiFineMax = cms.double(0.025),
    tanPhiMax = cms.double(0.8),
    dXclusBoxMax = cms.double(8.0),
    preClustering = cms.untracked.bool(False),
    chi2Max = cms.double(5000.0),
    maxRatioResidualPrune = cms.double(3.0),
    minHitsForPreClustering = cms.int32(10),
    CSCSegmentDebug = cms.untracked.bool(False),
    dRPhiFineMax = cms.double(12.0),
    nHitsPerClusterIsShower = cms.int32(20),
    minLayersApart = cms.int32(2),
    Pruning = cms.untracked.bool(False),
    dYclusBoxMax = cms.double(12.0)
)
process.HLTJetMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMCJetCorJetIcone5*_*_*', 
        'keep *_hltIterativeCone5CaloJets*_*_*', 
        'keep *_hltMet_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltHtMetIC5_*_*')
)
process.DTParametrizedDriftAlgo = cms.PSet(
    recAlgo = cms.string('DTParametrizedDriftAlgo'),
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
    )
)
process.RECOEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'drop recoSuperClusters_hybridSuperClusters_*_*', 'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_sisCone5CaloJets_*_*', 'keep *_sisCone7CaloJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoGsfTracks_pixelMatchGsfFit_*_*', 'keep recoGsfTrackExtras_pixelMatchGsfFit_*_*', 'keep recoTrackExtras_pixelMatchGsfFit_*_*', 'keep TrackingRecHitsOwned_pixelMatchGsfFit_*_*', 'keep recoElectronPixelSeeds_electronPixelSeeds_*_*', 'keep recoPhotons_photons_*_*', 'keep recoConversions_conversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*')+cms.untracked.vstring('keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*'))
)
process.zToMuMuGoldenEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToMuMuGoldenHLTPath')
    )
)
process.RecoMETAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
process.NominalCollision3VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.025),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.1),
    SigmaZ = cms.double(5.3)
)
process.SISConeJetParameters = cms.PSet(
    maxPasses = cms.int32(0),
    JetPtMin = cms.double(1.0),
    coneOverlapThreshold = cms.double(0.75),
    caching = cms.bool(False),
    protojetPtMin = cms.double(0.0),
    splitMergeScale = cms.string('pttilde')
)
process.RecoTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 
        'keep *_generalTracks_*_*', 
        'keep *_rsWithMaterialTracks_*_*')
)
process.HLTXchannelFEVT = cms.PSet(
    outputCommands = (cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*')+cms.untracked.vstring('keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*'))
)
process.higgsToZZ4LeptonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.electronPixelSeedConfiguration = cms.PSet(
    searchInTIDTEC = cms.bool(True),
    HighPtThreshold = cms.double(35.0),
    r2MinF = cms.double(-0.15),
    DeltaPhi1Low = cms.double(0.23),
    DeltaPhi1High = cms.double(0.08),
    ePhiMin1 = cms.double(-0.125),
    PhiMin2 = cms.double(-0.002),
    LowPtThreshold = cms.double(5.0),
    z2MinB = cms.double(-0.09),
    dynamicPhiRoad = cms.bool(True),
    ePhiMax1 = cms.double(0.075),
    DeltaPhi2 = cms.double(0.004),
    SizeWindowENeg = cms.double(0.675),
    rMaxI = cms.double(0.2),
    rMinI = cms.double(-0.2),
    r2MaxF = cms.double(0.15),
    pPhiMin1 = cms.double(-0.075),
    pPhiMax1 = cms.double(0.125),
    SCEtCut = cms.double(5.0),
    z2MaxB = cms.double(0.09),
    hcalRecHits = cms.InputTag("hbhereco"),
    maxHOverE = cms.double(0.2),
    PhiMax2 = cms.double(0.002)
)
process.HLTElectronTauAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelTrackIsolatedTauJetsSelectorElectronTau"), cms.InputTag("hltIsolatedTauJetsSelectorL3ElectronTau")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltElectronTrackIsolFilterElectronTau"), cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau"), cms.InputTag("hltFilterIsolatedTauJetsL3ElectronTau"), cms.InputTag("hltFilterPixelTrackIsolatedTauJetsElectronTau")),
    outputCommands = cms.untracked.vstring()
)
process.topDiLeptonMuonXEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topDiLeptonMuonXPath')
    )
)
process.RecoBTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*')
)
process.SimCalorimetryFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalDigis_*_*', 
        'keep *_ecalPreshowerDigis_*_*', 
        'keep *_ecalTriggerPrimitiveDigis_*_*', 
        'keep *_hcalDigis_*_*', 
        'keep *_hcalTriggerPrimitiveDigis_*_*')
)
process.combinedSecondaryVertexCommon = cms.PSet(
    useTrackWeights = cms.bool(True),
    pseudoMultiplicityMin = cms.uint32(2),
    correctVertexMass = cms.bool(True),
    trackPseudoSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(2.0),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    pseudoVertexV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.05)
    ),
    charmCut = cms.double(1.5),
    minimumTrackWeight = cms.double(0.5),
    trackPairV0Filter = cms.PSet(
        k0sMassWindow = cms.double(0.03)
    ),
    trackMultiplicityMin = cms.uint32(3),
    trackSelection = cms.PSet(
        totalHitsMin = cms.uint32(0),
        jetDeltaRMax = cms.double(0.3),
        qualityClass = cms.string('any'),
        pixelHitsMin = cms.uint32(0),
        sip3dSigMin = cms.double(-99999.9),
        sip3dSigMax = cms.double(99999.9),
        sip2dValMax = cms.double(99999.9),
        normChi2Max = cms.double(99999.9),
        ptMin = cms.double(0.0),
        sip2dSigMax = cms.double(99999.9),
        sip2dSigMin = cms.double(-99999.9),
        sip3dValMax = cms.double(99999.9),
        sip2dValMin = cms.double(-99999.9),
        sip3dValMin = cms.double(-99999.9)
    ),
    trackSort = cms.string('sip2dSig')
)
process.RecoEcalFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalRecHit_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_islandSuperClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'keep *_correctedIslandBarrelSuperClusters_*_*', 
        'keep *_correctedIslandEndcapSuperClusters_*_*', 
        'keep *_correctedHybridSuperClusters_*_*', 
        'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep *_preshowerClusterShape_*_*')
)
process.MaxHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxHitsTrajectoryFilter'),
    maxNumberOfHits = cms.int32(-1)
)
process.RecoLocalTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTEgamma_RECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*')
)
process.topSemiLepElectronEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTMuJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltMCJetCorJetIcone5*_*_*', 
        'keep *_hltIterativeCone5CaloJets*_*_*')
)
process.RecoParticleFlowAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep recoTracks_secStep_*_*', 
        'keep recoTracks_thStep_*_*')
)
process.CSCSegAlgoTC = cms.PSet(
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
        )),
    parameters_per_chamber_type = cms.vint32(2, 1, 1, 1, 1, 
        1, 1, 1, 1)
)
process.MIsoTrackAssociatorHits = cms.PSet(
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(1.0),
        dREcal = cms.double(1.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(1.0),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(1.0),
        useMuon = cms.bool(False),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    )
)
process.RecoGenJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenJets_*_*_*', 
        'keep *_genParticle_*_*')
)
process.HLTriggerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep *_hltL1GtRecord_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*')
)
process.RecoBTagAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*')
)
process.RecoLocalCaloAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 
        'drop *_electrontruth_*_*', 
        'keep *_mergedtruth_*_*')
)
process.MIsoCaloExtractorHcalBlock = cms.PSet(
    DR_Veto_H = cms.double(0.1),
    Vertex_Constraint_Z = cms.bool(False),
    Threshold_H = cms.double(0.5),
    ComponentName = cms.string('CaloExtractor'),
    Threshold_E = cms.double(0.2),
    DR_Max = cms.double(1.0),
    DR_Veto_E = cms.double(0.07),
    Weight_E = cms.double(0.0),
    Vertex_Constraint_XY = cms.bool(False),
    DepositLabel = cms.untracked.string('EcalPlusHcal'),
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    Weight_H = cms.double(1.0)
)
process.NominalCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.000142),
    BetaStar = cms.double(55.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.0),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.05),
    SigmaZ = cms.double(5.3)
)
process.NominalCollision1VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.025),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.05),
    SigmaZ = cms.double(5.3)
)
process.HLTEgamma_FEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltCkfL1IsoTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 
        'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*')
)
process.MaxConsecLostHitsTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MaxConsecLostHitsTrajectoryFilter'),
    maxConsecLostHits = cms.int32(1)
)
process.MEtoEDMConverterRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*')
)
process.HLTMuonPlusBLifetimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.RecoBTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.MIsoDepositGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("globalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('TrackCollection'),
    MuonTrackRefType = cms.string('track')
)
process.SiPixelGainCalibrationServiceParameters = cms.PSet(

)
process.CkfBaseTrajectoryFilter_block = cms.PSet(
    chargeSignificance = cms.double(-1.0),
    minHitsMinPt = cms.int32(3),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(-1),
    maxConsecLostHits = cms.int32(1),
    minimumNumberOfHits = cms.int32(5),
    nSigmaMinPt = cms.double(5.0),
    minPt = cms.double(0.9)
)
process.RecoEcalAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*')
)
process.HLTMuonPlusBSoftMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.HLTEgamma_AOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"), 
        cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"), cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"), cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltL1IsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonTrackIsolFilter"), cms.InputTag("hltL1IsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoDoublePhotonDoubleEtFilter"), cms.InputTag("hltL1NonIsoSingleEMHighEtTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoSinglePhotonEMVeryHighEtEtFilter"), cms.InputTag("hltL1IsoDoubleExclPhotonTrackIsolFilter"), cms.InputTag("hltL1IsoSinglePhotonPrescaledTrackIsolFilter"), cms.InputTag("hltL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1IsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleExclElectronTrackIsolFilter"), cms.InputTag("hltL1IsoDoubleElectronZeePMMassFilter"), cms.InputTag("hltL1IsoLargeWindowSingleElectronTrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoLargeWindowSingleElectronTrackIsolFilter"), cms.InputTag("hltL1IsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoLargeWindowDoubleElectronTrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt25TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt12TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt8TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt5PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt5TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTnoIsoSingleElectronEt8TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt18TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronLWEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSingleElectronEt12TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt45TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt30TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt8PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt10PixelMatchFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronLWonlyPMEt12PixelMatchFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSinglePhotonEt40TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoDoublePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTLooseIsoSingleElectronEt15TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoNoTrkIsoDoublePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt10TrackIsolFilter"), cms.InputTag("hltL1NonIsoSinglePhotonEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt20TrackIsolFilter"), cms.InputTag("hltL1NonIsoNoTrkIsoSinglePhotonEt25TrackIsolFilter"), 
        cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt15TrackIsolFilter"), cms.InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt5TrackIsolFilter")),
    outputCommands = cms.untracked.vstring()
)
process.MuonTrackLoaderForGLB = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        DoSmoothing = cms.bool(True),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        VertexConstraint = cms.bool(False)
    )
)
process.SimTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PixelDigiSimLinkedmDetSetVector_siPixelDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_siStripDigis_*_*', 
        'keep *_trackMCMatch_*_*')
)
process.HLTMuonPlusBSoftMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.heavyChHiggsToTauNuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.RecoLocalCaloFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hbhereco_*_*', 
        'keep *_hfreco_*_*', 
        'keep *_horeco_*_*', 
        'keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*')
)
process.HLTSpecialAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltIsolPixelTrackProd")),
    triggerFilters = cms.VInputTag(cms.InputTag("isolPixelTrackFilter")),
    outputCommands = cms.untracked.vstring()
)
process.RecoGenMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)
process.HLTMuonAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltL2MuonCandidates")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltSingleMuIsoL3IsoFiltered"), cms.InputTag("hltSingleMuNoIsoL3PreFiltered"), cms.InputTag("hltDiMuonIsoL3IsoFiltered"), cms.InputTag("hltDiMuonNoIsoL3PreFiltered"), cms.InputTag("hltZMML3Filtered"), 
        cms.InputTag("hltJpsiMML3Filtered"), cms.InputTag("hltUpsilonMML3Filtered"), cms.InputTag("hltMultiMuonNoIsoL3PreFiltered"), cms.InputTag("hltSameSignMuL3IsoFiltered"), cms.InputTag("hltExclDiMuonIsoL3IsoFiltered"), 
        cms.InputTag("hltSingleMuPrescale3L3PreFiltered"), cms.InputTag("hltSingleMuPrescale5L3PreFiltered"), cms.InputTag("hltSingleMuPrescale710L3PreFiltered"), cms.InputTag("hltSingleMuPrescale77L3PreFiltered"), cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2cm"), 
        cms.InputTag("hltDiMuonNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2cm"), cms.InputTag("hltSingleMuNoIsoL3PreFilteredRelaxedVtx2mm"), cms.InputTag("hltSingleMuStartupL2PreFiltered")),
    outputCommands = cms.untracked.vstring()
)
process.RecoLocalTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelClusters_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelRecHits_*_*', 
        'keep *_siStripRecHits_*_*', 
        'keep *_siStripMatchedRecHits_*_*')
)
process.j2tParametersVX = cms.PSet(
    tracks = cms.InputTag("generalTracks"),
    coneSize = cms.double(0.5)
)
process.MuonTrackLoaderForCosmic = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        PutTrajectoryIntoEvent = cms.untracked.bool(False),
        VertexConstraint = cms.bool(False),
        AllowNoVertex = cms.untracked.bool(True),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorAny'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        DoSmoothing = cms.bool(False)
    )
)
process.TC_ME1234 = cms.PSet(
    dPhiFineMax = cms.double(0.02),
    verboseInfo = cms.untracked.bool(True),
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    dPhiMax = cms.double(0.003),
    chi2ndfProbMin = cms.double(0.0001),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(6.0),
    dRPhiMax = cms.double(1.2)
)
process.MuonTrackLoaderForSTA = cms.PSet(
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
process.KtJetParameters = cms.PSet(
    Strategy = cms.string('Best')
)
process.SK_ME1234 = cms.PSet(
    dPhiFineMax = cms.double(0.025),
    verboseInfo = cms.untracked.bool(True),
    chi2Max = cms.double(99999.0),
    dPhiMax = cms.double(0.003),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(8.0),
    dRPhiMax = cms.double(8.0)
)
process.MumukHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMumukPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumuk_*_*', 
        'keep *_hltCtfWithMaterialTracksMumuk_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukAllConeTracks_*_*')
)
process.HLTElectronMuonAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1Iso"), cms.InputTag("hltPixelMatchElectronsL1NonIso"), cms.InputTag("hltL3MuonCandidates"), 
        cms.InputTag("hltL2MuonCandidates")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltemuL1IsoSingleElectronTrackIsolFilter"), cms.InputTag("hltemuNonIsoL1IsoTrackIsolFilter"), cms.InputTag("hltEMuL3MuonIsoFilter"), cms.InputTag("hltNonIsoEMuL3MuonPreFilter")),
    outputCommands = cms.untracked.vstring()
)
process.higgsTo2GammaEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.higgsToWW2LeptonsEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.RecoBTauAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.CaloJetParameters = cms.PSet(
    src = cms.InputTag("caloTowers"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    jetPtMin = cms.double(0.0),
    jetType = cms.untracked.string('CaloJet'),
    inputEMin = cms.double(0.0)
)
process.FEVTSIMDIGIEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep *_g4SimHits_*_*', 'keep edmHepMCProduct_source_*_*', 'keep recoGenJets_*_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep recoGenMETs_*_*_*', 'keep *_siPixelDigis_*_*', 'keep *_siStripDigis_*_*', 'keep *_trackMCMatch_*_*', 'keep *_muonCSCDigis_*_*', 'keep *_muonDTDigis_*_*', 'keep *_muonRPCDigis_*_*', 'keep *_ecalDigis_*_*', 'keep *_ecalPreshowerDigis_*_*', 'keep *_ecalTriggerPrimitiveDigis_*_*', 'keep *_hcalDigis_*_*', 'keep *_hcalTriggerPrimitiveDigis_*_*', 'keep *_cscTriggerPrimitiveDigis_*_*', 'keep *_dtTriggerPrimitiveDigis_*_*', 'keep *_rpcTriggerDigis_*_*', 'keep *_rctDigis_*_*', 'keep *_csctfDigis_*_*', 'keep *_dttfDigis_*_*', 'keep *_gctDigis_*_*', 'keep *_gmtDigis_*_*', 'keep *_gtDigis_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep *_MEtoEDMConverter_*_*'))
)
process.HLTAlcaRecoEcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)
process.DTCombinatorialPatternReco2DAlgo_ParamDrift = cms.PSet(
    Reco2DAlgoConfig = cms.PSet(
        segmCleanerMode = cms.int32(1),
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
        AlphaMaxPhi = cms.double(1.0),
        MaxAllowedHits = cms.uint32(50),
        nSharedHitsMax = cms.int32(2),
        AlphaMaxTheta = cms.double(0.1),
        debug = cms.untracked.bool(False),
        recAlgo = cms.string('DTParametrizedDriftAlgo'),
        nUnSharedHitsMin = cms.int32(2)
    ),
    Reco2DAlgoName = cms.string('DTCombinatorialPatternReco')
)
process.RecoGenJetsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*')
)
process.topFullyHadronicBJetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicBJetsPath')
    )
)
process.RecoGenMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGenMETs_*_*_*')
)
process.SimTrackerFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelDigis_*_*', 
        'keep *_siStripDigis_*_*', 
        'keep *_trackMCMatch_*_*')
)
process.CSCSegAlgoDF = cms.PSet(
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
    algo_psets = cms.VPSet(cms.PSet(
        preClustering = cms.untracked.bool(False),
        minHitsPerSegment = cms.int32(3),
        dPhiFineMax = cms.double(0.025),
        chi2Max = cms.double(5000.0),
        dXclusBoxMax = cms.double(8.0),
        tanThetaMax = cms.double(1.2),
        tanPhiMax = cms.double(0.5),
        maxRatioResidualPrune = cms.double(3.0),
        minHitsForPreClustering = cms.int32(10),
        CSCSegmentDebug = cms.untracked.bool(False),
        dRPhiFineMax = cms.double(8.0),
        nHitsPerClusterIsShower = cms.int32(20),
        minLayersApart = cms.int32(2),
        Pruning = cms.untracked.bool(False),
        dYclusBoxMax = cms.double(8.0)
    ), 
        cms.PSet(
            preClustering = cms.untracked.bool(False),
            minHitsPerSegment = cms.int32(3),
            dPhiFineMax = cms.double(0.025),
            chi2Max = cms.double(5000.0),
            dXclusBoxMax = cms.double(8.0),
            tanThetaMax = cms.double(2.0),
            tanPhiMax = cms.double(0.8),
            maxRatioResidualPrune = cms.double(3.0),
            minHitsForPreClustering = cms.int32(10),
            CSCSegmentDebug = cms.untracked.bool(False),
            dRPhiFineMax = cms.double(12.0),
            nHitsPerClusterIsShower = cms.int32(20),
            minLayersApart = cms.int32(2),
            Pruning = cms.untracked.bool(False),
            dYclusBoxMax = cms.double(12.0)
        ), 
        cms.PSet(
            preClustering = cms.untracked.bool(False),
            minHitsPerSegment = cms.int32(3),
            dPhiFineMax = cms.double(0.025),
            chi2Max = cms.double(5000.0),
            dXclusBoxMax = cms.double(8.0),
            tanThetaMax = cms.double(1.2),
            tanPhiMax = cms.double(0.5),
            maxRatioResidualPrune = cms.double(3.0),
            minHitsForPreClustering = cms.int32(30),
            CSCSegmentDebug = cms.untracked.bool(False),
            dRPhiFineMax = cms.double(8.0),
            nHitsPerClusterIsShower = cms.int32(20),
            minLayersApart = cms.int32(2),
            Pruning = cms.untracked.bool(False),
            dYclusBoxMax = cms.double(8.0)
        )),
    parameters_per_chamber_type = cms.vint32(3, 1, 2, 2, 1, 
        2, 1, 2, 1)
)
process.GlobalTrajectoryBuilderCommon = cms.PSet(
    Chi2ProbabilityCut = cms.double(30.0),
    Direction = cms.int32(0),
    Chi2CutCSC = cms.double(150.0),
    HitThreshold = cms.int32(1),
    MuonHitsOption = cms.int32(1),
    TrackRecHitBuilder = cms.string('WithTrackAngle'),
    RPCRecSegmentLabel = cms.InputTag("rpcRecHits"),
    Chi2CutRPC = cms.double(1.0),
    CSCRecSegmentLabel = cms.InputTag("cscSegments"),
    MuonTrackingRegionBuilder = cms.PSet(
        VertexCollection = cms.string('pixelVertices'),
        EtaR_UpperLimit_Par1 = cms.double(0.25),
        Eta_fixed = cms.double(0.2),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        Rescale_Dz = cms.double(3.0),
        Rescale_phi = cms.double(3.0),
        DeltaR = cms.double(0.2),
        DeltaZ_Region = cms.double(15.9),
        Rescale_eta = cms.double(3.0),
        PhiR_UpperLimit_Par2 = cms.double(0.2),
        Eta_min = cms.double(0.013),
        Phi_fixed = cms.double(0.2),
        EscapePt = cms.double(1.5),
        UseFixedRegion = cms.bool(False),
        PhiR_UpperLimit_Par1 = cms.double(0.6),
        EtaR_UpperLimit_Par2 = cms.double(0.15),
        Phi_min = cms.double(0.02),
        UseVertex = cms.bool(False)
    ),
    Chi2CutDT = cms.double(10.0),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True)
    ),
    GlobalMuonTrackMatcher = cms.PSet(
        MinP = cms.double(2.5),
        Chi2Cut = cms.double(50.0),
        MinPt = cms.double(1.0),
        DeltaDCut = cms.double(10.0),
        DeltaRCut = cms.double(0.2)
    ),
    PtCut = cms.double(1.0),
    TrackerPropagator = cms.string('SteppingHelixPropagatorAny'),
    DTRecSegmentLabel = cms.InputTag("dt4DSegments")
)
process.SimTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_trackMCMatch_*_*')
)
process.RecoVertexFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)
process.GeneratorInterfaceAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmGenInfoProduct_source_*_*', 
        'keep recoGenParticles_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)
process.TrackingToolsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*')
)
process.AODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'drop *', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_rsWithMaterialTracks_*_*', 
        'keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetExtender_*_*', 
        'keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*', 
        'keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*', 
        'keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*', 
        'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoGsfTracks_pixelMatchGsfFit_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*', 
        'drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep recoTracks_secStep_*_*', 
        'keep recoTracks_thStep_*_*', 
        'keep *_offlineBeamSpot_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep *_hltL1GtRecord_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*', 
        'keep *_MEtoEDMConverter_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep recoGenParticles_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*', 
        'keep *_trackMCMatch_*_*', 
        'keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*', 
        'keep recoGenMETs_*_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)
process.MuonUpdatorAtVertexAnyDirection = cms.PSet(
    MuonUpdatorAtVertexParameters = cms.PSet(
        MaxChi2 = cms.double(1000000.0),
        BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
        Propagator = cms.string('SteppingHelixPropagatorAny'),
        BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
    )
)
process.RecoJetsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 
        'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetTracksAssociatorAtVertex_*_*', 
        'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 
        'keep *_kt4JetExtender_*_*')
)
process.RecoEgammaAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoGsfTracks_pixelMatchGsfFit_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*')
)
process.zToEEEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToEEPath', 
            'zToEEOneTrackPath', 
            'zToEEOneSuperClusterPath')
    )
)
process.HLTElectronTauFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 
        'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 
        'keep *_hltConeIsolationL25ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 
        'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 
        'keep *_hltConeIsolationL3ElectronTau_*_*', 
        'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltCkfL1IsoTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 
        'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*')
)
process.RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep  FEDRawDataCollection_rawDataCollector_*_*')
)
process.RecoMuonIsolationRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_muGlobalIsoDepositCtfTk_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muGlobalIsoDepositJets_*_*')
)
process.RECOSIMANAEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'drop *', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'drop recoSuperClusters_hybridSuperClusters_*_*', 'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_sisCone5CaloJets_*_*', 'keep *_sisCone7CaloJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoGsfTracks_pixelMatchGsfFit_*_*', 'keep recoGsfTrackExtras_pixelMatchGsfFit_*_*', 'keep recoTrackExtras_pixelMatchGsfFit_*_*', 'keep TrackingRecHitsOwned_pixelMatchGsfFit_*_*', 'keep recoElectronPixelSeeds_electronPixelSeeds_*_*', 'keep recoPhotons_photons_*_*', 'keep recoConversions_conversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*')+cms.untracked.vstring('keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*')+cms.untracked.vstring('keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep edmHepMCProduct_source_*_*', 'keep SimTracks_g4SimHits_*_*', 'keep SimVertexs_g4SimHits_*_*', 'keep *_trackMCMatch_*_*', 'keep recoGenMETs_*_*_*', 'keep *_kt4GenJets_*_*', 'keep *_kt6GenJets_*_*', 'keep *_iterativeCone5GenJets_*_*', 'keep *_sisCone5GenJets_*_*', 'keep *_sisCone7GenJets_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep *_MEtoEDMConverter_*_*', 'keep recoCandidatesOwned_genParticles_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep *_goodMuons_*_*', 'keep *_goodTracks_*_*', 'keep *_goodStandAloneMuonTracks_*_*', 'keep *_muonIsolations_*_*', 'keep *_goodZToMuMu_*_*', 'keep *_goodZToMuMuOneTrack_*_*', 'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 'keep *_goodZMCMatch_*_*', 'drop *_*_*_HLT', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCandidatesOwned_allMuons_*_*', 'keep *_allMuonIsolations_*_*', 'keep *_zToMuMuGolden_*_*', 'keep *_allMuonsGenParticlesMatch_*_*', 'keep *_zToMuMuGoldenGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_zToEE_*_*', 'keep *_zToEEOneTrack_*_*', 'keep *_zToEEOneSuperCluster_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *_zToEEGenParticlesMatch_*_*', 'keep *_zToEEOneTrackGenParticlesMatch_*_*', 'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *'))
)
process.RecoMETFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*')
)
process.TrackingToolsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_GsfGlobalElectronTest_*_*')
)
process.HLTAlcaRecoHcalPhiSymStreamAOD = cms.PSet(
    triggerCollections = cms.VInputTag(),
    triggerFilters = cms.VInputTag(),
    outputCommands = cms.untracked.vstring()
)
process.AODSIMANAEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'drop *', 
        'drop *', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_rsWithMaterialTracks_*_*', 
        'keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetExtender_*_*', 
        'keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*', 
        'keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*', 
        'keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*', 
        'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoGsfTracks_pixelMatchGsfFit_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*', 
        'drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep recoTracks_secStep_*_*', 
        'keep recoTracks_thStep_*_*', 
        'keep *_offlineBeamSpot_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep *_hltL1GtRecord_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*', 
        'keep *_MEtoEDMConverter_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep recoGenParticles_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*', 
        'keep *_trackMCMatch_*_*', 
        'keep *_kt4GenJets_*_*', 
        'keep *_kt6GenJets_*_*', 
        'keep *_iterativeCone5GenJets_*_*', 
        'keep *_sisCone5GenJets_*_*', 
        'keep *_sisCone7GenJets_*_*', 
        'keep *_genParticle_*_*', 
        'keep recoGenMETs_*_*_*', 
        'keep *_MEtoEDMConverter_*_*', 
        'keep recoCandidatesOwned_genParticles_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep *_goodMuons_*_*', 
        'keep *_goodTracks_*_*', 
        'keep *_goodStandAloneMuonTracks_*_*', 
        'keep *_muonIsolations_*_*', 
        'keep *_goodZToMuMu_*_*', 
        'keep *_goodZToMuMuOneTrack_*_*', 
        'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 
        'keep *_goodZMCMatch_*_*', 
        'drop *_*_*_HLT', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCandidatesOwned_allMuons_*_*', 
        'keep *_allMuonIsolations_*_*', 
        'keep *_zToMuMuGolden_*_*', 
        'keep *_allMuonsGenParticlesMatch_*_*', 
        'keep *_zToMuMuGoldenGenParticlesMatch_*_*', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_zToEE_*_*', 
        'keep *_zToEEOneTrack_*_*', 
        'keep *_zToEEOneSuperCluster_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 
        'keep *_zToEEGenParticlesMatch_*_*', 
        'keep *_zToEEOneTrackGenParticlesMatch_*_*', 
        'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 
        'keep *')
)
process.zToMuMuGoldenEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCandidatesOwned_allMuons_*_*', 
        'keep *_allMuonIsolations_*_*', 
        'keep *_zToMuMuGolden_*_*', 
        'keep *_allMuonsGenParticlesMatch_*_*', 
        'keep *_zToMuMuGoldenGenParticlesMatch_*_*')
)
process.CompositeTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet()
)
process.MEtoEDMConverterFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*')
)
process.RecoMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MuonSeed_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_muGlobalIsoDepositCtfTk_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muGlobalIsoDepositJets_*_*')
)
process.HLTMuonPlusBSoftMuonAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBSoftmuonL3BJetTags")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltBSoftmuonL3filter"), cms.InputTag("hltMuBIsoL3IsoFiltered")),
    outputCommands = cms.untracked.vstring()
)
process.HLTAlcaRecoEcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*')
)
process.HLTAlcaRecoHcalPhiSymStreamFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 
        'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')
)
process.MIsoCaloExtractorEcalBlock = cms.PSet(
    DR_Veto_H = cms.double(0.1),
    Vertex_Constraint_Z = cms.bool(False),
    Threshold_H = cms.double(0.5),
    ComponentName = cms.string('CaloExtractor'),
    Threshold_E = cms.double(0.2),
    DR_Max = cms.double(1.0),
    DR_Veto_E = cms.double(0.07),
    Weight_E = cms.double(1.0),
    Vertex_Constraint_XY = cms.bool(False),
    DepositLabel = cms.untracked.string('EcalPlusHcal'),
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    Weight_H = cms.double(0.0)
)
process.FEVTSIMANAEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep *_g4SimHits_*_*', 'keep edmHepMCProduct_source_*_*', 'keep PixelDigiSimLinkedmDetSetVector_siPixelDigis_*_*', 'keep StripDigiSimLinkedmDetSetVector_siStripDigis_*_*', 'keep *_trackMCMatch_*_*', 'keep StripDigiSimLinkedmDetSetVector_muonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_*_*', 'keep EBSrFlagsSorted_*_*_*', 'keep EESrFlagsSorted_*_*_*', 'keep recoGenJets_*_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep recoGenMETs_*_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep *_MEtoEDMConverter_*_*', 'keep recoCandidatesOwned_genParticles_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep *_goodMuons_*_*', 'keep *_goodTracks_*_*', 'keep *_goodStandAloneMuonTracks_*_*', 'keep *_muonIsolations_*_*', 'keep *_goodZToMuMu_*_*', 'keep *_goodZToMuMuOneTrack_*_*', 'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 'keep *_goodZMCMatch_*_*', 'drop *_*_*_HLT', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCandidatesOwned_allMuons_*_*', 'keep *_allMuonIsolations_*_*', 'keep *_zToMuMuGolden_*_*', 'keep *_allMuonsGenParticlesMatch_*_*', 'keep *_zToMuMuGoldenGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_zToEE_*_*', 'keep *_zToEEOneTrack_*_*', 'keep *_zToEEOneSuperCluster_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *_zToEEGenParticlesMatch_*_*', 'keep *_zToEEOneTrackGenParticlesMatch_*_*', 'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *'))
)
process.HiggsAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *')
)
process.RecoMuonIsolationAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*')
)
process.DoubleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2DoubleTauJets_*_*', 
        'keep *_hltL2DoubleTauIsolation*_*_*', 
        'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 
        'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 
        'keep *_hltIsolatedL25PixelTau*_*_*')
)
process.HLTMuonTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 
        'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 
        'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 
        'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*')
)
process.EarlyCollisionVtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(200.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.0),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.0322),
    SigmaZ = cms.double(5.3)
)
process.SK_ME1A = cms.PSet(
    dPhiFineMax = cms.double(0.025),
    verboseInfo = cms.untracked.bool(True),
    chi2Max = cms.double(99999.0),
    dPhiMax = cms.double(0.025),
    wideSeg = cms.double(3.0),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(3.0),
    dRPhiMax = cms.double(8.0)
)
process.MIsoDepositParamGlobalViewIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
process.HLTriggerRECO = cms.PSet(
    outputCommands = (cms.untracked.vstring('keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*')+cms.untracked.vstring('keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*'))
)
process.ElectroWeakAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticles_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep *_goodMuons_*_*', 
        'keep *_goodTracks_*_*', 
        'keep *_goodStandAloneMuonTracks_*_*', 
        'keep *_muonIsolations_*_*', 
        'keep *_goodZToMuMu_*_*', 
        'keep *_goodZToMuMuOneTrack_*_*', 
        'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 
        'keep *_goodZMCMatch_*_*', 
        'drop *_*_*_HLT', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCandidatesOwned_allMuons_*_*', 
        'keep *_allMuonIsolations_*_*', 
        'keep *_zToMuMuGolden_*_*', 
        'keep *_allMuonsGenParticlesMatch_*_*', 
        'keep *_zToMuMuGoldenGenParticlesMatch_*_*', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_zToEE_*_*', 
        'keep *_zToEEOneTrack_*_*', 
        'keep *_zToEEOneSuperCluster_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 
        'keep *_zToEEGenParticlesMatch_*_*', 
        'keep *_zToEEOneTrackGenParticlesMatch_*_*', 
        'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*', 
        'keep recoCandidatesOwned_genParticleCandidates_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoCandidatesOwned_allElectrons_*_*', 
        'keep recoCandidatesOwned_allTracks_*_*', 
        'keep recoCandidatesOwned_allSuperClusters_*_*', 
        'keep *_allElectronIsolations_*_*', 
        'keep *_allTrackIsolations_*_*', 
        'keep *_allSuperClusterIsolations_*_*', 
        'keep *_allElectronsGenParticlesMatch_*_*', 
        'keep *_allTracksGenParticlesLeptonMatch_*_*', 
        'keep *_allSuperClustersGenParticlesLeptonMatch_*_*')
)
process.topSemiLepMuonPlus1JetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topSemiLepMuonPlus1JetPath')
    )
)
process.GeneratorInterfaceFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep edmHepMCProduct_source_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)
process.SingleTauHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2SingleTauJets_*_*', 
        'keep *_hltL2SingleTauIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau*_*_*', 
        'keep *_hltConeIsolationL25SingleTau*_*_*', 
        'keep *_hltIsolatedL25SingleTau*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 
        'keep *_hltAssociatorL3SingleTau*_*_*', 
        'keep *_hltConeIsolationL3SingleTau*_*_*', 
        'keep *_hltIsolatedL3SingleTau*_*_*')
)
process.higgsTo2GammaEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HggFilterPath')
    )
)
process.ecalLocalRecoAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTElectronPlusBLifetimeAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL1IsoRecoEcalCandidate"), cms.InputTag("hltL1NonIsoRecoEcalCandidate"), cms.InputTag("hltPixelMatchElectronsL1IsoForHLT"), cms.InputTag("hltPixelMatchElectronsL1NonIsoForHLT"), cms.InputTag("hltBLifetimeL3BJetTags")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltElBElectronTrackIsolFilter")),
    outputCommands = cms.untracked.vstring()
)
process.HLTAlcaRecoEcalPi0StreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_*_pi0EcalRecHitsEB_*')
)
process.higgsToInvisibleEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTMuJetsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*', 
        'keep *_hltMCJetCorJetIcone5*_*_*', 
        'keep *_hltIterativeCone5CaloJets*_*_*')
)
process.MIsoTrackExtractorBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("generalTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)
process.IconeJetParameters = cms.PSet(
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0)
)
process.RecoLocalTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.MuonTrackLoaderForL3 = cms.PSet(
    TrackLoaderParameters = cms.PSet(
        PutTkTrackIntoEvent = cms.untracked.bool(True),
        VertexConstraint = cms.bool(False),
        MuonSeededTracksInstance = cms.untracked.string('L2Seeded'),
        Smoother = cms.string('KFSmootherForMuonTrackLoader'),
        MuonUpdatorAtVertexParameters = cms.PSet(
            MaxChi2 = cms.double(1000000.0),
            BeamSpotPosition = cms.vdouble(0.0, 0.0, 0.0),
            Propagator = cms.string('SteppingHelixPropagatorOpposite'),
            BeamSpotPositionErrors = cms.vdouble(0.1, 0.1, 5.3)
        ),
        SmoothTkTrack = cms.untracked.bool(False),
        DoSmoothing = cms.bool(True)
    )
)
process.RecoBTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.RecoMuonIsolationParamGlobal = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_muParamGlobalIsoDepositGsTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalEcal_*_*', 
        'keep *_muParamGlobalIsoDepositCalHcal_*_*', 
        'keep *_muParamGlobalIsoDepositCtfTk_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muParamGlobalIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muParamGlobalIsoDepositJets_*_*')
)
process.topDiLepton2ElectronEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.GeneratorInterfaceRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_genParticles_*_*', 
        'keep *_genEventWeight_*_*', 
        'keep *_genEventScale_*_*', 
        'keep *_genEventPdfInfo_*_*', 
        'keep edmHepMCProduct_source_*_*', 
        'keep edmGenInfoProduct_source_*_*', 
        'keep *_genEventProcID_*_*', 
        'keep *_genEventRunInfo_*_*', 
        'keep edmAlpgenInfoProduct_source_*_*')
)
process.AODEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_islandBasicClusters_*_*', 
        'keep *_hybridSuperClusters_*_*', 
        'drop recoSuperClusters_hybridSuperClusters_*_*', 
        'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 
        'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 
        'keep recoSuperClusters_correctedEndcapSuperClustersWithPreshower_*_*', 
        'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 
        'keep recoTracks_GsfGlobalElectronTest_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_rsWithMaterialTracks_*_*', 
        'keep *_kt4CaloJets_*_*', 
        'keep *_kt6CaloJets_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_sisCone5CaloJets_*_*', 
        'keep *_sisCone7CaloJets_*_*', 
        'keep *_caloTowers_*_*', 
        'keep *_towerMaker_*_*', 
        'keep *_ic5JetTracksAssociatorAtVertex_*_*', 
        'keep *_iterativeCone5JetExtender_*_*', 
        'keep *_sisCone5JetExtender_*_*', 
        'keep *_kt4JetExtender_*_*', 
        'keep recoCaloMETs_*_*_*', 
        'keep recoMETs_*_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*', 
        'keep *_btagSoftElectrons_*_*', 
        'keep *_softElectronTagInfos_*_*', 
        'keep *_softMuonTagInfos_*_*', 
        'keep *_softElectronBJetTags_*_*', 
        'keep *_softMuonBJetTags_*_*', 
        'keep *_softMuonNoIPBJetTags_*_*', 
        'keep *_impactParameterTagInfos_*_*', 
        'keep *_jetProbabilityBJetTags_*_*', 
        'keep *_jetBProbabilityBJetTags_*_*', 
        'keep *_trackCountingHighPurBJetTags_*_*', 
        'keep *_trackCountingHighEffBJetTags_*_*', 
        'keep *_impactParameterMVABJetTags_*_*', 
        'keep *_combinedSVBJetTags_*_*', 
        'keep *_combinedSVMVABJetTags_*_*', 
        'keep *_secondaryVertexTagInfos_*_*', 
        'keep *_simpleSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexBJetTags_*_*', 
        'keep *_combinedSecondaryVertexMVABJetTags_*_*', 
        'keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*', 
        'keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*', 
        'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoGsfTracks_pixelMatchGsfFit_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*', 
        'drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep recoTracks_secStep_*_*', 
        'keep recoTracks_thStep_*_*', 
        'keep *_offlineBeamSpot_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*', 
        'keep edmTriggerResults_*_*_*', 
        'keep triggerTriggerEvent_*_*_*', 
        'keep *_hltL1GtRecord_*_*', 
        'keep *_hltL1GtObjectMap_*_*', 
        'keep *_hltL1extraParticles_*_*', 
        'keep *_hltOfflineBeamSpot_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)
process.ecalLocalRecoFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ecalWeightUncalibRecHit_*_*', 
        'keep *_ecalPreshowerRecHit_*_*', 
        'keep *_ecalRecHit_*_*')
)
process.GenJetParameters = cms.PSet(
    src = cms.InputTag("genParticlesForJets"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.0),
    jetPtMin = cms.double(5.0),
    jetType = cms.untracked.string('GenJet'),
    inputEMin = cms.double(0.0)
)
process.MIsoDepositGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("globalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('TrackCollection'),
    MuonTrackRefType = cms.string('track')
)
process.SimG4CoreFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_g4SimHits_*_*', 
        'keep edmHepMCProduct_source_*_*')
)
process.FastjetParameters = cms.PSet(

)
process.higgsToTauTauLeptonTauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *')
)
process.ThresholdPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ThresholdPtTrajectoryFilter'),
    nSigmaThresholdPt = cms.double(5.0),
    minHitsThresholdPt = cms.int32(3),
    thresholdPt = cms.double(10.0)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.RecoMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_standAloneMuons_*_*', 
        'keep recoTrackExtras_standAloneMuons_*_*', 
        'keep TrackingRecHitsOwned_standAloneMuons_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTrackExtras_globalMuons_*_*', 
        'keep TrackingRecHitsOwned_globalMuons_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoMuons_trackerMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep recoCaloMuons_calomuons_*_*', 
        'keep *_muIsoDepositTk_*_*', 
        'keep *_muIsoDepositCalByAssociatorTowers_*_*', 
        'keep *_muIsoDepositCalByAssociatorHits_*_*', 
        'keep *_muIsoDepositJets_*_*')
)
process.SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.RecoTauTagFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_coneIsolationTauJetTags_*_*', 
        'keep *_particleFlowJetCandidates_*_*', 
        'keep *_iterativeCone5PFJets_*_*', 
        'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 
        'keep *_pfRecoTauTagInfoProducer_*_*', 
        'keep *_pfRecoTauProducer_*_*', 
        'keep *_pfRecoTauDiscriminationByIsolation_*_*', 
        'keep *_pfRecoTauProducerHighEfficiency_*_*', 
        'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 
        'keep *_caloRecoTauTagInfoProducer_*_*', 
        'keep *_caloRecoTauProducer_*_*', 
        'keep *_caloRecoTauDiscriminationByIsolation_*_*')
)
process.MIsoDepositParamGlobalMultiIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(True),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('track')
)
process.FEVTSIMDIGIANAEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep *_g4SimHits_*_*', 'keep edmHepMCProduct_source_*_*', 'keep recoGenJets_*_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep recoGenMETs_*_*_*', 'keep *_siPixelDigis_*_*', 'keep *_siStripDigis_*_*', 'keep *_trackMCMatch_*_*', 'keep *_muonCSCDigis_*_*', 'keep *_muonDTDigis_*_*', 'keep *_muonRPCDigis_*_*', 'keep *_ecalDigis_*_*', 'keep *_ecalPreshowerDigis_*_*', 'keep *_ecalTriggerPrimitiveDigis_*_*', 'keep *_hcalDigis_*_*', 'keep *_hcalTriggerPrimitiveDigis_*_*', 'keep *_cscTriggerPrimitiveDigis_*_*', 'keep *_dtTriggerPrimitiveDigis_*_*', 'keep *_rpcTriggerDigis_*_*', 'keep *_rctDigis_*_*', 'keep *_csctfDigis_*_*', 'keep *_dttfDigis_*_*', 'keep *_gctDigis_*_*', 'keep *_gmtDigis_*_*', 'keep *_gtDigis_*_*', 'keep FEDRawDataCollection_source_*_*', 'keep FEDRawDataCollection_rawDataCollector_*_*', 'keep *_MEtoEDMConverter_*_*', 'keep recoCandidatesOwned_genParticles_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep *_goodMuons_*_*', 'keep *_goodTracks_*_*', 'keep *_goodStandAloneMuonTracks_*_*', 'keep *_muonIsolations_*_*', 'keep *_goodZToMuMu_*_*', 'keep *_goodZToMuMuOneTrack_*_*', 'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 'keep *_goodZMCMatch_*_*', 'drop *_*_*_HLT', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCandidatesOwned_allMuons_*_*', 'keep *_allMuonIsolations_*_*', 'keep *_zToMuMuGolden_*_*', 'keep *_allMuonsGenParticlesMatch_*_*', 'keep *_zToMuMuGoldenGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_zToEE_*_*', 'keep *_zToEEOneTrack_*_*', 'keep *_zToEEOneSuperCluster_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *_zToEEGenParticlesMatch_*_*', 'keep *_zToEEOneTrackGenParticlesMatch_*_*', 'keep *_zToEEOneSuperClusterGenParticlesMatch_*_*', 'keep recoCandidatesOwned_genParticleCandidates_*_*', 'keep recoTracks_ctfWithMaterialTracks_*_*', 'keep recoPixelMatchGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoCandidatesOwned_allElectrons_*_*', 'keep recoCandidatesOwned_allTracks_*_*', 'keep recoCandidatesOwned_allSuperClusters_*_*', 'keep *_allElectronIsolations_*_*', 'keep *_allTrackIsolations_*_*', 'keep *_allSuperClusterIsolations_*_*', 'keep *_allElectronsGenParticlesMatch_*_*', 'keep *_allTracksGenParticlesLeptonMatch_*_*', 'keep *_allSuperClustersGenParticlesLeptonMatch_*_*', 'keep *'))
)
process.RecoTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_generalTracks_*_*', 
        'keep recoTracks_rsWithMaterialTracks_*_*')
)
process.ReleaseValidation = cms.untracked.PSet(
    primaryDatasetName = cms.untracked.string('RelValRelValSingleMuPt1.cfiRAW2DIGI,RECO'),
    totalNumberOfEvents = cms.untracked.int32(5000),
    eventsPerJob = cms.untracked.int32(250)
)
process.MIsoTrackAssociatorJets = cms.PSet(
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(0.5),
        dREcal = cms.double(0.5),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(False),
        dREcalPreselection = cms.double(0.5),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(False),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(0.5),
        useMuon = cms.bool(False),
        useCalo = cms.bool(True),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(False)
    )
)
process.MIsoDepositParamGlobalIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("paramMuons","ParamGlobalMuons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('track')
)
process.RecoTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_ctfWithMaterialTracks_*_*', 
        'keep *_generalTracks_*_*', 
        'keep *_rsWithMaterialTracks_*_*')
)
process.DigiToRawFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep FEDRawDataCollection_source_*_*', 
        'keep FEDRawDataCollection_rawDataCollector_*_*')
)
process.MEtoEDMConverterAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*')
)
process.RecoEgammaRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 
        'keep recoGsfTracks_pixelMatchGsfFit_*_*', 
        'keep recoGsfTrackExtras_pixelMatchGsfFit_*_*', 
        'keep recoTrackExtras_pixelMatchGsfFit_*_*', 
        'keep TrackingRecHitsOwned_pixelMatchGsfFit_*_*', 
        'keep recoElectronPixelSeeds_electronPixelSeeds_*_*', 
        'keep recoPhotons_photons_*_*', 
        'keep recoConversions_conversions_*_*', 
        'keep recoTracks_ckfOutInTracksFromConversions_*_*', 
        'keep recoTracks_ckfInOutTracksFromConversions_*_*', 
        'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 
        'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 
        'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*')
)
process.MIsoTrackExtractorGsBlock = cms.PSet(
    Diff_z = cms.double(0.2),
    inputTrackCollection = cms.InputTag("ctfGSWithMaterialTracks"),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    ComponentName = cms.string('TrackExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(0.1),
    Chi2Prob_Min = cms.double(-1.0),
    DR_Veto = cms.double(0.01),
    NHits_Min = cms.uint32(0),
    Chi2Ndof_Max = cms.double(1e+64),
    Pt_Min = cms.double(-1.0),
    DepositLabel = cms.untracked.string(''),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)
process.HLTBTauRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltCaloTowersTau*_*_*', 
        'keep *_hltTowerMakerForAll_*_*', 
        'keep *_hltTowerMakerForTaus_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1*_*_*', 
        'keep *_hltIcone5Tau2*_*_*', 
        'keep *_hltIcone5Tau3*_*_*', 
        'keep *_hltIcone5Tau4*_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBSoftmuonHighestEtJets_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltBSoftmuonL25TagInfos_*_*', 
        'keep *_hltBSoftmuonL25BJetTags_*_*', 
        'keep *_hltBSoftmuonL25Jets_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltBSoftmuonL3TagInfos_*_*', 
        'keep *_hltBSoftmuonL3BJetTags_*_*', 
        'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 
        'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumu_*_*', 
        'keep *_hltCtfWithMaterialTracksMumu_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 
        'keep *_hltCkfTrackCandidatesMumuk_*_*', 
        'keep *_hltCtfWithMaterialTracksMumuk_*_*', 
        'keep *_hltMuTracks_*_*', 
        'keep *_hltMumukAllConeTracks_*_*', 
        'keep *_hltL2SingleTauJets_*_*', 
        'keep *_hltL2SingleTauIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau*_*_*', 
        'keep *_hltConeIsolationL25SingleTau*_*_*', 
        'keep *_hltIsolatedL25SingleTau*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 
        'keep *_hltAssociatorL3SingleTau*_*_*', 
        'keep *_hltConeIsolationL3SingleTau*_*_*', 
        'keep *_hltIsolatedL3SingleTau*_*_*', 
        'keep *_hltL2SingleTauMETJets_*_*', 
        'keep *_hltL2SingleTauMETIsolation*_*_*', 
        'keep *_hltMet*_*_*', 
        'keep *_hltAssociatorL25SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL25SingleTauMET*_*_*', 
        'keep *_hltIsolatedL25SingleTauMET*_*_*', 
        'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 
        'keep *_hltAssociatorL3SingleTauMET*_*_*', 
        'keep *_hltConeIsolationL3SingleTauMET*_*_*', 
        'keep *_hltIsolatedL3SingleTauMET*_*_*', 
        'keep *_hltL2DoubleTauJets_*_*', 
        'keep *_hltL2DoubleTauIsolation*_*_*', 
        'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 
        'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 
        'keep *_hltIsolatedL25PixelTau*_*_*')
)
process.MuonServiceProxy = cms.PSet(
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
    )
)
process.CondDBCommon = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    ),
    connect = cms.string('protocol://db/schema')
)
process.topDiLeptonMuonXEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.RecoPixelVertexingRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*')
)
process.RecoParticleFlowRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop CaloTowersSorted_towerMakerPF_*_*', 
        'keep recoPFClusters_*_*_*', 
        'keep recoPFBlocks_*_*_*', 
        'keep recoPFCandidates_*_*_*', 
        'keep *_secStep_*_*', 
        'keep *_thStep_*_*')
)
process.HLTElectronPlusBLifetimeRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*')
)
process.MinPtTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('MinPtTrajectoryFilter'),
    minPt = cms.double(1.0),
    nSigmaMinPt = cms.double(5.0),
    minHitsMinPt = cms.int32(3)
)
process.TopQuarkAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.higgsToWW2LeptonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HWWFilterPath')
    )
)
process.CondDBSetup = cms.PSet(
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('.'),
        connectionRetrialPeriod = cms.untracked.int32(10),
        idleConnectionCleanupPeriod = cms.untracked.int32(10),
        messageLevel = cms.untracked.int32(0),
        enablePoolAutomaticCleanUp = cms.untracked.bool(False),
        enableConnectionSharing = cms.untracked.bool(True),
        connectionRetrialTimeOut = cms.untracked.int32(60),
        connectionTimeOut = cms.untracked.int32(0),
        enableReadOnlySessionOnUpdateConnection = cms.untracked.bool(False)
    )
)
process.HLTMuonPlusBLifetimeAOD = cms.PSet(
    triggerCollections = cms.VInputTag(cms.InputTag("hltL3MuonCandidates"), cms.InputTag("hltBLifetimeL3BJetTags")),
    triggerFilters = cms.VInputTag(cms.InputTag("hltBLifetimeL3filter"), cms.InputTag("hltMuBIsoL3IsoFiltered")),
    outputCommands = cms.untracked.vstring()
)
process.RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
process.HLTJetMETRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltMCJetCorJetIcone5*_*_*', 
        'keep *_hltIterativeCone5CaloJets*_*_*', 
        'keep *_hltMet_*_*', 
        'keep *_hltHtMet_*_*', 
        'keep *_hltHtMetIC5_*_*')
)
process.MIsoDepositViewIOBlock = cms.PSet(
    ExtractForCandidate = cms.bool(False),
    inputMuonCollection = cms.InputTag("muons"),
    MultipleDepositsFlag = cms.bool(False),
    InputType = cms.string('MuonCollection'),
    MuonTrackRefType = cms.string('bestGlbTrkSta')
)
process.SimCalorimetryRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTAlcaRecoEcalPhiSymStreamRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 
        'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 
        'keep *_hltL1GtUnpack_*_*', 
        'keep *_hltGtDigis_*_*')
)
process.TC_ME1A = cms.PSet(
    dPhiFineMax = cms.double(0.013),
    verboseInfo = cms.untracked.bool(True),
    SegmentSorting = cms.int32(1),
    chi2Max = cms.double(6000.0),
    dPhiMax = cms.double(0.00198),
    chi2ndfProbMin = cms.double(0.0001),
    minLayersApart = cms.int32(2),
    dRPhiFineMax = cms.double(3.0),
    dRPhiMax = cms.double(0.6)
)
process.RECOSIMEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'drop *', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'drop recoSuperClusters_hybridSuperClusters_*_*', 'keep recoSuperClusters_islandSuperClusters_islandBarrelSuperClusters_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep recoPreshowerClusterShapes_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep *_kt4CaloJets_*_*', 'keep *_kt6CaloJets_*_*', 'keep *_iterativeCone5CaloJets_*_*', 'keep *_sisCone5CaloJets_*_*', 'keep *_sisCone7CaloJets_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep recoGsfElectrons_pixelMatchGsfElectrons_*_*', 'keep recoGsfTracks_pixelMatchGsfFit_*_*', 'keep recoGsfTrackExtras_pixelMatchGsfFit_*_*', 'keep recoTrackExtras_pixelMatchGsfFit_*_*', 'keep TrackingRecHitsOwned_pixelMatchGsfFit_*_*', 'keep recoElectronPixelSeeds_electronPixelSeeds_*_*', 'keep recoPhotons_photons_*_*', 'keep recoConversions_conversions_*_*', 'keep recoTracks_ckfOutInTracksFromConversions_*_*', 'keep recoTracks_ckfInOutTracksFromConversions_*_*', 'keep recoTrackExtras_ckfOutInTracksFromConversions_*_*', 'keep recoTrackExtras_ckfInOutTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfOutInTracksFromConversions_*_*', 'keep TrackingRecHitsOwned_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*')+cms.untracked.vstring('keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25LeptonTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1IsoStartUp_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoStartUp_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*')+cms.untracked.vstring('keep *_MEtoEDMConverter_*_*', 'keep *_genParticles_*_*', 'keep *_genEventWeight_*_*', 'keep *_genEventScale_*_*', 'keep *_genEventPdfInfo_*_*', 'keep edmHepMCProduct_source_*_*', 'keep edmGenInfoProduct_source_*_*', 'keep *_genEventProcID_*_*', 'keep *_genEventRunInfo_*_*', 'keep edmAlpgenInfoProduct_source_*_*', 'keep edmHepMCProduct_source_*_*', 'keep SimTracks_g4SimHits_*_*', 'keep SimVertexs_g4SimHits_*_*', 'keep *_trackMCMatch_*_*', 'keep recoGenMETs_*_*_*', 'keep *_kt4GenJets_*_*', 'keep *_kt6GenJets_*_*', 'keep *_iterativeCone5GenJets_*_*', 'keep *_sisCone5GenJets_*_*', 'keep *_sisCone7GenJets_*_*', 'keep *_genParticle_*_*', 'drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*', 'keep *_MEtoEDMConverter_*_*'))
)
process.higgsToZZ4LeptonsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('HZZFilterPath')
    )
)
process.NominalCollision2VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.000142),
    BetaStar = cms.double(55.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.025),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.05),
    SigmaZ = cms.double(5.3)
)
process.SimMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
process.HLTElectronPlusBLifetimeFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltHtMet_*_*', 
        'keep *_hltIterativeCone5CaloJets_*_*', 
        'keep *_hltBLifetimeHighestEtJets_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltBLifetimeL25Jets_*_*', 
        'keep *_hltBLifetimeL25Associator_*_*', 
        'keep *_hltBLifetimeL25TagInfos_*_*', 
        'keep *_hltBLifetimeL25BJetTags_*_*', 
        'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 
        'keep *_hltBLifetimeL3Jets_*_*', 
        'keep *_hltBLifetimeL3Associator_*_*', 
        'keep *_hltBLifetimeL3TagInfos_*_*', 
        'keep *_hltBLifetimeL3BJetTags_*_*', 
        'keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 
        'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 
        'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 
        'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 
        'keep *_hltL1IsoPhotonTrackIsol_*_*', 
        'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 
        'keep *_hltHcalDoubleCone_*_*', 
        'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 
        'keep *_hltL1IsoElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 
        'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltCkfL1IsoTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 
        'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 
        'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 
        'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 
        'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 
        'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*')
)
process.GaussVtxSmearingParameters = cms.PSet(
    MeanX = cms.double(0.0),
    MeanY = cms.double(0.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0015),
    SigmaX = cms.double(0.0015),
    SigmaZ = cms.double(5.3)
)
process.L1TriggerFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_cscTriggerPrimitiveDigis_*_*', 
        'keep *_dtTriggerPrimitiveDigis_*_*', 
        'keep *_rpcTriggerDigis_*_*', 
        'keep *_rctDigis_*_*', 
        'keep *_csctfDigis_*_*', 
        'keep *_dttfDigis_*_*', 
        'keep *_gctDigis_*_*', 
        'keep *_gmtDigis_*_*', 
        'keep *_gtDigis_*_*')
)
process.FastjetWithPU = cms.PSet(
    Active_Area_Repeats = cms.int32(5),
    UE_Subtraction = cms.string('yes'),
    Ghost_EtaMax = cms.double(6.0),
    GhostArea = cms.double(0.01)
)
process.RecoVertexAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)
process.HLTElectronMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltL1IsoRecoEcalCandidate_*_*', 
        'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 
        'keep *_hltL1IsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 
        'keep *_hltL1IsoElectronTrackIsol_*_*', 
        'keep *_hltL1NonIsoElectronTrackIsol_*_*', 
        'keep *_hltPixelMatchElectronsL1Iso_*_*', 
        'keep *_hltPixelMatchElectronsL1NonIso_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 
        'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 
        'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 
        'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 
        'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 
        'keep *_hltL2MuonSeeds_*_*', 
        'keep *_hltL2Muons_*_*', 
        'keep *_hltL3Muons_*_*', 
        'keep *_hltL2MuonCandidates_*_*', 
        'keep *_hltL3MuonCandidates_*_*', 
        'keep *_hltL2MuonIsolations_*_*', 
        'keep *_hltL3MuonIsolations_*_*')
)
process.FEVTEventContent = cms.PSet(
    outputCommands = (cms.untracked.vstring('drop *', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*', 'keep *_siPixelRecHits_*_*', 'keep *_siStripRecHits_*_*', 'keep *_siStripMatchedRecHits_*_*', 'keep *_dt1DRecHits_*_*', 'keep *_dt4DSegments_*_*', 'keep *_csc2DRecHits_*_*', 'keep *_cscSegments_*_*', 'keep *_rpcRecHits_*_*', 'keep *_hbhereco_*_*', 'keep *_hfreco_*_*', 'keep *_horeco_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalRecHit_*_*', 'keep *_ecalWeightUncalibRecHit_*_*', 'keep *_ecalPreshowerRecHit_*_*', 'keep *_islandBasicClusters_*_*', 'keep *_islandSuperClusters_*_*', 'keep *_hybridSuperClusters_*_*', 'keep *_correctedIslandBarrelSuperClusters_*_*', 'keep *_correctedIslandEndcapSuperClusters_*_*', 'keep *_correctedHybridSuperClusters_*_*', 'keep *_correctedEndcapSuperClustersWithPreshower_*_*', 'keep *_preshowerClusterShape_*_*', 'keep *_CkfElectronCandidates_*_*', 'keep *_GsfGlobalElectronTest_*_*', 'keep *_ctfWithMaterialTracks_*_*', 'keep *_generalTracks_*_*', 'keep *_rsWithMaterialTracks_*_*', 'keep recoCaloJets_*_*_*', 'keep *_caloTowers_*_*', 'keep *_towerMaker_*_*', 'keep *_ic5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtVertex_*_*', 'keep *_iterativeCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_iterativeCone5JetExtender_*_*', 'keep *_sisCone5JetTracksAssociatorAtVertex_*_*', 'keep *_sisCone5JetTracksAssociatorAtCaloFace_*_*', 'keep *_sisCone5JetExtender_*_*', 'keep *_kt4JetTracksAssociatorAtVertex_*_*', 'keep *_kt4JetTracksAssociatorAtCaloFace_*_*', 'keep *_kt4JetExtender_*_*', 'keep recoCaloMETs_*_*_*', 'keep recoMETs_*_*_*', 'keep *_MuonSeed_*_*', 'keep recoTracks_standAloneMuons_*_*', 'keep recoTrackExtras_standAloneMuons_*_*', 'keep TrackingRecHitsOwned_standAloneMuons_*_*', 'keep recoTracks_globalMuons_*_*', 'keep recoTrackExtras_globalMuons_*_*', 'keep TrackingRecHitsOwned_globalMuons_*_*', 'keep recoTracks_generalTracks_*_*', 'keep recoMuons_trackerMuons_*_*', 'keep recoMuons_muons_*_*', 'keep recoCaloMuons_calomuons_*_*', 'keep *_muIsoDepositTk_*_*', 'keep *_muIsoDepositCalByAssociatorTowers_*_*', 'keep *_muIsoDepositCalByAssociatorHits_*_*', 'keep *_muIsoDepositJets_*_*', 'keep *_muGlobalIsoDepositCtfTk_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorTowers_*_*', 'keep *_muGlobalIsoDepositCalByAssociatorHits_*_*', 'keep *_muGlobalIsoDepositJets_*_*', 'keep *_btagSoftElectrons_*_*', 'keep *_softElectronTagInfos_*_*', 'keep *_softMuonTagInfos_*_*', 'keep *_softElectronBJetTags_*_*', 'keep *_softMuonBJetTags_*_*', 'keep *_softMuonNoIPBJetTags_*_*', 'keep *_impactParameterTagInfos_*_*', 'keep *_jetProbabilityBJetTags_*_*', 'keep *_jetBProbabilityBJetTags_*_*', 'keep *_trackCountingHighPurBJetTags_*_*', 'keep *_trackCountingHighEffBJetTags_*_*', 'keep *_impactParameterMVABJetTags_*_*', 'keep *_combinedSVBJetTags_*_*', 'keep *_combinedSVMVABJetTags_*_*', 'keep *_secondaryVertexTagInfos_*_*', 'keep *_simpleSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexBJetTags_*_*', 'keep *_combinedSecondaryVertexMVABJetTags_*_*', 'keep *_coneIsolationTauJetTags_*_*', 'keep *_particleFlowJetCandidates_*_*', 'keep *_iterativeCone5PFJets_*_*', 'keep *_ic5PFJetTracksAssociatorAtVertex_*_*', 'keep *_pfRecoTauTagInfoProducer_*_*', 'keep *_pfRecoTauProducer_*_*', 'keep *_pfRecoTauDiscriminationByIsolation_*_*', 'keep *_pfRecoTauProducerHighEfficiency_*_*', 'keep *_pfRecoTauDiscriminationHighEfficiency_*_*', 'keep *_caloRecoTauTagInfoProducer_*_*', 'keep *_caloRecoTauProducer_*_*', 'keep *_caloRecoTauDiscriminationByIsolation_*_*', 'keep  *_offlinePrimaryVertices_*_*', 'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 'keep  *_nuclearInteractionMaker_*_*', 'keep *_pixelMatchGsfElectrons_*_*', 'keep *_pixelMatchGsfFit_*_*', 'keep *_electronPixelSeeds_*_*', 'keep *_conversions_*_*', 'keep *_photons_*_*', 'keep *_ckfOutInTracksFromConversions_*_*', 'keep *_ckfInOutTracksFromConversions_*_*', 'keep *_pixelTracks_*_*', 'keep *_pixelVertices_*_*', 'drop CaloTowersSorted_towerMakerPF_*_*', 'keep recoPFClusters_*_*_*', 'keep recoPFBlocks_*_*_*', 'keep recoPFCandidates_*_*_*', 'keep *_secStep_*_*', 'keep *_thStep_*_*', 'keep *_offlineBeamSpot_*_*', 'keep *_gtDigis_*_*', 'keep *_l1GtRecord_*_*', 'keep *_l1GtObjectMap_*_*', 'keep *_l1extraParticles_*_*', 'keep edmTriggerResults_*_*_*', 'keep triggerTriggerEvent_*_*_*', 'keep triggerTriggerEventWithRefs_*_*_*', 'keep edmEventTime_*_*_*', 'keep HLTPerformanceInfo_*_*_*', 'keep *_hltGctDigis_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltL1GtObjectMap_*_*', 'keep *_hltL1extraParticles_*_*', 'keep *_hltDt1DRecHits_*_*', 'keep *_hltDt4DSegments_*_*', 'keep *_hltCsc2DRecHits_*_*', 'keep *_hltCscSegments_*_*', 'keep *_hltRpcRecHits_*_*', 'keep *_hltHbhereco_*_*', 'keep *_hltHfreco_*_*', 'keep *_hltHoreco_*_*', 'keep *_hltEcalWeightUncalibRecHit_*_*', 'keep *_hltEcalPreshowerRecHit_*_*', 'keep *_hltEcalRecHit_*_*', 'keep *_hltSiPixelClusters_*_*', 'keep *_hltSiStripClusters_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep StripDigiSimLinkedmDetSetVector_hltMuonCSCDigis_*_*', 'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_hltMuonCSCDigis_*_*', 'keep *_hltOfflineBeamSpot_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltMet_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltHtMetIC5_*_*', 'keep *_hltCaloTowersTau*_*_*', 'keep *_hltTowerMakerForAll_*_*', 'keep *_hltTowerMakerForTaus_*_*', 'keep *_hltSiPixelRecHits_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltSiStripRecHits_*_*', 'keep *_hltSiStripMatchedRecHits_*_*', 'keep *_hltIcone5Tau1*_*_*', 'keep *_hltIcone5Tau2*_*_*', 'keep *_hltIcone5Tau3*_*_*', 'keep *_hltIcone5Tau4*_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*')+cms.untracked.vstring('keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltMumuPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumu_*_*', 'keep *_hltCtfWithMaterialTracksMumu_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukPixelSeedFromL2Candidate_*_*', 'keep *_hltCkfTrackCandidatesMumuk_*_*', 'keep *_hltCtfWithMaterialTracksMumuk_*_*', 'keep *_hltMuTracks_*_*', 'keep *_hltMumukAllConeTracks_*_*', 'keep *_hltL2SingleTauJets_*_*', 'keep *_hltL2SingleTauIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 'keep *_hltAssociatorL25SingleTau*_*_*', 'keep *_hltConeIsolationL25SingleTau*_*_*', 'keep *_hltIsolatedL25SingleTau*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTau*_*_*', 'keep *_hltAssociatorL3SingleTau*_*_*', 'keep *_hltConeIsolationL3SingleTau*_*_*', 'keep *_hltIsolatedL3SingleTau*_*_*', 'keep *_hltL2SingleTauMETJets_*_*', 'keep *_hltL2SingleTauMETIsolation*_*_*', 'keep *_hltMet*_*_*', 'keep *_hltAssociatorL25SingleTauMET*_*_*', 'keep *_hltConeIsolationL25SingleTauMET*_*_*', 'keep *_hltIsolatedL25SingleTauMET*_*_*', 'keep *_hltCtfWithMaterialTracksL3SingleTauMET*_*_*', 'keep *_hltAssociatorL3SingleTauMET*_*_*', 'keep *_hltConeIsolationL3SingleTauMET*_*_*', 'keep *_hltIsolatedL3SingleTauMET*_*_*', 'keep *_hltL2DoubleTauJets_*_*', 'keep *_hltL2DoubleTauIsolation*_*_*', 'keep *_hltAssociatorL25PixelTauIsolated*_*_*', 'keep *_hltConeIsolationL25PixelTauIsolated*_*_*', 'keep *_hltIsolatedL25PixelTau*_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltJetsPixelTracksAssociatorMuonTau_*_*', 'keep *_hltPixelTrackConeIsolationMuonTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorMuonTau_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIcone5Tau1_*_*', 'keep *_hltIcone5Tau2_*_*', 'keep *_hltIcone5Tau3_*_*', 'keep *_hltIcone5Tau4_*_*', 'keep *_hltL2TauJetsProvider_*_*', 'keep *_hltEMIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltJetsPixelTracksAssociatorElectronTau_*_*', 'keep *_hltPixelTrackConeIsolationElectronTau_*_*', 'keep *_hltPixelTrackIsolatedTauJetsSelectorElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL25ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL25ElectronTau_*_*', 'keep *_hltConeIsolationL25ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL25ElectronTau_*_*', 'keep *_hltCtfWithMaterialTracksL3ElectronTau_*_*', 'keep *_hltJetTracksAssociatorAtVertexL3ElectronTau_*_*', 'keep *_hltConeIsolationL3ElectronTau_*_*', 'keep *_hltIsolatedTauJetsSelectorL3ElectronTau_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBSoftmuonHighestEtJets_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltBSoftmuonL25TagInfos_*_*', 'keep *_hltBSoftmuonL25BJetTags_*_*', 'keep *_hltBSoftmuonL25Jets_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltBSoftmuonL3TagInfos_*_*', 'keep *_hltBSoftmuonL3BJetTags_*_*', 'keep *_hltBSoftmuonL3BJetTagsByDR_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltHtMet_*_*', 'keep *_hltIterativeCone5CaloJets_*_*', 'keep *_hltBLifetimeHighestEtJets_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltPixelVertices_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltBLifetimeL25Jets_*_*', 'keep *_hltBLifetimeL25Associator_*_*', 'keep *_hltBLifetimeL25TagInfos_*_*', 'keep *_hltBLifetimeL25BJetTags_*_*', 'keep *_hltBLifetimeRegionalCtfWithMaterialTracks_*_*', 'keep *_hltBLifetimeL3Jets_*_*', 'keep *_hltBLifetimeL3Associator_*_*', 'keep *_hltBLifetimeL3TagInfos_*_*', 'keep *_hltBLifetimeL3BJetTags_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltL1IsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1NonIsoLargeWindowElectronTrackIsol_*_*', 'keep *_hltL1IsoStartUpElectronTrackIsol_*_*', 'keep *_hltL1NonIsoStartupElectronTrackIsol_*_*', 'keep *_hltL1IsolatedPhotonEcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonEcalIsol_*_*', 'keep *_hltL1IsolatedPhotonHcalIsol_*_*', 'keep *_hltL1NonIsolatedPhotonHcalIsol_*_*', 'keep *_hltL1IsoPhotonTrackIsol_*_*', 'keep *_hltL1NonIsoPhotonTrackIsol_*_*', 'keep *_hltHcalDoubleCone_*_*', 'keep *_hltL1NonIsoEMHcalDoubleCone_*_*', 'keep *_hltL1IsoElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoElectronPixelSeeds_*_*', 'keep *_hltL1IsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoLargeWindowElectronPixelSeeds_*_*', 'keep *_hltL1IsoStartUpElectronPixelSeeds_*_*', 'keep *_hltL1NonIsoStartUpElectronPixelSeeds_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltPixelMatchElectronsL1IsoLargeWindow_*_*', 'keep *_hltPixelMatchElectronsL1NonIsoLargeWindow_*_*', 'keep *_hltPixelMatchStartUpElectronsL1Iso_*_*', 'keep *_hltPixelMatchStartUpElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoLargeWindowWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1IsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltCtfL1NonIsoStartUpWithMaterialTracks_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*')+cms.untracked.vstring('keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoLargeWindowElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoStartUpElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltCkfL1IsoTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoTrackCandidates_*_*', 'keep *_hltCkfL1IsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoLargeWindowTrackCandidates_*_*', 'keep *_hltCkfL1IsoStartUpTrackCandidates_*_*', 'keep *_hltCkfL1NonIsoStartUpTrackCandidates_*_*', 'keep *_hltL1IsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoEgammaRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalPixelSeedGenerator_*_*', 'keep *_hltL1IsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoEgammaRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoLargeWindowElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1NonIsoStartUpElectronsRegionalCkfTrackCandidates_*_*', 'keep *_hltL1IsoRecoEcalCandidate_*_*', 'keep *_hltL1NonIsoRecoEcalCandidate_*_*', 'keep *_hltL1IsolatedElectronHcalIsol_*_*', 'keep *_hltL1NonIsolatedElectronHcalIsol_*_*', 'keep *_hltL1IsoElectronTrackIsol_*_*', 'keep *_hltL1NonIsoElectronTrackIsol_*_*', 'keep *_hltPixelMatchElectronsL1Iso_*_*', 'keep *_hltPixelMatchElectronsL1NonIso_*_*', 'keep *_hltCorrectedHybridSuperClustersL1Isolated_*_*', 'keep *_hltCorrectedHybridSuperClustersL1NonIsolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1Isolated_*_*', 'keep *_hltCorrectedEndcapSuperClustersWithPreshowerL1NonIsolated_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksBarrel_*_*', 'keep recoTracks_hltCtfL1IsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltCtfL1NonIsoWithMaterialTracksEndcap_*_*', 'keep recoTracks_hltL1IsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoElectronsRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1IsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep recoTracks_hltL1NonIsoEgammaRegionalCTFFinalFitWithMaterial_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltL2MuonSeeds_*_*', 'keep *_hltL2Muons_*_*', 'keep *_hltL3Muons_*_*', 'keep *_hltL2MuonCandidates_*_*', 'keep *_hltL3MuonCandidates_*_*', 'keep *_hltL2MuonIsolations_*_*', 'keep *_hltL3MuonIsolations_*_*', 'keep *_hltMCJetCorJetIcone5*_*_*', 'keep *_hltIterativeCone5CaloJets*_*_*', 'keep *_hltPixelTracks_*_*', 'keep *_hltIsolPixelTrackProd_*_*', 'keep *_hltL1sIsoTrack_*_*', 'keep *_hltGtDigis_*_*', 'keep l1extraL1JetParticles_hltL1extraParticles_*_*', 'keep *_*_pi0EcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEB_*', 'keep *_hltAlCaPhiSymStream_phiSymEcalRecHitsEE_*', 'keep *_hltL1GtUnpack_*_*', 'keep *_hltGtDigis_*_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHBHE_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHO_*', 'keep *_hltAlCaHcalPhiSymStream_phiSymHcalRecHitsHF_*', 'keep *_MEtoEDMConverter_*_*'))
)
process.topFullyHadronicJetsEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('topFullyHadronicJetsPath')
    )
)
process.RecoVertexRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep  *_offlinePrimaryVertices_*_*', 
        'keep  *_offlinePrimaryVerticesFromCTFTracks_*_*', 
        'keep  *_nuclearInteractionMaker_*_*')
)
process.DoubleTauSiStripHLT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_hltPixelVertices_*_*', 
        'keep *_hltPixelTracks_*_*', 
        'keep *_hltIcone5Tau1_*_*', 
        'keep *_hltIcone5Tau2_*_*', 
        'keep *_hltIcone5Tau3_*_*', 
        'keep *_hltIcone5Tau4_*_*', 
        'keep *_hltL2TauJetsProvider_*_*', 
        'keep *_hltEcalSingleTauIsolated_*_*', 
        'keep *_hltEcalDoubleTauIsolated_*_*', 
        'keep *_hltMet_*_*', 
        'keep *_hltCtfWithMaterialTracksL25SingleTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL25DoubleTau_*_*', 
        'keep *_hltAssociatorL25SingleTau_*_*', 
        'keep *_hltAssociatorL25DoubleTau_*_*', 
        'keep *_hltConeIsolationL25SingleTau_*_*', 
        'keep *_hltConeIsolationL25DoubleTau_*_*', 
        'keep *_hltIsolatedL25SingleTau_*_*', 
        'keep *_hltIsolatedL25DoubleTau_*_*', 
        'keep *_hltCtfWithMaterialTracksL3DoubleTau_*_*', 
        'keep *_hltAssociatorL3DoubleTau_*_*', 
        'keep *_hltConeIsolationL3DoubleTau_*_*', 
        'keep *_hltIsolatedL3DoubleTau_*_*')
)
process.MIsoCaloExtractorByAssociatorHitsBlock = cms.PSet(
    Noise_HE = cms.double(0.2),
    DR_Veto_H = cms.double(0.1),
    UseRecHitsFlag = cms.bool(True),
    TrackAssociatorParameters = cms.PSet(
        muonMaxDistanceSigmaX = cms.double(0.0),
        muonMaxDistanceSigmaY = cms.double(0.0),
        CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
        dRHcal = cms.double(1.0),
        dREcal = cms.double(1.0),
        CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
        useEcal = cms.bool(True),
        dREcalPreselection = cms.double(1.0),
        HORecHitCollectionLabel = cms.InputTag("horeco"),
        dRMuon = cms.double(9999.0),
        crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
        muonMaxDistanceX = cms.double(5.0),
        muonMaxDistanceY = cms.double(5.0),
        useHO = cms.bool(True),
        accountForTrajectoryChangeCalo = cms.bool(False),
        DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
        EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
        dRHcalPreselection = cms.double(1.0),
        useMuon = cms.bool(False),
        useCalo = cms.bool(False),
        EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
        dRMuonPreselection = cms.double(0.2),
        truthMatch = cms.bool(False),
        HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
        useHcal = cms.bool(True)
    ),
    NoiseTow_EE = cms.double(0.15),
    Threshold_HO = cms.double(0.1),
    DR_Max = cms.double(1.0),
    PropagatorName = cms.string('SteppingHelixPropagatorAny'),
    Noise_HO = cms.double(0.2),
    Noise_EE = cms.double(0.1),
    Noise_EB = cms.double(0.025),
    DR_Veto_HO = cms.double(0.1),
    Noise_HB = cms.double(0.2),
    PrintTimeReport = cms.untracked.bool(False),
    NoiseTow_EB = cms.double(0.04),
    Threshold_H = cms.double(0.1),
    DR_Veto_E = cms.double(0.07),
    DepositLabel = cms.untracked.string('Cal'),
    ComponentName = cms.string('CaloExtractorByAssociator'),
    Threshold_E = cms.double(0.025),
    DepositInstanceLabels = cms.vstring('ecal', 
        'hcal', 
        'ho')
)
process.RecoPixelVertexingFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_pixelTracks_*_*', 
        'keep *_pixelVertices_*_*')
)
process.L1TriggerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_gtDigis_*_*', 
        'keep *_l1GtRecord_*_*', 
        'keep *_l1GtObjectMap_*_*', 
        'keep *_l1extraParticles_*_*')
)
process.NominalCollision4VtxSmearingParameters = cms.PSet(
    Phi = cms.double(0.0),
    BetaStar = cms.double(55.0),
    Z0 = cms.double(0.0),
    Emittance = cms.double(5.03e-08),
    Y0 = cms.double(0.025),
    Alpha = cms.double(0.0),
    X0 = cms.double(0.2),
    SigmaZ = cms.double(5.3)
)
process.TrackingToolsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*')
)
process.zToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoCandidatesOwned_genParticles_*_*', 
        'keep recoTracks_ctfWithMaterialTracks_*_*', 
        'keep recoTracks_globalMuons_*_*', 
        'keep recoTracks_standAloneMuons_*_*', 
        'keep recoMuons_muons_*_*', 
        'keep *_goodMuons_*_*', 
        'keep *_goodTracks_*_*', 
        'keep *_goodStandAloneMuonTracks_*_*', 
        'keep *_muonIsolations_*_*', 
        'keep *_goodZToMuMu_*_*', 
        'keep *_goodZToMuMuOneTrack_*_*', 
        'keep *_goodZToMuMuOneStandAloneMuonTrack_*_*', 
        'keep *_goodZMCMatch_*_*', 
        'drop *_*_*_HLT')
)
process.seqALCARECOMuAlZMuMu = cms.Sequence(process.ALCARECOMuAlZMuMuHLT+process.ALCARECOMuAlZMuMu)
process.genMETParticles = cms.Sequence(process.genCandidatesForMET)
process.recopixelvertexing = cms.Sequence(process.pixelTracks*process.pixelVertices)
process.islandClusteringSequence = cms.Sequence(process.islandBasicClusters*process.islandSuperClusters*process.correctedIslandBarrelSuperClusters*process.correctedIslandEndcapSuperClusters)
process.seqALCARECOHcalCalDijets = cms.Sequence(process.dijetsHLT*process.DiJProd)
process.hcalLocalRecoSequence = cms.Sequence(process.hbhereco+process.hfreco+process.horeco)
process.egammareco = cms.Sequence(process.electronSequence*process.conversionSequence*process.photonSequence)
process.seqALCARECORpcCalHLT = cms.Sequence(process.l1MuonHLTFilter)
process.highlevelreco = cms.Sequence(process.vertexreco*process.recoJetAssociations*process.btagging*process.tautagging*process.egammareco*process.particleFlowReco*process.PFTau)
process.muIsolation_ParamGlobalMuons = cms.Sequence(process.muIsoDeposits_ParamGlobalMuons)
process.secondStep = cms.Sequence(process.secClusters*process.secPixelRecHits*process.secStripRecHits*process.secTriplets*process.secTrackCandidates*process.secWithMaterialTracks*process.secStep)
process.reconstruction_plusGSF = cms.Sequence(process.reconstruction*process.GsfGlobalElectronTestSequence)
process.electronSequence = cms.Sequence(process.pixelMatchGsfElectronSequence)
process.ecalLocalRecoSequence = cms.Sequence(process.ecalWeightUncalibRecHit*process.ecalRecHit+process.ecalPreshowerRecHit)
process.calolocalreco = cms.Sequence(process.ecalLocalRecoSequence+process.hcalLocalRecoSequence)
process.rstracks = cms.Sequence(process.roadSearchSeeds*process.roadSearchClouds*process.rsTrackCandidates*process.rsWithMaterialTracks)
process.csclocalreco = cms.Sequence(process.csc2DRecHits*process.cscSegments)
process.seqALCARECOTkAlMinBias = cms.Sequence(process.ALCARECOTkAlMinBiasHLT+process.ALCARECOTkAlMinBias)
process.btagging = cms.Sequence(process.impactParameterTagInfos*process.jetBProbabilityBJetTags+process.jetProbabilityBJetTags+process.trackCountingHighPurBJetTags+process.trackCountingHighEffBJetTags+process.impactParameterMVABJetTags*process.secondaryVertexTagInfos*process.simpleSecondaryVertexBJetTags+process.combinedSecondaryVertexBJetTags+process.combinedSecondaryVertexMVABJetTags+process.btagSoftElectrons*process.softElectronTagInfos*process.softElectronBJetTags+process.softMuonTagInfos*process.softMuonBJetTags+process.softMuonNoIPBJetTags)
process.VertexSmearing = cms.Sequence(process.VtxSmeared)
process.seqALCARECOHcalCalHO = cms.Sequence(process.isoMuonHLT*process.hoCalibProducer)
process.striptrackerlocalreco = cms.Sequence(process.siStripZeroSuppression*process.siStripClusters*process.siStripMatchedRecHits)
process.tracksWithQuality = cms.Sequence(process.withLooseQuality*process.withTightQuality*process.generalTracks)
process.iterativeCone5JTA = cms.Sequence(process.iterativeCone5JetTracksAssociatorAtVertex*process.iterativeCone5JetTracksAssociatorAtCaloFace*process.iterativeCone5JetExtender)
process.iterTracking = cms.Sequence(process.firstfilter*process.secondStep*process.thirdStep)
process.localreco = cms.Sequence(process.trackerlocalreco+process.muonlocalreco+process.calolocalreco)
process.elecPreId = cms.Sequence(process.elecpreid)
process.pfClusteringPS = cms.Sequence(process.particleFlowRecHitPS*process.particleFlowClusterPS)
process.conversionSequence = cms.Sequence(process.ckfTracksFromConversions*process.conversions)
process.recoGenJets = cms.Sequence(process.kt4GenJets+process.kt6GenJets+process.iterativeCone5GenJets+process.sisCone5GenJets+process.sisCone7GenJets)
process.dtlocalreco = cms.Sequence(process.dt1DRecHits*process.dt4DSegments)
process.globalreco_plusGSF = cms.Sequence(process.globalreco*process.GsfGlobalElectronTestSequence)
process.seqALCARECOTkAlUpsilonMuMu = cms.Sequence(process.ALCARECOTkAlUpsilonMuMuHLT+process.ALCARECOTkAlUpsilonMuMu)
process.sisCone5JTA = cms.Sequence(process.sisCone5JetTracksAssociatorAtVertex*process.sisCone5JetTracksAssociatorAtCaloFace*process.sisCone5JetExtender)
process.highlevelreco_woConv = cms.Sequence(process.vertexreco*process.recoJetAssociations*process.btagging*process.tautagging*process.egammareco_woConvPhotons*process.particleFlowReco*process.PFTau)
process.PFTau = cms.Sequence(process.particleFlowJetCandidates*process.iterativeCone5PFJets*process.ic5PFJetTracksAssociatorAtVertex*process.pfRecoTauTagInfoProducer*process.pfRecoTauProducer*process.pfRecoTauProducerHighEfficiency*process.pfRecoTauDiscriminationByIsolation*process.pfRecoTauDiscriminationHighEfficiency)
process.seqALCARECOTkAlCosmicsCTF = cms.Sequence(process.ALCARECOTkAlCosmicsCTF)
process.trackerlocalreco = cms.Sequence(process.pixeltrackerlocalreco*process.striptrackerlocalreco)
process.globalreco_plusRS = cms.Sequence(process.globalreco*process.rstracks)
process.pfClusteringECAL = cms.Sequence(process.particleFlowRecHitECAL*process.particleFlowClusterECAL)
process.muonreco_plus_isolation = cms.Sequence(process.muonreco*process.muIsolation)
process.pixeltrackerlocalreco = cms.Sequence(process.siPixelClusters*process.siPixelRecHits)
process.seqALCARECOTkAlZMuMu = cms.Sequence(process.ALCARECOTkAlZMuMuHLT+process.ALCARECOTkAlZMuMu)
process.muonreco = cms.Sequence(process.muontracking*process.muonIdProducerSequence)
process.photonSequence = cms.Sequence(process.photons)
process.seqALCARECOHcalCalIsoTrkNoHLT = cms.Sequence(process.IsoProd)
process.genJetMET = cms.Sequence(process.genJetParticles*process.recoGenJets+process.genMETParticles*process.recoGenMET)
process.GsfGlobalElectronTestSequence = cms.Sequence(process.CkfElectronCandidates*process.GsfGlobalElectronTest)
process.caloTowersMETOptRec = cms.Sequence(process.calotoweroptmaker*process.caloTowersOpt)
process.vertexreco = cms.Sequence(process.offlinePrimaryVertices*process.offlinePrimaryVerticesFromCTFTracks)
process.postreco_generator = cms.Sequence(process.trackMCMatchSequence)
process.seqALCARECOEcalCalElectron = cms.Sequence(process.ewkHLTFilter*process.electronFilter*process.seqALCARECOEcalCalElectronRECO)
process.muIsoDeposits_ParamGlobalMuons = cms.Sequence(process.muParamGlobalIsoDepositTk+process.muParamGlobalIsoDepositCalByAssociatorTowers+process.muParamGlobalIsoDepositJets)
process.pgen = cms.Sequence(process.VertexSmearing+process.GeneInfo+process.genJetMET)
process.recoJetAssociations = cms.Sequence(process.ic5JetTracksAssociatorAtVertex*process.iterativeCone5JTA+process.sisCone5JTA+process.kt4JTA)
process.tautagging = cms.Sequence(process.coneIsolationTauJetTags*process.caloRecoTauTagInfoProducer*process.caloRecoTauProducer*process.caloRecoTauDiscriminationByIsolation)
process.caloTowersPFRec = cms.Sequence(process.towerMakerPF*process.caloTowersPF)
process.muonlocalreco = cms.Sequence(process.dtlocalreco+process.csclocalreco+process.rpcRecHits)
process.seqALCARECOHcalCalIsoTrk = cms.Sequence(process.isoHLT*process.IsoProd)
process.muIsolation_muons = cms.Sequence(process.muIsoDeposits_muons)
process.preshowerClusteringSequence = cms.Sequence(process.correctedEndcapSuperClustersWithPreshower*process.preshowerClusterShape)
process.muIsolation_ParamGlobalMuonsOld = cms.Sequence(process.muIsoDeposits_ParamGlobalMuonsOld)
process.reconstruction_woConv = cms.Sequence(process.localreco*process.globalreco_plusRS*process.highlevelreco_woConv)
process.particleFlowTrackWithNuclear = cms.Sequence(process.elecPreId*process.gsfElCandidates*process.gsfPFtracks*process.pfTrackElec*process.pfNuclear)
process.reconstruction_withRS = cms.Sequence(process.localreco*process.globalreco_plusRS*process.highlevelreco)
process.dtlocalreco_with_2DSegments = cms.Sequence(process.dt1DRecHits*process.dt2DSegments*process.dt4DSegments)
process.muIsoDeposits_muons = cms.Sequence(process.muIsoDepositTk+process.muIsoDepositCalByAssociatorTowers+process.muIsoDepositJets)
process.seqALCARECOTkAlJpsiMuMu = cms.Sequence(process.ALCARECOTkAlJpsiMuMuHLT+process.ALCARECOTkAlJpsiMuMu)
process.seqALCARECOHcalCalMinBias = cms.Sequence(process.hcalminbiasHLT*process.MinProd)
process.kt4JTA = cms.Sequence(process.kt4JetTracksAssociatorAtVertex*process.kt4JetTracksAssociatorAtCaloFace*process.kt4JetExtender)
process.particleFlowReco = cms.Sequence(process.iterTracking*process.caloTowersPFRec*process.particleFlowCluster*process.particleFlowTrack*process.particleFlowBlock*process.particleFlow)
process.seqALCARECOTkAlMuonIsolated = cms.Sequence(process.ALCARECOTkAlMuonIsolatedHLT+process.ALCARECOTkAlMuonIsolated)
process.muontracking = cms.Sequence(process.MuonSeed*process.standAloneMuons*process.globalMuons)
process.metreco = cms.Sequence(process.caloTowersMETOptRec*process.metOpt*process.met*process.metNoHF*process.metOptNoHF*process.htMetSC5*process.htMetSC7*process.htMetKT4*process.htMetKT6*process.htMetIC5)
process.seqALCARECOMuAlOverlaps = cms.Sequence(process.ALCARECOMuAlOverlapsHLT+process.ALCARECOMuAlOverlapsMuonSelector*process.ALCARECOMuAlOverlaps)
process.dynamicHybridClusteringSequence = cms.Sequence(process.dynamicHybridSuperClusters*process.correctedDynamicHybridSuperClusters)
process.muonlocalreco_with_2DSegments = cms.Sequence(process.dtlocalreco_with_2DSegments+process.csclocalreco+process.rpcRecHits)
process.muonIdProducerSequence = cms.Sequence(process.muons*process.calomuons)
process.ecalClusters = cms.Sequence(process.islandClusteringSequence*process.hybridClusteringSequence*process.preshowerClusteringSequence*process.dynamicHybridClusteringSequence*process.fixedMatrixClusteringSequence*process.fixedMatrixPreshowerClusteringSequence)
process.ckfTracksFromConversions = cms.Sequence(process.conversionTrackCandidates*process.ckfOutInTracksFromConversions*process.ckfInOutTracksFromConversions)
process.fixedMatrixClusteringSequence = cms.Sequence(process.fixedMatrixBasicClusters*process.fixedMatrixSuperClusters*process.fixedMatrixSuperClustersWithPreshower)
process.RawToDigi = cms.Sequence(process.csctfDigis+process.dttfDigis+process.gctDigis+process.gtDigis+process.siPixelDigis+process.SiStripRawToDigis+process.ecalDigis+process.ecalPreshowerDigis+process.hcalDigis+process.muonCSCDigis+process.muonDTDigis+process.muonRPCDigis)
process.pixelMatchGsfElectronSequence = cms.Sequence(process.electronPixelSeeds*process.egammaCkfTrackCandidates*process.pixelMatchGsfFit*process.pixelMatchGsfElectrons)
process.reconstruction = cms.Sequence(process.localreco*process.globalreco*process.highlevelreco)
process.egammareco_woConvPhotons = cms.Sequence(process.electronSequence*process.photonSequence)
process.newTracking = cms.Sequence(process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.newTrackCandidateMaker*process.preFilterCmsTracks*process.tracksWithQuality)
process.seqALCARECOSiStripCalMinBias = cms.Sequence(process.ALCARECOSiStripCalMinBiasHLT+process.ALCARECOSiStripCalMinBias)
process.ckftracks = cms.Sequence(process.newTracking)
process.seqALCARECOEcalCalPhiSym = cms.Sequence(process.ecalphiSymHLT)
process.caloTowersRec = cms.Sequence(process.towerMaker*process.caloTowers)
process.seqALCARECOTkAlCosmicsCosmicTF = cms.Sequence(process.ALCARECOTkAlCosmicsCosmicTF)
process.trackMCMatchSequence = cms.Sequence(process.trackMCMatch*process.standAloneMuonsMCMatch*process.globalMuonsMCMatch*process.allTrackMCMatch)
process.reconstruction_standard_candle = cms.Sequence(process.localreco*process.globalreco*process.vertexreco*process.recoJetAssociations*process.btagging*process.coneIsolationTauJetTags*process.electronSequence*process.photonSequence)
process.siStripElectronSequence = cms.Sequence(process.siStripElectrons*process.egammaCTFFinalFitWithMaterial*process.siStripElectronToTrackAssociator)
process.globalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.recoJets+process.metreco+process.muonreco_plus_isolation)
process.recoJets = cms.Sequence(process.kt4CaloJets+process.kt6CaloJets+process.iterativeCone5CaloJets+process.sisCone5CaloJets+process.sisCone7CaloJets)
process.particleFlowCluster = cms.Sequence(process.pfClusteringECAL*process.pfClusteringHCAL*process.pfClusteringPS)
process.fixedMatrixPreshowerClusteringSequence = cms.Sequence(process.correctedFixedMatrixSuperClustersWithPreshower*process.fixedMatrixPreshowerClusterShape)
process.seqALCARECOEcalCalPi0Calib = cms.Sequence(process.ecalpi0CalibHLT)
process.GeneInfo = cms.Sequence(process.genParticles+process.genEventWeight+process.genEventScale+process.genEventPdfInfo)
process.seqALCARECOEcalCalElectronRECO = cms.Sequence(process.alCaIsolatedElectrons)
process.globalreco_plusRS_plusGSF = cms.Sequence(process.globalreco*process.rstracks*process.GsfGlobalElectronTestSequence)
process.seqALCARECOHcalCalGammaJet = cms.Sequence(process.gammajetHLT*process.GammaJetProd)
process.genJetParticles = cms.Sequence(process.genParticlesForJets)
process.trackingParticles = cms.Sequence(process.trackingtruthprod*process.electrontruth*process.mergedtruth)
process.particleFlowTrack = cms.Sequence(process.elecPreId*process.gsfElCandidates*process.gsfPFtracks*process.pfTrackElec)
process.hybridClusteringSequence = cms.Sequence(process.hybridSuperClusters*process.correctedHybridSuperClusters)
process.recoGenMET = cms.Sequence(process.genMet+process.genMetNoNuBSM*process.genMetIC5GenJets)
process.pfClusteringHCAL = cms.Sequence(process.particleFlowRecHitHCAL*process.particleFlowClusterHCAL)
process.thirdStep = cms.Sequence(process.thClusters*process.thPixelRecHits*process.thStripRecHits*process.thPLSeeds*process.thTrackCandidates*process.thWithMaterialTracks*process.thStep)
process.SiStripRawToDigis = cms.Sequence(process.siStripDigis)
process.validation = cms.Sequence(process.globaldigisanalyze*process.globalrechitsanalyze*process.globalhitsanalyze*process.MEtoEDMConverter)
process.ecalLocalRecoSequence_nopreshower = cms.Sequence(process.ecalWeightUncalibRecHit*process.ecalRecHit)
process.muIsolation = cms.Sequence(process.muIsolation_muons)
process.muIsoDeposits_ParamGlobalMuonsOld = cms.Sequence(process.muParamGlobalIsoDepositGsTk+process.muParamGlobalIsoDepositCalEcal+process.muParamGlobalIsoDepositCalHcal)
process.pathALCARECOHcalCalIsoTrkNoHLT = cms.Path(process.seqALCARECOHcalCalIsoTrkNoHLT)
process.pathALCARECOHcalCalDijets = cms.Path(process.seqALCARECOHcalCalDijets)
process.pathALCARECOTkAlZMuMu = cms.Path(process.seqALCARECOTkAlZMuMu)
process.pathALCARECOTkAlCosmicsCTF = cms.Path(process.seqALCARECOTkAlCosmicsCTF)
process.pathALCARECOHcalCalHO = cms.Path(process.seqALCARECOHcalCalHO)
process.pathALCARECOTkAlMuonIsolated = cms.Path(process.seqALCARECOTkAlMuonIsolated)
process.pathALCARECOTkAlUpsilonMuMu = cms.Path(process.seqALCARECOTkAlUpsilonMuMu)
process.pathALCARECOMuAlZMuMu = cms.Path(process.seqALCARECOMuAlZMuMu)
process.pathALCARECOEcalCalPi0Calib = cms.Path(process.seqALCARECOEcalCalPi0Calib)
process.pathALCARECOEcalCalElectron = cms.Path(process.seqALCARECOEcalCalElectron)
process.pathALCARECOHcalCalIsoTrk = cms.Path(process.seqALCARECOHcalCalIsoTrk)
process.pathALCARECOSiStripCalMinBias = cms.Path(process.seqALCARECOSiStripCalMinBias)
process.pathALCARECOTkAlMinBias = cms.Path(process.seqALCARECOTkAlMinBias)
process.pathALCARECORpcCalHLT = cms.Path(process.seqALCARECORpcCalHLT)
process.pathALCARECOHcalCalGammaJet = cms.Path(process.seqALCARECOHcalCalGammaJet)
process.pathALCARECOMuAlOverlaps = cms.Path(process.seqALCARECOMuAlOverlaps)
process.pathALCARECOTkAlCosmicsCosmicTF = cms.Path(process.seqALCARECOTkAlCosmicsCosmicTF)
process.pathALCARECOEcalCalPhiSym = cms.Path(process.seqALCARECOEcalCalPhiSym)
process.pathALCARECOHcalCalMinBias = cms.Path(process.seqALCARECOHcalCalMinBias)
process.pathALCARECOTkAlJpsiMuMu = cms.Path(process.seqALCARECOTkAlJpsiMuMu)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction)
process.outpath = cms.EndPath(process.out_step)
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.outpath)


