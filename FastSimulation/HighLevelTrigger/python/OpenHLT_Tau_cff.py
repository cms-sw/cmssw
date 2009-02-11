import FWCore.ParameterSet.Config as cms

# import the whole HLT menu
from FastSimulation.Configuration.HLT_cff import *

hltTauPrescaler = cms.EDFilter("HLTPrescaler",
    makeFilterObject = cms.bool(True),
    eventOffset = cms.uint32(0),
    prescaleFactor = cms.uint32(1)
)

hltTauL1SeedFilter = cms.EDFilter("HLTLevel1GTSeed",
    L1SeedsLogicalExpression = cms.string('L1_SingleJet30'),
    L1GtReadoutRecordTag = cms.InputTag("gtDigis"),
    L1GtObjectMapTag = cms.InputTag("gtDigis"),
    L1CollectionsTag = cms.InputTag("l1extraParticles"),
    L1MuonCollectionTag = cms.InputTag("l1extraParticles"),
    L1TechTriggerSeeding = cms.bool(False)
)

hltL2TauJets = cms.EDFilter("L2TauJetsMerger",
    L1ParticlesJet = cms.InputTag("l1extraParticles","Central"),
    JetSrc = cms.VInputTag(cms.InputTag("hltIcone5Tau1"), cms.InputTag("hltIcone5Tau2"), cms.InputTag("hltIcone5Tau3"), cms.InputTag("hltIcone5Tau4"), cms.InputTag("hltIcone5Cen1"),cms.InputTag("hltIcone5Cen2"), cms.InputTag("hltIcone5Cen3"), cms.InputTag("hltIcone5Cen4")),
    EtMin = cms.double(5.0),
    L1ParticlesTau = cms.InputTag("l1extraParticles","Tau"),
    L1TauTrigger = cms.InputTag("hltTauL1SeedFilter")
)

hltL2TauIsolationProducer = cms.EDProducer("L2TauIsolationProducer",
    ECALIsolation = cms.PSet(
        innerCone = cms.double(0.15),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    TowerIsolation = cms.PSet(
        innerCone = cms.double(0.2),
        runAlgorithm = cms.bool(True),
        outerCone = cms.double(0.5)
    ),
    EERecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE"),
    EBRecHits = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"),
    L2TauJetCollection = cms.InputTag("hltL2TauJets"),
    ECALClustering = cms.PSet(
        runAlgorithm = cms.bool(True),
        clusterRadius = cms.double(0.08)
    ),
    towerThreshold = cms.double(0.2),
    crystalThreshold = cms.double(0.1)
)

hltL2TauIsolationSelector = cms.EDFilter("L2TauIsolationSelector",
    MinJetEt = cms.double(0.0),
    SeedTowerEt = cms.double(-10.0),
    ClusterEtaRMS = cms.double(1000.0),
    ClusterDRRMS = cms.double(1000.0),
    ECALIsolEt = cms.double(1000.0),
    TowerIsolEt = cms.double(1000.0),
    ClusterPhiRMS = cms.double(1000.0),
    ClusterNClusters = cms.int32(1000),
    L2InfoAssociation = cms.InputTag("hltL2TauIsolationProducer","L2TauIsolationInfoAssociator")
)

hltCaloTowers = cms.EDFilter("CaloTowerCandidateCreator",
    src = cms.InputTag("hltTowerMakerForAll"),
    minimumEt = cms.double(-1.0),
    minimumE = cms.double(-1.0)
)

hltCaloTowersCen1 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(0),
    TauTrigger = cms.InputTag("l1extraParticles","Central"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Cen1 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersCen1"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    inputEMin = cms.double(0.0)
)

hltCaloTowersCen2 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(1),
    TauTrigger = cms.InputTag("l1extraParticles","Central"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Cen2 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersCen2"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    inputEMin = cms.double(0.0)
)

hltCaloTowersCen3 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(2),
    TauTrigger = cms.InputTag("l1extraParticles","Central"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Cen3 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersCen3"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    inputEMin = cms.double(0.0)
)

hltCaloTowersCen4 = cms.EDFilter("CaloTowerCreatorForTauHLT",
    towers = cms.InputTag("hltTowerMakerForAll"),
    TauId = cms.int32(3),
    TauTrigger = cms.InputTag("l1extraParticles","Central"),
    minimumE = cms.double(0.8),
    UseTowersInCone = cms.double(0.8),
    minimumEt = cms.double(0.5)
)

hltIcone5Cen4 = cms.EDProducer("IterativeConeJetProducer",
    src = cms.InputTag("hltCaloTowersCen4"),
    verbose = cms.untracked.bool(False),
    inputEtMin = cms.double(0.5),
    coneRadius = cms.double(0.5),
    alias = cms.untracked.string('IC5CaloJet'),
    seedThreshold = cms.double(1.0),
    debugLevel = cms.untracked.int32(0),
    jetType = cms.untracked.string('CaloJet'),
    jetPtMin = cms.double(0.0),
    inputEMin = cms.double(0.0)
)


hltAssociatorL25Tau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltL2TauIsolationSelector","Isolated"),
    tracks = cms.InputTag("hltPixelTracks"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL25Tau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(2),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL25Tau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.1),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(1.0),
    useVertex = cms.bool(True)
)

hltIsolatedL25Tau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(True),
    UseIsolationDiscriminator = cms.bool(False), # ask Simone!!!
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.5),
    MatchingCone = cms.double(0.5),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL25Tau")),
    IsolationCone = cms.double(0.5),
    MinimumTransverseMomentumLeadingTrack = cms.double(1.0),
    UseVertex = cms.bool(False)
)

"""
hltL3TauPixelSeeds = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
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
            JetSrc = cms.InputTag("hltIsolatedL25Tau"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("hltPixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

hltCkfTrackCandidatesL3Tau = cms.EDFilter("CkfTrackCandidateMaker",
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryCleaner = cms.string('TrajectoryCleanerBySharedHits'),
    SeedLabel = cms.string(''),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),                                      
    SeedProducer = cms.string('hltL3TauPixelSeeds'),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    TrajectoryBuilder = cms.string('trajBuilderL3'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)

hltCtfWithMaterialTracksL3Tau = cms.EDProducer("TrackProducer",
    src = cms.InputTag('hltCkfTrackCandidatesL3Tau'),
    producer = cms.string(''),
    Fitter = cms.string('FittingSmootherRK'),
    useHitsSplitting = cms.bool(False),
    alias = cms.untracked.string('ctfWithMaterialTracks'),
    TrajectoryInEvent = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    beamSpot = cms.InputTag( "offlineBeamSpot" ),
    AlgorithmName = cms.string( "undefAlgorithm" ),                                           
    Propagator = cms.string('RungeKuttaTrackerPropagator')
)
"""

hltL3TauPixelSeeds = cms.Sequence(dummyModule)
hltCkfTrackCandidatesL3Tau = cms.Sequence(globalPixelTracking)
hltCtfWithMaterialTracksL3Tau = cms.EDFilter("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks")),
    ptMin = cms.untracked.double(1.0)
)

hltAssociatorL3Tau = cms.EDFilter("JetTracksAssociatorAtVertex",
    jets = cms.InputTag("hltIsolatedL25Tau"),
    tracks = cms.InputTag("hltCtfWithMaterialTracksL3Tau"),
    coneSize = cms.double(0.5)
)

hltConeIsolationL3Tau = cms.EDFilter("ConeIsolation",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    MaximumTransverseImpactParameter = cms.double(300.0),
    VariableConeParameter = cms.double(3.5),
    MinimumNumberOfHits = cms.int32(5),
    MinimumTransverseMomentum = cms.double(1.0),
    JetTrackSrc = cms.InputTag("hltAssociatorL3Tau"),
    VariableMaxCone = cms.double(0.17),
    VariableMinCone = cms.double(0.05),
    DeltaZetTrackVertex = cms.double(0.2),
    MaximumChiSquared = cms.double(100.0),
    SignalCone = cms.double(0.1),
    MatchingCone = cms.double(0.1),
    vertexSrc = cms.InputTag("hltPixelVertices"),
    MinimumNumberOfPixelHits = cms.int32(2),
    useBeamSpot = cms.bool(True),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    UseFixedSizeCone = cms.bool(True),
    BeamSpotProducer = cms.InputTag("offlineBeamSpot"),
    IsolationCone = cms.double(0.1),
    MinimumTransverseMomentumLeadingTrack = cms.double(1.0),
    useVertex = cms.bool(True)
)

hltIsolatedL3Tau = cms.EDFilter("IsolatedTauJetsSelector",
    MinimumTransverseMomentumInIsolationRing = cms.double(1.0),
    UseInHLTOpen = cms.bool(True),
    DeltaZetTrackVertex = cms.double(0.2),
    SignalCone = cms.double(0.5),
    MatchingCone = cms.double(0.5),
    VertexSrc = cms.InputTag("hltPixelVertices"),
    MaximumNumberOfTracksIsolationRing = cms.int32(0),
    JetSrc = cms.VInputTag(cms.InputTag("hltConeIsolationL3Tau")),
    IsolationCone = cms.double(0.5),
    MinimumTransverseMomentumLeadingTrack = cms.double(1.0),
    UseIsolationDiscriminator = cms.bool(False), # ask Simone!!!
    UseVertex = cms.bool(False)
)

TauOpenHLT = cms.EDProducer("HLTTauProducer",
    L25TrackIsoJets = cms.InputTag("hltConeIsolationL25Tau"),
    L3TrackIsoJets = cms.InputTag("hltConeIsolationL3Tau"),
    SignalCone = cms.double(0.15),
    MatchingCone = cms.double(0.1),
    L2EcalIsoJets = cms.InputTag("hltL2TauIsolationProducer","L2TauIsolationInfoAssociator"),
    IsolationCone = cms.double(0.5)
)

OpenHLTDoCaloSequence = cms.Sequence(hltEcalPreshowerDigis+hltEcalRegionalRestFEDs+hltEcalRegionalRestDigis+hltEcalRegionalRestWeightUncalibRecHit+hltEcalRegionalRestRecHitTmp+hltEcalRecHitAll+hltEcalPreshowerRecHit+HLTDoLocalHcalSequence+hltTowerMakerForAll+hltCaloTowers)
OpenHLTCaloTausCreatorSequence = cms.Sequence(OpenHLTDoCaloSequence+hltCaloTowersTau1+hltIcone5Tau1+hltCaloTowersTau2+hltIcone5Tau2+hltCaloTowersTau3+hltIcone5Tau3+hltCaloTowersTau4+hltIcone5Tau4+hltCaloTowersCen1+hltIcone5Cen1+hltCaloTowersCen2+hltIcone5Cen2+hltCaloTowersCen3+hltIcone5Cen3+hltCaloTowersCen4+hltIcone5Cen4)
