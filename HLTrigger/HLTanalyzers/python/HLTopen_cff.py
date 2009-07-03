import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
#from HLTrigger.Configuration.HLT_8E29_cff import *
#from HLTrigger.Configuration.HLT_1E31_cff import *
from HLTrigger.Configuration.HLT_FULL_cff import *

# create the jetMET HLT reco path
DoHLTJets = cms.Path(HLTBeginSequence + 
    HLTRecoJetMETSequence + 
    HLTDoHTRecoSequence)

# create the muon HLT reco path
DoHltMuon = cms.Path(
    HLTBeginSequence +
    HLTL2muonrecoSequence + 
    HLTL2muonisorecoSequence + 
    HLTL3muonrecoSequence + 
    HLTL3muonisorecoSequence +
    HLTEndSequence )

# create the Egamma HLT reco paths
DoHLTPhoton = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    hltL1IsolatedPhotonEcalIsol + 
    hltL1NonIsolatedPhotonEcalIsol + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedPhotonHcalIsol + 
    hltL1NonIsolatedPhotonHcalIsol + 
    HLTDoLocalTrackerSequence + 
    HLTL1IsoEgammaRegionalRecoTrackerSequence + 
    HLTL1NonIsoEgammaRegionalRecoTrackerSequence + 
    hltL1IsoPhotonHollowTrackIsol + 
    hltL1NonIsoPhotonHollowTrackIsol )

##DoHLTElectron = cms.Path( 
##    HLTBeginSequence + 
##    HLTDoRegionalEgammaEcalSequence + 
##    HLTL1IsolatedEcalClustersSequence + 
##    HLTL1NonIsolatedEcalClustersSequence + 
##    hltL1IsoRecoEcalCandidate + 
##    hltL1NonIsoRecoEcalCandidate + 
##    HLTDoLocalHcalWithoutHOSequence + 
##    hltL1IsolatedElectronHcalIsol + 
##    hltL1NonIsolatedElectronHcalIsol + 
##    HLTDoLocalPixelSequence + 
##    HLTDoLocalStripSequence + 
##    HLTPixelMatchElectronL1IsoSequence + 
##    HLTPixelMatchElectronL1NonIsoSequence + 
##    HLTPixelMatchElectronL1IsoTrackingSequence + 
##    HLTPixelMatchElectronL1NonIsoTrackingSequence + 
##    HLTL1IsoElectronsRegionalRecoTrackerSequence + 
##    HLTL1NonIsoElectronsRegionalRecoTrackerSequence + 
##    hltL1IsoElectronTrackIsol + 
##    hltL1NonIsoElectronTrackIsol )

##DoHLTElectronStartUpWindows = cms.Path( 
##    HLTBeginSequence + 
##    HLTDoRegionalEgammaEcalSequence + 
##    HLTL1IsolatedEcalClustersSequence + 
##    HLTL1NonIsolatedEcalClustersSequence + 
##    hltL1IsoRecoEcalCandidate + 
##    hltL1NonIsoRecoEcalCandidate + 
##    HLTDoLocalHcalWithoutHOSequence + 
##    hltL1IsolatedElectronHcalIsol + 
##    hltL1NonIsolatedElectronHcalIsol + 
##    HLTDoLocalPixelSequence + 
##    HLTDoLocalStripSequence + 
##    hltL1IsoStartUpElectronPixelSeeds + 
##    hltL1NonIsoStartUpElectronPixelSeeds + 
##    HLTPixelMatchStartUpElectronL1IsoTrackingSequence + 
##    HLTPixelMatchElectronL1NonIsoStartUpTrackingSequence + 
##    HLTL1IsoStartUpElectronsRegionalRecoTrackerSequence + 
##    HLTL1NonIsoStartUpElectronsRegionalRecoTrackerSequence + 
##    hltL1IsoStartUpElectronTrackIsol + 
##    hltL1NonIsoStartupElectronTrackIsol )

## LW ele RECO L1ISO
hltCkfL1IsoLWTC = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltL1IsoLargeWindowElectronPixelSeeds" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    cleanTrajectoryAfterInOut = cms.bool( False )
)

hltCtfL1IsoLW = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltCtfL1IsoLW" ),
    Fitter = cms.string( "hltKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltCkfL1IsoLWTC" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1IsoLW = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1IsoLW" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)

## LW ele RECO L1NonISO
hltCkfL1NonIsoLWTC = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltL1NonIsoLargeWindowElectronPixelSeeds" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    cleanTrajectoryAfterInOut = cms.bool( False )
)

hltCtfL1NonIsoLW = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( True ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltCtfL1NonIsoLW" ),
    Fitter = cms.string( "hltKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltCkfL1NonIsoLWTC" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)
hltPixelMatchElectronsL1NonIsoLW = cms.EDProducer( "EgammaHLTPixelMatchElectronProducers",
    TrackProducer = cms.InputTag( "hltCtfL1NonIsoLW" ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" )
)

## LW ele TrackIso L1Iso


hltL1IsoLWEleRegPSG = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1IsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1IsoLW" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "MixedLayerPairs" )
    )
)

hltL1IsoLWEleRegioCkfTC = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltL1IsoLWEleRegPSG" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    cleanTrajectoryAfterInOut = cms.bool( False )
)
hltL1IsoLWEleRegioCTF = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltL1IsoLWEleRegioCTF" ),
    Fitter = cms.string( "hltKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltL1IsoLWEleRegioCkfTC" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)


## LW ele TrackIso L1NonIso
hltL1NonIsoLWEleRegPSG = cms.EDProducer( "EgammaHLTRegionalPixelSeedGeneratorProducers",
    ptMin = cms.double( 1.5 ),
    vertexZ = cms.double( 0.0 ),
    originRadius = cms.double( 0.02 ),
    originHalfLength = cms.double( 0.5 ),
    deltaEtaRegion = cms.double( 0.3 ),
    deltaPhiRegion = cms.double( 0.3 ),
    candTag = cms.InputTag( "hltL1NonIsoRecoEcalCandidate" ),
    candTagEle = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLW" ),
    UseZInVertex = cms.bool( True ),
    BSProducer = cms.InputTag( "hltOfflineBeamSpot" ),
    OrderedHitsFactoryPSet = cms.PSet( 
      ComponentName = cms.string( "StandardHitPairGenerator" ),
      SeedingLayers = cms.string( "MixedLayerPairs" )
    )
)


hltL1NonIsoLWEleRegioCkfTC = cms.EDProducer( "CkfTrackCandidateMaker",
    src = cms.InputTag( "hltL1NonIsoLWEleRegPSG" ),
    TrajectoryBuilder = cms.string( "CkfTrajectoryBuilder" ),
    TrajectoryCleaner = cms.string( "TrajectoryCleanerBySharedHits" ),
    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
    useHitsSplitting = cms.bool( False ),
    doSeedingRegionRebuilding = cms.bool( False ),
    TransientInitialStateEstimatorParameters = cms.PSet( 
      propagatorAlongTISE = cms.string( "PropagatorWithMaterial" ),
      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialOpposite" )
    ),
    cleanTrajectoryAfterInOut = cms.bool( False )
)
hltL1NonIsoLWEleRegioCTF = cms.EDProducer( "TrackProducer",
    TrajectoryInEvent = cms.bool( False ),
    useHitsSplitting = cms.bool( False ),
    clusterRemovalInfo = cms.InputTag( "" ),
    alias = cms.untracked.string( "hltL1NonIsoLWEleRegioCTF" ),
    Fitter = cms.string( "hltKFFittingSmoother" ),
    Propagator = cms.string( "PropagatorWithMaterial" ),
    src = cms.InputTag( "hltL1NonIsoLWEleRegioCkfTC" ),
    beamSpot = cms.InputTag( "hltOfflineBeamSpot" ),
    TTRHBuilder = cms.string( "WithTrackAngle" ),
    AlgorithmName = cms.string( "undefAlgorithm" )
)


## track iso for LW
hltL1IsoLWEleTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1IsoLW" ),
    trackProducer = cms.InputTag( "hltL1IsoLWEleRegioCTF" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)

hltL1NonIsoLWEleTrackIsol = cms.EDProducer( "EgammaHLTElectronTrackIsolationProducers",
    electronProducer = cms.InputTag( "hltPixelMatchElectronsL1NonIsoLW" ),
    trackProducer = cms.InputTag( "hltL1NonIsoLWEleRegioCTF" ),
    egTrkIsoPtMin = cms.double( 1.5 ),
    egTrkIsoConeSize = cms.double( 0.2 ),
    egTrkIsoZSpan = cms.double( 0.1 ),
    egTrkIsoRSpan = cms.double( 999999.0 ),
    egTrkIsoVetoConeSize = cms.double( 0.02 )
)


DoLWTracking = cms.Sequence(
    hltCkfL1IsoLWTC +
    hltCtfL1IsoLW +
    hltPixelMatchElectronsL1IsoLW +
    hltCkfL1NonIsoLWTC +
    hltCtfL1NonIsoLW +
    hltPixelMatchElectronsL1NonIsoLW +
    hltL1IsoLWEleRegPSG +
    hltL1IsoLWEleRegioCkfTC +
    hltL1IsoLWEleRegioCTF +
    hltL1IsoLWEleTrackIsol +
    hltL1NonIsoLWEleRegPSG +
    hltL1NonIsoLWEleRegioCkfTC +
    hltL1NonIsoLWEleRegioCTF +
    hltL1NonIsoLWEleTrackIsol     
    )

DoHLTElectronStartUpWindows = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedElectronHcalIsol + 
    hltL1NonIsolatedElectronHcalIsol + 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence + 
    hltL1IsoStartUpElectronPixelSeeds + 
    hltL1NonIsoStartUpElectronPixelSeeds + 
    HLTPixelMatchElectronL1IsoTrackingSequence + 
    HLTPixelMatchElectronL1NonIsoTrackingSequence + 
    HLTL1IsoElectronsRegionalRecoTrackerSequence + 
    HLTL1NonIsoElectronsRegionalRecoTrackerSequence + 
    hltL1IsoElectronTrackIsol + 
    hltL1NonIsoElectronTrackIsol )

DoHLTElectronLargeWindows = cms.Path( 
    HLTBeginSequence + 
    HLTDoRegionalEgammaEcalSequence + 
    HLTL1IsolatedEcalClustersSequence + 
    HLTL1NonIsolatedEcalClustersSequence + 
    hltL1IsoRecoEcalCandidate + 
    hltL1NonIsoRecoEcalCandidate + 
    HLTDoLocalHcalWithoutHOSequence + 
    hltL1IsolatedElectronHcalIsol + 
    hltL1NonIsolatedElectronHcalIsol + 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence +
    hltL1IsoLargeWindowElectronPixelSeeds +
    hltL1NonIsoLargeWindowElectronPixelSeeds +
    DoLWTracking
    )

# create the tau HLT reco path
##from HLTrigger.HLTanalyzers.OpenHLT_Tau_cff import *
##DoHLTTau = cms.Path(HLTBeginSequence +
##                    hltTauPrescaler +
##                    OpenHLTCaloTausCreatorSequence +
##                    hltL2TauJets +
##                    hltL2TauIsolationProducer +
##                    hltL2TauRelaxingIsolationSelector +
##                    HLTDoLocalPixelSequence +
##                    HLTRecopixelvertexingSequence +
##                    HLTL25TauTrackReconstructionSequence +
##                    HLTL25TauTrackIsolation +
##                    TauOpenHLT+
##                    HLTEndSequence)

# create the b-jet HLT paths
##from HLTrigger.HLTanalyzers.OpenHLT_BJet_cff import *
##DoHLTBTag = cms.Path(
##    HLTBeginSequence +
##    HLTBCommonL2recoSequence +
##    OpenHLTBLifetimeL25recoSequence +
##    OpenHLTBSoftmuonL25recoSequence +
##    OpenHLTBLifetimeL3recoSequence +
##    OpenHLTBLifetimeL3recoSequenceRelaxed +
##    OpenHLTBSoftmuonL3recoSequence +
##    HLTEndSequence )


DoHLTAlCaPi0Eta1E31 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta1E31 +
    hltPreAlCaEcalPi01E31 +
    HLTDoRegionalPi0EtaESSequence +
    HLTDoRegionalPi0EtaEcalSequence +
    HLTEndSequence )

DoHLTAlCaPi0Eta8E29 = cms.Path(
    HLTBeginSequence +
    hltL1sAlCaEcalPi0Eta8E29 +
    hltPreAlCaEcalPi08E29 +
    HLTDoRegionalPi0EtaESSequence +
    HLTDoRegionalPi0EtaEcalSequence +
    HLTEndSequence )


##DoHLTAlCaECALPhiSym = cms.Path(
##    HLTBeginSequence +
##    # ccla hltL1sAlCaEcalPhiSym +
##    hltPreAlCaEcalPhiSym +
##    hltEcalDigis +
##    hltEcalWeightUncalibRecHit +
##    hltEcalRecHit +
##    hltAlCaPhiSymStream +
##    HLTDoLocalHcalSequence +
##    HLTEndSequence
##    )

DoHLTIsoTrack = cms.Path(
    HLTBeginSequence +
    hltL1sIsoTrack1E31 +
    hltPreIsoTrack1E31 +
    HLTL2HcalIsolTrackSequence +
    hltIsolPixelTrackProd1E31 +
    hltIsolPixelTrackL2Filter1E31 +
    HLTDoLocalStripSequence +
    hltHITPixelPairSeedGenerator1E31 +
    hltHITPixelTripletSeedGenerator1E31 +
    hltHITSeedCombiner1E31 +
    hltHITCkfTrackCandidates1E31 +
    hltHITCtfWithMaterialTracks1E31 +
    hltHITIPTCorrector1E31 +
    HLTEndSequence)

