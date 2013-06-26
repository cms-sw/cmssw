import FWCore.ParameterSet.Config as cms

ecalCosmicsHists = cms.EDAnalyzer("EcalCosmicsHists",
    histogramMinRange = cms.untracked.double(0.0),
    L1GlobalMuonReadoutRecord = cms.untracked.string('gtDigis'),
    ecalRecHitCollectionEB = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    ecalRecHitCollectionEE = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    barrelClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters"),
    L1GlobalReadoutRecord = cms.untracked.string('gtDigis'),
    ecalRawDataColl_ = cms.InputTag("ecalEBunpacker"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    TrackAssociatorParameterBlock = cms.PSet(
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
        )
    ),
    histogramMaxRange = cms.untracked.double(1.8),
    fileName = cms.untracked.string('EcalCosmicsHists'),
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
    #MinTimingAmpEB = cms.untracked.double(0.1),   # for adcToGeV=0.009, gain 200
    #MinRecHitAmpEB = cms.untracked.double(0.027), # for adcToGeV=0.009, gain 200
    MinTimingAmpEB = cms.untracked.double(0.35),  # for adcToGeV=0.035, gain 50
    MinRecHitAmpEB = cms.untracked.double(0.070), # for adcToGeV=0.035, gain 50                               
    MinTimingAmpEE = cms.untracked.double(0.9),   # for adcToGeV=0.06
    MinRecHitAmpEE = cms.untracked.double(0.180), # for adcToGeV=0.06
    MinHighEnergy = cms.untracked.double(2.0),
    runInFileName = cms.untracked.bool(True),
    TimeStampBins = cms.untracked.int32(1800),
    maskedEBs = cms.untracked.vstring('-1'),
    maskedChannels = cms.untracked.vint32(-1),
    TimeStampStart = cms.untracked.double(1215107133.0),
    TimeStampLength = cms.untracked.double(3.0),
    maskedFEDs = cms.untracked.vint32(-1),
    endcapClusterCollection = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
)


