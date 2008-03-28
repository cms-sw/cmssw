import FWCore.ParameterSet.Config as cms

# identify electrons from the ECAL energy deposits associate to the tracks
btagSoftElectrons = cms.EDProducer("SoftElectronProducer",
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
    TrackTag = cms.InputTag("generalTracks"),
    BasicClusterTag = cms.InputTag("islandBasicClusters","islandBarrelBasicClusters"),
    BasicClusterShapeTag = cms.InputTag("islandBasicClusters","islandBarrelShape"),
    HBHERecHitTag = cms.InputTag("hbhereco"),
    DiscriminatorCut = cms.double(0.9),
    HOverEConeSize = cms.double(0.3)
)

btagSoftElectrons.TrackAssociatorParameters.useEcal = True
btagSoftElectrons.TrackAssociatorParameters.useHcal = False
btagSoftElectrons.TrackAssociatorParameters.useHO = False
btagSoftElectrons.TrackAssociatorParameters.useMuon = False

