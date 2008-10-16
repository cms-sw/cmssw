import FWCore.ParameterSet.Config as cms

PhotonIDProd = cms.EDProducer("PhotonIDProducer",
    #required inputs
    #What collection of photons do I run on?
    photonProducer = cms.string('photons'),                              
    photonLabel = cms.string(''),
    #What labels do I use for my products?
    photonIDAssociationLabel = cms.string('PhotonAssociatedID'),
    photonIDLabel = cms.string('PhotonIDCutBasedProducer'),
    #What rechit collection do I use for ECAL iso?                          
    barrelEcalRecHitProducer = cms.string('ecalRecHit'),
    barrelEcalRecHitCollection = cms.string('EcalRecHitsEB'),
    endcapEcalRecHitProducer = cms.string('ecalRecHit'),
    endcapEcalRecHitCollection = cms.string('EcalRecHitsEE'),
    #What tower collection do I use for HCAL iso?
    HcalRecHitProducer = cms.string('towerMaker'),
    HcalRecHitCollection = cms.string(''),
    # Photon will be marked as being near phi module boundary if
    #  it is closer than this.  Currently half a crystal.
    #  1 Ecal Crystal = 0.0174 radians = 1 degree
    modulePhiBoundary =   cms.double(0.0087),
    # Photon will be marked as being near an eta boundary if
    #  it is between the 0th and 1st element, or the 2nd and 3rd, or the 4th and 5th...
    moduleEtaBoundary = cms.vdouble(0.0, 0.02, 0.43, 0.46, 0.78, 0.81, 1.13, 1.15, 1.45, 1.58),
    #What collection of tracks do I use for Track Isolation?
    trackProducer = cms.InputTag("generalTracks"),
    doCutBased = cms.bool(True),
    #switches, turn on quality cuts for various quantities.
    RequireFiducial = cms.bool(False),
    DoHollowConeTrackIsolationCut = cms.bool(True),
    DoSolidConeTrackIsolationCut = cms.bool(False),
    DoHollowConeNTrkCut = cms.bool(False),
    DoSolidConeNTrkCut = cms.bool(False),
    DoHadOverEMCut = cms.bool(False),
    DoEtaWidthCut = cms.bool(False),
    DoHcalTowerIsolationCut = cms.bool(True),
    DoEcalRecHitIsolationCut = cms.bool(True),
    DoR9Cut = cms.bool(True),                               
    #configuration of parameters for isolations
    #tracks
    isolationtrackThresholdA = cms.double(0.0),
    TrackConeOuterRadiusA = cms.double(0.4),
    TrackConeInnerRadiusA = cms.double(0.04),
    #Ecal rechits 
    EcalRecHitInnerRadiusA = cms.double(0.06),
    EcalRecHitOuterRadiusA = cms.double(0.4),
    EcalRecHitEtaSliceA = cms.double(0.04),
    EcalRecThreshEA = cms.double(0.0),
    EcalRecThreshEtA = cms.double(0.0),
    #Hcal towers
    HcalTowerInnerRadiusA = cms.double(0.1),
    HcalTowerOuterRadiusA = cms.double(0.4),
    HcalTowerThreshEA = cms.double(0.0),
    isolationtrackThresholdB = cms.double(0.0),
    TrackConeOuterRadiusB = cms.double(0.3),
    TrackConeInnerRadiusB = cms.double(0.04),
    #Ecal rechits 
    EcalRecHitInnerRadiusB = cms.double(0.06),
    EcalRecHitOuterRadiusB = cms.double(0.3),
    EcalRecHitEtaSliceB = cms.double(0.04),
    EcalRecThreshEB = cms.double(0.0),
    EcalRecThreshEtB = cms.double(0.0),
    #Hcal towers
    HcalTowerInnerRadiusB = cms.double(0.1),
    HcalTowerOuterRadiusB = cms.double(0.3),
    HcalTowerThreshEB = cms.double(0.0),
    #cuts
    #cuts, two sets, EE and EB
    #LooseEM cuts EB
    LooseEMHollowTrkEB = cms.double(999.9),
    LooseEMSolidTrkEB  = cms.double(999.9),
    LooseEMSolidNTrkEB = cms.int32(999),
    LooseEMHollowNTrkEB = cms.int32(999),
    LooseEMEtaWidthEB = cms.double(999.9),
    LooseEMHadOverEMEB = cms.double(999.9),
    LooseEMEcalRecHitIsoEB = cms.double(5.0),
    LooseEMHcalTowerIsoEB = cms.double(5.0),
    LooseEMR9CutEB = cms.double(0.0),
    #LoosePhoton cuts EB  
    LoosePhotonHollowTrkEB = cms.double(9.0),
    LoosePhotonSolidTrkEB  = cms.double(999.9),
    LoosePhotonSolidNTrkEB = cms.int32(999),
    LoosePhotonHollowNTrkEB = cms.int32(999),
    LoosePhotonEtaWidthEB = cms.double(999.9),
    LoosePhotonHadOverEMEB = cms.double(999.9),
    LoosePhotonEcalRecHitIsoEB = cms.double(5.0),
    LoosePhotonHcalTowerIsoEB = cms.double(5.0),
    LoosePhotonR9CutEB = cms.double(0.0),
    #TightPhoton cuts EB
    TightPhotonHollowTrkEB = cms.double(9.0),
    TightPhotonSolidTrkEB  = cms.double(999.9),
    TightPhotonSolidNTrkEB = cms.int32(999),
    TightPhotonHollowNTrkEB = cms.int32(999),
    TightPhotonEtaWidthEB = cms.double(999.9),
    TightPhotonHadOverEMEB = cms.double(999.9),
    TightPhotonEcalRecHitIsoEB = cms.double(5.0),
    TightPhotonHcalTowerIsoEB = cms.double(5.0),
    TightPhotonR9CutEB = cms.double(0.8),
    #LooseEM cuts EE
    LooseEMHollowTrkEE = cms.double(999.9),
    LooseEMSolidTrkEE  = cms.double(999.9),
    LooseEMSolidNTrkEE = cms.int32(999),
    LooseEMHollowNTrkEE = cms.int32(999),
    LooseEMEtaWidthEE = cms.double(999.9),
    LooseEMHadOverEMEE = cms.double(999.9),
    LooseEMEcalRecHitIsoEE = cms.double(5.0),
    LooseEMHcalTowerIsoEE = cms.double(5.0),
    LooseEMR9CutEE = cms.double(0.0),
    #LoosePhoton cuts EE  
    LoosePhotonHollowTrkEE = cms.double(9.0),
    LoosePhotonSolidTrkEE  = cms.double(999.9),
    LoosePhotonSolidNTrkEE = cms.int32(999),
    LoosePhotonHollowNTrkEE = cms.int32(999),
    LoosePhotonEtaWidthEE = cms.double(999.9),
    LoosePhotonHadOverEMEE = cms.double(999.9),
    LoosePhotonEcalRecHitIsoEE = cms.double(5.0),
    LoosePhotonHcalTowerIsoEE = cms.double(5.0),
    LoosePhotonR9CutEE = cms.double(0.0),
    #TightPhoton cuts EE
    TightPhotonHollowTrkEE = cms.double(9.0),
    TightPhotonSolidTrkEE  = cms.double(999.9),
    TightPhotonSolidNTrkEE = cms.int32(999),
    TightPhotonHollowNTrkEE = cms.int32(999),
    TightPhotonEtaWidthEE = cms.double(999.9),
    TightPhotonHadOverEMEE = cms.double(999.9),
    TightPhotonEcalRecHitIsoEE = cms.double(5.0),
    TightPhotonHcalTowerIsoEE = cms.double(5.0),
    TightPhotonR9CutEE = cms.double(0.8)
)


