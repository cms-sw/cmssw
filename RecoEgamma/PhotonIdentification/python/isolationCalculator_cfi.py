import FWCore.ParameterSet.Config as cms

isolationSumsCalculator = cms.PSet(
    #required inputs
    ComponentName = cms.string('isolationSumsCalculator'),
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
#    DoHollowConeTrackIsolationCut = cms.bool(True),
#    DoSolidConeTrackIsolationCut = cms.bool(False),
#    DoHollowConeNTrkCut = cms.bool(False),
#    DoSolidConeNTrkCut = cms.bool(False),
#    DoHadOverEMCut = cms.bool(False),
#    DoEtaWidthCut = cms.bool(False),
#    DoHcalTowerIsolationCut = cms.bool(True),
#    DoEcalRecHitIsolationCut = cms.bool(True),
#    DoR9Cut = cms.bool(True),                               
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
    HcalTowerThreshEB = cms.double(0.0)

)


