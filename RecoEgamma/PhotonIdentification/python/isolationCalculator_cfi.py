import FWCore.ParameterSet.Config as cms


isolationSumsCalculator = cms.PSet(
    #required inputs
    ComponentName = cms.string('isolationSumsCalculator'),

    barrelEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEB'),
    endcapEcalRecHitCollection = cms.InputTag('ecalRecHit:EcalRecHitsEE'),
    HBHERecHitCollection = cms.InputTag('hbhereco'),
    HFRecHitCollection = cms.InputTag('hfreco'),
    maxHcalRecHitSeverity = cms.int32(9), # FIXME recheck thresholds

    # Photon will be marked as being near phi module boundary if
    #  it is closer than this.  Currently half a crystal.
    #  1 Ecal Crystal = 0.0174 radians = 1 degree
    modulePhiBoundary =   cms.double(0.0087),
    # Photon will be marked as being near an eta boundary if
    #  it is between the 0th and 1st element, or the 2nd and 3rd, or the 4th and 5th...
    moduleEtaBoundary = cms.vdouble(0.0, 0.02, 0.43, 0.46, 0.78, 0.81, 1.13, 1.15, 1.45, 1.58),
    #What collection of tracks do I use for Track Isolation?
    trackProducer = cms.InputTag("generalTracks"),
    #use beam spot for track isolation
    beamSpotProducer = cms.InputTag("offlineBeamSpot"),
    #switches, turn on quality cuts for various quantities.
    vetoClustered  = cms.bool(False),  #will remove clustered rechits from ecal iso sum
    useNumCrystals = cms.bool(True),  #will define the veto region by number of crystals in stead of geometrically
    #configuration of parameters for isolation
#### BARREL
    #tracks
    isolationtrackThresholdA_Barrel = cms.double(0.0),
    TrackConeOuterRadiusA_Barrel    = cms.double(0.4),
    TrackConeInnerRadiusA_Barrel    = cms.double(0.04),
    isolationtrackEtaSliceA_Barrel  = cms.double(0.015),
    longImpactParameterA_Barrel     = cms.double(0.2),
    transImpactParameterA_Barrel    = cms.double(0.1),
#
    isolationtrackThresholdB_Barrel = cms.double(0.0),
    TrackConeOuterRadiusB_Barrel    = cms.double(0.3),
    TrackConeInnerRadiusB_Barrel    = cms.double(0.04),
    isolationtrackEtaSliceB_Barrel  = cms.double(0.015),
    longImpactParameterB_Barrel     = cms.double(0.2),
    transImpactParameterB_Barrel    = cms.double(0.1),
    #Ecal rechits 
    EcalRecHitInnerRadiusA_Barrel   = cms.double(3.5),
    EcalRecHitOuterRadiusA_Barrel   = cms.double(0.4),
    EcalRecHitEtaSliceA_Barrel      = cms.double(2.5),
    EcalRecHitThreshEA_Barrel       = cms.double(0.095),
    EcalRecHitThreshEtA_Barrel      = cms.double(0.0),
#
    EcalRecHitInnerRadiusB_Barrel   = cms.double(3.5),
    EcalRecHitOuterRadiusB_Barrel   = cms.double(0.3),
    EcalRecHitEtaSliceB_Barrel      = cms.double(2.5),
    EcalRecHitThreshEB_Barrel       = cms.double(0.095),
    EcalRecHitThreshEtB_Barrel      = cms.double(0.0),
    #Hcal rechits
    HcalDepth1RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth1RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth1RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth2RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth2RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth2RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth3RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth3RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth3RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth4RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth4RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth4RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth5RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth5RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth5RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth6RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth6RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth6RecHitThreshEA_Barrel     = cms.double(0.0),
    HcalDepth7RecHitInnerRadiusA_Barrel = cms.double(0.15),
    HcalDepth7RecHitOuterRadiusA_Barrel = cms.double(0.4),
    HcalDepth7RecHitThreshEA_Barrel     = cms.double(0.0),
#
    HcalDepth1RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth1RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth1RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth2RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth2RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth2RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth3RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth3RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth3RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth4RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth4RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth4RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth5RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth5RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth5RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth6RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth6RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth6RecHitThreshEB_Barrel     = cms.double(0.0),
    HcalDepth7RecHitInnerRadiusB_Barrel = cms.double(0.15),
    HcalDepth7RecHitOuterRadiusB_Barrel = cms.double(0.3),
    HcalDepth7RecHitThreshEB_Barrel     = cms.double(0.0),

#### ENDCAP
    #tracks
    isolationtrackThresholdA_Endcap  = cms.double(0.0),
    TrackConeOuterRadiusA_Endcap     = cms.double(0.4),
    TrackConeInnerRadiusA_Endcap     = cms.double(0.04),
    isolationtrackEtaSliceA_Endcap   = cms.double(0.015),
    longImpactParameterA_Endcap      = cms.double(0.2),
    transImpactParameterA_Endcap     = cms.double(0.1),
##
    isolationtrackThresholdB_Endcap  = cms.double(0.0),
    TrackConeOuterRadiusB_Endcap     = cms.double(0.3),
    TrackConeInnerRadiusB_Endcap     = cms.double(0.04),
    isolationtrackEtaSliceB_Endcap   = cms.double(0.015),
    longImpactParameterB_Endcap      = cms.double(0.2),
    transImpactParameterB_Endcap     = cms.double(0.1),
    #Ecal rechits 
    EcalRecHitInnerRadiusA_Endcap    = cms.double(3.5),
    EcalRecHitOuterRadiusA_Endcap    = cms.double(0.4),
    EcalRecHitEtaSliceA_Endcap       = cms.double(2.5),
    EcalRecHitThreshEA_Endcap        = cms.double(0.0),
    EcalRecHitThreshEtA_Endcap       = cms.double(0.110),
#
    EcalRecHitInnerRadiusB_Endcap    = cms.double(3.5),
    EcalRecHitOuterRadiusB_Endcap    = cms.double(0.3),
    EcalRecHitEtaSliceB_Endcap       = cms.double(2.5),
    EcalRecHitThreshEB_Endcap        = cms.double(0.0),
    EcalRecHitThreshEtB_Endcap       = cms.double(0.110),

    #severityLevelCut = cms.int32(4),
    #severityRecHitThreshold = cms.double(5.0),
    #spikeIdString = cms.string('kSwissCrossBordersIncluded'),
    #spikeIdThreshold = cms.double(0.95),

    #Hcal towers
    HcalDepth1RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth1RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth1RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth2RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth2RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth2RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth3RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth3RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth3RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth4RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth4RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth4RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth5RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth5RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth5RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth6RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth6RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth6RecHitThreshEA_Endcap     = cms.double(0.0),
    HcalDepth7RecHitInnerRadiusA_Endcap = cms.double(0.15),
    HcalDepth7RecHitOuterRadiusA_Endcap = cms.double(0.4),
    HcalDepth7RecHitThreshEA_Endcap     = cms.double(0.0),
#
    HcalDepth1RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth1RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth1RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth2RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth2RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth2RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth3RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth3RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth3RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth4RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth4RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth4RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth5RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth5RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth5RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth6RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth6RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth6RecHitThreshEB_Endcap     = cms.double(0.0),
    HcalDepth7RecHitInnerRadiusB_Endcap = cms.double(0.15),
    HcalDepth7RecHitOuterRadiusB_Endcap = cms.double(0.3),
    HcalDepth7RecHitThreshEB_Endcap     = cms.double(0.0),

    #recHitFlagsToBeExcluded = cms.vstring(
    #    'kFaultyHardware',
    #    'kPoorCalib',
    #    ecalRecHitFlag_kSaturated,
    #    ecalRecHitFlag_kLeadingEdgeRecovered,
    #    ecalRecHitFlag_kNeighboursRecovered,
    #    'kTowerRecovered',
    #    'kDead'
    #),

    


)


