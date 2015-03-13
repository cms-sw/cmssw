import FWCore.ParameterSet.Config as cms

ecalDrivenElectronSeedsParameters = cms.PSet(

    # steering
    fromTrackerSeeds = cms.bool(True),
    initialSeeds = cms.InputTag("newCombinedSeeds"),
    preFilteredSeeds = cms.bool(False),
    useRecoVertex = cms.bool(False),
    vertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    dynamicPhiRoad = cms.bool(True),
    searchInTIDTEC = cms.bool(True), ##  possibility to inhibit extended forward coverage

    # specify where to get the hits from
    measurementTrackerName = cms.string(""),

    # SC filtering
    #SCEtCut = cms.double(4.0),
    SCEtCutBarrel = cms.double(4.0),
    SCEtCutEndcap = cms.double(9.0), 

    # H/E
    applyHOverECut = cms.bool(True),
    #hOverEMethod = cms.int32(0),  # 0 = cone #1 = single tower #2 = towersBehindCluster (max is 4)
    hOverEMethodBarrel = cms.int32(0),  # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4)
    hOverEMethodEndcap = cms.int32(1),  # 0 = cone #1 = single tower #2 = towersBehindCluster #3 = clusters (max is 4)
    hOverEConeSizeBarrel = cms.double(0.15),
    hOverEConeSizeEndcap = cms.double(0.15),
    #maxHOverE = cms.double(0.1),
    maxHOverEBarrel = cms.double(0.15),
    maxHOverEEndcaps = cms.double(0.1), 
    maxHOverEOuterEndcaps = cms.double(0.2),
    maxHBarrel = cms.double(0.0),
    maxHEndcaps = cms.double(0.0),
    # H/E rechits
    hcalRecHits = cms.InputTag("hbhereco"), # OBSOLETE
    hOverEHBMinE = cms.double(0.7),         # OBSOLETE
    hOverEHFMinE = cms.double(0.8),         # OBSOLETE
    # H/E towers
    hcalTowers = cms.InputTag("towerMaker"),
    hOverEPtMin = cms.double(0.),
    # cluster sources
    barrelHCALClusters = cms.InputTag('particleFlowClusterHCAL'),
    endcapHCALClusters = cms.InputTag('particleFlowClusterHCAL'),
    
    # r/z windows
    nSigmasDeltaZ1 = cms.double(5.), ## in case beam spot is used for the matching
    deltaZ1WithVertex = cms.double(25.), ## in case reco vertex is used for the matching
    z2MinB = cms.double(-0.09), ## barrel
    z2MaxB = cms.double(0.09), ## barrel
    
    r2MinF = cms.double(-0.15), ## forward
    r2MaxF = cms.double(0.15), ## forward
    rMinI = cms.double(-0.2), ## intermediate region SC in EB and 2nd hits in PXF
    rMaxI = cms.double(0.2), ## intermediate region SC in EB and 2nd hits in PXF

    # phi windows (dynamic)
    LowPtThreshold = cms.double(5.0),
    HighPtThreshold = cms.double(35.0),
    SizeWindowENeg = cms.double(0.675),
    DeltaPhi1Low = cms.double(0.23),
    DeltaPhi1High = cms.double(0.08),
#    DeltaPhi2 = cms.double(0.008),
    DeltaPhi2B = cms.double(0.008), ## barrel
    DeltaPhi2F = cms.double(0.012), ## forward

    # phi windows (non dynamic, overwritten in case dynamic is selected)
    ePhiMin1 = cms.double(-0.125),
    ePhiMax1 = cms.double(0.075),
    pPhiMin1 = cms.double(-0.075),
    pPhiMax1 = cms.double(0.125),
#    PhiMin2 = cms.double(-0.002),
#    PhiMax2 = cms.double(0.002),
    PhiMin2B = cms.double(-0.002), ## barrel
    PhiMax2B = cms.double(0.002), ## barrel
    PhiMin2F = cms.double(-0.003), ## forward
    PhiMax2F = cms.double(0.003), ## forward
    
)

