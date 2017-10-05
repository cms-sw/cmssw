import FWCore.ParameterSet.Config as cms
#customisation for pp reco of XeXe run in Oct 2017

# Add caloTowers to AOD event content
def storeCaloTowersAOD(process):

    process.load('Configuration.EventContent.EventContent_cff')
    
    # extend AOD content
    if hasattr(process,'AODoutput'):
        process.AODoutput.outputCommands.extend(['keep *_towerMaker_*_*'])

    if hasattr(process,'AODSIMoutput'):
        process.AODSIMoutput.outputCommands.extend(['keep *_towerMaker_*_*'])

    return process

# Customize process to run HI-style photon isolation in the pp RECO sequences
def addHIIsolationProducer(process):

    process.load('Configuration.EventContent.EventContent_cff')
    
    # extend RecoEgammaFEVT content
    process.RecoEgammaFEVT.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*'
                                                  ])
    
    # extend RecoEgammaRECO content
    process.RECOEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    
    process.FEVTEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    process.FEVTSIMEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    # extend RecoEgammaRECO content
    process.RAWRECOEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.RECOSIMEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.RAWRECOSIMHLTEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    
    process.RAWRECODEBUGHLTEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.FEVTHLTALLEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.FEVTDEBUGEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    # extend RecoEgammaAOD content
    process.AODEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                 'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*'
                                                  ])

    process.AODSIMEventContent.outputCommands.extend(['keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppGED_*_*',
                                                 'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerpp_*_*'
                                                  ])

    # add HI Photon isolation sequence to pp RECO
    process.load('RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi')
    process.load('RecoEcal.EgammaClusterProducers.islandBasicClusters_cfi')

    process.photonIsolationHISequencePP = cms.Sequence(process.islandBasicClusters 
                                                       * process.photonIsolationHIProducerpp 
                                                       * process.photonIsolationHIProducerppGED)
    
    process.reconstruction *= process.photonIsolationHISequencePP
    
    return process

# Customize process to run and add photons reconstructed with Island Clustering
def addIslandPhotons(process):

    process.load('Configuration.EventContent.EventContent_cff')

    # extend RecoEgammaFEVT content
    process.RecoEgammaFEVT.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*'
                                                  ])
    
    # extend RecoEgammaRECO content
    process.RECOEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    
    process.FEVTEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    process.FEVTSIMEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    # extend RecoEgammaRECO content
    process.RAWRECOEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.RECOSIMEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.RAWRECOSIMHLTEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.RECODEBUGEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])
    
    process.RAWRECODEBUGHLTEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.FEVTHLTALLEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    process.FEVTDEBUGEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                  'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*',
                                                  'keep recoCaloClusters_islandBasicClusters_*_*'
                                                  ])

    # extend RecoEgammaAOD content
    process.AODEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                 'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*'
                                                  ])

    process.AODSIMEventContent.outputCommands.extend(['keep recoPhotons_islandPhotons_*_*',
                                                 'keep recoHIPhotonIsolationedmValueMap_photonIsolationHIProducerppIsland_*_*'
                                                  ])

    process.load('RecoEgamma.EgammaPhotonProducers.photonSequence_cff')
    
    process.islandPhotonCore = process.photonCore.clone()
    process.islandPhotonCore.scHybridBarrelProducer = cms.InputTag("correctedIslandBarrelSuperClusters")
    process.islandPhotonCore.scIslandEndcapProducer = cms.InputTag("correctedIslandEndcapSuperClusters")
    process.islandPhotonCore.minSCEt = cms.double(8.0)

    process.islandPhotons = process.photons.clone()
    process.islandPhotons.photonCoreProducer = cms.InputTag("islandPhotonCore")
    process.islandPhotons.minSCEtBarrel = cms.double(5.0)
    process.islandPhotons.minSCEtEndcap = cms.double(15.0)
    process.islandPhotons.minR9Barrel = cms.double(10.)
    process.islandPhotons.minR9Endcap = cms.double(10.)
    process.islandPhotons.maxHoverEEndcap = cms.double(0.5)
    process.islandPhotons.maxHoverEBarrel = cms.double(0.99)

    process.photonSequenceIsland = cms.Sequence(process.islandPhotonCore+process.islandPhotons)

    process.load('RecoHI.HiEgammaAlgos.photonIsolationHIProducer_cfi')

    process.photonIsolationHIProducerppIsland = process.photonIsolationHIProducer.clone(
                                                          trackCollection = cms.InputTag("generalTracks")
                                                          )

    process.load('RecoHI.HiEgammaAlgos.HiIslandClusteringSequence_cff')

    process.islandSequencePP = cms.Sequence(process.islandClusteringSequence 
                                                       * process.photonSequenceIsland 
                                                       * process.photonIsolationHIProducerppIsland)
    
    process.reconstruction *= process.islandSequencePP
    
    return process

#delete a lot of features out of PF to save on timing
def customisePF(process):
    process.load("RecoParticleFlow.Configuration.RecoParticleFlow_cff")
    process.particleFlowBlock.useNuclear = cms.bool(False)

    #kill this because it uses huge amount of timing and HI doesn't need it
    process.load("RecoParticleFlow.PFTracking.particleFlowDisplacedVertexCandidate_cfi")
    process.particleFlowDisplacedVertexCandidate.tracksSelectorParameters.pt_min = 999999.0
    process.particleFlowDisplacedVertexCandidate.tracksSelectorParameters.nChi2_max = 0.0
    process.particleFlowDisplacedVertexCandidate.tracksSelectorParameters.pt_min_prim = 999999.0
    process.particleFlowDisplacedVertexCandidate.tracksSelectorParameters.dxy = 999999.0

    #kill the entire Tau sequence as well, takes too long to run
    process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
    process.PFTau=cms.Sequence()#replace with an empty sequence

    #kill charged hadron subtraction for in-time PU mitigation
    process.pfNoPileUpIso.enable = False
    process.pfPileUpIso.Enable = False
    process.pfNoPileUp.enable = False
    process.pfPileUp.Enable = False

    #make it very hard to reconstruct conversions in this step (was bailing out in central events anyways)
    process.load("RecoTracker.ConversionSeedGenerators.PhotonConversionTrajectorySeedProducerFromSingleLeg_cfi")
    process.photonConvTrajSeedFromSingleLeg.RegionFactoryPSet.RegionPSet.ptMin = 999999.0
    process.photonConvTrajSeedFromSingleLeg.RegionFactoryPSet.RegionPSet.originRadius = 0
    process.photonConvTrajSeedFromSingleLeg.RegionFactoryPSet.RegionPSet.originHalfLength = 0

    #get rid of low pt tracker electrons
    process.load("RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi")
    process.trackerDrivenElectronSeeds.MinPt = 5.0

    return process

def customiseVertexing(process):
    #Primary Vtxing
    #change vertexing to use gap fitter w/ zsep of 1cm (more robust in central evts)
    process.load("RecoVertex.Configuration.RecoVertex_cff")
    process.unsortedOfflinePrimaryVertices.TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(3.0), #used to be 5
        minPt = cms.double(0.0),
        maxEta = cms.double(2.4),
        trackQuality = cms.string("any")
    )
    process.unsortedOfflinePrimaryVertices.TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)        ## 1 cm max separation between clusters
        )
    )

    #also hit the vtx made after the initial step
    process.load("RecoTracker.IterativeTracking.InitialStep_cff")
    process.firstStepPrimaryVerticesUnsorted.TkFilterParameters = cms.PSet(
        algorithm=cms.string('filter'),
        maxNormalizedChi2 = cms.double(10.0),
        minPixelLayersWithHits=cms.int32(2),
        minSiliconLayersWithHits = cms.int32(5),
        maxD0Significance = cms.double(3.0), #used to be 5
        minPt = cms.double(0.0),
        maxEta = cms.double(2.4),
        trackQuality = cms.string("any")
    )
    process.firstStepPrimaryVerticesUnsorted.TkClusParameters = cms.PSet(
        algorithm = cms.string("gap"),
        TkGapClusParameters = cms.PSet(
            zSeparation = cms.double(1.0)        ## 1 cm max separation between clusters
        )
    )
    

    #b-tagging vtxing
    #pull back inclusive vertex finder, should help w/ timing on TrackVertexArbitrator
    process.load("RecoVertex.AdaptiveVertexFinder.inclusiveVertexing_cff")
    process.inclusiveVertexFinder.minHits = 10
    process.inclusiveVertexFinder.minPt = 1.0
    process.inclusiveCandidateVertexFinderCvsL.minHits = 10
    process.inclusiveCandidateVertexFinderCvsL.minPt = 1.0


    return process

#don't try tracking under 0.3 GeV (unused for analysis anyways)
#higher threshhold for mixedtriplet/tobtec/pixelless steps to save on timing
def customiseTracking(process):

    from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cfi import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
    #initial step
    process.load("RecoTracker.IterativeTracking.InitialStep_cff")
    process.initialStepTrajectoryFilterBase.minPt=0.6  # ptmin of tracking region is 0.5
     
    #jet core
    process.load("RecoTracker.IterativeTracking.JetCoreRegionalStep_cff")
    process.jetCoreRegionalStepTrajectoryFilter.minPt = 5.0 #ptmin of tracking region is 10

    #high pt triplet
    process.load("RecoTracker.IterativeTracking.HighPtTripletStep_cff")
    process.highPtTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 0.2,
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.6,
        useFoundVertices = True,
        originRadius = 0.02
    ))  
    process.highPtTripletStepTrajectoryFilterBase.minPt=0.7 # ptmin of tracking region is 0.6

    #detached triplet
    process.load("RecoTracker.IterativeTracking.DetachedTripletStep_cff")
    process.detachedTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 3,
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.8,
        useFoundVertices = True,
        originRadius = 1.5
    ))
    process.detachedTripletStepTrajectoryFilterBase.minPt = 0.9 

    #detached quad
    process.load("RecoTracker.IterativeTracking.DetachedQuadStep_cff")
    process.detachedQuadStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 3.75,
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.8,
        useFoundVertices = True,
        originRadius = 1.5
    ))
    process.detachedQuadStepTrajectoryFilterBase.minPt = 0.9

    #low pt quad step
    process.load("RecoTracker.IterativeTracking.LowPtQuadStep_cff")
    process.lowPtQuadStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 0.5,
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.25,
        useFoundVertices = True,
        originRadius = 0.02
    ))  
    process.lowPtQuadStepTrajectoryFilterBase.minPt=0.3  

    #low pt triplet step
    process.load("RecoTracker.IterativeTracking.LowPtTripletStep_cff")
    process.lowPtTripletStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 0.2,
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.25,
        useFoundVertices = True,
        originRadius = 0.02
    ))
    process.lowPtTripletStepStandardTrajectoryFilter.minPt = 0.3
   
    #mixed triplet step
    process.load("RecoTracker.IterativeTracking.MixedTripletStep_cff")
    process.mixedTripletStepTrackingRegionsA = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,#this means use fixedErrorBelow
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 3.75,#a fourth the size of the pp version
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 0.4,
        useFoundVertices = True,
        originRadius = 1.5
    ))
    process.mixedTripletStepTrackingRegionsB = process.mixedTripletStepTrackingRegionsA.clone(RegionPSet = dict(ptMin=0.6, fixedError=2.5))
    process.mixedTripletStepTrajectoryFilter.minPt = 0.4
    process.mixedTripletStepPropagator.ptMin = 0.4
    process.mixedTripletStepPropagatorOpposite.ptMin = 0.4

    #pixelless step
    process.load("RecoTracker.IterativeTracking.PixelLessStep_cff")
    process.pixelLessStepTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,#this means use fixedErrorBelow
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 3.0,#a fourth the size of the pp version
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 2.0,
        useFoundVertices = True,
        originRadius = 1.0
    ))
    process.pixelLessStepTrajectoryFilter.minPt = 2.0

    #tobtec step
    process.load("RecoTracker.IterativeTracking.TobTecStep_cff")
    process.tobTecStepTrackingRegionsTripl = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,#this means use fixedErrorBelow
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 5.0,#a fourth the size of the pp version
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 2.0,
        useFoundVertices = True,
        originRadius = 3.5
))
    process.tobTecStepTrackingRegionsPair = _globalTrackingRegionWithVertices.clone(RegionPSet=dict(
        precise = True,
        useMultipleScattering = False,
        useFakeVertices       = False,
        beamSpot = "offlineBeamSpot",
        useFixedError = True,#this means use fixedErrorBelow
        nSigmaZ = 4.0,
        sigmaZVertex = 4.0,
        fixedError = 7.5,#a fourth the size of the pp version
        VertexCollection = "firstStepPrimaryVertices",
        ptMin = 2.0,
        useFoundVertices = True,
        originRadius = 6.0
))
    process.tobTecStepTrajectoryFilter.minPt = 2.0

    return process

#copied almost exactly from RecoTracker/Configuration/python/customiseClusterCheckForHighPileup.py
#some threshholds have been retuned however
#also need to reenable doClusterCheck because it is turned off by default in phase 1
def customiseClusterCheck(process):
    _maxPixel = 100000

    #this cut was tuned with XeXe MC: The old cut used for peripheral PbPb in 2015 is listed in comment below    
    #_cut = "strip < 1000000 && pixel < 100000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/7.)"
    _cut = "strip < 1000000 && pixel < 100000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/2.)"

    _maxElement = 1000000

    for module in process._Process__producers.values():
        cppType = module._TypedParameterizable__type

        # cluster multiplicity check
        if cppType == "ClusterCheckerEDProducer":
            module.doClusterCheck = True #added this line (not in pp config)
            module.MaxNumberOfPixelClusters = _maxPixel
            module.cut = _cut
        if hasattr(module, "ClusterCheckPSet"):
            module.ClusterCheckPSet.MaxNumberOfPixelClusters = _maxPixel
            module.ClusterCheckPSet.doClusterCheck = True #added this line (not in pp config)
            # PhotonConversionTrajectorySeedProducerFromQuadruplets does not have "cut"...
            if hasattr(module.ClusterCheckPSet, "cut"):
                module.ClusterCheckPSet.cut = _cut


        if cppType in ["PixelTripletLargeTipEDProducer", "MultiHitFromChi2EDProducer"]:
            module.maxElement = _maxElement
        if hasattr(module, "OrderedHitsFactoryPSet") and hasattr(module.OrderedHitsFactoryPSet, "GeneratorPSet"):
            #next line is in pp config but we comment it to be safe by changing more modules...
            #if module.OrderedHitsFactoryPSet.GeneratorPSet.ComponentName.value() in ["PixelTripletLargeTipGenerator", "MultiHitGeneratorFromChi2"]:
            module.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = _maxElement
    
    return process

def customisePPwithHI(process):

    process=storeCaloTowersAOD(process)
    process=addHIIsolationProducer(process)
    process=addIslandPhotons(process)
    process=customisePF(process)
    process=customiseVertexing(process)
    process=customiseTracking(process)
    process=customiseClusterCheck(process)

    return process
