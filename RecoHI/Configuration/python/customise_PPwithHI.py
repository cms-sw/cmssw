import FWCore.ParameterSet.Config as cms

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


    # modify cluster limits to run pp reconstruction on peripheral PbPb
def modifyClusterLimits(process):

    hiClusterCut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")

    if hasattr(process,'initialStepSeedsPreSplitting'): process.initialStepSeedsPreSplitting.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'initialStepSeeds'): process.initialStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'lowPtTripletStepSeeds'): process.lowPtTripletStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'globalSeedsFromTriplets'): process.globalSeedsFromTriplets.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'detachedTripletStepSeeds'): process.detachedTripletStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'mixedTripletStepSeedsA'): process.mixedTripletStepSeedsA.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'mixedTripletStepSeedsB'): process.mixedTripletStepSeedsB.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'globalMixedSeeds'): process.globalMixedSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'pixelLessStepSeeds'): process.pixelLessStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'globalPixelLessSeeds'): process.globalPixelLessSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'globalPixelSeeds'): process.globalPixelSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'pixelPairStepSeeds'): process.pixelPairStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'globalSeedsFromPairsWithVertices'): process.globalSeedsFromPairsWithVertices.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'tobTecStepSeedsPair'): process.tobTecStepSeedsPair.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'tobTecStepSeedsTripl'): process.tobTecStepSeedsTripl.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'pixelPairElectronSeeds'): process.pixelPairElectronSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'regionalCosmicTrackerSeeds'): process.regionalCosmicTrackerSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'stripPairElectronSeeds'): process.stripPairElectronSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'photonConvTrajSeedFromSingleLeg'): process.photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'photonConvTrajSeedFromQuadruplets'): process.photonConvTrajSeedFromQuadruplets.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'tripletElectronSeeds'): process.tripletElectronSeeds.ClusterCheckPSet.cut = hiClusterCut
    if hasattr(process,'jetCoreRegionalStepSeeds'): process.jetCoreRegionalStepSeeds.ClusterCheckPSet.cut = hiClusterCut


    maxElement = cms.uint32(1000000)
    
    if hasattr(process,'initialStepSeedsPreSplitting'): process.initialStepSeedsPreSplitting.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'initialStepSeeds'): process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'lowPtTripletStepSeeds'): process.lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'mixedTripletStepSeedsA'): process.mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'mixedTripletStepSeedsB'): process.mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'detachedTripletStepSeeds'): process.detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'pixelLessStepSeeds'): process.pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'tobTecStepSeedsTripl'): process.tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    if hasattr(process,'tobTecStepSeedsPair'): process.tobTecStepSeedsPair.OrderedHitsFactoryPSet.maxElement = maxElement
    if hasattr(process,'pixelPairStepSeeds'): process.pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement = maxElement
    if hasattr(process,'jetCoreRegionalStepSeeds'): process.jetCoreRegionalStepSeeds.OrderedHitsFactoryPSet.maxElement = maxElement

    return process


# Add caloTowers to AOD event content
def storeCaloTowersAOD(process):

    process.load('Configuration.EventContent.EventContent_cff')
    
    # extend AOD content
    if hasattr(process,'AODoutput'):
        process.AODoutput.outputCommands.extend(['keep *_towerMaker_*_*'])

    if hasattr(process,'AODSIMoutput'):
        process.AODSIMoutput.outputCommands.extend(['keep *_towerMaker_*_*'])

    return process

def customisePPwithHI(process):

    process=addHIIsolationProducer(process)
    process=modifyClusterLimits(process)
    process=storeCaloTowersAOD(process)

    return process

