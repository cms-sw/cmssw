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

    process.initialStepSeedsPreSplitting.ClusterCheckPSet.cut = hiClusterCut
    process.initialStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.lowPtTripletStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.globalSeedsFromTriplets.ClusterCheckPSet.cut = hiClusterCut
    process.detachedTripletStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.mixedTripletStepSeedsA.ClusterCheckPSet.cut = hiClusterCut
    process.mixedTripletStepSeedsB.ClusterCheckPSet.cut = hiClusterCut
    process.globalMixedSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.pixelLessStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.globalPixelLessSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.globalPixelSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.pixelPairStepSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.globalSeedsFromPairsWithVertices.ClusterCheckPSet.cut = hiClusterCut
    process.tobTecStepSeedsPair.ClusterCheckPSet.cut = hiClusterCut
    process.tobTecStepSeedsTripl.ClusterCheckPSet.cut = hiClusterCut
    process.pixelPairElectronSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.regionalCosmicTrackerSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.stripPairElectronSeeds.ClusterCheckPSet.cut = hiClusterCut
    process.photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.cut = hiClusterCut
    process.photonConvTrajSeedFromQuadruplets.ClusterCheckPSet.cut = hiClusterCut
    process.tripletElectronSeeds.ClusterCheckPSet.cut = hiClusterCut

    maxElement = cms.uint32(1000000)
    
    process.initialStepSeedsPreSplitting.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = maxElement
    process.tobTecStepSeedsPair.OrderedHitsFactoryPSet.maxElement = maxElement
    process.pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement = maxElement
    process.jetCoreRegionalStepSeeds.OrderedHitsFactoryPSet.maxElement = maxElement

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

