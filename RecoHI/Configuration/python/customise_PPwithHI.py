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

    process.initialStepSeedsPreSplitting.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.initialStepSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.lowPtTripletStepSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.globalSeedsFromTriplets.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.detachedTripletStepSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.mixedTripletStepSeedsA.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.mixedTripletStepSeedsB.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.globalMixedSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.pixelLessStepSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.globalPixelLessSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.globalPixelSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.pixelPairStepSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.globalSeedsFromPairsWithVertices.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.tobTecStepSeedsPair.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.tobTecStepSeedsTripl.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.pixelPairElectronSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.regionalCosmicTrackerSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.stripPairElectronSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.photonConvTrajSeedFromSingleLeg.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.photonConvTrajSeedFromQuadruplets.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    process.tripletElectronSeeds.ClusterCheckPSet.cut = cms.string("strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)")
    
    process.initialStepSeedsPreSplitting.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.detachedTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(1000000)
    process.tobTecStepSeedsPair.OrderedHitsFactoryPSet.maxElement = cms.uint32(1000000)
    process.pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement = cms.uint32(1000000)
    process.jetCoreRegionalStepSeeds.OrderedHitsFactoryPSet.maxElement = cms.uint32(1000000)

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

