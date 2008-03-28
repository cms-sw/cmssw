import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits with gsf fit
#
globalGsfElectrons = cms.EDProducer("GsfElectronProducer",
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    AssocShapeBarrelLabel = cms.string('hybridSuperClusters'),
    # electron preselection parameters
    maxEOverPBarrel = cms.double(3.0),
    SCLEndcapLabel = cms.string('electronPixelSeeds'),
    highPtPreselection = cms.bool(False),
    minEOverPEndcaps = cms.double(0.35),
    minEOverPBarrel = cms.double(0.35),
    applyEtaCorrection = cms.bool(True),
    hbheInstance = cms.string(''),
    hOverEConeSize = cms.double(0.1),
    hbheModule = cms.string('hbhereco'),
    AssocShapeBarrelProducer = cms.string('hybridShapeAssoc'),
    EtCut = cms.double(0.0), ## in Gev

    maxHOverE = cms.double(0.2),
    highPtMin = cms.double(150.0),
    maxEOverPEndcaps = cms.double(5.0),
    SCLBarrelProducer = cms.string('correctedHybridSuperClusters'),
    maxDeltaPhi = cms.double(0.1),
    TrackProducer = cms.string(''),
    AssocShapeEndcapProducer = cms.string('islandEndcapShapeAssoc'),
    TrackLabel = cms.string('pixelMatchGsfFitForGlobalGsfElectrons'),
    SCLBarrelLabel = cms.string('electronPixelSeeds'),
    maxDeltaEta = cms.double(0.02),
    AssocShapeEndcapLabel = cms.string('islandBasicClusters'),
    ElectronType = cms.string('GlobalGsfElectron'),
    SCLEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower')
)


