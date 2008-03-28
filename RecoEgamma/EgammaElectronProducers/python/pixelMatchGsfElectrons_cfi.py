import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits with gsf fit
#
pixelMatchGsfElectrons = cms.EDProducer("GsfElectronProducer",
    # nested parameter set for TransientInitialStateEstimator
    #FIXME!
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    AssocShapeBarrelLabel = cms.string('hybridSuperClusters'),
    # electron preselection parameters
    maxEOverPBarrel = cms.double(10000.0),
    SCLEndcapLabel = cms.string('electronPixelSeeds'),
    highPtPreselection = cms.bool(False),
    minEOverPEndcaps = cms.double(0.0),
    minEOverPBarrel = cms.double(0.0),
    applyEtaCorrection = cms.bool(False),
    hbheInstance = cms.string(''),
    hbheModule = cms.string('hbhereco'),
    AssocShapeBarrelProducer = cms.string('hybridShapeAssoc'),
    AssocShapeEndcapLabel = cms.string('islandBasicClusters'),
    highPtMin = cms.double(150.0),
    maxEOverPEndcaps = cms.double(10000.0),
    SCLBarrelProducer = cms.string('correctedHybridSuperClusters'),
    maxDeltaPhi = cms.double(0.1),
    TrackProducer = cms.string(''),
    AssocShapeEndcapProducer = cms.string('islandEndcapShapeAssoc'),
    TrackLabel = cms.string('pixelMatchGsfFit'),
    SCLBarrelLabel = cms.string('electronPixelSeeds'),
    maxDeltaEta = cms.double(0.02),
    ElectronType = cms.string(''),
    SCLEndcapProducer = cms.string('correctedEndcapSuperClustersWithPreshower')
)


