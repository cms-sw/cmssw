import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits with gsf fit
#
pixelMatchGsfElectrons = cms.EDProducer("GsfElectronProducer",
    #  InputTag endcapSuperClusters = correctedEndcapSuperClustersWithPreshower:electronPixelSeeds
    endcapSuperClusters = cms.InputTag("correctedFixedMatrixSuperClustersWithPreshower","electronPixelSeeds"),
    maxDeltaPhi = cms.double(0.1),
    minEOverPEndcaps = cms.double(0.0),
    maxEOverPEndcaps = cms.double(10000.0),
    minEOverPBarrel = cms.double(0.0),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters","electronPixelSeeds"),
    applyEtaCorrection = cms.bool(False),
    applyAmbResolution = cms.bool(True),
    tracks = cms.InputTag("pixelMatchGsfFit"),
    maxDeltaEta = cms.double(0.02),
    ElectronType = cms.string(''),
    # electron preselection parameters
    maxEOverPBarrel = cms.double(10000.0),
    highPtPreselection = cms.bool(False),
    hcalRecHits = cms.InputTag("hbhereco"),
    highPtMin = cms.double(150.0),
    # nested parameter set for TransientInitialStateEstimator
    #FIXME!
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    )
)


