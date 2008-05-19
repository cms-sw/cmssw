import FWCore.ParameterSet.Config as cms

# produce electrons based on matched pixel hits with gsf fit
#
globalGsfElectrons = cms.EDProducer("GlobalGsfElectronProducer",
    endcapSuperClusters = cms.InputTag("correctedIslandEndcapSuperClusters"),
    maxDeltaPhi = cms.double(0.1),
    minEOverPEndcaps = cms.double(0.35),
    # nested parameter set for TransientInitialStateEstimator
    TransientInitialStateEstimatorParameters = cms.PSet(
        propagatorAlongTISE = cms.string('PropagatorWithMaterial'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialOpposite')
    ),
    minEOverPBarrel = cms.double(0.35),
    barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters"),
    applyEtaCorrection = cms.bool(False),
    applyAmbResolution = cms.bool(False),
    EtCut = cms.double(0.0), ## in Gev

    tracks = cms.InputTag("pixelMatchGsfFitForGlobalGsfElectrons"),
    maxDeltaEta = cms.double(0.02),
    # electron preselection parameters
    maxEOverPBarrel = cms.double(3.0),
    highPtPreselection = cms.bool(False),
    hcalRecHits = cms.InputTag("hbhereco"),
    maxHOverE = cms.double(0.2),
    highPtMin = cms.double(150.0),
    hOverEConeSize = cms.double(0.1),
    maxEOverPEndcaps = cms.double(5.0)
)


