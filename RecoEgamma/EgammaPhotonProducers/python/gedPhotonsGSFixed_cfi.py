import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaTools.regressionModifier_cfi import *

gedPhotonsGSFixed = cms.EDProducer('GEDPhotonGSCrysFixer',
    photons = cms.InputTag('gedPhotons', '', cms.InputTag.skipCurrentProcess()),
    newCores = cms.InputTag('gedPhotonCoreGSFixed'),
    barrelEcalHits=cms.InputTag("ecalMultiAndGSGlobalRecHitEB"),
    primaryVertexProducer = cms.InputTag('offlinePrimaryVerticesWithBS'),
    # rest for regression
    minR9Barrel = cms.double(0.94),
    minR9Endcap = cms.double(0.95),
    endcapEcalHits = cms.InputTag('reducedEcalRecHitsEE'),
    superClusterEnergyCorrFunction =  cms.string("EcalClusterEnergyCorrection"),                  
    superClusterEnergyErrorFunction = cms.string("EcalClusterEnergyUncertainty"),
    superClusterCrackEnergyCorrFunction =  cms.string("EcalClusterCrackCorrection"),
    photonEcalEnergyCorrFunction = cms.string("EcalClusterEnergyCorrectionObjectSpecific"),
    regressionConfig = regressionModifier.clone(rhoCollection = cms.InputTag("fixedGridRhoFastjetAllTmp")),
    regressionWeightsFromDB = cms.bool(True),
    energyRegressionWeightsFileLocation = cms.string('that this is required is dumb'),
    energyRegressionWeightsDBLocation = cms.string('wgbrph')
)
