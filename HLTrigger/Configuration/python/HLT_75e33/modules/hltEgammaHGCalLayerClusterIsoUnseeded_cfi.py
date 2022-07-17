import FWCore.ParameterSet.Config as cms

hltEgammaHGCalLayerClusterIsoUnseeded = cms.EDProducer("EgammaHLTHGCalLayerClusterIsolationProducer",
    doRhoCorrection = cms.bool(False),
    drMax = cms.double(0.2),
    drVetoEM = cms.double(0.02),
    drVetoHad = cms.double(0.0),
    layerClusterProducer = cms.InputTag("hgcalLayerClusters"),
    minEnergyEM = cms.double(0.02),
    minEnergyHad = cms.double(0.07),
    recoEcalCandidateProducer = cms.InputTag("hltEgammaCandidatesUnseeded"),
    rhoMax = cms.double(99999999.0),
    rhoScale = cms.double(1.0),
    useEt = cms.bool(False)
)
