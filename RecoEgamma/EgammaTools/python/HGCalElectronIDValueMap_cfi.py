import FWCore.ParameterSet.Config as cms
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX_weights as dEdX

HGCalElectronIDValueMap = cms.EDProducer("HGCalElectronIDValueMapProducer",
    electrons = cms.InputTag("ecalDrivenGsfElectronsFromMultiCl"),
    pcaRadius = cms.double(3.),
    EERecHits = cms.InputTag('HGCalRecHit:HGCEERecHits'),
    FHRecHits = cms.InputTag('HGCalRecHit:HGCHEFRecHits'),
    BHRecHits = cms.InputTag('HGCalRecHit:HGCHEBRecHits'),
    dEdXWeights = dEdX,
    variables = cms.vstring([
        # Energies / pT
        "gsfTrackPt",
        "pOutPt",
        "scEt",
        "scEnergy",
        "ecOrigEt",
        "ecOrigEnergy",
        # energies calculated in an cylinder around the axis of the electron cluster
        "ecEt",
        "ecEnergy",
        "ecEnergyEE",
        "ecEnergyFH",
        "ecEnergyBH",
        # Track-based
        "fbrem",
        "gsfTrackHits",
        "gsfTrackChi2",
        "kfTrackHits",
        "kfTrackChi2",
        # Track-matching
        "dEtaTrackClust",
        "dPhiTrackClust",
        # Cluster shapes
        # PCA related
        "pcaEig1",
        "pcaEig2",
        "pcaEig3",
        "pcaSig1",
        "pcaSig2",
        "pcaSig3",
        # transverse shapes
        "sigmaUU",
        "sigmaVV",
        "sigmaEE",
        "sigmaPP",
        # long energy profile
        "nLayers",
        "firstLayer",
        "lastLayer",
        "e4oEtot",
        "layerEfrac10",
        "layerEfrac90",
        # depth
        "measuredDepth",
        "expectedDepth",
        "expectedSigma",
        "depthCompatibility",
        # Isolation (staggered rings)
        "caloIsoRing0",
        "caloIsoRing1",
        "caloIsoRing2",
        "caloIsoRing3",
        "caloIsoRing4",
    ])
)
