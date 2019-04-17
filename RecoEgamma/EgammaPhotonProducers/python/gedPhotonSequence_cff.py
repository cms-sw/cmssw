import FWCore.ParameterSet.Config as cms

#
# sequence to make photons from clusters in ECAL
#
# photon producer
from RecoEgamma.EgammaPhotonProducers.gedPhotonCore_cfi import *
from RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi import *

import RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi 

gedPhotonsTmp = RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi.gedPhotons.clone()
gedPhotonsTmp.photonProducer = cms.InputTag("gedPhotonCore")
gedPhotonsTmp.candidateP4type = cms.string("fromEcalEnergy")
del gedPhotonsTmp.regressionConfig
gedPhotonsTmp.outputPhotonCollection = cms.string("")
gedPhotonsTmp.reconstructionStep = cms.string("tmp")
gedPhotonSequenceTmp = cms.Sequence(gedPhotonCore+gedPhotonsTmp)


gedPhotons = RecoEgamma.EgammaPhotonProducers.gedPhotons_cfi.gedPhotons.clone()
gedPhotons.photonProducer = cms.InputTag("gedPhotonsTmp")
gedPhotons.outputPhotonCollection = cms.string("")
gedPhotons.reconstructionStep = cms.string("final")
gedPhotons.pfECALClusIsolation = cms.InputTag("photonEcalPFClusterIsolationProducer")
gedPhotons.pfHCALClusIsolation = cms.InputTag("photonHcalPFClusterIsolationProducer")
gedPhotons.pfIsolCfg = cms.PSet(
    chargedHadronIso = cms.InputTag("photonIDValueMaps","phoChargedIsolation"),
    neutralHadronIso = cms.InputTag("photonIDValueMaps","phoNeutralHadronIsolation"),
    photonIso = cms.InputTag("photonIDValueMaps","phoPhotonIsolation"),
    chargedHadronWorstVtxIso = cms.InputTag("photonIDValueMaps","phoWorstChargedIsolation"),
    chargedHadronWorstVtxGeomVetoIso = cms.InputTag("photonIDValueMaps","phoWorstChargedIsolationConeVeto"),
    chargedHadronPFPVIso = cms.InputTag("egmPhotonIsolationCITK:h+-DR030-"),
    )
    

gedPhotonSequence    = cms.Sequence(gedPhotons)



