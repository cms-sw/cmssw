import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaIsolationAlgos.pfBlockBasedIsolation_cfi import *

#
# particle based isolatio 
#
particleBasedIsolation = cms.EDProducer("ParticleBasedIsoProducer",
    photonTmpProducer = cms.InputTag("gedPhotonsTmp"),
    photonProducer = cms.InputTag("gedPhotons"),
    electronTmpProducer = cms.InputTag("gedGsfElectronsTmp"),                                                                        
    electronProducer = cms.InputTag("gedGsfElectrons"),                                    
    pfEgammaCandidates = cms.InputTag("particleFlowEGamma"),
    pfCandidates = cms.InputTag("particleFlow"),
    valueMapPhoToEG = cms.string("valMapPFEgammaCandToPhoton"),             
    valueMapPhoPFblockIso = cms.string("gedPhotons"),
    valueMapEleToEG = cms.string(""),
    valueMapElePFblockIso = cms.string("gedGsfElectrons"),
    pfBlockBasedIsolationSetUp=cms.PSet(pfBlockBasedIsolation)                                 
)


