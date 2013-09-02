import FWCore.ParameterSet.Config as cms

'''
Various modules that will produce collections of MC objects
useful for TauAnalysis
'''

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

from PhysicsTools.JetMCAlgos.TauGenJets_cfi import tauGenJets

trueHadronicTaus = cms.EDFilter("TauGenJetDecayModeSelector",
     src = cms.InputTag("tauGenJets"),
     select = cms.vstring('oneProng0Pi0', 'oneProng1Pi0', 'oneProng2Pi0', 'oneProngOther',
                          'threeProng0Pi0', 'threeProng1Pi0', 'threeProngOther', 'rare'),
     filter = cms.bool(False)
)

trueMuonicTaus = cms.EDFilter("TauGenJetDecayModeSelector",
      src = cms.InputTag("tauGenJets"),
      select = cms.vstring('muon'),
      filter = cms.bool(False)
)

trueElectronicTaus = cms.EDFilter("TauGenJetDecayModeSelector",
      src = cms.InputTag("tauGenJets"),
      select = cms.vstring('electron'),
      filter = cms.bool(False)
)

trueElectrons = cms.EDProducer("GenParticlePruner",
      src = cms.InputTag("genParticles"),
      select = cms.vstring(
         'drop *',
         'keep++ pdgId = 11',
         'keep++ pdgId = -11'
      )
)

trueMuons = cms.EDProducer("GenParticlePruner",
      src = cms.InputTag("genParticles"),
      select = cms.vstring(
         'drop *',
         'keep++ pdgId = 13',
         'keep++ pdgId = -13'
      )
)

makeTruthCollections = cms.Sequence(genParticlesForJets*iterativeCone5GenJets*tauGenJets*trueHadronicTaus*trueMuonicTaus*trueElectronicTaus*
      trueElectrons*trueMuons)
      
