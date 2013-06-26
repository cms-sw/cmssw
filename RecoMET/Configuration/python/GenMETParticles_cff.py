import FWCore.ParameterSet.Config as cms

# File: GenMET.cff
# Author: R. Remington
# Date: 10/21/08
#
# Form Missing ET from Generator Information and store into event as a GenMET
# product.  Exclude calo invisible final state particles like neutrinos, muons
#
#
# F.R. Mar. 22, 2007 IMPORTANT: this configuration assumes that some
#                    GenParticle collections are made via GenJet's configuration
from PhysicsTools.HepMCCandAlgos.genParticleCandidatesFast_cfi import *
#from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *

genCandidatesForMET = cms.EDProducer(
    "InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(False),
    excludeFromResonancePids = cms.vuint32(),
    tausAsJets = cms.bool(False),
    
    ###These ID's will contribute to MET because they will be skipped in the negative vector sum Et calculation performed by the MET Algorithm  
    ignoreParticleIDs = cms.vuint32(
    1000022,
    1000012, 1000014, 1000016,
    2000012, 2000014, 2000016,
    1000039, 5100039,
    4000012, 4000014, 4000016,
    9900012, 9900014, 9900016,
    39, 12, 13, 14, 16
    )  
    )

genParticlesForMETAllVisible = cms.EDProducer(
    "InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(False),
    excludeFromResonancePids = cms.vuint32(),
    tausAsJets = cms.bool(False),
    
    ignoreParticleIDs = cms.vuint32(
    1000022,
    1000012, 1000014, 1000016,
    2000012, 2000014, 2000016,
    1000039, 5100039,
    4000012, 4000014, 4000016,
    9900012, 9900014, 9900016,
    39, 12, 14, 16
    )
    )                                        

genMETParticles = cms.Sequence(genCandidatesForMET+genParticlesForMETAllVisible)
