import FWCore.ParameterSet.Config as cms

#
# $Id: GenJetParticles.cff,v 1.9 2008/03/25 14:42:03 oehler Exp $
#
# ShR 27 Mar 07: move modules producing candidates for Jets from RecoGenJets.cff
# 
#
genParticlesForJets = cms.EDFilter("InputGenJetsParticleSelector",
    src = cms.InputTag("genParticles"),
    ignoreParticleIDs = cms.vuint32(1000022, 2000012, 2000014, 2000016, 1000039, 5000039, 4000012, 9900012, 9900014, 9900016, 39),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(True),
    excludeFromResonancePids = cms.vuint32(12, 13, 14, 16),
    tausAsJets = cms.bool(False)
)

genJetParticles = cms.Sequence(genParticlesForJets)

