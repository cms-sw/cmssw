import FWCore.ParameterSet.Config as cms

# Exact copy of GenJetInputParticleSelector extended to configure heavy flavour hadrons that should be added to input particles
#
# $Id: GenJetParticles_cfi.py,v 1.1 2013/03/14 16:29:00 nbartosi Exp $
#
# ShR 27 Mar 07: move modules producing candidates for Jets from RecoGenJets.cff
# 
#
genParticlesForJetsPlusNoHadron = cms.EDProducer("InputGenJetsParticlePlusHadronSelector",
    src = cms.InputTag("genParticles"),
    ignoreParticleIDs = cms.vuint32(
         1000022,
         1000012, 1000014, 1000016,
         2000012, 2000014, 2000016,
         1000039, 5100039,
         4000012, 4000014, 4000016,
         9900012, 9900014, 9900016,
         39),
    partonicFinalState = cms.bool(False),
    excludeResonances = cms.bool(True),
    excludeFromResonancePids = cms.vuint32(11, 12, 13, 14, 16),
    tausAsJets = cms.bool(False),
    injectHadronFlavours = cms.vint32()
)
