import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5GenJets_cfi import sisCone5GenJets
from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
from RecoJets.JetProducers.gk5GenJets_cfi import gk5GenJets
from RecoJets.JetProducers.kt4GenJets_cfi import kt4GenJets
from RecoJets.JetProducers.ca4GenJets_cfi import ca4GenJets

from RecoHI.HiJetAlgos.HiGenJets_cff import *


sisCone8GenJets = sisCone5GenJets.clone( rParam = 0.8 )
ak8GenJets      = ak5GenJets.clone( rParam = 0.8 )
gk8GenJets      = gk5GenJets.clone( rParam = 0.8 )
kt6GenJets      = kt4GenJets.clone( rParam = 0.6 )
ca6GenJets      = ca4GenJets.clone( rParam = 0.6 )


sisCone5GenJetsNoNu = sisCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
sisCone8GenJetsNoNu = sisCone8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
kt4GenJetsNoNu = kt4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
kt6GenJetsNoNu = kt6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
iterativeCone5GenJetsNoNu = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ak5GenJetsNoNu = ak5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ak8GenJetsNoNu = ak8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
gk5GenJetsNoNu = gk5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
gk8GenJetsNoNu = gk8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ca4GenJetsNoNu = ca4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ca6GenJetsNoNu = ca6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )

sisCone5GenJetsNoMuNoNu = sisCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
sisCone8GenJetsNoMuNoNu = sisCone8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
kt4GenJetsNoMuNoNu = kt4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
kt6GenJetsNoMuNoNu = kt6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
iterativeCone5GenJetsNoMuNoNu = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ak5GenJetsNoMuNoNu = ak5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ak8GenJetsNoMuNoNu = ak8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
gk5GenJetsNoMuNoNu = gk5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
gk8GenJetsNoMuNoNu = gk8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ca4GenJetsNoMuNoNu = ca4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ca6GenJetsNoMuNoNu = ca6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )

recoGenJets   =cms.Sequence(kt4GenJets+kt6GenJets+iterativeCone5GenJets+
                            ak5GenJets+ak8GenJets)

recoAllGenJets=cms.Sequence(kt4GenJets+kt6GenJets+
                            ak5GenJets+ak8GenJets)

recoAllGenJetsNoNu=cms.Sequence(kt4GenJetsNoNu+kt6GenJetsNoNu+
                                ak5GenJetsNoNu+ak8GenJetsNoNu)

recoAllGenJetsNoMuNoNu=cms.Sequence(kt4GenJetsNoMuNoNu+kt6GenJetsNoMuNoNu+
                                    ak5GenJetsNoMuNoNu+ak8GenJetsNoMuNoNu)
