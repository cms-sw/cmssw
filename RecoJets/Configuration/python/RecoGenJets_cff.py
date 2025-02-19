import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5GenJets_cfi import sisCone5GenJets
from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
from RecoJets.JetProducers.gk5GenJets_cfi import gk5GenJets
from RecoJets.JetProducers.kt4GenJets_cfi import kt4GenJets
from RecoJets.JetProducers.ca4GenJets_cfi import ca4GenJets

from RecoHI.HiJetAlgos.HiGenJets_cff import *


sisCone7GenJets = sisCone5GenJets.clone( rParam = 0.7 )
ak7GenJets      = ak5GenJets.clone( rParam = 0.7 )
gk7GenJets      = gk5GenJets.clone( rParam = 0.7 )
kt6GenJets      = kt4GenJets.clone( rParam = 0.6 )
ca6GenJets      = ca4GenJets.clone( rParam = 0.6 )


sisCone5GenJetsNoNu = sisCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
sisCone7GenJetsNoNu = sisCone7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
kt4GenJetsNoNu = kt4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
kt6GenJetsNoNu = kt6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
iterativeCone5GenJetsNoNu = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ak5GenJetsNoNu = ak5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ak7GenJetsNoNu = ak7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
gk5GenJetsNoNu = gk5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
gk7GenJetsNoNu = gk7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ca4GenJetsNoNu = ca4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ca6GenJetsNoNu = ca6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )

sisCone5GenJetsNoMuNoNu = sisCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
sisCone7GenJetsNoMuNoNu = sisCone7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
kt4GenJetsNoMuNoNu = kt4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
kt6GenJetsNoMuNoNu = kt6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
iterativeCone5GenJetsNoMuNoNu = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ak5GenJetsNoMuNoNu = ak5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ak7GenJetsNoMuNoNu = ak7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
gk5GenJetsNoMuNoNu = gk5GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
gk7GenJetsNoMuNoNu = gk7GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ca4GenJetsNoMuNoNu = ca4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ca6GenJetsNoMuNoNu = ca6GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )

recoGenJets   =cms.Sequence(kt4GenJets+kt6GenJets+
                            iterativeCone5GenJets+
                            ak5GenJets+ak7GenJets)

recoAllGenJets=cms.Sequence(sisCone5GenJets+sisCone7GenJets+
                            kt4GenJets+kt6GenJets+
                            iterativeCone5GenJets+
                            ak5GenJets+ak7GenJets+
                            gk5GenJets+gk7GenJets+
                            ca4GenJets+ca6GenJets)

recoAllGenJetsNoNu=cms.Sequence(sisCone5GenJetsNoNu+sisCone7GenJetsNoNu+
                                kt4GenJetsNoNu+kt6GenJetsNoNu+
                                iterativeCone5GenJetsNoNu+
                                ak5GenJetsNoNu+ak7GenJetsNoNu+
                                gk5GenJetsNoNu+gk7GenJetsNoNu+
                                ca4GenJetsNoNu+ca6GenJetsNoNu)

recoAllGenJetsNoMuNoNu=cms.Sequence(sisCone5GenJetsNoMuNoNu+sisCone7GenJetsNoMuNoNu+
                                    kt4GenJetsNoMuNoNu+kt6GenJetsNoMuNoNu+
                                    iterativeCone5GenJetsNoMuNoNu+
                                    ak5GenJetsNoMuNoNu+ak7GenJetsNoMuNoNu+
                                    gk5GenJetsNoMuNoNu+gk7GenJetsNoMuNoNu+
                                    ca4GenJetsNoMuNoNu+ca6GenJetsNoMuNoNu)
