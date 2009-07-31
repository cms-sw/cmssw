import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.sc5GenJets_cfi import sisCone5GenJets
from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
from RecoJets.JetProducers.ak5GenJets_cfi import ak5GenJets
from RecoJets.JetProducers.gk5GenJets_cfi import gk5GenJets
from RecoJets.JetProducers.kt4GenJets_cfi import kt4GenJets
from RecoJets.JetProducers.ca4GenJets_cfi import ca4GenJets


sisCone7GenJets = sisCone5GenJets.clone( rParam = 0.7 )
ak7GenJets      = ak5GenJets.clone( rParam = 0.7 )
gk7GenJets      = gk5GenJets.clone( rParam = 0.7 )
kt6GenJets      = kt4GenJets.clone( rParam = 0.6 )
ca6GenJets      = ca4GenJets.clone( rParam = 0.6 )


sisCone5GenJetsNoNu = sisCone5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
sisCone7GenJetsNoNu = sisCone7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
kt4GenJetsNoNu = kt4GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
kt6GenJetsNoNu = kt6GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
iterativeCone5GenJetsNoNu = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
ak5GenJetsNoNu = ak5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
ak7GenJetsNoNu = ak7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
gk5GenJetsNoNu = gk5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
gk7GenJetsNoNu = gk7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
ca4GenJetsNoNu = ca4GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )
ca6GenJetsNoNu = ca6GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNu") )

sisCone5GenJetsNoNuBSM = sisCone5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
sisCone7GenJetsNoNuBSM = sisCone7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
kt4GenJetsNoNuBSM = kt4GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
kt6GenJetsNoNuBSM = kt6GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
iterativeCone5GenJetsNoNuBSM = iterativeCone5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
ak5GenJetsNoNuBSM = ak5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
ak7GenJetsNoNuBSM = ak7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
gk5GenJetsNoNuBSM = gk5GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
gk7GenJetsNoNuBSM = gk7GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
ca4GenJetsNoNuBSM = ca4GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )
ca6GenJetsNoNuBSM = ca6GenJets.clone( src = cms.InputTag("genParticlesAllStableNoNuBSM") )

recoGenJets   =cms.Sequence(sisCone5GenJets+sisCone7GenJets+
                            kt4GenJets+kt6GenJets+
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

recoAllGenJetsNoNuBSM=cms.Sequence(sisCone5GenJetsNoNuBSM+sisCone7GenJetsNoNuBSM+
                                kt4GenJetsNoNuBSM+kt6GenJetsNoNuBSM+
                                iterativeCone5GenJetsNoNuBSM+
                                ak5GenJetsNoNuBSM+ak7GenJetsNoNuBSM+
                                gk5GenJetsNoNuBSM+gk7GenJetsNoNuBSM+
                                ca4GenJetsNoNuBSM+ca6GenJetsNoNuBSM)
