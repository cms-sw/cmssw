import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJets

from RecoHI.HiJetAlgos.HiGenJets_cff import *


ak4GenJetsNoNu = ak4GenJets.clone( src = "genParticlesForJetsNoNu" )
ak8GenJetsNoNu = ak8GenJets.clone( src = "genParticlesForJetsNoNu" )

ak4GenJetsNoMuNoNu = ak4GenJets.clone( src = "genParticlesForJetsNoMuNoNu" )
ak8GenJetsNoMuNoNu = ak8GenJets.clone( src = "genParticlesForJetsNoMuNoNu" )

recoGenJetsTask = cms.Task(ak4GenJets,
                           ak8GenJets,
                           ak4GenJetsNoNu,
                           ak8GenJetsNoNu
                           )
recoGenJets  = cms.Sequence(recoGenJetsTask)

recoAllGenJetsTask=cms.Task(ak4GenJets,
                            ak8GenJets)
recoAllGenJets=cms.Sequence(recoAllGenJetsTask)

recoAllGenJetsNoNuTask=cms.Task(ak4GenJetsNoNu,
                                ak8GenJetsNoNu)
recoAllGenJetsNoNu=cms.Sequence(recoAllGenJetsNoNuTask)

recoAllGenJetsNoMuNoNuTask=cms.Task(ak4GenJetsNoMuNoNu,
                                    ak8GenJetsNoMuNoNu)
recoAllGenJetsNoMuNoNu=cms.Sequence(recoAllGenJetsNoMuNoNuTask)
