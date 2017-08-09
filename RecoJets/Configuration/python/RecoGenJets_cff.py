import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets
from RecoJets.JetProducers.ak8GenJets_cfi import ak8GenJets

from RecoHI.HiJetAlgos.HiGenJets_cff import *


ak4GenJetsNoNu = ak4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )
ak8GenJetsNoNu = ak8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoNu") )

ak4GenJetsNoMuNoNu = ak4GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )
ak8GenJetsNoMuNoNu = ak8GenJets.clone( src = cms.InputTag("genParticlesForJetsNoMuNoNu") )

recoGenJets  = cms.Sequence(ak4GenJets+
                            ak8GenJets+
                            ak4GenJetsNoNu+
                            ak8GenJetsNoNu
			    )

recoAllGenJets=cms.Sequence(ak4GenJets+
                            ak8GenJets)

recoAllGenJetsNoNu=cms.Sequence(ak4GenJetsNoNu+
                                ak8GenJetsNoNu)

recoAllGenJetsNoMuNoNu=cms.Sequence(ak4GenJetsNoMuNoNu+
                                    ak8GenJetsNoMuNoNu)
