import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *

kt4PFJets = cms.EDProducer("KtJetProducer",
                           PFJetParameters,
                           FastjetNoPU,
                           KtJetParameters,
                           JetPtMin = cms.double( 1.0 ),
                           alias = cms.untracked.string( 'KT4PFJet' ),
                           FJ_ktRParam = cms.double( 0.4 )
                           )

kt6PFJets = cms.EDProducer("KtJetProducer",
                           PFJetParameters,
                           FastjetNoPU,
                           KtJetParameters,
                           JetPtMin = cms.double( 1.0 ),
                           alias = cms.untracked.string( 'KT6PFJet' ),
                           FJ_ktRParam = cms.double( 0.6 )
                           )

iterativeCone5PFJets = cms.EDProducer("IterativeConeJetProducer",
                                      PFJetParameters,
                                      IconeJetParameters,
                                      alias = cms.untracked.string( 'IC5PFJet' ),
                                      coneRadius = cms.double( 0.5 )
                                      )

sisCone5PFJets = cms.EDProducer("SISConeJetProducer",
                                PFJetParameters,
                                SISConeJetParameters,
                                FastjetNoPU,
                                alias = cms.untracked.string( 'SISC5PFJet' ),
                                coneRadius = cms.double( 0.5 )
                                )

sisCone7PFJets = cms.EDProducer("SISConeJetProducer",
                                PFJetParameters,
                                SISConeJetParameters,
                                FastjetNoPU,
                                alias = cms.untracked.string( 'SISC7PFJet' ),
                                coneRadius = cms.double( 0.7 )
                                )

recoPFJets = cms.Sequence(kt4PFJets+kt6PFJets+iterativeCone5PFJets+sisCone5PFJets+sisCone7PFJets)

