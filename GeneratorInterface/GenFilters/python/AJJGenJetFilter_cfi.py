import FWCore.ParameterSet.Config as cms

ajjGenJetFilterPhotonInBarrelMjj300 = cms.EDFilter("AJJGenJetFilter",
                                                   GenJetCollection = cms.InputTag('ak4GenJetsNoNu'),
                                                   genParticles = cms.InputTag("genParticles"),
                                                   
                                                   #following cuts are applied to select jets
                                                   minPt = cms.untracked.double( 40 ),
                                                   minEta = cms.untracked.double( -4.5 ),
                                                   maxEta = cms.untracked.double( 4.5 ),
                                                   deltaRJetLep = cms.untracked.double( 0.3 ),

                                                   #following cuts are applied on the first two leading jets
                                                   minDeltaEta = cms.untracked.double( 3.0 ),
                                                   maxDeltaEta = cms.untracked.double( 999.0 ),
                                                   MinInvMass = cms.untracked.double( 300 ),
                                                   
                                                   #the cut on the photon eta
                                                   maxPhotonEta = cms.untracked.double( 1.48 ),
                                                   minPhotonPt  = cms.untracked.double( 50 ),
                                                   maxPhotonPt  = cms.untracked.double( 10000 )
)

ajjGenJetFilterPhoton = cms.EDFilter("AJJGenJetFilter",
                                     GenJetCollection = cms.InputTag('ak4GenJetsNoNu'),
                                     genParticles = cms.InputTag("genParticles"),
                                                   
                                     #following cuts are applied to select jets
                                     #if minPt is negative, no criteri on jets (including njets and delta_eta and invmasss) is applied
                                     minPt = cms.untracked.double( -1 ),
                                     minEta = cms.untracked.double( -4.5 ),
                                     maxEta = cms.untracked.double( 4.5 ),
                                     deltaRJetLep = cms.untracked.double( 0.3 ),

                                     #following cuts are applied on the first two leading jets
                                     minDeltaEta = cms.untracked.double( 3.0 ),
                                     maxDeltaEta = cms.untracked.double( 999.0 ),
                                     MinInvMass = cms.untracked.double( 300 ),
                                     
                                     #the cut on the photon eta
                                     maxPhotonEta = cms.untracked.double( 1.48 ),
                                     minPhotonPt  = cms.untracked.double( 50 ),
                                     maxPhotonPt  = cms.untracked.double( 10000 )
)
