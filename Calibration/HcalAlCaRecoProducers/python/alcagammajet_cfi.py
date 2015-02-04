import FWCore.ParameterSet.Config as cms

# producer for alcadijets (HCAL gamma-jet)
GammaJetProd = cms.EDProducer("AlCaGammaJetProducer",
                              PhoInput = cms.InputTag("photons"),
                              PFjetInput = cms.InputTag("ak5PFJets"),
                              HBHEInput = cms.InputTag("hbhereco"),
                              HFInput = cms.InputTag("hfreco"),
                              HOInput = cms.InputTag("horeco"),
                              METInput = cms.InputTag("pfMet"),
                              TriggerResults = cms.InputTag("TriggerResults::HLT"),
                              gsfeleInput = cms.InputTag("gedGsfElectrons"),
                              particleFlowInput = cms.InputTag("particleFlow"),
                              VertexInput = cms.InputTag("offlinePrimaryVertices"),
                              ConversionsInput = cms.InputTag("allConversions"),
                              rhoInput = cms.InputTag("fixedGridRhoFastjetAll"),
                               BeamSpotInput = cms.InputTag("offlineBeamSpot"),
                              PhoLoose = cms.InputTag("PhotonIDProd", "PhotonCutBasedIDLoose"),
                              PhoTight = cms.InputTag("PhotonIDProd", "PhotonCutBasedIDTight"),
                              MinPtJet = cms.double(10.0),
                              MinPtPhoton = cms.double(10.0)
                              )


