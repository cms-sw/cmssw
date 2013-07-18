import FWCore.ParameterSet.Config as cms

PFTauTransverseImpactParameters = cms.EDProducer("PFTauTransverseImpactParameters",
                                                 PFTauTag =  cms.InputTag("hpsPFTauProducer"),
                                                 PFTauPVATag = cms.InputTag("PFTauPrimaryVertexProducer"),
                                                 PFTauSVATag = cms.InputTag("PFTauSecondaryVertexProducer"),
                                                 useFullCalculation = cms.bool(False)
                                                 )

