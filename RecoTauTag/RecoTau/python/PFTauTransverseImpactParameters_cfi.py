import FWCore.ParameterSet.Config as cms

PFTauTransverseImpactParameters = cms.EDProducer("PFTauTransverseImpactParameters",
    PFTauTag =  cms.InputTag("hpsPFTauProducer"),
    PFTauPVATag = cms.InputTag("PFTauPrimaryVertexProducer"),
    PFTauSVATag = cms.InputTag("PFTauSecondaryVertexProducer"),
    ##useFullCalculation = cms.bool(True)
    useFullCalculation = cms.bool(False) # CV: keep 'useFullCalculation' disabled in CMSSW_5_3_X branch because the tau ID MVA3 has been trained this way
)

