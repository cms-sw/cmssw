import FWCore.ParameterSet.Config as cms

tauMVADiscriminatorHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
)

tauMVADiscriminatorInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
)

