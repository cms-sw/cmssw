import FWCore.ParameterSet.Config as cms

tauMVADiscriminatorHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
    computerName      = cms.string('ZTauTauTraining')
)

tauMVADiscriminatorInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
    computerName      = cms.string('ZTauTauTraining')
)

