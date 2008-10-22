import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByMVAHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency"),
    computerName      = cms.string('ZTauTauTraining')
)

pfRecoTauDiscriminationByMVAInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut"),
    computerName      = cms.string('ZTauTauTraining')
)

