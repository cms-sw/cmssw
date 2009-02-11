import FWCore.ParameterSet.Config as cms

pfRecoTauDiscriminationByMVAHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency"),
    computerName      = cms.string('ZTauTauTraining'),
    prefailValue      = cms.double(-1.0)    #specifies the value to set if one of the preDiscriminats fails (should match the minimum MVA output)

)

pfRecoTauDiscriminationByMVAInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut"),
    computerName      = cms.string('ZTauTauTraining'),
    prefailValue      = cms.double(-1.0)    #specifies the value to set if one of the preDiscriminats fails (should match the minimum MVA output)
)

