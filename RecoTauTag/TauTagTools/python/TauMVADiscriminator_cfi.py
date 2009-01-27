import FWCore.ParameterSet.Config as cms

#Define the mapping of Decay mode IDs onto the names of trained MVA files
#Note that one category can apply to multiple decay modes, a decay mode can not have multiple categories

# Get MVA configuration defintions (edit MVAs here)
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *

# Define vectors of the DecayMode->MVA implementaions associations you want to use
# Note: any decay mode not associated to an MVA will be marked as failing the MVA!
DecayModeBasedTauID = cms.VPSet(
      OneProngNoPiZero,
      OneProngOnePiZero,
      OneProngTwoPiZero,
      ThreeProngNoPiZero,
      ThreeProngOnePiZero
      )

SingleNetBasedTauID = cms.VPSet(
      SingleNet
)

pfRecoTauDiscriminationByMVAHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutHighEfficiency"),
    computers         = DecayModeBasedTauID,
    prefailValue      = cms.double(-2.0)    #specifies the value to set if one of the preDiscriminats fails (should match the minimum MVA output)
)

pfRecoTauDiscriminationByMVAInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingTrackPtCutInsideOut"),
    computers         = DecayModeBasedTauID,
    prefailValue      = cms.double(-2.0)    #specifies the value to set if one of the preDiscriminats fails (should match the minimum MVA output)
)

