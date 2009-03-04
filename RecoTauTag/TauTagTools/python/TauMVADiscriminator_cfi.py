import FWCore.ParameterSet.Config as cms

#Define the mapping of Decay mode IDs onto the names of trained MVA files
#Note that one category can apply to multiple decay modes, a decay mode can not have multiple categories

# Get MVA configuration defintions (edit MVAs here)
from RecoTauTag.TauTagTools.TauMVAConfigurations_cfi import *


pfRecoTauDiscriminationByMVAHighEfficiency = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeHighEfficiency"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingPionPtCutHighEfficiency"),
    computers         = TaNC,
    prefailValue      = cms.double(-2.0),    #specifies the value to set if one of the preDiscriminats fails (should be lower than minimum MVA output, -1)
    MakeBinaryDecision = cms.bool(False)     #If this is enabled, the discriminator will test whether the output satisifies the cut defined in the neural net configuration,
                                             # and sets the output as 0. or 1. respecitively.
                                             # for example, if TaNC.OneProngOnePiZero.cut = 0.7 and this is enabled,
                                             # any one prongs with NN output < 0.7 will fail with discriminator value 0
)

pfRecoTauDiscriminationByMVAInsideOut = cms.EDProducer("TauMVADiscriminator",
    pfTauDecayModeSrc = cms.InputTag("pfTauDecayModeInsideOut"),
    preDiscriminants  = cms.VInputTag("pfRecoTauDiscriminationByLeadingPionCutInsideOut"),
    computers         = TaNC,
    prefailValue      = cms.double(-2.0),    
    MakeBinaryDecision = cms.bool(False)
)

