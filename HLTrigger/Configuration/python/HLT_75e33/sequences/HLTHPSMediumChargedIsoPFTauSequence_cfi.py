import FWCore.ParameterSet.Config as cms

from ..modules.hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator_cfi import *
from ..modules.hltHpsPFTauMediumRelativeChargedIsolationDiscriminator_cfi import *
from ..modules.hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminator_cfi import *

HLTHPSMediumChargedIsoPFTauSequence = cms.Sequence( 
    hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator + 
    hltHpsPFTauMediumRelativeChargedIsolationDiscriminator + 
    hltHpsPFTauMediumAbsOrRelChargedIsolationDiscriminator 
)
