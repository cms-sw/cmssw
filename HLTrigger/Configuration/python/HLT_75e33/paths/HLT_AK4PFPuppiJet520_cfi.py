import FWCore.ParameterSet.Config as cms

from ..modules.hltSingleAK4PFPuppiJet520_cfi import *
from ..modules.hltL1SeedsForPuppiJetFilter_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTAK4PFPuppiJetsReconstruction_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

HLT_AK4PFPuppiJet520 = cms.Path(
    HLTBeginSequence
    + hltL1SeedsForPuppiJetFilter
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTAK4PFPuppiJetsReconstruction
    + hltSingleAK4PFPuppiJet520
    + HLTEndSequence
)
