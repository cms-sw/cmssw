import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..sequences.HLTHgcalLocalRecoSequence_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgamma_cfi import *
from ..sequences.HLTJMESequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *
from ..sequences.HLTLocalrecoSequence_cfi import *
from ..sequences.HLTRawToDigiSequence_cfi import *

MC_JME = cms.Path(
    HLTBeginSequence
    + HLTRawToDigiSequence
    + HLTHgcalLocalRecoSequence
    + HLTLocalrecoSequence
    + HLTTrackingSequence
    + HLTMuonsSequence
    + HLTParticleFlowSequence
    + HLTHgcalTiclPFClusteringForEgamma
    + HLTJMESequence
    + hltPFPuppiHT
    + hltPFPuppiMHT
)
