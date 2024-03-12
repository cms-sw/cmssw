import FWCore.ParameterSet.Config as cms

from ..modules.hltPFPuppiHT_cfi import *
from ..modules.hltPFPuppiMHT_cfi import *
from ..sequences.hgcalLocalRecoSequence_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgamma_cfi import *
from ..sequences.HLTJMESequence_cfi import *
from ..sequences.HLTMuonsSequence_cfi import *
from ..sequences.HLTParticleFlowSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..sequences.localrecoSequence_cfi import *
from ..sequences.RawToDigiSequence_cfi import *

MC_JME = cms.Path(HLTBeginSequence+RawToDigiSequence+hgcalLocalRecoSequence+localrecoSequence+HLTTrackingV61Sequence+HLTMuonsSequence+HLTParticleFlowSequence+HLTHgcalTiclPFClusteringForEgamma+HLTJMESequence+hltPFPuppiHT+hltPFPuppiMHT)
