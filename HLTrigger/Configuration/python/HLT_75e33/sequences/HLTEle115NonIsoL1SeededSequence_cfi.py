import FWCore.ParameterSet.Config as cms

from ..modules.hltEG115EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForSingleEleNonIsolatedFilter_cfi import *
from ..modules.hltEle115NonIsoClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoHEL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle115NonIsoPMS2L1SeededFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..tasks.HLTEle115NonIsoL1SeededTask_cfi import *

HLTEle115NonIsoL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForSingleEleNonIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG115EtL1SeededFilter +
    hltEle115NonIsoClusterShapeL1SeededFilter +
    hltEle115NonIsoClusterShapeSigmavvL1SeededFilter +
    hltEle115NonIsoClusterShapeSigmawwL1SeededFilter +
    hltEle115NonIsoHgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltEle115NonIsoHEL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltEle115NonIsoPixelMatchL1SeededFilter +
    hltEle115NonIsoPMS2L1SeededFilter +
    HLTGsfElectronL1SeededSequence +
    hltEle115NonIsoGsfDetaL1SeededFilter +
    hltEle115NonIsoGsfDphiL1SeededFilter,
    HLTEle115NonIsoL1SeededTask
)
