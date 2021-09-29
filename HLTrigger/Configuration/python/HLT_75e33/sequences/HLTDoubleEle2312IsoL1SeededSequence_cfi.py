import FWCore.ParameterSet.Config as cms

from ..modules.hltDiEG12EtL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoClusterShapeL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoEcalIsoL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoHcalIsoL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoHEL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoHgcalHEL1SeededFilter_cfi import *
from ..modules.hltDiEG2312IsoHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoGsfDetaL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoGsfDphiL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoGsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoPixelMatchL1SeededFilter_cfi import *
from ..modules.hltDiEle2312IsoPMS2L1SeededFilter_cfi import *
from ..modules.hltEG23EtL1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEGL1SeedsForDoubleEleIsolatedFilter_cfi import *
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgamma_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1Seeded_cfi import *
from ..sequences.HLTPFHcalClusteringForEgamma_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *
from ..tasks.HLTDoubleEle2312IsoL1SeededTask_cfi import *

HLTDoubleEle2312IsoL1SeededSequence = cms.Sequence(
    HLTL1Sequence +
    hltEGL1SeedsForDoubleEleIsolatedFilter +
    HLTDoFullUnpackingEgammaEcalL1SeededSequence +
    HLTPFClusteringForEgammaL1Seeded +
    HLTHgcalTiclPFClusteringForEgammaL1Seeded +
    hltEgammaCandidatesWrapperL1Seeded +
    hltEG23EtL1SeededFilter +
    hltDiEG12EtL1SeededFilter +
    hltDiEG2312IsoClusterShapeL1SeededFilter +
    hltDiEG2312IsoClusterShapeSigmavvL1SeededFilter +
    hltDiEG2312IsoClusterShapeSigmawwL1SeededFilter +
    hltDiEG2312IsoHgcalHEL1SeededFilter +
    HLTDoLocalHcalSequence +
    HLTFastJetForEgamma +
    hltDiEG2312IsoHEL1SeededFilter +
    hltDiEG2312IsoEcalIsoL1SeededFilter +
    hltDiEG2312IsoHgcalIsoL1SeededFilter +
    HLTPFHcalClusteringForEgamma +
    hltDiEG2312IsoHcalIsoL1SeededFilter +
    HLTElePixelMatchL1SeededSequence +
    hltDiEle2312IsoPixelMatchL1SeededFilter +
    hltDiEle2312IsoPMS2L1SeededFilter +
    HLTGsfElectronL1SeededSequence +
    hltDiEle2312IsoGsfOneOEMinusOneOPL1SeededFilter +
    hltDiEle2312IsoGsfDetaL1SeededFilter +
    hltDiEle2312IsoGsfDphiL1SeededFilter +
    hltDiEle2312IsoBestGsfNLayerITL1SeededFilter +
    hltDiEle2312IsoBestGsfChi2L1SeededFilter +
    hltDiEle2312IsoGsfTrackIsoFromL1TracksL1SeededFilter +
    HLTTrackingV61Sequence +
    hltDiEle2312IsoGsfTrackIsoL1SeededFilter,
    HLTDoubleEle2312IsoL1SeededTask
)
