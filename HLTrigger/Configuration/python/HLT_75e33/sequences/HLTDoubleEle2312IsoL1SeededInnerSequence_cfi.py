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
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6L1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

HLTDoubleEle2312IsoL1SeededInnerSequence = cms.Sequence(hltEgammaCandidatesL1Seeded
    +hltEgammaClusterShapeL1Seeded
    +hltEgammaHGCALIDVarsL1Seeded
    +hltEgammaHoverEL1Seeded
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG23EtL1SeededFilter
    +hltDiEG12EtL1SeededFilter
    +hltDiEG2312IsoClusterShapeL1SeededFilter
    +hltDiEG2312IsoClusterShapeSigmavvL1SeededFilter
    +hltDiEG2312IsoClusterShapeSigmawwL1SeededFilter
    +hltDiEG2312IsoHgcalHEL1SeededFilter
    +hltDiEG2312IsoHEL1SeededFilter
    +hltDiEG2312IsoEcalIsoL1SeededFilter
    +hltDiEG2312IsoHgcalIsoL1SeededFilter
    +hltDiEG2312IsoHcalIsoL1SeededFilter
    +hltDiEle2312IsoPixelMatchL1SeededFilter
    +hltDiEle2312IsoPMS2L1SeededFilter
    +hltDiEle2312IsoGsfOneOEMinusOneOPL1SeededFilter
    +hltDiEle2312IsoGsfDetaL1SeededFilter
    +hltDiEle2312IsoGsfDphiL1SeededFilter
    +hltDiEle2312IsoBestGsfNLayerITL1SeededFilter
    +hltDiEle2312IsoBestGsfChi2L1SeededFilter
    +hltDiEle2312IsoGsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoV6L1Seeded
    +hltDiEle2312IsoGsfTrackIsoL1SeededFilter)

