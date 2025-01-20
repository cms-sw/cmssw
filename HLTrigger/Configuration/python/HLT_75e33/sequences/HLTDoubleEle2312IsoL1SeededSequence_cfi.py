import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTL1Sequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTTrackingV61Sequence_cfi import *

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
from ..modules.hltEgammaEleGsfTrackIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *


HLTDoubleEle2312IsoL1SeededSequence = cms.Sequence(HLTL1Sequence
    +hltEGL1SeedsForDoubleEleIsolatedFilter
    +HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +hltEgammaCandidatesL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG23EtL1SeededFilter
    +hltDiEG12EtL1SeededFilter
    +hltEgammaClusterShapeL1Seeded
    +hltDiEG2312IsoClusterShapeL1SeededFilter
    +hltEgammaHGCALIDVarsL1Seeded
    +hltDiEG2312IsoClusterShapeSigmavvL1SeededFilter
    +hltDiEG2312IsoClusterShapeSigmawwL1SeededFilter
    +hltDiEG2312IsoHgcalHEL1SeededFilter
    +HLTEGammaDoLocalHcalSequence
    +HLTFastJetForEgammaSequence
    +hltEgammaHoverEL1Seeded
    +hltDiEG2312IsoHEL1SeededFilter
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltDiEG2312IsoEcalIsoL1SeededFilter
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltDiEG2312IsoHgcalIsoL1SeededFilter
    +HLTPFHcalClusteringForEgammaSequence
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltDiEG2312IsoHcalIsoL1SeededFilter
    +HLTElePixelMatchL1SeededSequence
    +hltDiEle2312IsoPixelMatchL1SeededFilter
    +hltDiEle2312IsoPMS2L1SeededFilter
    +HLTGsfElectronL1SeededSequence
    +hltDiEle2312IsoGsfOneOEMinusOneOPL1SeededFilter
    +hltDiEle2312IsoGsfDetaL1SeededFilter
    +hltDiEle2312IsoGsfDphiL1SeededFilter
    +hltDiEle2312IsoBestGsfNLayerITL1SeededFilter
    +hltDiEle2312IsoBestGsfChi2L1SeededFilter
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltDiEle2312IsoGsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingV61Sequence
    +hltEgammaEleGsfTrackIsoL1Seeded
    +hltDiEle2312IsoGsfTrackIsoL1SeededFilter
)
