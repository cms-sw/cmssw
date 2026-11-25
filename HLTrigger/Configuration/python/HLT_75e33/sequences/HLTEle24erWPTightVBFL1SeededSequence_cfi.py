import FWCore.ParameterSet.Config as cms

# Clone of HLTEle32WPTightL1SeededSequence without its L1 e/gamma seed filter,
# with ET > 24 GeV and |eta| < 2.1.
from ..sequences.HLTDoFullUnpackingEgammaEcalL1SeededSequence_cfi import *
from ..sequences.HLTEGammaDoLocalHcalSequence_cfi import *
from ..sequences.HLTElePixelMatchL1SeededSequence_cfi import *
from ..sequences.HLTFastJetForEgammaSequence_cfi import *
from ..sequences.HLTGsfElectronL1SeededSequence_cfi import *
from ..sequences.HLTHgcalTiclPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFClusteringForEgammaL1SeededSequence_cfi import *
from ..sequences.HLTPFHcalClusteringForEgammaSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

from ..modules.hltEgammaEcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoL1Seeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoL1Seeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoL1Seeded_cfi import *
from ..modules.hltEG24EtEta2p1L1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesWrapperL1Seeded_cfi import *
from ..modules.hltEle24erWPTightVBFBestGsfChi2L1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFBestGsfNLayerITL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFClusterShapeL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFClusterShapeSigmavvL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFClusterShapeSigmawwL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFEcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFGsfDetaL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFGsfDphiL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFGsfOneOEMinusOneOPL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFGsfTrackIsoFromL1TracksL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFGsfTrackIsoL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFHcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFHEL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFHgcalHEL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFHgcalIsoL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFPixelMatchL1SeededFilter_cfi import *
from ..modules.hltEle24erWPTightVBFPMS2L1SeededFilter_cfi import *
from ..modules.hltEgammaCandidatesL1Seeded_cfi import *
from ..modules.hltEgammaClusterShapeL1Seeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsL1Seeded_cfi import *
from ..modules.hltEgammaHoverEL1Seeded_cfi import *

HLTEle24erWPTightVBFL1SeededSequence = cms.Sequence(
    HLTDoFullUnpackingEgammaEcalL1SeededSequence
    +HLTPFClusteringForEgammaL1SeededSequence
    +HLTHgcalTiclPFClusteringForEgammaL1SeededSequence
    +hltEgammaCandidatesL1Seeded
    +hltEgammaCandidatesWrapperL1Seeded
    +hltEG24EtEta2p1L1SeededFilter
    +hltEgammaClusterShapeL1Seeded
    +hltEle24erWPTightVBFClusterShapeL1SeededFilter
    +hltEgammaHGCALIDVarsL1Seeded
    +hltEle24erWPTightVBFClusterShapeSigmavvL1SeededFilter
    +hltEle24erWPTightVBFClusterShapeSigmawwL1SeededFilter
    +hltEle24erWPTightVBFHgcalHEL1SeededFilter
    +HLTEGammaDoLocalHcalSequence
    +HLTFastJetForEgammaSequence
    +hltEgammaHoverEL1Seeded
    +hltEle24erWPTightVBFHEL1SeededFilter
    +hltEgammaEcalPFClusterIsoL1Seeded
    +hltEle24erWPTightVBFEcalIsoL1SeededFilter
    +hltEgammaHGCalLayerClusterIsoL1Seeded
    +hltEle24erWPTightVBFHgcalIsoL1SeededFilter
    +HLTPFHcalClusteringForEgammaSequence
    +hltEgammaHcalPFClusterIsoL1Seeded
    +hltEle24erWPTightVBFHcalIsoL1SeededFilter
    +HLTElePixelMatchL1SeededSequence
    +hltEle24erWPTightVBFPixelMatchL1SeededFilter
    +hltEle24erWPTightVBFPMS2L1SeededFilter
    +HLTGsfElectronL1SeededSequence
    +hltEle24erWPTightVBFGsfOneOEMinusOneOPL1SeededFilter
    +hltEle24erWPTightVBFGsfDetaL1SeededFilter
    +hltEle24erWPTightVBFGsfDphiL1SeededFilter
    +hltEle24erWPTightVBFBestGsfNLayerITL1SeededFilter
    +hltEle24erWPTightVBFBestGsfChi2L1SeededFilter
    +hltEgammaEleL1TrkIsoL1Seeded
    +hltEle24erWPTightVBFGsfTrackIsoFromL1TracksL1SeededFilter
    +HLTTrackingSequence
    +hltEgammaEleGsfTrackIsoL1Seeded
    +hltEle24erWPTightVBFGsfTrackIsoL1SeededFilter

)
