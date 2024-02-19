import FWCore.ParameterSet.Config as cms

from ..modules.hltEgammaCandidatesUnseeded_cfi import *
from ..modules.hltEgammaClusterShapeUnseeded_cfi import *
from ..modules.hltEgammaEcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaEleGsfTrackIsoV6Unseeded_cfi import *
from ..modules.hltEgammaEleL1TrkIsoUnseeded_cfi import *
from ..modules.hltEgammaHcalPFClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHGCALIDVarsUnseeded_cfi import *
from ..modules.hltEgammaHGCalLayerClusterIsoUnseeded_cfi import *
from ..modules.hltEgammaHoverEUnseeded_cfi import *

HLTEle26WP70UnseededInnerSequence = cms.Sequence(hltEgammaCandidatesUnseeded+hltEgammaClusterShapeUnseeded+hltEgammaHGCALIDVarsUnseeded+hltEgammaHoverEUnseeded+hltEgammaEcalPFClusterIsoUnseeded+hltEgammaHGCalLayerClusterIsoUnseeded+hltEgammaHcalPFClusterIsoUnseeded+hltEgammaEleL1TrkIsoUnseeded+hltEgammaEleGsfTrackIsoV6Unseeded)
