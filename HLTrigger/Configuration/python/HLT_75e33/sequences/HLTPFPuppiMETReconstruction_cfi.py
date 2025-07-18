import FWCore.ParameterSet.Config as cms

from ..modules.hltGoodOfflinePrimaryVertices_cfi import *
from ..modules.hltPFPuppiMET_cfi import *
from ..modules.hltPFPuppiNoLep_cfi import *
from ..modules.hltPixelClustersMultiplicity_cfi import *

HLTPFPuppiMETReconstruction = cms.Sequence(hltGoodOfflinePrimaryVertices+hltPixelClustersMultiplicity+hltPFPuppiNoLep+hltPFPuppiMET)
