import FWCore.ParameterSet.Config as cms

from ..modules.ak4CaloJetsForTrk_cfi import *
from ..modules.caloTowerForTrk_cfi import *
from ..modules.firstStepPrimaryVertices_cfi import *
from ..modules.firstStepPrimaryVerticesUnsorted_cfi import *
from ..modules.initialStepTrackRefsForJets_cfi import *

initialStepPVSequence = cms.Sequence(firstStepPrimaryVerticesUnsorted+initialStepTrackRefsForJets+caloTowerForTrk+ak4CaloJetsForTrk+firstStepPrimaryVertices)
