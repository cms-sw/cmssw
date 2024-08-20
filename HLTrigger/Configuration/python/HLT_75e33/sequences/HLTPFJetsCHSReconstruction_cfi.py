import FWCore.ParameterSet.Config as cms

from ..modules.hltGoodOfflinePrimaryVertices_cfi import *
from ..modules.hltAK4PFCHSJetCorrector_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL1_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL2_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL3_cfi import *
from ..modules.hltAK4PFCHSJets_cfi import *
from ..modules.hltAK4PFCHSJetsCorrected_cfi import *
from ..modules.hltAK8PFCHSJetCorrector_cfi import *
from ..modules.hltAK8PFCHSJetCorrectorL1_cfi import *
from ..modules.hltAK8PFCHSJetCorrectorL2_cfi import *
from ..modules.hltAK8PFCHSJetCorrectorL3_cfi import *
from ..modules.hltAK8PFCHSJets_cfi import *
from ..modules.hltAK8PFCHSJetsCorrected_cfi import *
from ..modules.hltParticleFlowPtrs_cfi import *
from ..modules.hltPfNoPileUpJME_cfi import *
from ..modules.hltPfPileUpJME_cfi import *

HLTPFJetsCHSReconstruction = cms.Sequence(hltParticleFlowPtrs+hltGoodOfflinePrimaryVertices+hltPfPileUpJME+hltPfNoPileUpJME+hltAK4PFCHSJets+hltAK4PFCHSJetCorrectorL1+hltAK4PFCHSJetCorrectorL2+hltAK4PFCHSJetCorrectorL3+hltAK4PFCHSJetCorrector+hltAK4PFCHSJetsCorrected+hltAK8PFCHSJets+hltAK8PFCHSJetCorrectorL1+hltAK8PFCHSJetCorrectorL2+hltAK8PFCHSJetCorrectorL3+hltAK8PFCHSJetCorrector+hltAK8PFCHSJetsCorrected)
