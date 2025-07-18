import FWCore.ParameterSet.Config as cms

from ..modules.hltPFMET_cfi import *
from ..modules.hltAK4PFCHSJetCorrector_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL1_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL2_cfi import *
from ..modules.hltAK4PFCHSJetCorrectorL3_cfi import *
from ..modules.hltPFMETTypeOne_cfi import *
from ..modules.hltPFMETTypeOneCorrector_cfi import *

HLTPFMETsReconstruction = cms.Sequence(hltPFMET+hltAK4PFCHSJetCorrectorL1+hltAK4PFCHSJetCorrectorL2+hltAK4PFCHSJetCorrectorL3+hltAK4PFCHSJetCorrector+hltPFMETTypeOneCorrector+hltPFMETTypeOne)
