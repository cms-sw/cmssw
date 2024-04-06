import FWCore.ParameterSet.Config as cms

from ..modules.hltPFMET_cfi import *
from ..modules.hltPFMETJetCorrector_cfi import *
from ..modules.hltPFMETJetCorrectorL1_cfi import *
from ..modules.hltPFMETJetCorrectorL2_cfi import *
from ..modules.hltPFMETJetCorrectorL3_cfi import *
from ..modules.hltPFMETTypeOne_cfi import *
from ..modules.hltPFMETTypeOneCorrector_cfi import *

HLTPFMETsReconstruction = cms.Sequence(hltPFMET+hltPFMETJetCorrectorL1+hltPFMETJetCorrectorL2+hltPFMETJetCorrectorL3+hltPFMETJetCorrector+hltPFMETTypeOneCorrector+hltPFMETTypeOne)
