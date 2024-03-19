import FWCore.ParameterSet.Config as cms

from ..modules.hltAK8PFJetCorrector_cfi import *
from ..modules.hltAK8PFJetCorrectorL1_cfi import *
from ..modules.hltAK8PFJetCorrectorL2_cfi import *
from ..modules.hltAK8PFJetCorrectorL3_cfi import *
from ..modules.hltAK8PFJets_cfi import *
from ..modules.hltAK8PFJetsCorrected_cfi import *

HLTAK8PFJetsReconstruction = cms.Sequence(hltAK8PFJets+hltAK8PFJetCorrectorL1+hltAK8PFJetCorrectorL2+hltAK8PFJetCorrectorL3+hltAK8PFJetCorrector+hltAK8PFJetsCorrected)
