import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFJetCorrector_cfi import *
from ..modules.hltAK4PFJetCorrectorL1_cfi import *
from ..modules.hltAK4PFJetCorrectorL2_cfi import *
from ..modules.hltAK4PFJetCorrectorL3_cfi import *
from ..modules.hltAK4PFJets_cfi import *
from ..modules.hltAK4PFJetsCorrected_cfi import *

HLTAK4PFJetsReconstruction = cms.Sequence(hltAK4PFJets+hltAK4PFJetCorrectorL1+hltAK4PFJetCorrectorL2+hltAK4PFJetCorrectorL3+hltAK4PFJetCorrector+hltAK4PFJetsCorrected)
