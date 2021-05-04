import FWCore.ParameterSet.Config as cms

from ..modules.goodOfflinePrimaryVertices_cfi import *
from ..modules.hltAK4PFPuppiJetCorrector_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL1_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL2_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL3_cfi import *
from ..modules.hltAK4PFPuppiJets_cfi import *
from ..modules.hltAK4PFPuppiJetsCorrected_cfi import *
from ..modules.hltPFPuppi_cfi import *
from ..modules.hltPixelClustersMultiplicity_cfi import *

HLTAK4PFPuppiJetsReconstruction = cms.Sequence(
    goodOfflinePrimaryVertices +
    hltPixelClustersMultiplicity +
    hltPFPuppi +
    hltAK4PFPuppiJets +
    hltAK4PFPuppiJetCorrectorL1 +
    hltAK4PFPuppiJetCorrectorL2 +
    hltAK4PFPuppiJetCorrectorL3 +
    hltAK4PFPuppiJetCorrector +
    hltAK4PFPuppiJetsCorrected
)
