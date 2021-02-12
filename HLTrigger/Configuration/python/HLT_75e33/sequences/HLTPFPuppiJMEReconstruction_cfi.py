import FWCore.ParameterSet.Config as cms

from ..modules.hltAK4PFPuppiJetCorrector_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL1_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL2_cfi import *
from ..modules.hltAK4PFPuppiJetCorrectorL3_cfi import *
from ..modules.hltAK4PFPuppiJets_cfi import *
from ..modules.hltAK4PFPuppiJetsCorrected_cfi import *
from ..modules.hltAK8PFPuppiJetCorrector_cfi import *
from ..modules.hltAK8PFPuppiJetCorrectorL1_cfi import *
from ..modules.hltAK8PFPuppiJetCorrectorL2_cfi import *
from ..modules.hltAK8PFPuppiJetCorrectorL3_cfi import *
from ..modules.hltAK8PFPuppiJets_cfi import *
from ..modules.hltAK8PFPuppiJetsCorrected_cfi import *
from ..modules.hltPFPuppi_cfi import *
from ..modules.hltPFPuppiMET_cfi import *
from ..modules.hltPFPuppiMETTypeOne_cfi import *
from ..modules.hltPFPuppiMETTypeOneCorrector_cfi import *
from ..modules.hltPFPuppiMETv0_cfi import *
from ..modules.hltPFPuppiNoLep_cfi import *
from ..modules.hltPixelClustersMultiplicity_cfi import *

HLTPFPuppiJMEReconstruction = cms.Sequence(hltPixelClustersMultiplicity+hltPFPuppiNoLep+hltPFPuppiMET+hltPixelClustersMultiplicity+hltPFPuppi+hltPFPuppiMETv0+hltAK4PFPuppiJets+hltAK4PFPuppiJetCorrectorL1+hltAK4PFPuppiJetCorrectorL2+hltAK4PFPuppiJetCorrectorL3+hltAK4PFPuppiJetCorrector+hltAK4PFPuppiJetsCorrected+hltPFPuppiMETTypeOneCorrector+hltPFPuppiMETTypeOne+hltAK8PFPuppiJets+hltAK8PFPuppiJetCorrectorL1+hltAK8PFPuppiJetCorrectorL2+hltAK8PFPuppiJetCorrectorL3+hltAK8PFPuppiJetCorrector+hltAK8PFPuppiJetsCorrected)
