import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus that need to be rerun in order to update Monte Carlo/Data samples produced with CMSSW_7_0_x RecoTauTag tags
to the latest tau id. developments recommended by the Tau POG

authors: Evan Friis, Wisconsin
         Christian Veelken, LLR

'''


from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation2_cff                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA5_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronDeadECAL_cfi          import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon2_cfi                     import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuonMVA_cfi                   import *

from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauChargedIsoPtSum 
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauNeutralIsoPtSum
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauPUcorrPtSum
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauNeutralIsoPtSumWeight
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauFootprintCorrection
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauPhotonPtSumOutsideSignalCone
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone
from RecoTauTag.Configuration.HPSPFTaus_cff import hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits


updateHPSPFTaus = cms.Sequence()

