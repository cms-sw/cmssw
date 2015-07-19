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

updateHPSPFTaus = cms.Sequence(
     hpsPFTauChargedIsoPtSum*
     hpsPFTauNeutralIsoPtSum*
     hpsPFTauPUcorrPtSum*
     hpsPFTauNeutralIsoPtSumWeight*
     hpsPFTauFootprintCorrection*
     hpsPFTauPhotonPtSumOutsideSignalCone*
     hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits*
     hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits*
     hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits*
     hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits*
     hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone*
     hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits
)
