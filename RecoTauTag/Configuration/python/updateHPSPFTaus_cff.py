import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus that need to be rerun in order to update Monte Carlo/Data samples produced with CMSSW_5_2_x RecoTauTag tags
to the latest tau id. developments recommended by the Tau POG

authors: Evan Friis, Wisconsin
         Christian Veelken, LLR

'''

from RecoTauTag.Configuration.HPSPFTaus_cff import *

updateHPSPFTaus = cms.Sequence(
    hpsPFTauDiscriminationByChargedIsolationSeq*
    hpsPFTauDiscriminationByMVAIsolationSeq*

    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr*

    hpsPFTauDiscriminationByMVAElectronRejection*
    hpsPFTauDiscriminationByMVA2rawElectronRejection*
    hpsPFTauDiscriminationByMVA2VLooseElectronRejection*
    hpsPFTauDiscriminationByMVA2LooseElectronRejection*    
    hpsPFTauDiscriminationByMVA2MediumElectronRejection*
    hpsPFTauDiscriminationByMVA2TightElectronRejection*
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByMediumMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection
)
