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
    hpsPFTauDiscriminationByMVA2Loose1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Loose2ElectronRejection*
    hpsPFTauDiscriminationByMVA2Medium1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Medium2ElectronRejection*
    hpsPFTauDiscriminationByMVA2Tight1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Tight2ElectronRejection*
    hpsPFTauDiscriminationByMVA2VTight1ElectronRejection*
    hpsPFTauDiscriminationByMVA2VTight2ElectronRejection*    
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByMediumMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection
)
