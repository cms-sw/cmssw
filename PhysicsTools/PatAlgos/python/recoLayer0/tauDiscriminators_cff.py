import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

patFixedConePFTauDiscrimination = cms.Sequence(
    fixedConePFTauDiscriminationByLeadingTrackFinding +
    fixedConePFTauDiscriminationByLeadingTrackPtCut +
    fixedConePFTauDiscriminationByLeadingPionPtCut +
    fixedConePFTauDiscriminationByIsolation +
    fixedConePFTauDiscriminationByTrackIsolation +
    fixedConePFTauDiscriminationByECALIsolation +
    fixedConePFTauDiscriminationByIsolationUsingLeadingPion +
    fixedConePFTauDiscriminationByTrackIsolationUsingLeadingPion +
    fixedConePFTauDiscriminationByECALIsolationUsingLeadingPion +
    fixedConePFTauDiscriminationAgainstElectron +
    fixedConePFTauDiscriminationAgainstMuon
)
patHPSPFTauDiscrimination = cms.Sequence(
    hpsPFTauDiscriminationByDecayModeFinding +
    hpsPFTauDiscriminationByVLooseIsolation +
    hpsPFTauDiscriminationByLooseIsolation +
    hpsPFTauDiscriminationByMediumIsolation +
    hpsPFTauDiscriminationByTightIsolation +
    hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByTightIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr +
    hpsPFTauDiscriminationByLooseElectronRejection +
    hpsPFTauDiscriminationByMediumElectronRejection +
    hpsPFTauDiscriminationByTightElectronRejection +
    hpsPFTauDiscriminationByMVAElectronRejection +
    hpsPFTauDiscriminationByLooseMuonRejection +
    hpsPFTauDiscriminationByMediumMuonRejection +
    hpsPFTauDiscriminationByTightMuonRejection
)
patShrinkingConePFTauDiscrimination = cms.Sequence(
    shrinkingConePFTauDiscriminationByLeadingTrackFinding +
    shrinkingConePFTauDiscriminationByLeadingTrackPtCut +
    shrinkingConePFTauDiscriminationByLeadingPionPtCut +
    shrinkingConePFTauDiscriminationByIsolation +
    shrinkingConePFTauDiscriminationByTrackIsolation +
    shrinkingConePFTauDiscriminationByECALIsolation +
    shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion +
    shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion +
    shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion +
    shrinkingConePFTauDiscriminationAgainstElectron +
    shrinkingConePFTauDiscriminationAgainstMuon +
    shrinkingConePFTauDecayModeProducer +
    shrinkingConePFTauDecayModeIndexProducer +
    shrinkingConePFTauDiscriminationByTaNC +
    shrinkingConePFTauDiscriminationByTaNCfrOnePercent +
    shrinkingConePFTauDiscriminationByTaNCfrHalfPercent +
    shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent +
    shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
)

patCaloTauDiscrimination = cms.Sequence(
    caloRecoTauDiscriminationAgainstElectron  +
    caloRecoTauDiscriminationByIsolation +
    caloRecoTauDiscriminationByLeadingTrackFinding +
    caloRecoTauDiscriminationByLeadingTrackPtCut
)
