import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

patFixedConePFTauDiscrimination = cms.Sequence(
    fixedConePFTauDiscriminationByIsolation +
    fixedConePFTauDiscriminationByLeadingTrackFinding +
    fixedConePFTauDiscriminationByLeadingTrackPtCut +
    fixedConePFTauDiscriminationByTrackIsolation +
    fixedConePFTauDiscriminationByECALIsolation +
    fixedConePFTauDiscriminationAgainstElectron +
    fixedConePFTauDiscriminationAgainstMuon
)
patFixedConeHighEffPFTauDiscrimination = cms.Sequence(
    fixedConeHighEffPFTauDiscriminationByIsolation +
    fixedConeHighEffPFTauDiscriminationByLeadingTrackFinding +
    fixedConeHighEffPFTauDiscriminationByLeadingTrackPtCut +
    fixedConeHighEffPFTauDiscriminationByTrackIsolation +
    fixedConeHighEffPFTauDiscriminationByECALIsolation +
    fixedConeHighEffPFTauDiscriminationAgainstElectron +
    fixedConeHighEffPFTauDiscriminationAgainstMuon
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
    shrinkingConePFTauDiscriminationByTaNC +
    shrinkingConePFTauDiscriminationByTaNCfrOnePercent +
    shrinkingConePFTauDiscriminationByTaNCfrHalfPercent +
    shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent +
    shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
)

patCaloTauDiscrimination = cms.Sequence(
    #caloRecoTauDiscriminationAgainstElectron  +  # Not on AOD
    caloRecoTauDiscriminationByIsolation +
    caloRecoTauDiscriminationByLeadingTrackFinding +
    caloRecoTauDiscriminationByLeadingTrackPtCut 
)
