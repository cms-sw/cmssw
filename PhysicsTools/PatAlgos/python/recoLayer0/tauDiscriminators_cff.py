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
patFixedConeHighEffPFTauDiscrimination = cms.Sequence(
    fixedConeHighEffPFTauDiscriminationByLeadingTrackFinding +
    fixedConeHighEffPFTauDiscriminationByLeadingTrackPtCut +
    fixedConeHighEffPFTauDiscriminationByLeadingPionPtCut +
    fixedConeHighEffPFTauDiscriminationByIsolation +
    fixedConeHighEffPFTauDiscriminationByTrackIsolation +
    fixedConeHighEffPFTauDiscriminationByECALIsolation +
    fixedConeHighEffPFTauDiscriminationByIsolationUsingLeadingPion +
    fixedConeHighEffPFTauDiscriminationByTrackIsolationUsingLeadingPion +
    fixedConeHighEffPFTauDiscriminationByECALIsolationUsingLeadingPion +
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
