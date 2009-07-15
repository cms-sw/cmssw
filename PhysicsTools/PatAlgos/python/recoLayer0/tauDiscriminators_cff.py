import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

## FIXME: are they already on AOD?? (yes they are)
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
    shrinkingConePFTauDiscriminationByIsolation +
    shrinkingConePFTauDiscriminationByLeadingTrackFinding +
    shrinkingConePFTauDiscriminationByLeadingTrackPtCut +
    shrinkingConePFTauDiscriminationByTrackIsolation +
    shrinkingConePFTauDiscriminationByECALIsolation +
    shrinkingConePFTauDiscriminationAgainstElectron +
    shrinkingConePFTauDiscriminationAgainstMuon
)

patCaloTauDiscrimination = cms.Sequence(
    #caloRecoTauDiscriminationAgainstElectron  +  # Not on AOD
    caloRecoTauDiscriminationByIsolation +
    caloRecoTauDiscriminationByLeadingTrackFinding +
    caloRecoTauDiscriminationByLeadingTrackPtCut 
)
