import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

## FIXME: are they already on AOD?? (yes they are)
patPFTauDiscrimination = cms.Sequence(
    fixedConePFTauDiscriminationByIsolation +
    fixedConePFTauDiscriminationByLeadingTrackFinding +
    fixedConePFTauDiscriminationByLeadingTrackPtCut +
    fixedConePFTauDiscriminationByTrackIsolation +
    fixedConePFTauDiscriminationByECALIsolation +
    fixedConePFTauDiscriminationAgainstElectron +
    fixedConePFTauDiscriminationAgainstMuon
)

patCaloTauDiscrimination = cms.Sequence(
    #caloRecoTauDiscriminationAgainstElectron  +  # Not on AOD
    caloRecoTauDiscriminationByIsolation +
    caloRecoTauDiscriminationByLeadingTrackFinding +
    caloRecoTauDiscriminationByLeadingTrackPtCut 
)

#patTauDiscrimination = cms.Sequence ()  # Empty sequences not yet supported
