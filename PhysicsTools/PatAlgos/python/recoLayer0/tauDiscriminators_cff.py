import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

## FIXME: are they already on AOD??
patPFTauDiscrimination = cms.Sequence(
    pfRecoTauDiscriminationByIsolation +
    pfRecoTauDiscriminationByLeadingTrackFinding +
    pfRecoTauDiscriminationByLeadingTrackPtCut +
    pfRecoTauDiscriminationByTrackIsolation +
    pfRecoTauDiscriminationByECALIsolation +
    pfRecoTauDiscriminationAgainstElectron +
    pfRecoTauDiscriminationAgainstMuon
)

patCaloTauDiscrimination = cms.Sequence(
    #caloRecoTauDiscriminationAgainstElectron  +  # Not on AOD
    caloRecoTauDiscriminationByIsolation +
    caloRecoTauDiscriminationByLeadingTrackFinding +
    caloRecoTauDiscriminationByLeadingTrackPtCut 
)

#patTauDiscrimination = cms.Sequence ()  # Empty sequences not yet supported
