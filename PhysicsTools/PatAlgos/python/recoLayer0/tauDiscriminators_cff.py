import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *

#patAODPFRecoTauDiscriminationByIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
#patAODFRecoTauDiscriminationByIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
patAODTauDiscrimination = cms.Sequence(
    caloRecoTauDiscriminationByIsolation *
    pfRecoTauDiscriminationByIsolation
)

#copy the PFTauDiscriminator producer;
#instead of the AOD reco::PFTau collection, set the reco::PFTaus collection produced by PAT layer 0 cleaning
#as reference for PFTauDiscriminator objects
patPFRecoTauDiscriminationByIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
patPFRecoTauDiscriminationByIsolation.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationByLeadingTrackFinding = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
patPFRecoTauDiscriminationByLeadingTrackFinding.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationByLeadingTrackPtCut = copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
patPFRecoTauDiscriminationByLeadingTrackPtCut.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationByTrackIsolation = copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
patPFRecoTauDiscriminationByTrackIsolation.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationByECALIsolation = copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
patPFRecoTauDiscriminationByECALIsolation.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationAgainstElectron = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
patPFRecoTauDiscriminationAgainstElectron.PFTauProducer = cms.InputTag('allLayer0Taus')
patPFRecoTauDiscriminationAgainstMuon = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
patPFRecoTauDiscriminationAgainstMuon.PFTauProducer = cms.InputTag('allLayer0Taus')

#copy the CaloTauDiscriminator producer;
#instead of the AOD reco::CaloTau collection, set the reco::CaloTaus collection produced by PAT layer 0 cleaning
#as reference for CaloTauDiscriminator objects
patCaloRecoTauDiscriminationByIsolation = copy.deepcopy(caloRecoTauDiscriminationByIsolation)
patCaloRecoTauDiscriminationByIsolation.CaloTauProducer = 'allLayer0CaloTaus'
patCaloRecoTauDiscriminationByLeadingTrackFinding = copy.deepcopy(caloRecoTauDiscriminationByLeadingTrackFinding)
patCaloRecoTauDiscriminationByLeadingTrackFinding.CaloTauProducer = 'allLayer0CaloTaus'
patCaloRecoTauDiscriminationByLeadingTrackPtCut = copy.deepcopy(caloRecoTauDiscriminationByLeadingTrackPtCut)
patCaloRecoTauDiscriminationByLeadingTrackPtCut.CaloTauProducer = 'allLayer0CaloTaus'
#patCaloRecoTauDiscriminationAgainstElectron = copy.deepcopy(caloRecoTauDiscriminationAgainstElectron)  # Not on AOD
#patCaloRecoTauDiscriminationAgainstElectron.CaloTauProducer = 'allLayer0CaloTaus'                      # Not on AOD

patPFTauDiscrimination = cms.Sequence(
    patPFRecoTauDiscriminationByIsolation +
    patPFRecoTauDiscriminationByLeadingTrackFinding +
    patPFRecoTauDiscriminationByLeadingTrackPtCut +
    patPFRecoTauDiscriminationByTrackIsolation +
    patPFRecoTauDiscriminationByECALIsolation +
    patPFRecoTauDiscriminationAgainstElectron +
    patPFRecoTauDiscriminationAgainstMuon
)

patCaloTauDiscrimination = cms.Sequence(
    #patCaloRecoTauDiscriminationAgainstElectron  +  # Not on AOD
    patCaloRecoTauDiscriminationByIsolation +
    patCaloRecoTauDiscriminationByLeadingTrackFinding +
    patCaloRecoTauDiscriminationByLeadingTrackPtCut 
)

