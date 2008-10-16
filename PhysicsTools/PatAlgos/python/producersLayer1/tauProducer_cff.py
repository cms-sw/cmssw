import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cfi import *

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
patPFTauDiscrimination = cms.Sequence(
    patPFRecoTauDiscriminationByIsolation*
    patPFRecoTauDiscriminationByLeadingTrackFinding*
    patPFRecoTauDiscriminationByLeadingTrackPtCut*
    patPFRecoTauDiscriminationByTrackIsolation*
    patPFRecoTauDiscriminationByECALIsolation*
    patPFRecoTauDiscriminationAgainstElectron*
    patPFRecoTauDiscriminationAgainstMuon
)

from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi import *
from PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cff import *
layer1Taus = cms.Sequence(patPFTauDiscrimination * allLayer1Taus * selectedLayer1Taus * countLayer1Taus)


