import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.JPTRecoTauProducer_cff import *
from RecoTauTag.RecoTau.TCRecoTauProducer_cfi import *

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
caloRecoTauDiscriminationByLeadingTrackFinding.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
caloRecoTauDiscriminationByLeadingTrackPtCut.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
caloRecoTauDiscriminationByIsolation.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
caloRecoTauDiscriminationAgainstElectron.CaloTauProducer = cms.InputTag('tcRecoTauProducer')

from JetMETCorrections.TauJet.TCRecoTauDiscriminationAgainstHadronicJets_cfi import *
tcRecoTauDiscriminationAgainstHadronicJets.CaloTauProducer = cms.InputTag('tcRecoTauProducer')

from JetMETCorrections.TauJet.TCRecoTauDiscriminationAlgoComponent_cfi import *
tcRecoTauDiscriminationAlgoComponent.CaloTauProducer = cms.InputTag('tcRecoTauProducer')

TCTau = cms.Sequence(jptRecoTauProducer*
                     tcRecoTauProducer *
                     caloRecoTauTagInfoProducer *
                     caloRecoTauDiscriminationByLeadingTrackFinding *
                     caloRecoTauDiscriminationByLeadingTrackPtCut *
                     caloRecoTauDiscriminationByIsolation *
                     caloRecoTauDiscriminationAgainstElectron *
                     tcRecoTauDiscriminationAgainstHadronicJets *
                     tcRecoTauDiscriminationAlgoComponent
)

