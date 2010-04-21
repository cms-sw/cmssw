import FWCore.ParameterSet.Config as cms

from JetMETCorrections.TauJet.JPTRecoTauProducer_cff import *
from JetMETCorrections.TauJet.TCRecoTauProducer_cfi import *

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
caloRecoTauDiscriminationByLeadingTrackFinding.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
caloRecoTauDiscriminationByLeadingTrackPtCut.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
caloRecoTauDiscriminationByIsolation.CaloTauProducer = cms.InputTag('tcRecoTauProducer')
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
caloRecoTauDiscriminationAgainstElectron.CaloTauProducer = cms.InputTag('tcRecoTauProducer')

from JetMETCorrections.TauJet.TCRecoTauDiscriminationAgainstHadronicJets_cfi import *

TCTau = cms.Sequence(jptRecoTauProducer*
                     tcRecoTauProducer *
                     caloRecoTauTagInfoProducer *
                     caloRecoTauDiscriminationByLeadingTrackFinding *
                     caloRecoTauDiscriminationByLeadingTrackPtCut *
                     caloRecoTauDiscriminationByIsolation *
                     caloRecoTauDiscriminationAgainstElectron *
                     tcRecoTauDiscriminationAgainstHadronicJets)

