import FWCore.ParameterSet.Config as cms
import copy

from JetMETCorrections.TauJet.JPTRecoTauProducer_cff import *
from JetMETCorrections.TauJet.TCRecoTauProducer_cfi import *

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
from JetMETCorrections.TauJet.TCRecoTauDiscriminationAgainstHadronicJets_cfi import *

caloRecoTauProducer = copy.deepcopy(tcRecoTauProducer)

tautagging = cms.Sequence(jptRecoTauProducer*
                          caloRecoTauProducer *
                          caloRecoTauDiscriminationByLeadingTrackFinding *
                          caloRecoTauDiscriminationByLeadingTrackPtCut *
                          caloRecoTauDiscriminationByIsolation *
                          caloRecoTauDiscriminationAgainstElectron *
                          tcRecoTauDiscriminationAgainstHadronicJets)

