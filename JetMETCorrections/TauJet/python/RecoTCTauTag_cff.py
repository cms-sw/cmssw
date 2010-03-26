import FWCore.ParameterSet.Config as cms
import copy

from JetMETCorrections.TauJet.JPTRecoTauProducer_cff import *
from JetMETCorrections.TauJet.TCRecoTauProducer_cfi import *
#from JetMETCorrections.TauJet.TCTauProducer_cff import *
#originalCaloRecoTauProducer = copy.deepcopy(caloRecoTauProducer)
caloRecoTauProducer = copy.deepcopy(tcRecoTauProducer)

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *

tautagging = cms.Sequence(jptRecoTauProducer*
                          caloRecoTauProducer *
                          caloRecoTauTagInfoProducer *
                          caloRecoTauDiscriminationByLeadingTrackFinding *
                          caloRecoTauDiscriminationByLeadingTrackPtCut *
                          caloRecoTauDiscriminationByIsolation *
                          caloRecoTauDiscriminationAgainstElectron)

