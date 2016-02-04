import FWCore.ParameterSet.Config as cms
import copy

from RecoTauTag.RecoTau.JPTRecoTauProducer_cff import *
from RecoTauTag.RecoTau.TCRecoTauProducer_cfi import *

from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstMuon_cfi import *
#from RecoTauTag.RecoTau.TCRecoTauDiscriminationAgainstHadronicJets_cfi import *

caloRecoTauProducer = copy.deepcopy(tcRecoTauProducer)

JPTJetsCollectionForTCTaus = cms.Sequence(
    jptRecoTauProducer
)

TCTau = cms.Sequence(
    caloRecoTauProducer *
    caloRecoTauDiscriminationByLeadingTrackFinding *
    caloRecoTauDiscriminationByLeadingTrackPtCut *
    caloRecoTauDiscriminationByTrackIsolation *
    caloRecoTauDiscriminationByECALIsolation *
    caloRecoTauDiscriminationByIsolation *
    caloRecoTauDiscriminationAgainstElectron *
    caloRecoTauDiscriminationAgainstMuon
    #tcRecoTauDiscriminationAgainstHadronicJets
)

tautagging = cms.Sequence(
    JPTJetsCollectionForTCTaus *
    TCTau
)

