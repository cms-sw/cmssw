import FWCore.ParameterSet.Config as cms
 
#CaloTauTagInfo Producer
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
#CaloTau Producer
from RecoTauTag.RecoTau.CaloRecoTauProducer_cfi import *
#CaloTauDiscriminatorByIsolation Producer
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationAgainstMuon_cfi import *
 
tautagging = cms.Sequence(
    caloRecoTauTagInfoProducer*
    caloRecoTauProducer*
    caloRecoTauDiscriminationByLeadingTrackFinding*
    caloRecoTauDiscriminationByLeadingTrackPtCut*                          
    caloRecoTauDiscriminationByTrackIsolation*
    caloRecoTauDiscriminationByECALIsolation*
    caloRecoTauDiscriminationByIsolation*    
    caloRecoTauDiscriminationAgainstElectron *
    caloRecoTauDiscriminationAgainstMuon
)
