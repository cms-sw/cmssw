import FWCore.ParameterSet.Config as cms

#CaloTauTagInfo Producer
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
#CaloTau Producer
from RecoTauTag.RecoTau.CaloRecoTauProducer_cfi import *
#CaloTauDiscriminatorByIsolation Producer
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
tautagging = cms.Sequence(caloRecoTauTagInfoProducer*caloRecoTauProducer*caloRecoTauDiscriminationByIsolation)

