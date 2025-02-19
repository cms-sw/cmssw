import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
from RecoTauTag.RecoTau.CaloRecoTauProducer_cfi import *
CaloTau = cms.Sequence(caloRecoTauTagInfoProducer*caloRecoTauProducer)

