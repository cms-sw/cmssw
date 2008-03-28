import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
#OLD dataformats ConeIsolation
from RecoTauTag.ConeIsolation.coneIsolationTauJetTags_cfi import *
#CaloTauTagInfo Producer
from RecoTauTag.RecoTau.CaloRecoTauTagInfoProducer_cfi import *
#CaloTau Producer
from RecoTauTag.RecoTau.CaloRecoTauProducer_cfi import *
#CaloTauDiscriminatorByIsolation Producer
from RecoTauTag.RecoTau.CaloRecoTauDiscriminationByIsolation_cfi import *
tautagging = cms.Sequence(coneIsolationTauJetTags*caloRecoTauTagInfoProducer*caloRecoTauProducer*caloRecoTauDiscriminationByIsolation)

