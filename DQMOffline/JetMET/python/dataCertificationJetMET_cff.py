import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.jetMETDQMOfflineClient_cfi import *
from DQMOffline.JetMET.dataCertificationJetMET_cfi import *

dataCertificationJetMETSequence = cms.Sequence(jetMETDQMOfflineClient + qTesterJet + qTesterMET + dataCertificationJetMET)

#dataCertificationJetMETSequence = cms.Sequence(jetMETDQMOfflineClient + qTesterJet + qTesterMET + dataCertificationJetMET)
#dataCertificationJetMETSequence = cms.Sequence(jetMETDQMOfflineClient + qTesterJet + qTesterMET )
