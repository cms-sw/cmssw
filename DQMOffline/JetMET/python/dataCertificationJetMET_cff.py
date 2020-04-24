import FWCore.ParameterSet.Config as cms

from DQMOffline.JetMET.dataCertificationJetMET_cfi import *

dataCertificationJetMETSequence = cms.Sequence(qTesterJet + qTesterMET + dataCertificationJetMET)
