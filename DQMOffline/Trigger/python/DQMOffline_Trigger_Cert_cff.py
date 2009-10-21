import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MuonHLTValidation_cfi import *


dqmOfflineTriggerCert = cms.Sequence(muonHLTCertSeq)


