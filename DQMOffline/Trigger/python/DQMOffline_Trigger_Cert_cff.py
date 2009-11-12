import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MuonHLTValidation_cfi import *

from DQMOffline.Trigger.EgHLTOfflineSummaryClient_cfi import *

dqmOfflineTriggerCert = cms.Sequence(muonHLTCertSeq*egHLTOffCertSeq)


