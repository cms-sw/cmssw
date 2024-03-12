import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_Certification_cff import *

DQMOfflineHeavyIons_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

DQMOfflineHeavyIons_Certification.remove(dataCertificationJetMET)
# foo bar baz
# rE7w8YsPzNq3A
# 4G6lHmBhfJZgD
