import FWCore.ParameterSet.Config as cms

from DQMOffline.Configuration.DQMOffline_Certification_cff import *

DQMOfflineCosmics_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

DQMOfflineCosmics_CertificationFakeHLT = cms.Sequence( DQMOfflineCosmics_Certification )
DQMOfflineCosmics_CertificationFakeHLT.remove( dqmOfflineTriggerCert )
