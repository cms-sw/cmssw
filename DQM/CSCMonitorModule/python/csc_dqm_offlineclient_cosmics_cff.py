import FWCore.ParameterSet.Config as cms

from DQM.CSCMonitorModule.csc_dqm_offlineclient_cfi import *
from DQM.CSCMonitorModule.csc_daq_info_cfi import *
from DQM.CSCMonitorModule.csc_dcs_info_cfi import *
from DQM.CSCMonitorModule.csc_certification_info_cfi import *

cscOfflineCosmicsClients = cms.Sequence(dqmCSCOfflineClient + cscDaqInfo + cscDcsInfo + cscCertificationInfo)
