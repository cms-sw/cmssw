import FWCore.ParameterSet.Config as cms

from DQMServices.Components.DQMDaqInfo_cfi import *

from DQMOffline.Configuration.DQMOffline_DAQ_cff import *
from DQMOffline.Configuration.DQMOffline_DCS_cff import *
from DQMOffline.Configuration.DQMOffline_CRT_cff import *

#DQMOffline_Certification = cms.Sequence(dqmDaqInfo)

DQMOffline_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

