import FWCore.ParameterSet.Config as cms

from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cfi import *

castorSources = cms.Sequence(castorOfflineMonitor)

