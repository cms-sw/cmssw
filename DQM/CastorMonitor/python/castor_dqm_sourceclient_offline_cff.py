import FWCore.ParameterSet.Config as cms

from DQM.CastorMonitor.castor_dqm_sourceclient_offline_cfi import *
castorSources = cms.Sequence(castorOfflineMonitor)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toReplaceWith(castorSources, castorSources.copyAndExclude([castorOfflineMonitor]))

