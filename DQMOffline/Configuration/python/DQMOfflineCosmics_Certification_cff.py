import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from DQMOffline.Configuration.DQMOffline_Certification_cff import *

# Remove items that do not yet work with the Phase-2 cosmics setup 
# TODO: Fix missing dependencies to restore them to working
phase2_tracker.toReplaceWith(daq_dqmoffline,daq_dqmoffline.copyAndExclude([siStripDaqInfo]))
phase2_tracker.toReplaceWith(dcs_dqmoffline,dcs_dqmoffline.copyAndExclude([siStripDcsInfo]))
phase2_tracker.toReplaceWith(crt_dqmoffline,crt_dqmoffline.copyAndExclude([siStripCertificationInfo,trackingCertificationInfo]))


DQMOfflineCosmics_Certification = cms.Sequence(daq_dqmoffline*dcs_dqmoffline*crt_dqmoffline)

DQMOfflineCosmics_CertificationFakeHLT = cms.Sequence( DQMOfflineCosmics_Certification )
DQMOfflineCosmics_CertificationFakeHLT.remove( dqmOfflineTriggerCert )
