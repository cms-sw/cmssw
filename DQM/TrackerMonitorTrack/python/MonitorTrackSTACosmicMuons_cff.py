# The following comments couldn't be translated into the new config version:

# services for the DQM (Analyzer)

import FWCore.ParameterSet.Config as cms

#DQM global tarcks analyzer
from DQM.TrackerMonitorTrack.MonitorTrackSTACosmicMuons_cfi import *
DaqMonitorROOTBackEnd = cms.Service("DaqMonitorROOTBackEnd")

# DQM Online File saver module
# PUT THIS MODULE INTO YOUR PATH FOR OPERATION AT P5
dqmSaver = cms.EDFilter("DQMFileSaver",
    # Save file every N events (-1: disabled)
    prescaleEvt = cms.untracked.int32(-1),
    # Save file every N lumi sections (-1: disabled)
    prescaleLS = cms.untracked.int32(-1),
    # Save at end of job, no runnumber in filename (default: false)
    saveAtJobEnd = cms.untracked.bool(False),
    # subsystem name (appears in filename)
    # default is: filename is taken from SourceName
    #             (toplevel of me-dir-tree) fixme
    fileName = cms.untracked.string('SubsystemName'),
    # environment flag (default: "Online")
    # FIXME should be implemented in dbe::save first
    environment = cms.untracked.string('Online'),
    # Save at end of run, runnumber in filename (default: true)
    saveAtRunEnd = cms.untracked.bool(True),
    # Save file every N minutes (-1: disabled)
    prescaleTime = cms.untracked.int32(-1)
)

DQMShipMonitoring = cms.Service("DQMShipMonitoring",
    #  // event-period for shipping monitoring to collector (default: 25)
    period = cms.untracked.uint32(500)
)


