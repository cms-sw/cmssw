import FWCore.ParameterSet.Config as cms

# load the DQM service
from DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi import *

# The following is not used any more; there are no clients for this output
# so it wastes a lot of memory, possibly also cpu power
# Needs also hasSharedMemory to be false for the FUEventProcessor xdaq config
## DQM output via the shared memory
#FUShmDQMOutputService = cms.Service("FUShmDQMOutputService",
#    initialMessageBufferSize = cms.untracked.int32(1000000),
#    compressionLevel = cms.int32(1),
#    lumiSectionInterval = cms.untracked.int32(2000000),
#    lumiSectionsPerUpdate = cms.double(1.0),
#    useCompression = cms.bool(True)
#)

# needed in the online because the otherwise default initiated one gets ill-configured (missing parameter lvl1Labels)
# it doesn't really belong here, but ok, it doesn't hurt either
from FWCore.PrescaleService.PrescaleService_cfi import *
PrescaleService.prescaleTable = cms.VPSet()
