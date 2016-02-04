import FWCore.ParameterSet.Config as cms

from OnlineDB.SiStripConfigDb.SiStripPartitions_cff import *

SiStripConfigDb = cms.Service("SiStripConfigDb",

    UsingDb = cms.untracked.bool(False),
    ConfDb = cms.untracked.string(''),

    UsingDbCache = cms.untracked.bool(False),
    SharedMemory = cms.untracked.string(''),

    TNS_ADMIN = cms.untracked.string('/afs/cern.ch/project/oracle/admin'),

    Partitions = cms.untracked.PSet(
        SiStripPartitions
    )

)

# UsingDb:           Connect to DB (true) or use XML files (false)
# ConfDB:            Database connection parameters
# UsingDbCache:      Use database cache (true) or connect directly (false)
# SharedMemory:      Name of shared memory used by database cache
# TNS_ADMIN:         Overrides environmental variable "TNS_ADMIN" 
# SiStripPartitions: Contains list of PSets defining database partitions (defined within top-level PSet called "Partitions")


