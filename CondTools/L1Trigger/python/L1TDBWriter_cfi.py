import FWCore.ParameterSet.Config as cms

# L1TWriter module
# Module can be run several times provided that the following conditions are met
# * No sinceRun is provided - illogical to save data for the same IOV several times
# * No keyValue provided - we cann't store different IOV intervals with the same keyValue, so use generated one
from CondTools.L1Trigger.L1DBParams_cfi import *
L1TDBWriter = cms.EDFilter("L1TWriter",
    # Tag to use for the key. Recuired
    keyTag = cms.string('current'),
    # Catalog to use. Same as in PoolDBOutputService, but only valid catalogs
    # accepted, no fancy "local" or other shortcuts.        
    catalog = cms.string('file:test.xml'),
    # Determins what run assign as start of validity interval.
    #        int32 sinceRun = 10                            
    # Optional, if missing use one from EventSetup ???
    # Optional, but not much point to skip it.
    toSave = cms.VPSet(),
    # Database to use. Same as in PoolDBOutputService
    connect = cms.string('sqlite_file:l1config.db')
)

L1TDBWriter.toSave.extend(L1DBParams.validItems)

