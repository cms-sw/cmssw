# Config file template to write new/update AlCaRecoTriggerBits stored
# in AlCaRecoTriggerBitsRcd that is used to get selected HLT paths for
# the HLTHighLevel filter for AlCaReco production.
#
# Please understand that there are two IOVs involved:
# 1) One for the output tag. Here the usually used default is 1->inf,
#    changed by process.AlCaRecoTriggerBitsRcdUpdate.firstRunIOV
#    and process.AlCaRecoTriggerBitsRcdUpdate.lastRunIOV.
# 2) The IOV of the tag of the input AlCaRecoTriggerBitsRcd.
#    That is chosen by process.source.firstRun (but irrelevant if 
#    process.AlCaRecoTriggerBitsRcdUpdate.startEmpty = True)
#
# See also further comments below, especially the WARNING.
#
#  Author    : Gero Flucke
#  Date      : February 2009
#  $Revision: 1.3 $
#  $Date: 2009/04/21 14:42:50 $
#  (last update by $Author: flucke $)

import FWCore.ParameterSet.Config as cms

process = cms.Process("UPDATEDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1)
    ))

# the module writing to DB
process.load("CondTools.HLT.AlCaRecoTriggerBitsRcdUpdate_cfi")
# The IOV that you want to write out, defaut is 1 to -1/inf. 
#process.AlCaRecoTriggerBitsRcdUpdate.firstRunIOV = 1 # docu see...
#process.AlCaRecoTriggerBitsRcdUpdate.lastRunIOV = -1 # ...cfi
# If you want to start from scratch, comment the next line:
process.AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
# In case you want to remove 'keys', use this possibly comma separated list.
# Also if you want to replace settings for one 'key', you have to remove it first.
process.AlCaRecoTriggerBitsRcdUpdate.listNamesRemove = ["SiStripCalZeroBias"]
# Here specifiy 'key' and corresponding paths for new entries or updated ones:
process.AlCaRecoTriggerBitsRcdUpdate.triggerListsAdd = [
    cms.PSet(listName = cms.string('SiStripCalZeroBias'), # to be updated
             hltPaths = cms.vstring('HLT_ZeroBias','RandomPath')),
    cms.PSet(listName = cms.string('NewAlCaReco'),        # to be added
             hltPaths = cms.vstring('HLT_path1','HLT_path2', 'HLT_path3')),
    cms.PSet(listName = cms.string('NewAlCaRecoEmpty'),   # to be added
             hltPaths = cms.vstring())
    ]

# Here specify the 'keys' to be replaced 
process.AlCaRecoTriggerBitsRcdUpdate.alcarecoToReplace = []

# No data, but have to specify run number if you do not want 1, see below:
process.source = cms.Source("EmptySource",
                            #numberEventsInRun = cms.untracked.uint32(1),
                            #firstRun = cms.untracked.uint32(1) # 1 is default
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# DB input - needed only for AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
# WARNING:
# Take care in case the input tag has several IOVs: The run number that will be 
# used to define which payload you get is defined by the run number in the
# EmptySource above!
# Either a global tag...
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = "90X_dataRun2_Express_v0" # may choose non-default tag
# ...or (recommended since simpler) directly from DB/sqlite

process.load("CondCore.CondDB.CondDB_cfi")

# from local sqlite file
#process.CondDB.connect = 'sqlite_file:AlCaRecoTriggerBits.db'
# from conditons Database
process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'
 
process.dbInput = cms.ESSource(
    "PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(record = cms.string('AlCaRecoTriggerBitsRcd'),
                               # tag = cms.string('TestTag') # choose tag to update
                               tag = cms.string('AlCaRecoHLTpaths8e29_1e31_v7_hlt')
                               )
                      )
    )

# DB output service:
process.CondDB.connect = 'sqlite_file:AlCaRecoTriggerBits.db'
#process.CondDB.connect = 'sqlite_file:AlCaRecoTriggerBitsUpdate.db'

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    process.CondDB,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(record = cms.string('AlCaRecoTriggerBitsRcd'),
                               tag = cms.string('TestTag') # choose output tag you want
                               )
                      )
    )

# Put module in path:
process.p = cms.Path(process.AlCaRecoTriggerBitsRcdUpdate)


