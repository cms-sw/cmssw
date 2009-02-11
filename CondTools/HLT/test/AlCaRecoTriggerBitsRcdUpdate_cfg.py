# Config file template to write new/update AlCaRecoTriggerBits stored
# in AlCaRecoTriggerBitsRcd that is used to get selected HLT paths for
# the HLTHighLevel filter for AlCaReco production.
# See comments inside, especially the WARNING.
# 
#  Author    : Gero Flucke
#  Date      : February 2009
#  $Revision: 1.42 $
#  $Date: 2008/11/10 14:48:42 $
#  (last update by $Author: henderle $)

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
# If you want to update, uncomment the next line:
#process.AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
# In case you want to remove 'keys', use this possibly comma separated list.
# Also if you want to replace settings for one 'key', you have to remove it first.
#process.AlCaRecoTriggerBitsRcdUpdate.listNamesRemove = ["TkAlZMuMu"]
# Here specifiy 'key' and corresponding paths for new entries or updated ones:
#process.AlCaRecoTriggerBitsRcdUpdate.triggerListsAdd = [
#    cms.PSet(listName = cms.string('TkAlZMuMu'),
#             hltPaths = cms.vstring('path_1','path_2','path_3')),
#    cms.PSet(listName = cms.string('Bla'),
#             hltPaths = cms.vstring('p1','p2'))
#    ]


# No data, but have to specify run number if you do not want 1, see below:
process.source = cms.Source("EmptySource",
                            #numberEventsInRun = cms.untracked.uint32(1),
                            #firstRun = cms.untracked.uint32(1) # 1 is default
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# DB input - needed only for AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
# WARNING:
# Take care in case the input tag has an IOV: The run number that will be used
# to define which payload you get is defined by the run number in the
# EmptySource above!
# import CondCore.DBCommon.CondDBSetup_cfi
#process.dbInput = cms.ESSource(
#    "PoolDBESSource",
#    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
#    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('AlCaRecoTriggerBitsRcd'),
#        tag = cms.string('TestTag') # choose old tag to update
#        )
#                      )
#    )

# DB output service:
import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
#    connect = cms.string('sqlite_file:AlCaRecoTriggerBitsUpdate.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('AlCaRecoTriggerBitsRcd'),
        tag = cms.string('TestTag') # choose tag you want
        )
                      )
    )


# Put module in path:
process.p = cms.Path(process.AlCaRecoTriggerBitsRcdUpdate)


