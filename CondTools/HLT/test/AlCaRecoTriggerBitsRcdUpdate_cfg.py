import FWCore.ParameterSet.Config as cms

process = cms.Process("UPDATEDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1)
    ))

# the module writing to DB
process.load("CondTools.HLT.AlCaRecoTriggerBitsRcdUpdate_cfi")
# default process.AlCaRecoTriggerBitsRcdUpdate.firstRunIOV = 1 # docu see...
# default
#process.AlCaRecoTriggerBitsRcdUpdate.lastRunIOV = 10 # ...cfi
process.AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
process.AlCaRecoTriggerBitsRcdUpdate.listNamesRemove = ["TkAlZMuMu"]
process.AlCaRecoTriggerBitsRcdUpdate.triggerListsAdd = [
    cms.PSet(listName = cms.string('TkAlZMuMu'),
             hltPaths = cms.vstring('path_1','path_2','path_3')),
    cms.PSet(listName = cms.string('Bla'),
             hltPaths = cms.vstring('p1','p2'))
    ]


# No data, but have to specify run number (for reading):
process.source = cms.Source("EmptySource",
                            #numberEventsInRun = cms.untracked.uint32(1),
                            #firstRun = cms.untracked.uint32(5)
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# DB input - needed only for AlCaRecoTriggerBitsRcdUpdate.startEmpty = False
import CondCore.DBCommon.CondDBSetup_cfi
process.dbInput = cms.ESSource(
    "PoolDBESSource",
    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('AlCaRecoTriggerBitsRcd'),
        tag = cms.string('TestTag') # choose old tag to update
        )
                      )
    )

# DB output service:
import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:AlCaRecoTriggerBitsUpdate.db'),
#    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('AlCaRecoTriggerBitsRcd'),
        tag = cms.string('TestTag') # choose tag you want
        )
                      )
    )


# Put module in path:
process.p = cms.Path(process.AlCaRecoTriggerBitsRcdUpdate)


