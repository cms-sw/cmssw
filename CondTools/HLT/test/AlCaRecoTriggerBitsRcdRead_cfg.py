import FWCore.ParameterSet.Config as cms

process = cms.Process("WRITEDB")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1)
    ))

# the module writing to DB
process.load("CondTools.HLT.AlCaRecoTriggerBitsRcdRead_cfi")
# 'twiki' is default - others are text, python (future: html?)
process.AlCaRecoTriggerBitsRcdRead.outputType = 'twiki'

# No data, but have to specify run number (default is 1):
process.source = cms.Source("EmptySource",
                            numberEventsInRun = cms.untracked.uint32(1),
                            #firstRun = cms.untracked.uint32(5)
                            )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Input for AlCaRecoTriggerBitsRcd,
# either via GloblalTag (use of _cfi instead of _cff sufficient and faster):
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cfi")
process.GlobalTag.globaltag = "MC_3XY_V14::All" # choose your tag

# ...or specify database and tag:  
#import CondCore.DBCommon.CondDBSetup_cfi
#process.dbInput = cms.ESSource(
#    "PoolDBESSource",
#    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
#    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('AlCaRecoTriggerBitsRcd'),
#        tag = cms.string('AlCaRecoHLTpaths8e29_1e31_v2') # choose tag you want
#        )
#                      )
#    )

# Put module in path:
process.p = cms.Path(process.AlCaRecoTriggerBitsRcdRead)
