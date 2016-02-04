import FWCore.ParameterSet.Config as cms

process = cms.Process("GMTTEST")
# load and configure modules
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtTextToRaw_cfi")

# FED Id (default 813)
#replace l1GtTextToRaw.FedId = 813
# FED raw data size (in 8bits units, including header and trailer)
# If negative value, the size is retrieved from the trailer.        
#replace l1GtTextToRaw.RawDataSize = 872
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtUnpack_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('l1GtUnpack'), ## DEBUG mode

    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'), ## DEBUG mode

        DEBUG = cms.untracked.PSet( ## DEBUG mode, all messages

            limit = cms.untracked.int32(-1)
        ),
        #          untracked PSet DEBUG = { untracked int32 limit = 10}  // DEBUG mode, max 10 messages
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(10)
)

process.dump = cms.EDAnalyzer("L1MuGMTDump",
    GMTInputTag = cms.untracked.InputTag("l1GtUnpack")
)

process.p = cms.Path(process.l1GtTextToRaw*process.l1GtUnpack*process.dump)
process.l1GtTextToRaw.TextFileName = 'testGt_DumpSpyToText_output.txt'
process.l1GtUnpack.DaqGtInputTag = 'l1GtTextToRaw'
process.l1GtUnpack.ActiveBoardsMask = 0x0101

