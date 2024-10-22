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
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    debugModules = cms.untracked.vstring('l1GtUnpack')
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

