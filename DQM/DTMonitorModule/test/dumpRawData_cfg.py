
import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDTRaw")

# the source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    'file:/data/c/cerminar/data/GlobalRun/run61642_BeamSplash/Run61642_EventNumberSkim_RAW.root'
    ),
                            skipEvents = cms.untracked.uint32(0) )


# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.dump = cms.EDAnalyzer("DumpFEDRawDataProduct",
                              feds = cms.untracked.vint32(770, 771, 772, 773, 774, 775),
                              dumpPayload = cms.untracked.bool(True)
                              )


process.dtRawDump = cms.Path(process.dump)
