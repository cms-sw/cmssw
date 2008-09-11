
import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDTRaw")

# the source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/data/Commissioning08/BeamHalo/RECO/StuffAlmostToP5_v1/000/061/642/10A0FE34-A67D-DD11-AD05-000423D94E1C.root'
    ),
                            skipEvents = cms.untracked.int32(0)
                            )

# process.source = cms.Source("NewEventStreamFileReader",
#                             fileNames = cms.untracked.vstring(
#     'file:/directory/pippo.dat'
#     ))

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
                                    )

process.dump = cms.EDAnalyzer("DumpFEDRawDataProduct",
                              feds = cms.untracked.vint32(770, 771, 772, 773, 774, 775),
                              dumpPayload = cms.untracked.bool(True)
                              )


process.dtRawDump = cms.Path(process.dump)
