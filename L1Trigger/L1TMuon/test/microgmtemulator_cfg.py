import FWCore.ParameterSet.Config as cms

process = cms.Process("MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
            # 'file:test/fwd_iso_scan.root'
            # 'file:test/ttbar_large_sample.root'
            # 'file:test/many_events.root'
            'file:microgmt_input_many_events.root'
    )
)

process.load("L1Trigger.L1TMuon.microgmtemulator_cfi")

process.out = cms.OutputModule("PoolOutputModule",
    # fileName = cms.untracked.string('fwd_iso_scan.root')
    # fileName = cms.untracked.string('ttbar_small_sample.root')
    # fileName = cms.untracked.string('many_events.root')
    fileName = cms.untracked.string('microgmt_out_iso_test.root')
)

#process.content = cms.EDAnalyzer("EventContentAnalyzer")
process.p = cms.Path(process.microGMTEmulator)

process.e = cms.EndPath(process.out)
