import FWCore.ParameterSet.Config as cms

process = cms.Process("MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")

# max events has to match what is in the .dat file:
# fwd_iso_scan: 10, iso_test: 20, many_events: 91
FILENAME = "iso_test"
n_events_dict = {"fwd_iso_scan":10, "iso_test":21, "many_events":91}
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(n_events_dict[FILENAME]))

process.load("L1Trigger.L1TMuon.microgmtinputproducer_cfi")
process.load("L1Trigger.L1TMuon.microgmtemulator_cfi")

process.MicroGMTInputProducer.inputFileName = "patterns/{fname}.dat".format(fname=FILENAME)

process.source = cms.Source("EmptySource",)


process.out = cms.OutputModule("PoolOutputModule",
                               fileName=cms.untracked.string('microgmt_{fname}.root'.format(fname=FILENAME))
)

process.p = cms.Path(process.MicroGMTInputProducer+process.microGMTEmulator)

process.e = cms.EndPath(process.out)
