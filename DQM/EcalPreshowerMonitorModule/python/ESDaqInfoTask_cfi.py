import FWCore.ParameterSet.Config as cms

ecalPreshowerDaqInfoTask = cms.EDAnalyzer("ESDaqInfoTask",
      esMapping = cms.PSet(LookupTable = cms.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat")),
      prefixME = cms.untracked.string('EcalPreshower'),
      mergeRuns = cms.untracked.bool(False),
      ESFedRangeMin = cms.untracked.int32(520),
      ESFedRangeMax = cms.untracked.int32(575)
)

