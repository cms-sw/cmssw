import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("DTVDriftAnalyzer",eras.Run3)

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.dtVDriftAnalyzer = cms.EDAnalyzer("DTVDriftAnalyzer",
    rootFileName = cms.untracked.string(''),
    readLegacyVDriftDB =cms.bool(True),
)

process.p = cms.Path(process.dtVDriftAnalyzer)
# foo bar baz
# YJCA8S4wZF2f2
# JHmOOb6UQdQYI
