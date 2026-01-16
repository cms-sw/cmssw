import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("DTVDriftAnalyzer",eras.Run3)

process.load("CondCore.CondDB.CondDB_cfi")

import FWCore.ParameterSet.Config as cms
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True

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
                                          #readLegacyVDriftDB =cms.bool(False),
)

process.p = cms.Path(process.dtVDriftAnalyzer)
