import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

chamberEfficiencyTest = DQMEDHarvester("DTChamberEfficiencyTest",
    runningStandalone = cms.untracked.bool(True),
    #Names of the quality tests: they must match those specified in "qtList"
    XEfficiencyTestName = cms.untracked.string('ChEfficiencyInRangeX'),
    YEfficiencyTestName = cms.untracked.string('ChEfficiencyInRangeY'),
    folderRoot = cms.untracked.string(''),
    debug = cms.untracked.bool(False),
    diagnosticPrescale = cms.untracked.int32(1)
)


