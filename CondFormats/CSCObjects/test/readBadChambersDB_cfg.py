# Read Bad Chambers values - Tim Cox - 05.05.2009
# I intend this to read from the standard cond data files, whatever they are.
# This is for CMSSW_31X

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_31X::All'

process.load("CalibMuon.Configuration.getCSCConditions_frontier_cff")

# Must change connect string in CalibMuon.Configuration.getCSCConditions_frontier_cff
# (and note that Final 31X will use //FrontierProd/CMS_COND_31X_CSC)
process.cscConditions.connect = cms.string('frontier://FrontierPrep/CMS_COND_31X_ALL')

process.cscConditions.toGet = cms.VPSet(
        cms.PSet(record = cms.string('CSCBadChambersRcd'),
                 tag = cms.string('CSCBadChambers_none_FiveLiveME42'))
)
process.es_prefer_cscConditions = cms.ESPrefer("PoolDBESSource","cscConditions")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.analyze = cms.EDAnalyzer("CSCReadBadChambersAnalyzer",
    outputToFile = cms.bool(True),
    readBadChambers = cms.bool(True),
    me42installed = cms.bool(True)                                 
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.analyze)
process.ep = cms.EndPath(process.printEventNumber)

