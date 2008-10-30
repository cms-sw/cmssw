#
import FWCore.ParameterSet.Config as cms

process = cms.Process("RawDumper")

# process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('dumper'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.MessageLogger.cerr.threshold = 'Debug'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/0089689A-0A9E-DD11-ABC3-001D09F2512C.root')
)

#process.out = cms.OutputModule("PoolOutputModule",
#    fileName =  cms.untracked.string('file:histos.root')
#)


process.dumper = cms.EDAnalyzer("SiPixelRawDumper", 
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.untracked.bool(True),
    InputLabel = cms.untracked.string('source'),
#    InputLabel = cms.untracked.string('siPixelRawData'),
    CheckPixelOrder = cms.untracked.bool(False)
)

# process.s = cms.Sequence(process.dumper)

process.p = cms.Path(process.dumper)

# process.ep = cms.EndPath(process.out)


