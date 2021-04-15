import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('INFO')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source ("PoolSource", 
	duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
    	fileNames = cms.untracked.vstring(
	'file:/tmp/barvic/csc_00221766_Cosmic.root'
    )
)

process.cscdumper = cms.EDAnalyzer("CSCFileDumper",
    source = cms.InputTag("rawDataCollector"),
    output = cms.untracked.string("/tmp/barvic/raw_dump"),
#    events = cms.untracked.string("1073500,1166393")
)

process.p = cms.Path(process.cscdumper)

