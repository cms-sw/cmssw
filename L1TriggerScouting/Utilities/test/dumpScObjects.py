import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing ('analysis')

options.register ('numOrbits',
                  -1,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "Number of orbits to process")

options.register ('filePath',
                  "file:/dev/shm/PoolOutputTest.root",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Sub lumisection number to process")

options.parseArguments()

process = cms.Process( "DUMP" )


process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(options.numOrbits)
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(options.filePath)
)

process.dump = cms.EDAnalyzer("DumpScObjects",
  gmtMuonsTag      = cms.InputTag("GmtUnpacker", "", "SCPU"),
  caloJetsTag      = cms.InputTag("CaloUnpacker", "", "SCPU"),
  caloEGammasTag   = cms.InputTag("CaloUnpacker", "", "SCPU"),
  caloTausTag      = cms.InputTag("CaloUnpacker", "", "SCPU"),
  caloEtSumsTag    = cms.InputTag("CaloUnpacker", "", "SCPU"),
  minBx            = cms.untracked.uint32(0),
  maxBx            = cms.untracked.uint32(3564),

  skipEmptyBx      = cms.untracked.bool(True), # don't show empty BX

  #checkMuons       = cms.untracked.bool(False), # test removing a collection

  searchEvent      = cms.untracked.bool(True),
  orbitNumber      = cms.untracked.uint32(88981531),
  searchStartBx    = cms.untracked.uint32(1027-2),
  searchStopBx     = cms.untracked.uint32(1027+2),
)

process.p = cms.Path(
  process.dump
)