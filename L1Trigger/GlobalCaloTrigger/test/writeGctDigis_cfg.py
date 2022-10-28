# writeGctDigis_cfg.py
#
# J Brooke 9/07/07
#
# Python translation 15/07/08
#
# Create RCT/GCT digis from trigger primitive digis and write to file
# Recommended for use with a RelVal file as input
#

import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
 
# Geometry
from Configuration.StandardSequences.MagneticField_cff import *
from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.GeometryDB_cff import *

# include L1 emulator
from L1Trigger.Configuration.L1CaloEmulator_cff import *

# Input data source
process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring("file:single_e_pt35.root")
  )

maxEvents = cms.untracked.PSet ( input = cms.untracked.int32(10) )

test = cms.EDAnalyzer("L1GctTestAnalyzer", 
    rawInput     = cms.untracked.InputTag("none"),
    emuInput     = cms.untracked.InputTag("gctDigis"),
    outFile      = cms.untracked.string("writeGctDigis.txt"),
    doHardware   = cms.untracked.bool(False),
    doEmulated   = cms.untracked.bool(True),
    doRctEm      = cms.untracked.bool(False),
    doInternEm   = cms.untracked.bool(False),
    doEm         = cms.untracked.bool(True),
    doJets       = cms.untracked.bool(False),
    rctEmMinRank = cms.untracked.uint32(0)
  )

# write out Root file
outputEvents = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring (
      "drop *",
      "keep *_ecalTriggerPrimitiveDigis_*_*",
      "keep *_hcalTriggerPrimitiveDigis_*_*",
      "keep *_rctDigis_*_*",
      "keep *_gctDigis_*_*"
      ),
      fileName = cms.untracked.string("writeGctDigis.root")
  )

process.p = cms.Path(L1CaloEmulator*test)

process.outpath = cms.EndPath(outputEvents)



