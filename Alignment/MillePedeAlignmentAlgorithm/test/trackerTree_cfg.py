from __future__ import print_function
##########################################################################
# Creates the TrackerTree.root file.
# Configuration file from TrackerAlignment/test/trackerTreeGenerator_cfg.py
##

import FWCore.ParameterSet.Config as cms
import os

from FWCore.ParameterSet.VarParsing import VarParsing

# argument parsing
options = VarParsing ("analysis")
options.register("globalTag",
                 "auto:phase1_2017_design",
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "Global tag -> provides tracker geometry")
options.register("firstRun",
                 1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "run to define tracker-geometry IOV")
options.parseArguments()

# Process definition
process = cms.Process("TrackerTreeGeneration")

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.TrackerTreeGenerator=dict()
process.MessageLogger.cerr.INFO.limit = 0
process.MessageLogger.cerr.default.limit = -1
process.MessageLogger.cerr.TrackerTreeGenerator = cms.untracked.PSet(limit = cms.untracked.int32(-1))

# Process options
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Input source
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.firstRun))

# Number of events
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Geometry
process.load("Configuration.Geometry.GeometryRecoDB_cff")

# Conditions
# use always ideal conditions to get no influence from Alignment on absolute Positions, Orientations...
# so it is clear that when choosing special regions in e.g. globalPhi, Modules of the same Rod are contained in the same region
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag)
print("Using global tag:", process.GlobalTag.globaltag.value())

# Analyzer
process.load("Alignment.TrackerAlignment.TrackerTreeGenerator_cfi")

# Output File Configuration
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

# Path
process.p = cms.Path(process.TrackerTreeGenerator)
