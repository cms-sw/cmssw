import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("source_cfi")
process.load("m1a_cfi")
process.load("testout1_cfi")
process.load("e1_cfi")
process.load("p1_cfi")
process.load("maxEvents_cfi")
process.load("maxLuminosityBlocks_cfi")
process.load("options_cfi")
process.load("MessageLogger_cfi")
