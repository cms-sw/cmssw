#!/usr/bin/env pythonma
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

# Event Setup
#############
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_3XY_V10::All'

# Module Definition for Detector Simulation
###########################################
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt  = cms.double(100.01),
        MinPt  = cms.double(1.99),
        PartID = cms.vint32(13),
        MaxEta = cms.double(0.9),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(2.5),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
    AddAntiParticle = cms.bool(False)
)

#from L1Trigger.Configuration.L1Trigger_EventContent_cff import *

process.FEVT = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("SingleMuPtHigh_EtaOverlap.root"),
	outputCommands = cms.untracked.vstring(
		"drop *",
		"keep SimTracks_*_*_*",
		"keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_*_MPCSORTED_*",
		"keep L1MuDTChambPhContainer_*_*_*"
	)	
)

# Run Path Definition
#####################
process.p = cms.Path(process.generator*process.pgen*process.psim*process.pdigi)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p,process.outpath)
