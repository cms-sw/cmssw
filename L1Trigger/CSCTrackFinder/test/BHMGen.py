#!/usr/bin/env pythona
import FWCore.ParameterSet.Config as cms

# process name
##############
process = cms.Process("Rec")

# number of events
##################
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# services
##########
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# halo source
#############
process.load("L1Trigger.CSCTrackFinder.BHMParam_cfi")
#source = cms.Source("BeamHaloSource",
#	GENMOD = cms.untracked.int32(1),
#	LHC_B1 = cms.untracked.int32(1),
#	LHC_B2 = cms.untracked.int32(0),
#	IW_MUO = cms.untracked.int32(1),
#	IW_HAD = cms.untracked.int32(0),
#	EG_MIN = cms.untracked.double(10.0),
#	EG_MAX = cms.untracked.double(5000.0),
#	shift_bx = cms.untracked.int32(0),
#	BXNX = cms.untracked.double(25.0)
#)		

# standard sequences
####################
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.VtxSmearedBetafuncEarlyCollision_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")

# Geometry & Magnetic Field
###########################
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

# Digitization
##############
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

# L1Emulator
############
process.load("L1Trigger.Configuration.L1Emulator_cff")
#process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")

# Reconstruction
################
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi")
process.load("RecoMuon.CosmicMuonProducer.cosmicMuons_cff")

# Output Module Definition
##########################
process.FEVT = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("BeamHaloMuGenToRec.root"),
	basketSize = cms.untracked.int32(4096),
	outputCommands = cms.untracked.vstring(
		"keep *"
#		"drop *",
#	    	"keep *_muonCSCDigis_*_*",
#	    	"keep *_csc2DRecHits_*_*",
#	    	"keep *_cscSegments_*_*",
#	    	"keep *_cosmicMuons_*_*",
#	    	"keep *_cosmicMuonsEndCapsOnly_*_*",
#	    	"keep *_*_MuonCSCHits_*",
#	    	"keep SimTracks_*_*_*",
#		"keep edmHepMCProduct_*_*_*",
#	 	"keep CrossingFrame_*_*_*"
	)
)

# Schedule
##########
process.p1 = cms.Path(process.pgen)
process.p2 = cms.Path(process.psim)
#process.p3 = cms.Path(process.mix*process.doAllDigi*process.trackingParticles)
process.p3 = cms.Path(process.pdigi)
process.p4 = cms.Path(process.L1Emulator)
process.p5 = cms.Path(process.muonlocalreco*process.CosmicMuonSeed*process.cosmicMuons)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.p2,process.p3,process.p4,process.p5,process.outpath)
