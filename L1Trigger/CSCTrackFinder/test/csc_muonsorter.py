#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
		fileNames = cms.untracked.vstring(
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/64159D00-0878-DF11-A575-0026189438DE.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/6454B9B8-0878-DF11-A989-002618943923.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/66C3C49E-0A78-DF11-B7A9-0030486792A8.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/963C8A21-0A78-DF11-B9E8-003048678FA0.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/9A4492B1-0B78-DF11-8534-0018F3D096E0.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/9CE4D72A-0A78-DF11-BE15-003048678FFA.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/DEF4B715-0B78-DF11-B011-002618943953.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/E2788F9D-0A78-DF11-A762-001A9281171E.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/E47ABDA5-0C78-DF11-B5A4-001A92810AA4.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/E8F6EE9B-0A78-DF11-B0B3-002618943833.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/EA0C7D29-0978-DF11-8E76-003048678FF2.root',
			'/store/relval/CMSSW_3_6_3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V10-v1/0005/F25B3091-0A78-DF11-88CE-0026189438DE.root'
	),
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START36_V10::All'

# L1 Emulator
#############
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.joeTrackOut = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
process.joeTrackOut.SectorReceiverInput = 'simCscTriggerPrimitiveDigis:MPCSORTED'
process.joeTrackOut.DTproducer = 'simDtTriggerPrimitiveDigis'
#process.joeTrackOut.SectorProcessor.trigger_on_ME1a = True
#process.joeTrackOut.SectorProcessor.trigger_on_ME1b = True
#process.joeTrackOut.SectorProcessor.mindeta113_accp = 25
process.joeTrackOut.SectorProcessor.initializeFromPSet = True
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
process.joeOut = L1Trigger.CSCTrackFinder.csctfDigis_cfi.csctfDigis.clone()
process.joeOut.CSCTrackProducer = 'joeTrackOut'


# Analysis Module Definition
############################
process.effic = cms.EDAnalyzer("CSCTFEfficiencies",
	OutFile = cms.untracked.string("Validation.root")
)
	
process.FEVT = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("test.root"),
	outputCommands = cms.untracked.vstring(	
		"keep *"
	)
)
	
# Path Definition
#################
process.p = cms.Path(process.joeTrackOut*process.joeOut*process.effic)
#process.p = cms.Path(process.simCscTriggerPrimitiveDigis*process.simDtTriggerPrimitiveDigis*process.simCsctfTrackDigis*process.simCsctfDigis*process.effic)
#process.p = cms.Path(process.simCscTriggerPrimitiveDigis*process.simDtTriggerPrimitiveDigis*process.simCsctfTrackDigis*process.simCsctfDigis)
#process.outpath = cms.EndPath(process.FEVT)
#process.schedule = cms.Schedule(process.p, process.outpath)
