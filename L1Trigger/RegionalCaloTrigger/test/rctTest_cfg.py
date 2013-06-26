# The following comments couldn't be translated into the new config version:

#    path p4 = {maskedRctInputDigis, rctDigis}
#    path p4 = {L1Emulator}
# Event output
#include "Configuration/EventContent/data/EventContent.cff"
#   module FEVT = PoolOutputModule 
#   { 
#     	using FEVTSIMEventContent
#	untracked string fileName = "rctTest.root"
#    	#untracked PSet dataset ={	
#        #	untracked string dataTier = "GEN-SIM-DIGI-RECO"
#    	#	}
#   }
#   endpath outpath = {FEVT}

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#	untracked PSet maxEvents = {untracked int32 input = 2}
#include "Configuration/ReleaseValidation/data/Services.cff"
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.Generator_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

#needed to prevent exception
process.load("MagneticField.Engine.uniformMagneticField_cfi")

#include "L1Trigger/RegionalCaloTrigger/data/maskedRctInputDigis.cfi"
process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")

#include "L1Trigger/RegionalCaloTrigger/data/L1RCTTestAnalyzer.cfi"
#replace L1RCTTestAnalyzer.showRegionSums = false
#include "Configuration/StandardSequences/data/L1Emulator.cff"
process.load("L1Trigger.RegionalCaloTrigger.rctDigis_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(64)
)
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('rctTest.root')
)

process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(35.001),
        #		untracked double MinPhi = 0.
        #		untracked double MaxPhi = 0.5
        MinPt = cms.untracked.double(34.999),
        # You can request more than one particle
        # since PartID is a vector, you can place in as many
        # PDG id's as you wish, comma-separated
        #
        #untracked vint32 PartID = {11,-11,13,-13,211,-211}         # e-e+mu-mu+pi+pi-
        #untracked vint32 PartID = {11,-11}         # electron, positron
        #untracked vint32 PartID = {211, -211}         # pi+ pi-
        PartID = cms.untracked.vint32(2112),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MinPhi = cms.untracked.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater) for printouts

    psethack = cms.string('single e pt 35'),
    firstRun = cms.untracked.uint32(1)
)

process.p0 = cms.Path(process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p4 = cms.Path(process.rctDigis)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p4)



