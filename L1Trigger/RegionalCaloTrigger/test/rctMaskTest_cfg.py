import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Generator_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

#needed to prevent exception
process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.load("L1Trigger.RegionalCaloTrigger.L1RCTTestAnalyzer_cfi")

#include "Configuration/StandardSequences/data/L1Emulator.cff"
process.load("L1Trigger.RegionalCaloTrigger.rctDigis_cfi")

#replace simRctDigis.ecalDigisLabel = maskedRctInputDigis
#replace simRctDigis.hcalDigisLabel = maskedRctInputDigis
process.load("L1TriggerConfig.L1ScalesProducers.L1CaloScalesConfig_cff")

process.load("L1TriggerConfig.L1ScalesProducers.L1CaloInputScalesConfig_cff")

process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")

#    path p4 = {L1Emulator}
# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
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
    fileName = cms.string('rct.root')
)

process.source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(35.001),
        MinPt = cms.untracked.double(34.999),
        # You can request more than one particle
        # since PartID is a vector, you can place in as many
        # PDG id's as you wish, comma-separated
        #
        PartID = cms.untracked.vint32(11, 11, 11, 11, -11, 
            -11, -11, -11),
        MaxEta = cms.untracked.double(2.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-2.5),
        MinPhi = cms.untracked.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater) for printouts

    psethack = cms.string('single e pt 35'),
    firstRun = cms.untracked.uint32(1)
)

process.maskedRctInputDigis = cms.EDProducer("MaskedRctInputDigiProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    maskFile = cms.FileInPath('L1Trigger/RegionalCaloTrigger/test/data/testMaskOutEtaMinus.txt'),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis")
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO')
    ),
    fileName = cms.untracked.string('rctTest.root')
)

process.p0 = cms.Path(process.pgen)
process.p1 = cms.Path(process.psim)
process.p2 = cms.Path(process.pdigi)
process.p4 = cms.Path(process.maskedRctInputDigis*process.rctDigis)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p0,process.p1,process.p2,process.p4)

process.L1RCTTestAnalyzer.showRegionSums = False
process.rctDigis.ecalDigisLabel = 'maskedRctInputDigis'
process.rctDigis.hcalDigisLabel = 'maskedRctInputDigis'


