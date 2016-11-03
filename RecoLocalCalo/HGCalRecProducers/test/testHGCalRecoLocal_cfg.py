import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("testHGCalRecoLocal",eras.Phase2C2_timing)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D3_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### setup HGCal local reco
# get uncalibrechits with weights method
process.load("RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi")
process.HGCalUncalibRecHit.HGCEEdigiCollection  = 'mix:HGCDigisEE'
process.HGCalUncalibRecHit.HGCHEFdigiCollection = 'mix:HGCDigisHEfront'
process.HGCalUncalibRecHit.HGCHEBdigiCollection = 'mix:HGCDigisHEback'

# get rechits e.g. from the weights
process.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
process.HGCalRecHit.HGCEEuncalibRecHitCollection  = 'HGCalUncalibRecHit:HGCEEUncalibRecHits'
process.HGCalRecHit.HGCHEFuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEFUncalibRecHits'
process.HGCalRecHit.HGCHEBuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEBUncalibRecHits'

#PF Rec Hit
process.load("RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cff") 


process.HGCalRecoLocal = cms.Sequence(process.HGCalUncalibRecHit +
                                      process.HGCalRecHit)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_mix_*_*',
        'keep *_HGCalUncalibRecHit_*_*',
        'keep *_HGCalRecHit_*_*',
        'keep *_particleFlowRecHitHGC_*_*'
        ),
    fileName = cms.untracked.string('file:testHGCalLocalReco.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.0),
        MinPt = cms.double(35.0),
        PartID = cms.vint32(11),
        MaxEta = cms.double(2.9),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.6),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single electron pt 35'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

process.mix.digitizers = cms.PSet(process.theDigitizersValid)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.recotest_step = cms.Path(process.HGCalRecoLocal+process.particleFlowRecHitHGC)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.recotest_step,process.out_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

# customisation of the process.
# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforFullSim

#call to customisation function customizeHLTforFullSim imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforFullSim(process)

# End of customisation functions
