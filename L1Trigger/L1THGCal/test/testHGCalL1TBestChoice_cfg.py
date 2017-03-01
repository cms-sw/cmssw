import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('DIGI',eras.Phase2C2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D4_cff')
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


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('SingleElectronPt50_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Output definition
process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    #outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    outputCommands = cms.untracked.vstring(
        'keep *_*_HGCHitsEE_*',
        'keep *_*_HGCHitsHEback_*',
        'keep *_*_HGCHitsHEfront_*',
        'keep *_mix_*_*',
        'keep *_genParticles_*_*',
        'keep *_hgcalTriggerPrimitiveDigiProducer_*_*',
        'keep *_hgcalTriggerPrimitiveDigiFEReproducer_*_*'
    ),
    fileName = cms.untracked.string('file:junk.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("test_bestchoice.root")
    )



# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MinPt = cms.double(49.99),
        MaxPt = cms.double(50.01),
        PartID = cms.vint32(11),
        MinEta = cms.double(1.5),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single electron pt 50'),
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
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
## define trigger emulator without trigger cell selection
process.hgcalTriggerPrimitiveDigiProducer.FECodec.NData = cms.uint32(999) # put number larger than max number of trigger cells in module
cluster_algo_all =  cms.PSet( AlgorithmName = cms.string('SingleCellClusterAlgoBestChoice'),
                              FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec,
                              HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                              HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),
                              calib_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters
                                 )
process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms = cms.VPSet( cluster_algo_all )
process.hgcl1tpg_step1 = cms.Path(process.hgcalTriggerPrimitives)

## define trigger emulator with trigger cell selection
process.hgcalTriggerPrimitiveDigiFEReproducer.FECodec.triggerCellTruncationBits = cms.uint32(0)
cluster_algo_select =  cms.PSet( AlgorithmName = cms.string('SingleCellClusterAlgo'),
                                 FECodec = process.hgcalTriggerPrimitiveDigiFEReproducer.FECodec,
                                 HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                                 HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),
                                 calib_parameters = process.hgcalTriggerPrimitiveDigiProducer.BEConfiguration.algorithms[0].calib_parameters

                                 )
process.hgcalTriggerPrimitiveDigiFEReproducer.BEConfiguration.algorithms = cms.VPSet( cluster_algo_select )
process.hgcl1tpg_step2 = cms.Path(process.hgcalTriggerPrimitives_reproduce)

# define best choice tester
process.hgcaltriggerbestchoicetester = cms.EDAnalyzer(
    "HGCalTriggerBestChoiceTester",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    #bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    isSimhitComp = cms.bool(True),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    #bhSimHits = cms.InputTag('g4SimHits:HGCHitsHEback'),
    beClustersAll = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:SingleCellClusterAlgo'),
    beClustersSelect = cms.InputTag('hgcalTriggerPrimitiveDigiFEReproducer:SingleCellClusterAlgo'),
    FECodec = process.hgcalTriggerPrimitiveDigiProducer.FECodec.clone()
    )
process.test_step = cms.Path(process.hgcaltriggerbestchoicetester)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.hgcl1tpg_step1,process.hgcl1tpg_step2,process.digi2raw_step,process.test_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
