import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('SIM',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
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
    annotation = cms.untracked.string('SingleElectronPt10_cfi nevts:10'),
    name = cms.untracked.string('Applications')
)

# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("test_triggergeom.root")
    )

MessageLogger = cms.Service("MessageLogger")

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

process.endjob_step = cms.EndPath(process.endOfProcess)

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
# Eventually modify default geometry parameters
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V11_Imp3
process = custom_geometry_V11_Imp3(process)

tester_cells = cms.PSet(
    TesterName = cms.string('HGCalTriggerGeoTesterCells')
)
tester_triggercells = cms.PSet(
    TesterName = cms.string('HGCalTriggerGeoTesterTriggerCells')
)
tester_modules = cms.PSet(
    TesterName = cms.string('HGCalTriggerGeoTesterModules')
)

tester_stage1 = cms.PSet(
    TesterName = cms.string('HGCalTriggerGeoTesterBackendStage1')
)

tester_stage2 = cms.PSet(
    TesterName = cms.string('HGCalTriggerGeoTesterBackendStage2')
)

process.L1THGCaltriggergeomtester = cms.EDAnalyzer(
    "HGCalTriggerGeoTesterManager", 
    Testers = cms.VPSet(
        tester_cells,
        tester_triggercells,
        tester_modules,
        tester_stage1,
        tester_stage2,
    )
)
process.test_step = cms.Path(process.L1THGCaltriggergeomtester)

# Schedule definition
process.schedule = cms.Schedule(process.test_step,process.endjob_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
