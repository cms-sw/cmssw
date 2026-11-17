import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('DIGI',Phase2C17I13M9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110_cff')
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

# Output definition


# Additional output definition
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("test_triggergeom.root")
    )



# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T35', '')

process.endjob_step = cms.EndPath(process.endOfProcess)

process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')
# Eventually modify default geometry parameters
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V16_Imp1
process = custom_geometry_V16_Imp1(process)

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
