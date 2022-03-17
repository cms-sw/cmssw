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

# Output definition

# load HGCAL TPG simulation
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

# Eventually modify default geometry parameters
from L1Trigger.L1THGCal.customTriggerGeometry import custom_geometry_V11_Imp3
process = custom_geometry_V11_Imp3(process)

# Fetch stage 1 truncation parameters
from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import stage1truncation_proc

# ordered u/v coordinated of TCs in a module, for consistent TC index definition
ordered_tcu = [4, 5, 4, 3, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 7, 6, 6, 7, 5, 4, 4, 5, 5, 4, 4, 5, 7, 6, 6, 7, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
ordered_tcv = [0, 1, 1, 0, 2, 3, 3, 2, 2, 3, 3, 2, 0, 1, 1, 0, 7, 7, 6, 6, 7, 7, 6, 6, 5, 5, 4, 4, 5, 5, 4, 4, 3, 2, 3, 4, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6]

# ID of tested FPGA
fpga_id = 1

# geometry version
from re import match
geomXMLcontent = process.XMLIdealGeometryESSource.geomXMLFiles.value()
hgcal_xml = 'Geometry/HGCalCommonData/data/hgcal/v.*/hgcal.xml'
geometry_version = list(filter(lambda v: match(hgcal_xml,v), geomXMLcontent))
if len(geometry_version) != 1:
    raise ValueError('Can\'t find geometry version - expected one xml file, but got {}'.format(len(geometry_version)))

geometry_version = geometry_version[0].split('/')[4]

process.hgcalbackendstage1parameterextractor = cms.EDAnalyzer(
    "HGCalBackendStage1ParameterExtractor",
    outJSONname = cms.string("geometryConfig_backendStage1.json"),
    testedFpga = cms.int32(fpga_id),
    BackendStage1Params = stage1truncation_proc.truncation_parameters,
    TriggerGeometryParam = process.hgcalTriggerGeometryESProducer.TriggerGeometry,
    TCcoord_UV = cms.PSet(
        TCu = cms.vuint32(ordered_tcu),
        TCv = cms.vuint32(ordered_tcv)
    ),
    MetaData = cms.PSet(
        geometryVersion = cms.string(geometry_version)
    ),
)

# Path and EndPath definitions
process.test_step = cms.Path(process.hgcalbackendstage1parameterextractor)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.test_step,process.endjob_step)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
