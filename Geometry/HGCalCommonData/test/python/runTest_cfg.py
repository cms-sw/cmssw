import FWCore.ParameterSet.Config as cms

process = cms.Process('SIM')

# import of standard configurations
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Geometry.HGCalCommonData.testTB181V1XML_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedFlat_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBAnalyzer_cfi')
process.load('SimG4CMS.HGCalTestBeam.HGCalTBCheckGunPosition_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('HGCalGeom')
    process.MessageLogger.categories.append('SimG4CoreGeometry')
    process.MessageLogger.categories.append('HGCSim')

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleMuonE200_cfi nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Additional output definition
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('TBGenSim.root')
)

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomEThetaGunProducer",
    AddAntiParticle = cms.bool(False),
    PGunParameters = cms.PSet(
        MinE = cms.double(99.99),
        MaxE = cms.double(100.01),
        MinTheta = cms.double(0.0),
        MaxTheta = cms.double(0.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        PartID = cms.vint32(211)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single pion E 100')
)

process.VtxSmeared.MinZ                 = -800.0
process.VtxSmeared.MaxZ                 = -800.0
process.VtxSmeared.MinX                 = -7.5
process.VtxSmeared.MaxX                 =  7.5
process.VtxSmeared.MinY                 = -7.5
process.VtxSmeared.MaxY                 =  7.5
process.g4SimHits.HGCSD.RejectMouseBite = True
process.g4SimHits.HGCSD.RotatedWafer    = True
process.HGCalTBAnalyzer.doDigis         = False
process.HGCalTBAnalyzer.doRecHits       = False
process.HGCalTBAnalyzer.useFH           = True
process.HGCalTBAnalyzer.useEE           = False
process.HGCalTBAnalyzer.useBeam         = False
process.HGCalTBAnalyzer.zFrontEE        = 1110.0
process.HGCalTBAnalyzer.zFrontFH        = 1176.5
process.HGCalTBAnalyzer.zFrontFH        = 1307.5
process.HGCalTBAnalyzer.maxDepth        = 39
process.HGCalTBAnalyzer.deltaZ          = 26.2
process.HGCalTBAnalyzer.zFirst          = 22.8

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.gunfilter_step  = cms.Path(process.HGCalTBCheckGunPostion)
process.analysis_step = cms.Path(process.HGCalTBAnalyzer)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
                                process.simulation_step,
                                process.gunfilter_step,
                                process.analysis_step,
                                process.endjob_step
)

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq 
