# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM,DIGI --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --eventcontent FEVTDEBUGHLT --era Run2_25ns --customise=SimMuon/GEMDigitizer/customizeGEMDigi.customize_digi_addGEM_muon_only,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions,SLHCUpgradeSimulations/Configuration/me0Customs.customize_Digi -n 100 --no_exec --fileout out_digi.root --python_filename SingleMuPt100_cfi_GEM-SIM-DIGI_Extended2015MuonGEMDev_cfg.py
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
process = cms.Process('DIGI',Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("EmptySource")
process.options = cms.untracked.PSet()

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('SingleMuPt100_cfi nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('out_digi.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

### TO ACTIVATE LogDebug NEED TO COMPILE IT WITH:                 
### --------------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"                
### Make sure that you first cleaned your CMSSW version:          
### --> scram b clean                                             
### before issuing the scram command above                        
### --------------------------------------------------------------
### LogDebug output goes to cout; all other output to "junk.log"  
### Code/Configuration with thanks to Tim Cox                     
##################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.files.junk = dict()
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    GEMGeometryBuilderFromDDD = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0GeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    RPCGeometryBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)
##################################################################

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    AddAntiParticle = cms.bool(True),
    PGunParameters = cms.PSet(
        MaxEta = cms.double(3.0),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(100.01),
        MinEta = cms.double(2.0),
        MinPhi = cms.double(-3.14159265359),
        MinPt = cms.double(99.99),
        PartID = cms.vint32(-13)
    ),
    Verbosity = cms.untracked.int32(0),
    firstRun = cms.untracked.uint32(1),
    psethack = cms.string('single mu pt 100')
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.digitisation_step = cms.Path(process.pdigi)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from SimMuon.GEMDigitizer.customizeGEMDigi
from SimMuon.GEMDigitizer.customizeGEMDigi import customize_digi_addGEM_muon_only 

#call to customisation function customize_digi_addGEM_muon_only imported from SimMuon.GEMDigitizer.customizeGEMDigi
process = customize_digi_addGEM_muon_only(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixRPCConditions 

#call to customisation function fixRPCConditions imported from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads
process = fixRPCConditions(process)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.me0Customs
from SLHCUpgradeSimulations.Configuration.me0Customs import customise_Digi

#call to customisation function customise_Digi imported from SLHCUpgradeSimulations.Configuration.me0Customs
process = customise_Digi(process)

# End of customisation functions

