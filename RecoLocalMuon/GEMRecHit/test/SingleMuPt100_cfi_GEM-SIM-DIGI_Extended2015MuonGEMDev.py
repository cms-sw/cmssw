# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM,DIGI --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM-DIGI --geometry Extended2015MuonGEMDev,Extended2015MuonGEMDevReco --eventcontent FEVTDEBUGHLT --era Run2_25ns --customise=SimMuon/GEMDigitizer/customizeGEMDigi.customize_digi_addGEM_muon_only,SLHCUpgradeSimulations/Configuration/fixMissingUpgradeGTPayloads.fixRPCConditions,SLHCUpgradeSimulations/Configuration/me0Customs.customize_Digi -n 100 --no_exec --fileout out_digi.root --python_filename SingleMuPt100_cfi_GEM-SIM-DIGI_Extended2015MuonGEMDev_cfg.py
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('DIGI',eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    # input = cms.untracked.int32(3)
    # input = cms.untracked.int32(5000)
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
    # fileName = cms.untracked.string('out_digi_5000evt.root'),
    # fileName = cms.untracked.string('out_digi_noise.root'),
    # fileName = cms.untracked.string('out_digi_alldigitized.root'),
    # fileName = cms.untracked.string('out_digi_test.root'),
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
# process.MessageLogger.categories.append("GEMGeometryBuilderFromDDD")
# process.MessageLogger.categories.append("ME0GeometryBuilderFromDDD")
# process.MessageLogger.categories.append("RPCGeometryBuilderFromDDD")
# process.MessageLogger.categories.append("GEMDigiProducer")
# process.MessageLogger.categories.append("GEMSimpleModel")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMGeometryBuilderFromDDD = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0GeometryBuilderFromDDD = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # RPCGeometryBuilderFromDDD = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMDigiProducer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSimpleModel = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    
)
##################################################################

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    AddAntiParticle = cms.bool(True),
    PGunParameters = cms.PSet(
        MaxEta = cms.double(2.45),
        MaxPhi = cms.double(3.14159265359),
        MaxPt = cms.double(100.01),
        MinEta = cms.double(1.65),
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

# Manual customization to switch off background hits
process.simMuonGEMDigis.digitizeOnlyMuons      = cms.bool(True)  # default: false
process.simMuonGEMDigis.doBkgNoise             = cms.bool(False) # default: true    # comment Roumyana: #False == No background simulation
process.simMuonGEMDigis.doNoiseCLS             = cms.bool(False) # default: true
# process.simMuonGEMDigis.simulateIntrinsicNoise = cms.bool(False) # default: false
# process.simMuonGEMDigis.simulateElectronBkg    = cms.bool(True), # default: true    # comment Roumyana: #False=simulate only neutral Bkg
# process.simMuonGEMDigis.simulateLowNeutralRate = cms.bool(False) # default: true    # comment Roumyana: #True=neutral_Bkg at L=1x10^{34}, False at L=5x10^{34}cm^{-2}s^{-1}

# End of customisation functions

