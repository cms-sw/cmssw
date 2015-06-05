# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: SingleMuPt100_cfi -s GEN,SIM --conditions auto:run2_mc --magField 38T_PostLS1 --datatier GEN-SIM --geometry Extended2015Muon,Extended2015MuonReco --eventcontent FEVTDEBUGHLT --era Run2_25ns -n 100 --no_exec --fileout out_sim.root --python_filename SingleMuPt100_cfi_GEM-SIM_Extended2015Muon_cfg.py
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('SIM',eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2015MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015Muon_cff')
# for future releases: use GEMDev
# process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
# process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedNominalCollision2015_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("EmptySource")
process.options = cms.untracked.PSet()

### TO ACTIVATE LogVerbatim in Simulation Packages NEED TO:       
### --------------------------------------------------------------
### scram b disable-biglib                                        
### scram b clean
### scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"                    
###                                                               
### TO ACTIVATE LogTrace IN GEMRecHit NEED TO COMPILE IT WITH:
### --------------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"                
### Make sure that you first cleaned your CMSSW version:          
### --> scram b clean                                             
### before issuing the scram command above                        
### --------------------------------------------------------------
### !!! If you want to compile any CSC-related code with LogDebug ON, 
### you need to explicitly compile and build the CSCDetId package too,
### i.e. do first: git cms-addpkg DataFormats/MuonDetId.              
### This problem can occur at other places as well, so check carefully
### the compilation process when switching on the debug flags     
### --------------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"  
### Code/Configuration with thanks to Tim Cox                     
### --------------------------------------------------------------
### to have a handle on the loops inside RPCSimSetup              
### I have split the LogDebug stream in several streams           
### that can be activated independentl                            
##################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append("ME0GeometryBuilderfromDDD") # in Geometry/GEMGeometryBuilder
process.MessageLogger.categories.append("ME0NumberingScheme")        # in Geometry/MuonNumbering
process.MessageLogger.categories.append("MuonSimDebug")              # in SimG4CMS/Muon
process.MessageLogger.categories.append("MuonME0FrameRotation")      # in SimG4CMS/Muon
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0GeometryBuilderfromDDD = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0NumberingScheme        = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    MuonME0FrameRotation      = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    MuonSimDebug              = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)

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
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(1048576),
    fileName = cms.untracked.string('out_sim.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

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
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGHLToutput_step)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.generator * getattr(process,path)._seq 


