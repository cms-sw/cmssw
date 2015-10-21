# Auto generated configuration file
# using: 
# Revision: 1.20 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --filein file:JME-TP2023SHCALDR-00001_step1.root --mc --eventcontent FEVTDEBUG --customise SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023HGCalMuon,Configuration/DataProcessing/Utils.addMonitoring --datatier RECO --conditions PH2_1K_FB_V6::All --step RAW2DIGI,L1Reco,RECO --geometry Extended2023HGCalMuon,Extended2023HGCalMuonReco --python_filename HGCal_config_RECO.py --no_exec -n 50
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### TO ACTIVATE LogTrace IN GEMSegment NEED TO COMPILE IT WITH:
### -----------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"             
### Make sure that you first cleaned your CMSSW version:       
### --> scram b clean                                          
### before issuing the scram command above                     
### -----------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"
### Code/Configuration with thanks to Tim Cox                   
### -----------------------------------------------------------
###############################################################
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
### process.MessageLogger.categories.append("ME0Segment")
### process.MessageLogger.categories.append("ME0SegmentBuilder")
# process.MessageLogger.categories.append("ME0SegAlgoMM")   
### process.MessageLogger.categories.append("ME0SegFit")      
### process.MessageLogger.categories.append("ME0SegFitMatrixDetails")      
# process.MessageLogger.debugModules = cms.untracked.vstring("*")
# process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
# process.MessageLogger.cout = cms.untracked.PSet(
#     threshold = cms.untracked.string("DEBUG"),
#     default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
#     FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     # ME0Segment             = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     # ME0SegmentBuilder      = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     ME0SegAlgoMM             = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     # ME0SegFit              = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     # ME0SegFitMatrixDetails = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
# )
###############################################################

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    # fileNames = cms.untracked.vstring('file:JME-TP2023SHCALDR-00001_step1.root')
    # fileNames = cms.untracked.vstring('file:out_raw_182_1_pjB.root')
    # fileNames = cms.untracked.vstring('file:out_raw_198_1_kGQ.root')
    # fileNames = cms.untracked.vstring('file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RAW_500ps_amandeep_100.root')
    fileNames = cms.untracked.vstring('file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RAW_5ns_amandeep_100.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.20 $'),
    annotation = cms.untracked.string('step2 nevts:50'),
    name = cms.untracked.string('Applications')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('out_reco.root'),
    # fileName = cms.untracked.string('step2_RAW2DIGI_L1Reco_RECO.root'),
    # fileName = cms.untracked.string('DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_100_v2.root'),
    # fileName = cms.untracked.string('DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_116_v2.root'),
    # fileName = cms.untracked.string('DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_500ps_amandeep_Random.root'),
    # fileName = cms.untracked.string('DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_5ns_amandeep_Random.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('RECO')
    )
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag ### process.load done earlier ...
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V6::All', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step,process.FEVTDEBUGoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023HGCalMuon 

#call to customisation function cust_2023HGCalMuon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023HGCalMuon(process)

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions

# from RecoLocalMuon.GEMRecHit.me0Segments import me0Segments
process.load('RecoLocalMuon.GEMRecHit.me0Segments_cfi')
process.me0Segments.algo_pset.dTimeChainBoxMax = cms.double(1.0)
