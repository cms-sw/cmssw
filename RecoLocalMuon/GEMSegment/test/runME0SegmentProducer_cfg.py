import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMSegmentRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2))
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
# process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2019_cff')
# process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# process.GlobalTag.globaltag = 'auto:upgrade2019'
# process.GlobalTag.globaltag = 'DES19_62_V7::All'
# process.GlobalTag.globaltag = 'POSTLS161_V12::All'
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
# from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


process.load('RecoLocalMuon.GEMSegment.me0Segments_cfi')

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
### to have a handle on the loops inside RPCSimSetup           
### I have split the LogDebug stream in several streams        
### that can be activated independentl                         
###############################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.junk = dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0Segment          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0SegmentBuilder   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0SegAlgoMM      = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    ME0SegFit         = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)

### Input and Output Files
##########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_local_reco.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_local_reco_me0segment.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('me0segment_step')
    )
)

### Paths and Schedules
#######################
process.me0segment_step  = cms.Path(process.me0Segments)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    process.me0segment_step,
    process.endjob_step,
    process.out_step
)

