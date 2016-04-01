import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMSegmentRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
# process.load('Configuration.Geometry.GeometryExtended2019Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2019_cff')
# process.load('Configuration.Geometry.GeometryExtended2023Reco_cff')
# process.load('Configuration.Geometry.GeometryExtended2023_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
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

#process.load('RecoLocalMuon.GEMSegment.me0Segments_cfi')
process.load('RecoLocalMuon.GEMSegment.gemSegments_cfi')

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
#process.MessageLogger.categories.append("ME0Segment")
#process.MessageLogger.categories.append("ME0SegmentBuilder")
# process.MessageLogger.categories.append("ME0SegAlgoMM")   
# process.MessageLogger.categories.append("ME0SegFit")      
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    #ME0Segment          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    #ME0SegmentBuilder   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0SegAlgoMM      = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # ME0SegFit         = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)

### Input and Output Files
##########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #'file:/cms/scratch/jskim/cmssw/CMSSW_8_1_0_pre1/src/RecoLocalMuon/GEMRecHit/test/out_local_reco_GEMRechit_GE11_8and8.root'
        'file:/cms/scratch/jskim/cmssw/CMSSW_8_1_0_pre1/src/RecoLocalMuon/GEMRecHit/test/out_local_reco_GEMRechit_GE11_9and10.root'
    ),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    skipBadFiles = cms.untracked.bool(True),
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        #'out_local_reco_GEMSegment_GE11_8and8.root'
        'out_local_reco_GEMSegment_GE11_9and10.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('gemsegment_step')
    )
)

### Paths and Schedules
#######################
process.gemsegment_step  = cms.Path(process.gemSegments)
#process.me0segment_step  = cms.Path(process.me0Segments)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    #process.me0segment_step,
    process.gemsegment_step,
    process.endjob_step,
    process.out_step
)

# Automatic addition of the customisation function from Geometry.GEMGeometry.gemGeometryCustoms
#from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_8and8partitions_v2

#call to customisation function custom_GE11_8and8partitions_v2 imported from Geometry.GEMGeometry.gemGeometryCustoms
#process = custom_GE11_8and8partitions_v2(process)

# Automatic addition of the customisation function from Geometry.GEMGeometry.gemGeometryCustoms
from Geometry.GEMGeometry.gemGeometryCustoms import custom_GE11_9and10partitions_v1

#call to customisation function custom_GE11_9and10partitions_v1 imported from Geometry.GEMGeometry.gemGeometryCustoms
process = custom_GE11_9and10partitions_v1(process)
