import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMSegmentRECO")


### Input and Output Files
#######################################
process.maxEvents = cms.untracked.PSet( 
     input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_local_reco.root'
        # 'file:out_local_reco_5000evt.root'
        # 'file:out_local_reco_noise.root'
        # 'file:out_local_reco_alldigitized.root'
    ),
    # eventsToProcess = cms.untracked.VEventRange('1:1:2',) 
    # eventsToProcess = cms.untracked.VEventRange('1:1:23',) # debug evt 23 [run=1,ls=1,evt=23]
)
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_local_reco_gemsegment.root'
        # 'file:out_local_reco_gemsegment_5000evt.root'
        # 'file:out_local_reco_noise_gemsegment.root'
        # 'file:out_local_reco_test_gemsegment.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('gemsegment_step')
    )
)
#######################################


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
# process.MessageLogger.categories.append("GEMSegment")
# process.MessageLogger.categories.append("GEMSegmentBuilder")
# process.MessageLogger.categories.append("GEMSegAlgoPV")   
# process.MessageLogger.categories.append("GEMSegFit")      
# process.MessageLogger.categories.append("GEMSegFitMatrixDetails")      
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSegment             = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSegmentBuilder      = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSegAlgoPV           = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSegFit              = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMSegFitMatrixDetails = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)

### Paths and Schedules
#######################
process.gemsegment_step  = cms.Path(process.gemSegments)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)


process.schedule = cms.Schedule(
    process.gemsegment_step,
    process.endjob_step,
    process.out_step
)

