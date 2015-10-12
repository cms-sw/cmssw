import FWCore.ParameterSet.Config as cms

process = cms.Process("MyMuonRECO")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

# Process options :: 
# - wantSummary helps to understand which module crashes
# - skipEvent skips event in case a product was not found
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True), 
                                      # SkipEvent = cms.untracked.vstring('ProductNotFound') 
                                    )



process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

# Ideal geometry, needed for transient ECAL alignement
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Geometry.CMSCommonData.cmsExtendedGeometry2015MuonGEMDevXML_cfi')
process.load('Geometry.GEMGeometryBuilder.gemGeometry_cfi')
process.load('Geometry.GEMGeometryBuilder.me0Geometry_cfi')
process.load('Geometry.RPCGeometryBuilder.rpcGeometry_cfi')
process.load('Geometry.MuonNumbering.muonNumberingInitialization_cfi')

process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")               
process.load("Configuration.StandardSequences.MagneticField_cff")                # recommended configuration


process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


### Try to do RecoLocalMuon on all muon detectors ###
#####################################################
process.load('RecoLocalMuon.GEMRecHit.gemRecHits_cfi')
#process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')
process.load('RecoLocalMuon.GEMRecHit.me0LocalReco_cff')
process.load("RecoLocalMuon.Configuration.RecoLocalMuon_cff")
### me0Muon reco now
process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')


# Fix DT and CSC Alignment #
############################
# does this work actually?
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
process = fixDTAlignmentConditions(process)
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
process = fixCSCAlignmentConditions(process)


# Skip Digi2Raw and Raw2Digi steps for Al Muon detectors #
##########################################################
process.gemRecHits.gemDigiLabel = cms.InputTag("simMuonGEMDigis","","GEMDIGI")
process.rpcRecHits.rpcDigiLabel = cms.InputTag('simMuonRPCDigis')
process.csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
process.dt1DRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")
process.dt1DCosmicRecHits.dtDigiLabel = cms.InputTag("simMuonDTDigis")


# Explicit configuration of CSC for postls1 = run2 #
####################################################
process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
process.CSCIndexerESProducer.AlgoName = cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName = cms.string("CSCChannelMapperPostls1")
process.CSCGeometryESModule.useGangedStripsInME1a = False
process.csc2DRecHits.readBadChannels = cms.bool(False)
process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
# process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
# process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")

process.gemRecHits = cms.EDProducer("GEMRecHitProducer",
    recAlgoConfig = cms.PSet(),
    recAlgo = cms.string('GEMRecHitStandardAlgo'),
    gemDigiLabel = cms.InputTag("simMuonGEMDigis"),
    # maskSource = cms.string('File'),
    # maskvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMMaskVec.dat'),
    # deadSource = cms.string('File'),
    # deadvecfile = cms.FileInPath('RecoLocalMuon/GEMRecHit/data/GEMDeadVec.dat')
)



### Input and Output Files
##########################
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_digi.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string( 
        'file:out_local_reco.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_*_*_*',
    ),
#    SelectEvents = cms.untracked.PSet(
#        SelectEvents = cms.vstring('localreco_step', 'globalreco_step')
#    )
)

### TO ACTIVATE LogTrace one NEEDS TO COMPILE IT WITH:
### -----------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"             
### Make sure that you first cleaned your CMSSW version:       
### --> scram b clean                                          
### before issuing the scram command above                     
### -----------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"
###############################################################
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# process.MessageLogger.categories.append("MuonIdentification")
# process.MessageLogger.categories.append("TrackAssociator")
# process.MessageLogger.debugModules = cms.untracked.vstring("*")
# process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
# process.MessageLogger.cout = cms.untracked.PSet(
#     threshold          = cms.untracked.string("DEBUG"),
#     default            = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
#     FwkReport          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     MuonIdentification = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     TrackAssociator    = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
# )

### Paths and Schedules
#######################
process.digi2raw_step   = cms.Path(process.siPixelRawData+process.SiStripDigiToRaw+process.rawDataCollector) 
process.raw2digi_step   = cms.Path(process.RawToDigi) 
process.muonlocalreco += process.gemRecHits
process.muonlocalreco += process.me0LocalReco
process.muonGlobalReco += process.me0MuonReco

process.reconstruction_step = cms.Path(process.reconstruction)

process.endjob_step     = cms.Path(process.endOfProcess)
process.out_step        = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.digi2raw_step,
    process.raw2digi_step,
    process.reconstruction_step,
    process.endjob_step,
    process.out_step
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1),
    oncePerEventMode = cms.untracked.bool(True)
)
process.Timing = cms.Service("Timing")
process.options.wantSummary = cms.untracked.bool(True)
