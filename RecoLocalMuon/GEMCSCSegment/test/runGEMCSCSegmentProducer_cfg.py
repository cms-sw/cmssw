import FWCore.ParameterSet.Config as cms

process = cms.Process("GEMCSCSegmentRECO")

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D1Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D1_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

# Fix DT and CSC Alignment #
############################
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixDTAlignmentConditions
process = fixDTAlignmentConditions(process)
from SLHCUpgradeSimulations.Configuration.fixMissingUpgradeGTPayloads import fixCSCAlignmentConditions
process = fixCSCAlignmentConditions(process)
############################

# Explicit configuration of CSC for postls1 = run2 #
####################################################
process.load("CalibMuon.CSCCalibration.CSCChannelMapper_cfi")
process.load("CalibMuon.CSCCalibration.CSCIndexer_cfi")
process.CSCIndexerESProducer.AlgoName = cms.string("CSCIndexerPostls1")
process.CSCChannelMapperESProducer.AlgoName = cms.string("CSCChannelMapperPostls1")
process.CSCGeometryESModule.useGangedStripsInME1a = False
# process.csc2DRecHits.readBadChannels = cms.bool(False)
# process.csc2DRecHits.CSCUseGasGainCorrections = cms.bool(False)
# process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
# process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")               



### TO ACTIVATE LogTrace IN GEMCSCSegment NEED TO COMPILE IT WITH: 
### --------------------------------------------------------------
### --> scram b -j8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
### Make sure that you first cleaned your CMSSW version: 
### --> scram b clean 
### before issuing the scram command above
### --------------------------------------------------------------
### LogTrace output goes to cout; all other output to "junk.log"
### Code/Configuration with thanks to Tim Cox
### --------------------------------------------------------------
### to have a handle on the loops inside RPCSimSetup 
### I have split the LogDebug stream in several streams
### that can be activated independentl
##################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append("GEMCSCSegment")
process.MessageLogger.categories.append("GEMCSCSegmentBuilder")
# process.MessageLogger.categories.append("GEMCSCSegAlgoRR")
# process.MessageLogger.categories.append("GEMCSCSegFit")
process.MessageLogger.debugModules = cms.untracked.vstring("*")
process.MessageLogger.destinations = cms.untracked.vstring("cout","junk")
process.MessageLogger.cout = cms.untracked.PSet(
    threshold = cms.untracked.string("DEBUG"),
    default = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
    FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    GEMCSCSegment          = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    GEMCSCSegmentBuilder   = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMCSCSegAlgoRR        = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    # GEMCSCSegFit           = cms.untracked.PSet( limit = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
)
##################################################################
### TO ACTIVATE LogVerbatim IN GEMCSCSegment
### --------------------------------------------------------------
# process.MessageLogger.categories.append("GEMCSCSegment")
# process.MessageLogger.categories.append("GEMCSCSegFit")
# process.MessageLogger.destinations = cms.untracked.vstring("cout")
# process.MessageLogger.cout = cms.untracked.PSet(
#     threshold = cms.untracked.string("INFO"),
#     default   = cms.untracked.PSet( limit = cms.untracked.int32(0)  ),
#     FwkReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     GEMCSCSegment = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
#     GEMCSCSegFit  = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
# )
##################################################################


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")


process.load('RecoLocalMuon.GEMCSCSegment.gemcscSegments_cfi')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # 'file:/afs/cern.ch/work/p/piet/Analysis/SLC6/GEMCSCSegment/CMSSW_7_5_0_pre2/src/RecoLocalMuon/GEMRecHit/test/out_local_reco_test.root' #  29 evts
        # 'file:/afs/cern.ch/work/p/piet/Analysis/SLC6/GEMCSCSegment/CMSSW_7_5_0_pre2/src/RecoLocalMuon/GEMRecHit/test/out_local_reco.root'        # 100 evts
        'file:/afs/cern.ch/work/p/piet/Analysis/SLC6/GEMCSCSegment/CMSSW_7_5_0_pre3/src/RecoLocalMuon/GEMRecHit/test/out_local_reco_100GeV_1000evts.root'
    )
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string(
        # 'file:out_rereco_gemcscsegment.root'
        'file:out_rereco_gemcscsegment_75_2023_100GeV_1evt.root'
    ),
    outputCommands = cms.untracked.vstring(
        'keep  *_gemcscSegments_*_*',
        'keep  *_*csc*_*_*',
        'keep  *_*dt*_*_*',
        'keep  *_*gem*_*_*',
        'keep  *_*rpc*_*_*',
        'keep  *_*muon*_*_*',
        'keep  *_*CSC*_*_*',
        'keep  *_*DT*_*_*',
        'keep  *_*GEM*_*_*',
        'keep  *_*RPC*_*_*',
        'keep  *_*MUON*_*_*',
        'keep  *_*_*csc*_*',
        'keep  *_*_*dt*_*',
        'keep  *_*_*gem*_*',
        'keep  *_*_*rpc*_*',
        'keep  *_*_*muon*_*',
        'keep  *_*_*CSC*_*',
        'keep  *_*_*DT*_*',
        'keep  *_*_*GEM*_*',
        'keep  *_*_*RPC*_*',
        'keep  *_*_*MUON*_*',
        'keep  *SimTrack*_*_*_*',
        # 'keep  *_*_*_*',
    )
)

process.contentAna = cms.EDAnalyzer("EventContentAnalyzer")
process.reco_step    = cms.Path(process.gemcscSegments)
process.endjob_step  = cms.Path(process.endOfProcess)
process.out_step     = cms.EndPath(process.output)

process.schedule = cms.Schedule(
    process.reco_step,
    process.endjob_step,
    process.out_step
)
