import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonL1RegionalFilter")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
       'file:SingleMuPt100_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco.root',
       '/store/relval/CMSSW_3_5_0_pre3/RelValJpsiMM/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V15-v2/0006/FED7DB1C-B405-DF11-B043-0030487CD6DA.root',
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#process.load("HLTrigger.Muon.HLTMuonL1RegionalFilter_cfi")

process.HLTMuonL1RegionalFilter = cms.EDFilter( "HLTMuonL1RegionalFilter",
    CandTag = cms.InputTag( "hltL1extraParticles" ),
    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu10" ),
    EtaBoundaries = cms.vdouble( -2.5, -1.6, 1.6, 2.5 ),
    MinPts = cms.vdouble( 20, 20, 20  ),
    QualityBitMasks = cms.vint32( 192, 128, 192 ),
    MinN = cms.int32( 1 ),
    SaveTag = cms.untracked.bool( True )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('HLTMuonL1RegionalFilter')
process.MessageLogger.categories = cms.untracked.vstring('HLTMuonL1RegionalFilter')
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.path = cms.Path(
  process.HLTMuonL1RegionalFilter
)

