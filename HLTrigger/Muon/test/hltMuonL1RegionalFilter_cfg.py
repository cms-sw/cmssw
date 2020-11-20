import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTMuonL1RegionalFilter")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
'file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT_RAW2DIGI_L1Reco.root'
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("HLTrigger.Muon.HLTMuonL1RegionalFilter_cfi")

#process.HLTMuonL1RegionalFilter = cms.EDFilter( "HLTMuonL1RegionalFilter",
#    CandTag = cms.InputTag( "hltL1extraParticles" ),
#    PreviousCandTag = cms.InputTag( "hltL1sL1SingleMu10" ),
#    MinN = cms.int32( 1 ),
#    saveTags = cms.bool( True ),
#    Cuts = cms.VPSet(
#        cms.PSet(
#            EtaRange = cms.vdouble( -2.5, -1.6 ),
#            MinPt  = cms.double( 20 ),
#            QualityBits = cms.vuint32( 6, 7 )
#        ),
#        cms.PSet(
#            EtaRange = cms.vdouble( -1.6,  1.6 ),
#            MinPt  = cms.double( 20 ),
#            QualityBits = cms.vuint32( 7 )
#        ),
#        cms.PSet(
#            EtaRange = cms.vdouble(  1.6,  2.5 ),
#            MinPt  = cms.double( 20 ),
#            QualityBits = cms.vuint32( 6, 7 )
#        )
#    )
#)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.debugModules = cms.untracked.vstring('HLTMuonL1RegionalFilter')
process.MessageLogger.HLTMuonL1RegionalFilter = dict()
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.path = cms.Path(
  process.HLTMuonL1RegionalFilter
)

