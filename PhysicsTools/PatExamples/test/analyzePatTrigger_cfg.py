import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:patTuple.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'analyzePatTrigger.root' )
)

process.triggerAnalysis = cms.EDAnalyzer( "PatTriggerAnalyzer",
    trigger      = cms.InputTag( "patTrigger" ),
    triggerEvent = cms.InputTag( "patTriggerEvent" ),
    muons        = cms.InputTag( "selectedPatMuons" ),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
    nBins    = cms.uint32( 20 ),
    binWidth = cms.double( 4. ),
    minID = cms.uint32( 81 ),
    maxID = cms.uint32( 96 )
)

process.p = cms.Path(
    process.triggerAnalysis
)
