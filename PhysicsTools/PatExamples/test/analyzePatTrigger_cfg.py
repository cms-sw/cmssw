import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:edmPatTrigger.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'rootPatTrigger.root' )
)

process.triggerAnalysis = cms.EDAnalyzer( "PatTriggerAnalyzer",
    hltProcessName = cms.string( 'HLT' ),
    processName    = cms.string( 'PAT' ),
    trigger        = cms.InputTag( "patTrigger" ),
    triggerEvent   = cms.InputTag( "patTriggerEvent" ),
    photons        = cms.InputTag( "selectedLayer1Photons" ),
    electrons      = cms.InputTag( "selectedLayer1Electrons" ),
    muons          = cms.InputTag( "selectedLayer1Muons" ),
    taus           = cms.InputTag( "selectedLayer1Taus" ),
    jets           = cms.InputTag( "selectedLayer1Jets" ),
    mets           = cms.InputTag( "layer1METs" )
)

process.p = cms.Path(
    process.triggerAnalysis
)
