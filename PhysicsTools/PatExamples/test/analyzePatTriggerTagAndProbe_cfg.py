import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.load( "FWCore.MessageService.MessageLogger_cfi" )
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( False )
)

process.source = cms.Source( "PoolSource",
    fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/cms/Tutorials/TWIKI_DATA/edmPatTrigger_zjets_patTutoial_june10.root'
    )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.TFileService = cms.Service( "TFileService",
    fileName = cms.string( 'analyzePatTriggerTagAndProbe.root' )
)

process.tagAndProbeAnalysis = cms.EDAnalyzer( "PatTriggerTagAndProbe",
    trigger      = cms.InputTag( "patTrigger" ),
    triggerEvent = cms.InputTag( "patTriggerEvent" ),
    muons        = cms.InputTag( "selectedPatMuons" ),
    muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' )
)

process.p = cms.Path(
    process.tagAndProbeAnalysis
)
