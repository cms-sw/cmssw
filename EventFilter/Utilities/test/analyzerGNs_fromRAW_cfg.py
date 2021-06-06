import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 10 ),
                                        output = cms.untracked.int32( 10 )
                                        )

process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring( 'ProductNotFound' ) )

process.source = cms.Source( "PoolSource",
                             #fileNames = cms.untracked.vstring( "/store/data/Run2016B/SingleMuon/RAW/v2/000/274/146/00000/040AA6B5-5C24-E611-BF6D-02163E013712.root" )
                             fileNames = cms.untracked.vstring( "/store/data/Run2016B/SingleMuon/RAW/v2/000/274/161/00000/00098E23-3825-E611-A603-02163E0134BD.root" )
                             )

process.a = cms.EDAnalyzer( "GlobalNumbersAnalysis",
                            inputTag = cms.untracked.InputTag( "rawDataCollector" ),
                            )

process.b = cms.EDAnalyzer( "DumpFEDRawDataProduct",
                            label = cms.untracked.string( "rawDataCollector" ),
                            feds = cms.untracked.vint32( 1024 ),
                            dumpPayload = cms.untracked.bool( True )
                            )

# path to be run
process.p = cms.Path(process.a+process.b)

