import FWCore.ParameterSet.Config as cms

process = cms.Process( "RctRawToDigi" )


process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))
                                    )
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cerr.INFO.limit = cms.untracked.int32(100)



#process.dumpRaw = cms.EDAnalyzer( 
#    "DumpFEDRawDataProduct",
#    label = cms.untracked.string("rawDataCollector"),
#    feds = cms.untracked.vint32 ( 1350 ),
#    dumpPayload = cms.untracked.bool ( True )
#)

# unpacker
process.load( "EventFilter.RctRawToDigi.l1RctHwDigis_cfi" )
process.l1RctHwDigis.inputLabel = cms.InputTag( "rawDataCollector" )
process.l1RctHwDigis.verbose = cms.untracked.bool( True )
process.l1RctHwDigis.rctFedId = cms.untracked.int32( 1350 )

# Other statements
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

process.output = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring ( 
    "keep *",
#    "keep *_l1RctHwDigis_*_*",
#    "keep *_rctDigiToRaw_*_*"
  ),
  
  fileName = cms.untracked.string( "rctDigis.root" )

)

process.p = cms.Path( 
    process.l1RctHwDigis 
    #+process.dumpRaw
 )

process.out = cms.EndPath( process.output )

# input
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1000 ) )

process.source = cms.Source ( "PoolSource",
   fileNames = cms.untracked.vstring(
        #'file:/hdfs/store/mc/RunIISpring15Digi74/WJetsToLNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/GEN-SIM-RAW/AVE_30_BX_50ns_tsg_MCRUN2_74_V6-v1/00000/048543B8-4FEF-E411-96E5-C4346BC7AAE0.root'
    'file:082DB4A9-D21C-E511-AAA8-02163E011AA5.root'
  )
)

