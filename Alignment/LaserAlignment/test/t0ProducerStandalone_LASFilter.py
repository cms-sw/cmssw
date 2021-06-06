
import FWCore.ParameterSet.Config as cms

process = cms.Process( "laserAlignmentT0ProducerProcess" )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source( "PoolSource",
  #skipEvents = cms.untracked.uint32( 7000 ), 
  fileNames = cms.untracked.vstring(
    '/store/data/Run2010A/TestEnables/RAW/v1/000/140/124/56E00D1B-308F-DF11-BE54-001D09F24691.root'
  )
)
process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "source"#"hltCalibrationRaw"

process.load( "Configuration.StandardSequences.FrontierConditions_GlobalTag_cff" )
process.GlobalTag.globaltag = cms.string('GR_R_37X_V6A::All')

# multiple sets can be given, only those will be output
process.load( "Alignment.LaserAlignment.LaserAlignmentT0Producer_cfi" )
process.laserAlignmentT0Producer.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)

process.load( "DQMServices.Core.DQM_cfg" )
process.load( "DQMOffline.Alignment.LaserAlignmentT0ProducerDQM_cfi" )
process.LaserAlignmentT0ProducerDQM.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)
process.LaserAlignmentT0ProducerDQM.OutputInPlainROOT = True;
process.LaserAlignmentT0ProducerDQM.UpperAdcThreshold = cms.uint32( 280 )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)

process.out = cms.OutputModule( "PoolOutputModule",
  outputCommands = cms.untracked.vstring(
    'drop *', 
    'keep *_laserAlignmentT0Producer_*_*'
  ),
  fileName = cms.untracked.string( '/tmp/aperiean/TkAlLAS_Run140124_LASFilter_test.root' )
)

process.load('Alignment.LaserAlignment.LaserAlignmentEventFilter_cfi')
#process.p = cms.Path( process.LaserAlignmentEventFilter )


#process.load('UserCode.BWittmer.LASFilter_cfi')
#from UserCode.BWittmer.LAS_defs_cff import *

#process.LASFilter.FED_IDs = cms.vint32()
#process.LASFilter.FED_IDs.extend(FED_TECp)
#process.LASFilter.FED_IDs.extend(FED_TECm)
#process.LASFilter.FED_IDs.extend(FED_AT_TOB)
#process.LASFilter.FED_IDs.extend(FED_AT_TIB)
#process.LASFilter.FED_IDs.extend(FED_AT_TECp)
#process.LASFilter.FED_IDs.extend(FED_AT_TECm)

#process.LASFilter.SIGNAL_IDs = cms.vint32()
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECp_R4)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECp_R6)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECm_R4)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECm_R6)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TOB)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TIB)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TECp)
#process.LASFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TECm)                                      

process.seqDigitization = cms.Path( process.siStripDigis )

process.seqAnalysis = cms.Path( process.LaserAlignmentEventFilter * 
                                (process.laserAlignmentT0Producer +
                                 process.LaserAlignmentT0ProducerDQM ))
process.outputPath = cms.EndPath( process.out )
