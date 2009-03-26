
import FWCore.ParameterSet.Config as cms

LaserAlignmentT0ProducerDQM = cms.EDAnalyzer( "LaserAlignmentT0ProducerDQM",
  # specify the input digi collections to run on
  DigiProducerList = cms.VPSet(
    cms.PSet(
      DigiLabel = cms.string( 'ZeroSuppressed' ),
      DigiType = cms.string( 'Processed' ),
      DigiProducer = cms.string( 'siStripDigis' )
    ), 
    cms.PSet(
      DigiLabel = cms.string( 'VirginRaw' ),
      DigiType = cms.string( 'Raw' ),
      DigiProducer = cms.string( 'siStripDigis' )
    ), 
    cms.PSet(
      DigiLabel = cms.string( 'ProcessedRaw' ),
      DigiType = cms.string( 'Raw' ),
      DigiProducer = cms.string( 'siStripDigis' )
    ), 
    cms.PSet(
      DigiLabel = cms.string( 'ScopeMode' ),
      DigiType = cms.string( 'Raw' ),
      DigiProducer = cms.string( 'siStripDigis' )
    )
  ),

  # the lower threshold for the strip amplitude;
  # profiles with digis above will be considered containing signal
  LowerAdcThreshold = cms.uint32( 15 ),

  # the upper threshold for the strip amplitude;
  # profiles with digis below will be considered containing a signal
  UpperAdcThreshold = cms.uint32( 220 ),

  # the dqm folder name to write to
  FolderName = cms.string( "TkAlLAS" ),

  # additionally dump in plain ROOT file?
  OutputInPlainROOT = cms.bool( False ),

  # if plain ROOT output, then write to this file
  PlainOutputFileName = cms.string( "TkAlLAS.dqm.root" )
                                              
)
