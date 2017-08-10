import FWCore.ParameterSet.Config as cms

# lumi from scalers
#hltScalersRawToDigi4DQM = cms.EDProducer( "ScalersRawToDigi",
#    scalersInputTag = cms.InputTag( "rawDataCollector" )
#)
hltLumiMonitor = cms.EDAnalyzer( "LumiMonitor",
    useBPixLayer1 = cms.bool( False ),
    minPixelClusterCharge = cms.double( 15000.0 ),
    histoPSet = cms.PSet(
      lsPSet = cms.PSet(  nbins = cms.int32( 2500 ) ),
      pixelClusterPSet = cms.PSet(
        nbins = cms.int32( 200 ),
        xmin = cms.double( -0.5 ),
        xmax = cms.double( 19999.5 )
      ),
      lumiPSet = cms.PSet(
        nbins = cms.int32( 5000 ),
        xmin = cms.double( 0.0 ),
        xmax = cms.double( 20000.0 )
      ),
      pixellumiPSet = cms.PSet(
        nbins = cms.int32( 300 ),
        xmin = cms.double( 0.0 ),
        xmax = cms.double( 3.0 )
      )
    ),
    minNumberOfPixelsPerCluster = cms.int32( 2 ),
    FolderName = cms.string( "HLT/LumiMonitoring" ),
    scalers = cms.InputTag( "scalersRawToDigi" ),
    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
    doPixelLumi = cms.bool( False )
)

lumiMonitorHLTsequence = cms.Sequence(
#    hltScalersRawToDigi4DQM +
    hltLumiMonitor
)
