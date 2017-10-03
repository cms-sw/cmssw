import FWCore.ParameterSet.Config as cms

# lumi from scalers
#hltScalersRawToDigi4DQM = cms.EDProducer( "ScalersRawToDigi",
#    scalersInputTag = cms.InputTag( "rawDataCollector" )
#)

from DQM.HLTEvF.lumiMonitor_cfi import lumiMonitor

hltLumiMonitor = lumiMonitor.clone()
hltLumiMonitor.useBPixLayer1 = cms.bool( False )
hltLumiMonitor.minPixelClusterCharge = cms.double( 15000.0 )
hltLumiMonitor.histoPSet = cms.PSet(
    lsPSet = cms.PSet(  
      nbins = cms.int32( 2500 ) 
    ),
    pixelClusterPSet = cms.PSet(
      nbins = cms.int32( 200 ),
      xmin = cms.double( -0.5 ),
      xmax = cms.double( 19999.5 )
    ),
    puPSet = cms.PSet(
      nbins = cms.int32( 130  ),
      xmin = cms.double(   0. ),
      xmax = cms.double( 130. )
    ),
    lumiPSet = cms.PSet(
      nbins = cms.int32(   440 ),
      xmin = cms.double(     0.0 ),
      xmax = cms.double( 22000.0 )
    ),
    pixellumiPSet = cms.PSet(
      nbins = cms.int32( 300 ),
      xmin = cms.double( 0.0 ),
      xmax = cms.double( 3.0 )
    )
)
hltLumiMonitor.minNumberOfPixelsPerCluster = cms.int32( 2 )
hltLumiMonitor.FolderName = cms.string( "HLT/LumiMonitoring" )
hltLumiMonitor.scalers = cms.InputTag( "scalersRawToDigi" )
hltLumiMonitor.pixelClusters = cms.InputTag( "hltSiPixelClusters" )
hltLumiMonitor.doPixelLumi = cms.bool( False )

lumiMonitorHLTsequence = cms.Sequence(
#    hltScalersRawToDigi4DQM +
    hltLumiMonitor
)
