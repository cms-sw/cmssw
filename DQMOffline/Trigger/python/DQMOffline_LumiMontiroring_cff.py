import FWCore.ParameterSet.Config as cms

# lumi from scalers
#hltScalersRawToDigi4DQM = cms.EDProducer( "ScalersRawToDigi",
#    scalersInputTag = cms.InputTag( "rawDataCollector" )
#)

from DQM.HLTEvF.lumiMonitor_cfi import lumiMonitor

hltLumiMonitor = lumiMonitor.clone(
    useBPixLayer1 =  False ,
    minPixelClusterCharge = 15000.0, 

    histoPSet = dict(
                lsPSet = dict(  
                        nbins = 2500 ), 
                            
                pixelClusterPSet = dict(
                        nbins = 200 ,
                        xmin = -0.5 ,
                        xmax = 19999.5),
    
                puPSet = dict(
                        nbins = 130,
                        xmin =   0. ,
                        xmax =  130.),
    
                lumiPSet = dict(
                        nbins =   440 ,
                        xmin =   0.0 ,
                        xmax = 22000.0),
    
                pixellumiPSet = dict(
                        nbins = 300 ,
                        xmin =  0.0 ,
                        xmax = 3.0 )
        ),

    minNumberOfPixelsPerCluster =  2,
    FolderName =  "HLT/LumiMonitoring",
    scalers = "scalersRawToDigi",
    pixelClusters =  "hltSiPixelClusters",
    doPixelLumi =  False
)
lumiMonitorHLTsequence = cms.Sequence(
#    hltScalersRawToDigi4DQM +
    hltLumiMonitor
)
