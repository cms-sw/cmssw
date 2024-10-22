import FWCore.ParameterSet.Config as cms

# lumi from scalers
#hltScalersRawToDigi4DQM = cms.EDProducer( "ScalersRawToDigi",
#    scalersInputTag = cms.InputTag( "rawDataCollector" )
#)

from DQM.HLTEvF.lumiMonitor_cfi import lumiMonitor as _lumiMonitor

hltLumiMonitor = _lumiMonitor.clone(
    folderName = 'HLT/LumiMonitoring',
    scalers = 'scalersRawToDigi',
    onlineMetaDataDigis = 'onlineMetaDataDigis',
    doPixelLumi = False,
    useBPixLayer1 = False,
    pixelClusters =  'hltSiPixelClusters',
    minNumberOfPixelsPerCluster = 2,
    minPixelClusterCharge = 15000,
    histoPSet = dict(
        lsPSet = dict(
            nbins = 2500
        ),
        pixelClusterPSet = dict(
            nbins = 200,
            xmin = -0.5,
            xmax = 19999.5
        ),
        puPSet = dict(
            nbins = 130,
            xmin = 0,
            xmax = 130
        ),
        lumiPSet = dict(
            nbins = 440,
            xmin = 0,
            xmax = 22000
        ),
        pixellumiPSet = dict(
            nbins = 300,
            xmin = 0,
            xmax = 3
        )
    )
)

lumiMonitorHLTsequence = cms.Sequence(
#    hltScalersRawToDigi4DQM +
    hltLumiMonitor
)
