import FWCore.ParameterSet.Config as cms

lumiMonitor = cms.EDAnalyzer("LumiMonitor",
   FolderName                  = cms.string("HLT/LumiMonitoring"),
   doPixelLumi                 = cms.bool(False),
   useBPixLayer1               = cms.bool(False),
   minNumberOfPixelsPerCluster = cms.int32(2), # from DQM/PixelLumi/python/PixelLumiDQM_cfi.py
   minPixelClusterCharge       = cms.double(15000.),
   pixelClusters               = cms.InputTag("hltSiPixelClusters"),
   scalers                     = cms.InputTag('scalersRawToDigi'),
   histoPSet                   = cms.PSet(
      pixelClusterPSet            = cms.PSet(
            nbins = cms.int32 (  200  ),
            xmin  = cms.double(   -0.5),
            xmax  = cms.double(19999.5),
      ),
      lumiPSet                    = cms.PSet(
            nbins = cms.int32 ( 3600 ),
            xmin  = cms.double( 3000.),
            xmax  = cms.double(12000.),
      ),
      pixellumiPSet               = cms.PSet(
            nbins = cms.int32 (300 ),
            xmin  = cms.double(  0.),
            xmax  = cms.double(  3.),
      ),
      lsPSet                      = cms.PSet(
            nbins = cms.int32 (2500 ),
            xmin  = cms.double(   0.),
            xmax  = cms.double(2500.),
      ),
   ),
)
