import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
SiStripMonitorTrack = DQMEDAnalyzer(
    "SiStripMonitorTrack",

    TopFolderName = cms.string('SiStrip'),
    TrackProducer = cms.string('generalTracks'),
    TrackLabel    = cms.string(''),
    TrajectoryInEvent = cms.bool(True),
    AlgoName      = cms.string('GenTk'),

    ADCDigi_src = cms.InputTag('siStripDigis','ZeroSuppressed'),
    Cluster_src = cms.InputTag('siStripClusters'),

    genericTriggerEventPSet = cms.PSet(),

    ModulesToBeExcluded = cms.vuint32(),

    Mod_On        = cms.bool(False),
    OffHisto_On   = cms.bool(True),
    Trend_On      = cms.bool(False),
    HistoFlag_On  = cms.bool(False),
    TkHistoMap_On = cms.bool(True),
    clchCMoriginTkHmap_On = cms.bool(False),
    ClusterConditions = cms.PSet( On       = cms.bool(False),
                                  minStoN  = cms.double(0.0),
                                  maxStoN  = cms.double(2000.0),
                                  minWidth = cms.double(0.0),
                                  maxWidth = cms.double(200.0)
                                  ),

    TH1nClustersOn = cms.PSet( Nbinx = cms.int32(50),
                             xmin  = cms.double(-0.5),
                             xmax  = cms.double(2999.5)
                             ),
    
    TH1nClustersOnStereo = cms.PSet( Nbinx = cms.int32(50),
                             xmin  = cms.double(-0.5),
                             xmax  = cms.double(2999.5)
                             ),

    TH1nClustersOff = cms.PSet( Nbinx = cms.int32(100),
                             xmin  = cms.double(-0.5),
                             xmax  = cms.double(99999.5)
                             ),

    TH1ClusterGain = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(3.5)
    ),

    TH1ClusterCharge = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(999.5)
    ),

    TH1ClusterChargeRaw = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(999.5)
    ),

    TH1ClusterStoN = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(299.5)
    ),

    TH1ClusterChargeCorr = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(399.5)
    ),

    TH1ClusterStoNCorr = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(99.5)
     ),

    TH1ClusterStoNCorrMod = cms.PSet(
        Nbinx = cms.int32(50),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(199.5)
    ),

    TH1ClusterNoise = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(20),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(9.5)
    ),

    TH1ClusterWidth = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
        Nbinx = cms.int32(20),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(19.5)
    ),

    TH1ClusterPos = cms.PSet(
        layerView = cms.bool(True),
        ringView  = cms.bool(False),
    ),

    TH1ClusterSymmEtaCC = cms.PSet( Nbinx = cms.int32(120),
                                    xmin  = cms.double(-0.1),
                                    xmax  = cms.double(1.1)
                                    ),

    TH1ClusterWidthCC = cms.PSet( Nbinx = cms.int32(10),
                                  xmin  = cms.double(-0.5),
                                  xmax  = cms.double(9.5)
                                  ),

    TH1ClusterEstimatorCC = cms.PSet( Nbinx = cms.int32(120),
                                      xmin  = cms.double(-0.1),
                                      xmax  = cms.double(1.1)
                                      ),

    TProfileClusterPGV = cms.PSet( Nbinx = cms.int32(20),
                                   xmin = cms.double(-10.0),
                                   xmax = cms.double(10.0),
                                   Nbiny = cms.int32(20),
                                   ymin = cms.double(-0.1),
                                   ymax = cms.double(1.2)
                                   ),

    Trending = cms.PSet(
        Nbins = cms.int32(2400),
        xmin = cms.double(0.0),
        xmax = cms.double(150)
        ),

    TH1ClusterChargePerCM = cms.PSet(
        layerView = cms.bool(False),
        ringView  = cms.bool(True),
        Nbinx = cms.int32(100),
        xmin  = cms.double(-0.5),
        xmax  = cms.double(9999.5)
    ),

    UseDCSFiltering = cms.bool(True)

)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(SiStripMonitorTrack,  TH1nClustersOn = dict(xmax = 5999.5))
