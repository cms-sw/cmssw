import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
SiStripMonitorTrack = cms.EDFilter(
    "SiStripMonitorTrack",
    
    TrackProducer = cms.string('generalTracks'),
    TrackLabel    = cms.string(''),
    AlgoName      = cms.string('GenTk'),
    
    RawDigis_On     = cms.bool(False),
    RawDigiProducer = cms.string('simSiStripDigis'),
    RawDigiLabel    = cms.string('VirginRaw'),
    
    MTCCData = cms.bool(False),
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('test_monitortrackparameters_rs.root'),    
    
    FolderName = cms.string('Track/GlobalParameters'),
    
    Cluster_src = cms.InputTag('siStripClusters'),
    
    ModulesToBeExcluded = cms.vuint32(),
        
    Mod_On      = cms.bool(False),
    OffHisto_On = cms.bool(True),
    Trend_On    = cms.bool(False),    
    
    ClusterConditions = cms.PSet( On       = cms.bool(False),
                                  minStoN  = cms.double(0.0),
                                  maxStoN  = cms.double(2000.0),
                                  minWidth = cms.double(0.0),
                                  maxWidth = cms.double(200.0)
                                  ),
    
    TH1nTracks = cms.PSet( Nbinx = cms.int32(10),
                           xmin  = cms.double(-0.5),
                           xmax  = cms.double(9.5)
                           ),
    
    TH1nRecHits = cms.PSet( Nbinx = cms.int32(16),
                            xmin  = cms.double(-0.5),
                            xmax  = cms.double(15.5)
                            ),

    TH1nClusters = cms.PSet( Nbinx = cms.int32(50),
                             xmin  = cms.double(-0.5),
                             xmax  = cms.double(99.5)
                             ),
    
    TH1ClusterCharge = cms.PSet( Nbinx = cms.int32(400),
                                 xmin  = cms.double(-10.0),
                                 xmax  = cms.double(800.0)
                                 ),
    
    TH1ClusterStoN = cms.PSet( Nbinx = cms.int32(300),
                               xmin  = cms.double(-10.0),
                               xmax  = cms.double(600.0)
                               ),
    
    TH1ClusterChargeCorr = cms.PSet( Nbinx = cms.int32(200),
                                     xmin  = cms.double(0.0),
                                     xmax  = cms.double(400.0)
                                     ),
    
    TH1ClusterStoNCorr = cms.PSet( Nbinx = cms.int32(200),
                                   xmin  = cms.double(0.0),
                                   xmax  = cms.double(200.0)
                                   ),
    
    TH1ClusterPos = cms.PSet( Nbinx = cms.int32(768),
                              xmin  = cms.double(-0.5),
                              xmax  = cms.double(767.5)
                              ),
    
    TH1ClusterNoise = cms.PSet( Nbinx = cms.int32(20),
                                xmin  = cms.double(0.0),
                                xmax  = cms.double(10.0)
                                ),
    
    TH1ClusterWidth = cms.PSet( Nbinx = cms.int32(20),
                                xmin  = cms.double(-0.5),
                                xmax  = cms.double(19.5)
                                ),
    
    TH1ClusterSymmEtaCC = cms.PSet( Nbinx = cms.int32(120),
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
    
    Trending = cms.PSet( Nbins      = cms.int32(10),
                         Steps      = cms.int32(5),
                         UpdateMode = cms.int32(1)
                         )
    
    )


