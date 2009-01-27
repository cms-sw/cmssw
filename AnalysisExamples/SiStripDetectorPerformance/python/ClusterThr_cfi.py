import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
ClusterThr = cms.EDFilter(
    "ClusterThr",
    Cluster_src = cms.InputTag('siStripClusters'),
    TrackProducer = cms.string('TrackRefitter'),
    TrackLabel    = cms.string(''),
    TrajectoryInEvent = cms.bool(True),
    
    ModulesToBeExcluded = cms.vuint32(),

    #Decide the SubDet to be analyzed    
    SubDets = cms.vstring('TIB'),
    #Decide the Layer to be analyzed
    Layers = cms.vuint32(4),

    NeighStrips = cms.int32(3),
    
    fileName = cms.string('testClusterThr.root'),

#================>>>>>>>>>>>>>#TIB: 12-17 / TOB: 17-22 / TID 10-17 / TEC 11-17
    StoNThr = cms.PSet( StoNBmax = cms.double(17),
                        StoNSmin = cms.double(22)
                        ),
    BadModuleStudies = cms.PSet( Bad = cms.bool(True),    # Set Bad true if you want to study good and bad modules separately
                                 justGood = cms.bool(True)# Set justGood true if you want to study only good modules, false if only bad modules
                                 ),
    NoiseMode = cms.uint32(1),
    
    ThC = cms.PSet( startThC = cms.double(5),
                    stopThC = cms.double(13),
                    stepThC = cms.double(2)
                    ),
    ThS = cms.PSet( startThS = cms.double(3),
                    stopThS = cms.double(10),
                    stepThS = cms.double(1)
                    ),
    ThN = cms.PSet( startThN = cms.double(2),
                    stopThN = cms.double(6),
                    stepThN = cms.double(1)
                    ),
    TH1ClusterSignal =  cms.PSet( Nbinx = cms.int32(200),
                          xmin = cms.double(-0.5),
                          xmax = cms.double(799.5)
                         ),      
    TH1ClusterNoise= cms.PSet( Nbinx= cms.int32(20),
                      xmin = cms.double(-0.5),
                      xmax = cms.double(10.5)
                      ),   
    TH1ClusterStoN = cms.PSet( Nbinx= cms.int32(50),
                       xmin= cms.double(-0.5),
                       xmax= cms.double(199.5)
                      ), 
    TH1ClusterWidth= cms.PSet( Nbinx= cms.int32(20),
                       xmin= cms.double(-0.5),
                       xmax= cms.double(19.5)
                      ),
    TH1ClusterPos =  cms.PSet( Nbinx= cms.int32(768),
                       xmin= cms.double(0.5),
                       xmax= cms.double(768.5)
                      ), 
    TH1ClusterNum =  cms.PSet( Nbinx= cms.int32(30),
                       xmin= cms.double(-0.5),
                       xmax= cms.double(29.5)
                      )
    
    )
