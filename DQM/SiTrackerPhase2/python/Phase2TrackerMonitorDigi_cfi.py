import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
digiMon = DQMEDAnalyzer('Phase2TrackerMonitorDigi',
    Verbosity = cms.bool(False),
    TopFolderName = cms.string("Ph2TkDigi"),
    PixelPlotFillingFlag = cms.bool(False),
    InnerPixelDigiSource   = cms.InputTag("simSiPixelDigis","Pixel"),
    OuterTrackerDigiSource = cms.InputTag("mix", "Tracker"),
    GeometryType = cms.string('idealForDigi'),
    NumberOfDigisPerDetH = cms.PSet(
           Nbins = cms.int32(100),
           xmin = cms.double(-0.5),
           xmax = cms.double(99.5),
           switch = cms.bool(True)
    ),
    DigiOccupancySH = cms.PSet(
           Nbins = cms.int32(51),
           xmin = cms.double(-0.001),
           xmax = cms.double(0.05),
           switch = cms.bool(True)
    ),
    DigiOccupancyPH = cms.PSet(
           Nbins = cms.int32(51),
           xmin = cms.double(-0.0001),
           xmax = cms.double(0.005),
           switch = cms.bool(True)
    ),
    ChargeXYMapH = cms.PSet(
           Nxbins = cms.int32(450),
           xmin   = cms.double(0.5),
           xmax   = cms.double(450.5),
           Nybins = cms.int32(1350),
           ymin   = cms.double(0.5),
           ymax   = cms.double(1350.5),
           switch = cms.bool(False)
    ),
    PositionOfDigisSH = cms.PSet(
           Nxbins = cms.int32(1016),
           xmin   = cms.double(0.5),
           xmax   = cms.double(1016.5),
           Nybins = cms.int32(2),
           ymin   = cms.double(0.5),
           ymax   = cms.double(2.5),
           switch = cms.bool(True)
    ),
    PositionOfDigisPH = cms.PSet(
           Nxbins = cms.int32(960),
           xmin   = cms.double(0.5),
           xmax   = cms.double(960.5),
           Nybins = cms.int32(32),
           ymin   = cms.double(0.5),
           ymax   = cms.double(32.5),
           switch = cms.bool(True)
    ),
    EtaH = cms.PSet(
        Nbins  = cms.int32(45),
        xmin   = cms.double(-4.5),
        xmax   = cms.double(4.5),
        switch = cms.bool(True)
    ),
    DigiChargeH = cms.PSet(
      Nbins = cms.int32(261),
      xmin   = cms.double(0.5),
      xmax   = cms.double(260.5),
        switch = cms.bool(True)
    ), 
    TotalNumberOfDigisPerLayerH = cms.PSet(
      Nbins = cms.int32(100),
      xmin   = cms.double(0.0),
      xmax   = cms.double(50000.0),
        switch = cms.bool(True)
    ),
    NumberOfHitDetsPerLayerH = cms.PSet(
      Nbins = cms.int32(2000),
      xmin   = cms.double(-0.5),
      xmax   = cms.double(2000.5),
        switch = cms.bool(True)
    ),
    NumberOfClustersPerDetH = cms.PSet(
           Nbins = cms.int32(100),
           xmin = cms.double(-0.5),
           xmax = cms.double(99.5),
           switch = cms.bool(True)
    ),
    ClusterWidthH = cms.PSet(
           Nbins = cms.int32(16),
           xmin   = cms.double(-0.5),
           xmax   = cms.double(15.5),
           switch = cms.bool(True)
    ),
    ClusterChargeH = cms.PSet(
        Nbins = cms.int32(1024),
        xmin   = cms.double(0.5),
        xmax   = cms.double(1024.5),
        switch = cms.bool(True)
    ),  
    ClusterPositionSH = cms.PSet(
        Nxbins = cms.int32(1016),
        xmin   = cms.double(0.5),
        xmax   = cms.double(1016.5),
        Nybins = cms.int32(2),
        ymin   = cms.double(0.5),
        ymax   = cms.double(2.5),
        switch = cms.bool(True)
    ),
    ClusterPositionPH = cms.PSet(
        Nxbins = cms.int32(1016),
        xmin   = cms.double(0.5),
        xmax   = cms.double(1016.5),
        Nybins = cms.int32(32),
        ymin   = cms.double(0.5),
        ymax   = cms.double(32.5),
        switch = cms.bool(True)
    ),
    XYPositionMapH = cms.PSet(
        Nxbins = cms.int32(1250),
        xmin   = cms.double(-1250.),
        xmax   = cms.double(1250.),
        Nybins = cms.int32(1250),
        ymin   = cms.double(-1250.),
        ymax   = cms.double(1250.),
        switch = cms.bool(False)
    ),
    RZPositionMapH = cms.PSet(
        Nxbins = cms.int32(3000),
        xmin   = cms.double(-3000.),
        xmax   = cms.double(3000.),
        Nybins = cms.int32(1250),
        ymin   = cms.double(0.),
        ymax   = cms.double(1250.),
        switch = cms.bool(False)
    )
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(digiMon,
                       InnerPixelDigiSource = "mixData:Pixel",
                       OuterTrackerDigiSource="mixData:Tracker"
                                                                 )
