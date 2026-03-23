import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cfi import *

pixDigiMon = digiMon.clone(
    PixelPlotFillingFlag = True,
    StandAloneClusteriserFlag = False,
    TopFolderName = "TrackerPhase2ITDigi",
    NumberOfDigisPerDetH = digiMon.NumberOfDigisPerDetH.clone(
        Nbins = 500,
        xmin = -0.5,
        xmax = 999.5,
        switch = True
    ),
    NumberOfClustersPerDetH = digiMon.NumberOfClustersPerDetH.clone(
        Nbins = 200,
        xmin = 0.0,
        xmax = 2000.,
        switch = True
    ),
    PositionOfDigisPH = digiMon.PositionOfDigisPH.clone(
        Nxbins = 1344,
        xmin = 0.5,
        xmax = 1344.5,
        Nybins = 432,
        ymin = 0.5,
        ymax = 432.5,
        switch = True
    ),
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        Nxbins = 340,
        xmin = -170.,
        xmax = 170.,
        Nybins = 340,
        ymin = -170.,
        ymax = 170.,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        Nxbins = 600,
        xmin = -3000.0,
        xmax = 3000.,
        Nybins = 280,
        ymin = 0.,
        ymax = 280.,
        switch = True
    ),
    ClusterPositionPH = digiMon.ClusterPositionPH.clone(
        Nxbins = 960,
        xmin = 0.5,
        xmax = 960.5,
        Nybins = 32,
        ymin = 0.5,
        ymax = 32.5,
        switch = True
    )
)

otDigiMon = digiMon.clone(
    PixelPlotFillingFlag = False,
    StandAloneClusteriserFlag = False,
    TopFolderName = "TrackerPhase2OTDigi",
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        Nxbins = 250,
        xmin = -125.,
        xmax = 125.,
        Nybins = 250,
        ymin = -125.,
        ymax = 125.,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        Nxbins = 600,
        xmin = -300.,
        xmax = 300.,
        Nybins = 250,
        ymin = 0.,
        ymax = 125.,
        switch = True
    )
)
CRACKDigiMon = digiMon.clone(
    PixelPlotFillingFlag = False,
    StandAloneClusteriserFlag = False,
    TopFolderName = "TrackerPhase2OTDigi",
    CrackOverview = digiMon.CrackOverview.clone(
        xmin = 0,
        xmax = 13.5,
        ymin = 0,
        ymax = 7.5,
        switch = True
    ),
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        Nxbins = 250,
        xmin = -7,
        xmax = 7,
        Nybins = 250,
        ymin = -10,
        ymax = 50,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        Nxbins = 600,
        xmin = -70.,
        xmax = 70.,
        Nybins = 250,
        ymin = 0.,
        ymax = 60.,
        switch = True
    ),
    TotalNumberOfDigisPerLayerH = digiMon.TotalNumberOfDigisPerLayerH.clone(
        xmax = 100
    )
)

