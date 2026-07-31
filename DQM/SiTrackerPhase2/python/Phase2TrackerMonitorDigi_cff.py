import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cfi import *

pixDigiMon = digiMon.clone(
    PixelPlotFillingFlag = True,
    StandAloneClusteriserFlag = False,
    TopFolderName = "InnerTracker",
    NumberOfDigisPerDetH = digiMon.NumberOfDigisPerDetH.clone(
        NxBins = 500,
        xmin = -0.5,
        xmax = 999.5,
        switch = True
    ),
    NumberOfClustersPerDetH = digiMon.NumberOfClustersPerDetH.clone(
        NxBins = 200,
        xmin = 0.0,
        xmax = 2000.,
        switch = True
    ),
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        NxBins = 340,
        xmin = -30.,
        xmax = 30.,
        NyBins = 340,
        ymin = -30.,
        ymax = 30.,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        NxBins = 600,
        xmin = -300.0,
        xmax = 300.,
        NyBins = 280,
        ymin = 0.,
        ymax = 28.,
        switch = True
    ),
    DigiChargeVsWidthH = digiMon.DigiChargeH.clone(
        name = "Digi_Charge_vs_Width",
        title = "Digi Charge vs Width {};Digi charge [ADC];Digi Width",
        NyBins = (digiMon.ClusterWidthH.NxBins),
        ymin = (digiMon.ClusterWidthH.xmin),
        ymax = (digiMon.ClusterWidthH.xmax),
        switch = (digiMon.DigiChargeH.switch and digiMon.ClusterWidthH.switch)
    ),
    DigiOccupancyVsEtaPH = digiMon.EtaH.clone(
        name = cms.string("Digi_Occupancy_Vs_Eta_P"),
        title = cms.string("Digi occupancy vs #eta pixels {};#eta;"),
        NyBins = digiMon.DigiOccupancyPH.NxBins,
        ymin = digiMon.DigiOccupancyPH.xmin,
        ymax = digiMon.DigiOccupancyPH.xmax,
        switch = (digiMon.DigiOccupancyPH.switch and digiMon.EtaH.switch)
    )
)

otDigiMon = digiMon.clone(
    PixelPlotFillingFlag = False,
    StandAloneClusteriserFlag = False,
    TopFolderName = "OuterTracker",
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        NxBins = 250,
        xmin = -125.,
        xmax = 125.,
        NyBins = 250,
        ymin = -125.,
        ymax = 125.,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        NxBins = 600,
        xmin = -300.,
        xmax = 300.,
        NyBins = 250,
        ymin = 0.,
        ymax = 125.,
        switch = True
    ),
    DigiChargeVsWidthH = digiMon.DigiChargeH.clone(
        name = "Digi_Charge_vs_Width",
        title = "Digi Charge vs Width {};Digi charge [ADC];Digi Width",
        NyBins = digiMon.ClusterWidthH.NxBins,
        ymin = digiMon.ClusterWidthH.xmin,
        ymax = digiMon.ClusterWidthH.xmax,
        switch = (digiMon.DigiChargeH.switch and digiMon.ClusterWidthH.switch)
    ),
    DigiOccupancyVsEtaPH = digiMon.EtaH.clone(
        name = cms.string("Digi_Occupancy_Vs_Eta_P"),
        title = cms.string("Digi occupancy vs #eta pixels {};#eta;"),
        NyBins = (digiMon.DigiOccupancyPH.NxBins),
        ymin = (digiMon.DigiOccupancyPH.xmin),
        ymax = (digiMon.DigiOccupancyPH.xmax),
        switch = (digiMon.DigiOccupancyPH.switch and digiMon.EtaH.switch)
    ),
    DigiOccupancyVsEtaSH = digiMon.EtaH.clone(
        name = cms.string("Digi_Occupancy_Vs_Eta_S"),
        title = cms.string("Digi occupancy vs #eta strips {};#eta;"),
        NyBins = (digiMon.DigiOccupancySH.NxBins),
        ymin = (digiMon.DigiOccupancySH.xmin),
        ymax = (digiMon.DigiOccupancySH.xmax),
        switch = (digiMon.DigiOccupancySH.switch and digiMon.EtaH.switch)
    ),
    DigiFractionOverThresholdVsEtaH = digiMon.EtaH.clone(
        name = cms.string("Digis_Fraction_Over_Threshold_vs_eta"),
        title = cms.string("Digi fraction over threshold vs #eta in {};#eta;"),
        NyBins = (digiMon.NumberOfDigisPerDetH.NxBins),
        ymin = (digiMon.NumberOfDigisPerDetH.xmin),
        ymax = (digiMon.NumberOfDigisPerDetH.xmax),
        switch = (digiMon.NumberOfDigisPerDetH.switch and digiMon.EtaH.switch)
    )
)
CRACKDigiMon = digiMon.clone(
    PixelPlotFillingFlag = False,
    StandAloneClusteriserFlag = False,
    TopFolderName = "OuterTracker",
    CrackOverview = digiMon.CrackOverview.clone(
        xmin = 0,
        xmax = 13.5,
        ymin = 0,
        ymax = 7.5,
        switch = True
    ),
    XYPositionMapH = digiMon.XYPositionMapH.clone(
        NxBins = 250,
        xmin = -7,
        xmax = 7,
        NyBins = 250,
        ymin = -10,
        ymax = 50,
        switch = True
    ),
    RZPositionMapH = digiMon.RZPositionMapH.clone(
        NxBins = 600,
        xmin = -70.,
        xmax = 70.,
        NyBins = 250,
        ymin = 0.,
        ymax = 60.,
        switch = True
    ),
    TotalNumberOfDigisPerLayerH = digiMon.TotalNumberOfDigisPerLayerH.clone(
        xmax = 100
    )
)

