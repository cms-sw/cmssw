import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cfi import *


pixDigiMon = digiMon.clone()
pixDigiMon.PixelPlotFillingFlag = cms.bool(True)
pixDigiMon.TopFolderName = cms.string("TrackerPhase2ITDigi")
pixDigiMon.NumberOfDigisPerDetH = cms.PSet(
    Nbins = cms.int32(200),
    xmin = cms.double(0.0),
    xmax = cms.double(2000.),
    switch = cms.bool(True))
pixDigiMon.NumberOfClustersPerDetH = cms.PSet(
    Nbins = cms.int32(200),
    xmin = cms.double(0.0),
    xmax = cms.double(2000.),
    switch = cms.bool(True))
pixDigiMon.ChargeXYMapH = cms.PSet(
    Nxbins = cms.int32(450),
    xmin   = cms.double(0.5),
    xmax   = cms.double(450.5),
    Nybins = cms.int32(1350),
    ymin   = cms.double(0.5),
    ymax   = cms.double(1350.5),
    switch = cms.bool(True))
pixDigiMon.PositionOfDigisPH = cms.PSet(
    Nxbins = cms.int32(1350),
    xmin   = cms.double(0.5),
    xmax   = cms.double(1350.5),
    Nybins = cms.int32(450),
    ymin   = cms.double(0.5),
    ymax   = cms.double(450.5),
    switch = cms.bool(True))
pixDigiMon.XYPositionMapH = cms.PSet(
    Nxbins = cms.int32(340),
    xmin   = cms.double(-170.),
    xmax   = cms.double(170.),
    Nybins = cms.int32(340),
    ymin   = cms.double(-170.),
    ymax   = cms.double(170.),
    switch = cms.bool(True))
pixDigiMon.RZPositionMapH = cms.PSet(
    Nxbins = cms.int32(3000),
    xmin   = cms.double(-3000.0),
    xmax   = cms.double(3000.),
    Nybins = cms.int32(280),
    ymin   = cms.double(0.),
    ymax   = cms.double(280.),
    switch = cms.bool(True))
pixDigiMon.ClusterPositionPH = cms.PSet(
    Nxbins = cms.int32(960),
    xmin   = cms.double(0.5),
    xmax   = cms.double(960.5),
    Nybins = cms.int32(32),
    ymin   = cms.double(0.5),
    ymax   = cms.double(32.5),
    switch = cms.bool(True))


otDigiMon = digiMon.clone()
otDigiMon.PixelPlotFillingFlag = cms.bool(False)
otDigiMon.TopFolderName = cms.string("TrackerPhase2OTDigi")
