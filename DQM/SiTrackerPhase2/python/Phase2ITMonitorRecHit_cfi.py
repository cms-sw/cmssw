import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
rechitMonitorIT = DQMEDAnalyzer('Phase2ITMonitorRecHit',
                                Verbosity = cms.bool(False),
                                TopFolderName = cms.string("TrackerPhase2ITRecHit"),
                                rechitsSrc = cms.InputTag("siPixelRecHits"),
                                GlobalNumberRecHits = cms.PSet(
                                    NxBins = cms.int32(50),
                                    xmin = cms.double(0.),
                                    xmax = cms.double(0.),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionXY_PXB = cms.PSet(
                                    NxBins = cms.int32(600),
                                    xmin = cms.double(-300.),
                                    xmax = cms.double(300.),
                                    NyBins = cms.int32(600),
                                    ymin = cms.double(-300.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionXY_PXEC = cms.PSet(
                                    NxBins = cms.int32(600),
                                    xmin = cms.double(-300.),
                                    xmax = cms.double(300.),
                                    NyBins = cms.int32(600),
                                    ymin = cms.double(-300.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionRZ_PXB = cms.PSet(
                                    NxBins = cms.int32(1500),
                                    xmin = cms.double(-3000.),
                                    xmax = cms.double(3000.),
                                    NyBins = cms.int32(300),
                                    ymin = cms.double(0.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionRZ_PXEC = cms.PSet(
                                    NxBins = cms.int32(1500),
                                    xmin = cms.double(-3000.),
                                    xmax = cms.double(3000.),
                                    NyBins = cms.int32(300),
                                    ymin = cms.double(0.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                LocalNumberRecHits = cms.PSet(
                                    NxBins = cms.int32(50),
                                    xmin = cms.double(0.),
                                    xmax = cms.double(0.),
                                    switch = cms.bool(True)
                                ),
                                RecHitPosX = cms.PSet(
                                    NxBins = cms.int32(100),
                                    xmin = cms.double(-2.5),
                                    xmax = cms.double(2.5),
                                    switch = cms.bool(True)
                                ),
                                RecHitPosY = cms.PSet(
                                    NxBins = cms.int32(100),
                                    xmin = cms.double(-2.5),
                                    xmax = cms.double(2.5),
                                    switch = cms.bool(True)
                                ),
                                RecHitPosErrorX = cms.PSet(
                                    NxBins = cms.int32(100),
                                    xmin = cms.double(0.),
                                    xmax = cms.double(0.2),
                                    switch = cms.bool(True)
                                ),
                                RecHitPosErrorY = cms.PSet(
                                    NxBins = cms.int32(100),
                                    xmin = cms.double(0.),
                                    xmax = cms.double(0.2),
                                    switch = cms.bool(True)
                                ),
                                LocalClusterSizeX = cms.PSet(
                                    NxBins = cms.int32(21),
                                    xmin = cms.double(-0.5),
                                    xmax = cms.double(20.5),
                                    switch = cms.bool(True)
                                ),
                                LocalClusterSizeY = cms.PSet(
                                    NxBins = cms.int32(21),
                                    xmin = cms.double(-0.5),
                                    xmax = cms.double(20.5),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionXY_perlayer = cms.PSet(
                                    NxBins = cms.int32(600),
                                    xmin = cms.double(-300.),
                                    xmax = cms.double(300.),
                                    NyBins = cms.int32(600),
                                    ymin = cms.double(-300.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                GlobalPositionRZ_perlayer = cms.PSet(
                                    NxBins = cms.int32(1500),
                                    xmin = cms.double(-3000.),
                                    xmax = cms.double(3000.),
                                    NyBins = cms.int32(300),
                                    ymin = cms.double(0.),
                                    ymax = cms.double(300.),
                                    switch = cms.bool(True)
                                ),
                                LocalPositionXY = cms.PSet(
                                    NxBins = cms.int32(500),
                                    xmin = cms.double(0.),
                                    xmax = cms.double(0.),
                                    NyBins = cms.int32(500),
                                    ymin = cms.double(0.),
                                    ymax = cms.double(0.),
                                    switch = cms.bool(True)
                                ),
                                
) 
