import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
digiMon = DQMEDAnalyzer('Phase2TrackerMonitorDigi',
                        Verbosity = cms.bool(False),
                        TopFolderName = cms.string("Ph2TkDigi"),
                        PixelPlotFillingFlag = cms.bool(False),
                        StandAloneClusteriserFlag = cms.bool(False),
                        InnerPixelDigiSource   = cms.InputTag("simSiPixelDigis","Pixel"),
                        OuterTrackerDigiSource = cms.InputTag("mix", "Tracker"),
                        GeometryType = cms.string('idealForDigi'),
                        NumberOfDigisPerDetH = cms.PSet(
                            name   = cms.string("Num_Digis_Per_Det"),
                            title  = cms.string("Number of digis per det per event {}"),
                            NxBins = cms.int32(100),
                            xmin   = cms.double(-0.5),
                            xmax   = cms.double(99.5),
                            switch = cms.bool(True)
                        ),
                        DigiOccupancySH = cms.PSet(
                            name   = cms.string("Digi_Occupancy_S"),
                            title  = cms.string("Digi occupancy Strips {}"),
                            NxBins = cms.int32(51),
                            xmin   = cms.double(-0.001),
                            xmax   = cms.double(0.05),
                            switch = cms.bool(True)
                        ),
                        DigiOccupancyPH = cms.PSet(
                            name   = cms.string("Digi_Occupancy_P"),
                            title  = cms.string("Digi occupancy Pixels {}"),
                            NxBins = cms.int32(51),
                            xmin   = cms.double(-0.0001),
                            xmax   = cms.double(0.005),
                            switch = cms.bool(True)
                        ),
                        ChargeXYMapH = cms.PSet(
                            name   = cms.string("Digi_Charge_XY"),
                            title  = cms.string("Digi_Charge_XY {};Row;Column"),
                            NxBins = cms.int32(450),
                            xmin   = cms.double(0.5),
                            xmax   = cms.double(450.5),
                            NyBins = cms.int32(1350),
                            ymin   = cms.double(0.5),
                            ymax   = cms.double(1350.5),
                            switch = cms.bool(True)
                        ),
                        EtaH = cms.PSet(
                            NxBins = cms.int32(45),
                            xmin   = cms.double(-4.5),
                            xmax   = cms.double(4.5),
                            switch = cms.bool(True)
                        ),
                        DigiChargeH = cms.PSet(
                            name   = cms.string("Digi_Charge"),
                            title  = cms.string("Digi charge {};Digi charge [ADC]"),
                            NxBins = cms.int32(16),
                            xmin   = cms.double(-0.5),
                            xmax   = cms.double(15.5),
                            switch = cms.bool(True)
                        ),
                        TotalNumberOfDigisPerLayerH = cms.PSet(
                            name   = cms.string("Num_Digis_Per_Layer"),
                            title  = cms.string("Number of digis per layer per event {}"),
                            NxBins = cms.int32(5000),
                            xmin   = cms.double(0.0),
                            xmax   = cms.double(100000.0),
                            switch = cms.bool(True)
                        ),
                        NumberOfHitDetsPerLayerH = cms.PSet(
                            name   = cms.string("Num_Digi_Hit_Detectors_Per_Layer"),
                            title  = cms.string("Number of hit detectors with digis per event {}"),
                            NxBins = cms.int32(5000),
                            xmin   = cms.double(-0.5),
                            xmax   = cms.double(2999.5),
                            switch = cms.bool(True)
                        ),
                        NumberOfClustersPerDetH = cms.PSet(
                            name   = cms.string("Num_Digi_Clusters_Per_Det"),
                            title  = cms.string("Number of clusters per det per event {}"),
                            NxBins = cms.int32(100),
                            xmin   = cms.double(-0.5),
                            xmax   = cms.double(99.5),
                            switch = cms.bool(True)
                        ),
                        ClusterWidthH = cms.PSet(
                            name   = cms.string("Digi_Cluster_Width"),
                            title  = cms.string("Digi Cluster Width {};Cluster width"),
                            NxBins = cms.int32(16),
                            xmin   = cms.double(-0.5),
                            xmax   = cms.double(15.5),
                            switch = cms.bool(True)
                        ),
                        ClusterChargeH = cms.PSet(
                            NxBins = cms.int32(1024),
                            xmin   = cms.double(0.5),
                            xmax   = cms.double(1024.5),
                            switch = cms.bool(True)
                        ),
                        DigisOverThresholdH = cms.PSet(
                            name   = cms.string("Digis_Fraction_Over_Threshold"),
                            title  = cms.string("Fraction of digis over threshold in {};"),
                            NxBins = cms.int32(11),
                            xmin   = cms.double(-0.05),
                            xmax   = cms.double(1.05),
                            switch = cms.bool(True)
                        ),
                        XYPositionMapH = cms.PSet(
                            name   = cms.string("Digi_Global_Position_XY"),
                            title  = cms.string("Digi_Global_Position_XY;Digi position X [cm];Digi position Y [cm]"),
                            NxBins = cms.int32(1250),
                            xmin   = cms.double(-125.),
                            xmax   = cms.double(125.),
                            NyBins = cms.int32(1250),
                            ymin   = cms.double(-125.),
                            ymax   = cms.double(125.),
                            switch = cms.bool(True)
                        ),
                        RZPositionMapH = cms.PSet(
                            name   = cms.string("Digi_Global_Position_RZ"),
                            title  = cms.string("Digi_Global_Position_RZ;Digi position z [cm];Digi position #rho [cm]"),
                            NxBins = cms.int32(3000),
                            xmin   = cms.double(-300.),
                            xmax   = cms.double(300.),
                            NyBins = cms.int32(1250),
                            ymin   = cms.double(0.),
                            ymax   = cms.double(125.),
                            switch = cms.bool(True)
                        ),
                        CrackOverview = cms.PSet(
                            name   = cms.string('Crack_Overview_digis'),
                            title  = cms.string('Crack_Overview_digis;Module;Layer'),
                            xmin   = cms.double(0),
                            xmax   = cms.double(13.5),
                            ymin   = cms.double(0),
                            ymax   = cms.double(7.5),
                            switch = cms.bool(False)
                        )
                    )

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(digiMon,
                       InnerPixelDigiSource = "mixData:Pixel",
                       OuterTrackerDigiSource="mixData:Tracker"
                                                                 )

