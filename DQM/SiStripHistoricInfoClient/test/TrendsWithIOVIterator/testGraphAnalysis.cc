{

gROOT->Reset();


GraphAnalysis A1("TotalNumberOfClusters_OnTrack_mean",false,"",-2.,16.);
A1.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A2("ClusterChargeCorr_OnTrack_landauPeak",false,"",99.,99.);
A2.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A3("ClusterChargeCorr_OnTrack_mean",false,"",-10.,210.);
A3.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A4("ClusterChargeCorr_OnTrack_landauChi2NDF",false,"",99.,99.);
A4.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A5("ClusterNoise_OnTrack_gaussMean",false,"",99.,99.);
A5.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A6("ClusterNoise_OnTrack_gaussChi2NDF",false,"",99.,99.);
A6.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A7("ClusterStoNCorr_OnTrack_landauPeak",false,"",99.,99.);
A7.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A8("ClusterStoNCorr_OnTrack_landauChi2NDF",false,"",99.,99.);
A8.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A9("ClusterStoNCorr_OnTrack_mean",false,"",-10.,50.);
A9.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A10("ClusterWidth_OnTrack_mean",false,"",99.,99.);
A10.plotGraphAnalysis("TIB,TOB,TID,TEC");


//

GraphAnalysis A21("ClusterChargeCorr__OnTrack_landauPeak",false,"_TIB_Layers",99.,99.);
A21.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A31("ClusterChargeCorr__OnTrack_mean",false,"_TIB_Layers",99.,99.);
A31.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A41("ClusterChargeCorr__OnTrack_landauChi2NDF",false,"_TIB_Layers",99.,99.);
A41.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A51("ClusterNoise__OnTrack_gaussMean",false,"_TIB_Layers",99.,99.);
A51.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A101("ClusterNoise__OnTrack_gaussChi2NDF",false,"_TIB_Layers",99.,99.);
A101.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A61("ClusterStoNCorr__OnTrack_landauPeak",false,"_TIB_Layers",99.,99.);
A61.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A71("ClusterStoNCorr__OnTrack_mean",false,"_TIB_Layers",99.,99.);
A71.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A81("ClusterStoNCorr__OnTrack_landauChi2NDF",false,"_TIB_Layers",99.,99.);
A81.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis A91("ClusterWidth__OnTrack_mean",false,"_TIB_Layers",99.,99.);
A91.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

//

GraphAnalysis A22("ClusterChargeCorr__OnTrack_landauPeak",false,"_TOB_Layers",99.,99.);
A22.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A32("ClusterChargeCorr__OnTrack_mean",false,"_TOB_Layers",99.,99.);
A32.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A42("ClusterChargeCorr__OnTrack_landauChi2NDF",false,"_TOB_Layers",99.,99.);
A42.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A52("ClusterNoise__OnTrack_gaussMean",false,"_TOB_Layers",99.,99.);
A52.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A102("ClusterNoise__OnTrack_gaussChi2NDF",false,"_TOB_Layers",99.,99.);
A102.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A62("ClusterStoNCorr__OnTrack_landauPeak",false,"_TOB_Layers",99.,99.);
A62.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A72("ClusterStoNCorr__OnTrack_mean",false,"_TOB_Layers",99.,99.);
A72.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A82("ClusterStoNCorr__OnTrack_landauChi2NDF",false,"_TOB_Layers",99.,99.);
A82.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis A92("ClusterWidth__OnTrack_mean",false,"_TOB_Layers",99.,99.);
A92.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

//

GraphAnalysis A23("ClusterChargeCorr__OnTrack_landauPeak",false,"_TID_Side1",99.,99.);
A23.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A33("ClusterChargeCorr__OnTrack_mean",false,"_TID_Side1",99.,99.);
A33.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A43("ClusterChargeCorr__OnTrack_landauChi2NDF",false,"_TID_Side1",99.,99.);
A43.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A53("ClusterNoise__OnTrack_gaussMean",false,"_TID_Side1",99.,99.);
A53.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A103("ClusterNoise__OnTrack_gaussChi2NDF",false,"_TID_Side1",99.,99.);
A103.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A63("ClusterStoNCorr__OnTrack_landauPeak",false,"_TID_Side1",99.,99.);
A63.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A73("ClusterStoNCorr__OnTrack_mean",false,"_TID_Side1",99.,99.);
A73.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A83("ClusterStoNCorr__OnTrack_landauChi2NDF",false,"_TID_Side1",99.,99.);
A83.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis A93("ClusterWidth__OnTrack_mean",false,"_TID_Side1",99.,99.);
A93.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");


GraphAnalysis A14("ClusterChargeCorr__OnTrack_landauPeak",false,"_TID_Side2",99.,99.);
A14.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A24("ClusterChargeCorr__OnTrack_mean",false,"_TID_Side2",99.,99.);
A24.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A34("ClusterChargeCorr__OnTrack_landauChi2NDF",false,"_TID_Side2",99.,99.);
A34.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A44("ClusterNoise__OnTrack_gaussMean",false,"_TID_Side2",99.,99.);
A44.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A104("ClusterNoise__OnTrack_gaussChi2NDF",false,"_TID_Side2",99.,99.);
A104.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A54("ClusterStoNCorr__OnTrack_landauPeak",false,"_TID_Side2",99.,99.);
A54.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A64("ClusterStoNCorr__OnTrack_mean",false,"_TID_Side2",99.,99.);
A64.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A74("ClusterStoNCorr__OnTrack_landauChi2NDF",false,"_TID_Side2",99.,99.);
A74.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis A84("ClusterWidth__OnTrack_mean",false,"_TID_Side2",99.,99.);
A84.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");


//

GraphAnalysis C1("TotalNumberOfClusters_OffTrack_mean",false,"",99.,99.);
C1.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis C2("ClusterCharge_OffTrack_mean",false,"",99.,99.);
C2.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis C3("ClusterNoise_OffTrack_gaussMean",false,"",99.,99.);
C3.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis C4("ClusterNoise_OffTrack_gaussChi2NDF",false,"",99.,99.);
C4.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis C5("ClusterStoN_OffTrack_mean",false,"",-10.,80.);
C5.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis C6("ClusterWidth_OffTrack_mean",false,"",99.,99.);
C6.plotGraphAnalysis("TIB,TOB,TID,TEC");

//

GraphAnalysis C11("ClusterCharge__OffTrack_mean",false,"_TIB_Layers",99.,99.);
C11.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis C12("ClusterNoise__OffTrack_gaussMean",false,"_TIB_Layers",99.,99.);
C12.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis C13("ClusterNoise__OffTrack_gaussChi2NDF",false,"_TIB_Layers",99.,99.);
C13.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis C14("ClusterStoN__OffTrack_mean",false,"_TIB_Layers",99.,99.);
C14.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

GraphAnalysis C15("ClusterWidth__OffTrack_mean",false,"_TIB_Layers",99.,99.);
C15.plotGraphAnalysis("TIB_Layer1,TIB_Layer2,TIB_Layer3,TIB_Layer4");

//

GraphAnalysis C21("ClusterCharge__OffTrack_mean",false,"_TOB_Layers",99.,99.);
C21.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis C22("ClusterNoise__OffTrack_gaussMean",false,"_TOB_Layers",99.,99.);
C22.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis C23("ClusterNoise__OffTrack_gaussChi2NDF",false,"_TOB_Layers",99.,99.);
C23.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis C24("ClusterStoN__OffTrack_mean",false,"_TOB_Layers",99.,99.);
C24.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

GraphAnalysis C25("ClusterWidth__OffTrack_mean",false,"_TOB_Layers",99.,99.);
C25.plotGraphAnalysis("TOB_Layer1,TOB_Layer2,TOB_Layer3,TOB_Layer4,TOB_Layer5,TOB_Layer6");

//

GraphAnalysis C31("ClusterCharge__OffTrack_mean",false,"_TID_Side1",99.,99.);
C31.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis C32("ClusterNoise__OffTrack_gaussMean",false,"_TID_Side1",99.,99.);
C32.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis C39("ClusterNoise__OffTrack_gaussChi2NDF",false,"_TID_Side1",99.,99.);
C39.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis C33("ClusterStoN__OffTrack_mean",false,"_TID_Side1",99.,99.);
C33.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis C34("ClusterWidth__OffTrack_mean",false,"_TID_Side1",99.,99.);
C34.plotGraphAnalysis("TID_Side1_Layer1,TID_Side1_Layer2,TID_Side1_Layer3");

GraphAnalysis C35("ClusterCharge__OffTrack_mean",false,"_TID_Side2",99.,99.);
C35.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis C36("ClusterNoise__OffTrack_gaussMean",false,"_TID_Side2",99.,99.);
C36.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis C40("ClusterNoise__OffTrack_gaussChi2NDF",false,"_TID_Side2",99.,99.);
C40.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis C37("ClusterStoN__OffTrack_mean",false,"_TID_Side2",99.,99.);
C37.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

GraphAnalysis C38("ClusterWidth__OffTrack_mean",false,"_TID_Side2",99.,99.);
C38.plotGraphAnalysis("TID_Side2_Layer1,TID_Side2_Layer2,TID_Side2_Layer3");

//


GraphAnalysis B1("Chi2",true,"_mean",0.,160.);
B1.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B2("NumberOfRecHitsPerTrack",true,"_mean",4.,16.);
B2.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B3("NumberOfTracks",true,"_mean",0.,0.16);
B3.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B4("TrackPt",true,"_mean",99.,99.);
B4.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B5("TrackPz",true,"_mean",99.,99.);
B5.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B6("TrackPx",true,"_mean",99.,99.);
B6.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B7("TrackPy",true,"_mean",99.,99.);
B7.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B8("TrackPhi",true,"_mean",99.,99.);
B8.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B9("TrackEta",true,"_mean",99.,99.);
B9.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B10("TrackTheta",true,"_mean",99.,99.);
B10.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B11("DistanceOfClosestApproach",true,"_mean",99.,99.);
B11.plotGraphAnalysis("CKFTk_mean,CosmicTk_mean,RSTk_mean");

GraphAnalysis B12("DistanceOfClosestApproach",true,"_rms",99.,99.);
B12.plotGraphAnalysis("CKFTk_rms,CosmicTk_rms,RSTk_rms");



}
