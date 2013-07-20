{

gROOT->Reset();


HistoricInspector A;
A.setDB("oracle://devdb10/CMS_COND_STRIP", "historicFromT0_V9_ReReco","cms_cond_strip","","COND/Services/TBufferBlobStreamingService");
//A.setDB("sqlite_file:historicDQM.db", "historicFromT0_pixel_test","cms_cond_strip","w3807dev","");

A.setDebug(1);
A.setDoStat(1);

//A.setBlackList("66706,66720,67033,67085,66692,66703,66714,66733,66739,66740,66878,66955,66985,66989,66993,66711,66716,66722,66746,66748,66756,66709,66783,66887,66893,66904,66910,66987,68286,69269,68286");
//A.setBlackList("68286");

// bad run list
//A.setBlackList("65941,65943,65945,65947,65948,65956,66480,67033,67038,67039,67049,67085,67114,68286,69253,69256,69351,69365,69522,69573,69788,69797,69800,69850,69874,69892");

// ./readDB_intervalOfRuns_HDQM.sh 42 65941 66746 : 65941, 65945, 65947, 65948, 65956, 66470, 66471, 66475, 66480, 66533, 66569,66604, 66615, 66637, 66644, 66676, 66692, 66703, 66706, 66709, 66711 ,66714, 66716, 66720, 66722, 66733, 66739, 66740, 66746 

// ./readDB_intervalOfRuns_HDQM.sh 43 66748 67647 : 66748, 66887 ,66893, 66904, 66910, 66987, 66989 ,66993 ,67033 ,67038 ,67039, 67043 ,67049 ,67085 ,67114 ,67122 ,67124 ,67126 ,67128 ,67139 ,67141 ,67147,67173, 67225,67534,67539,67541,67544,67548,67557,67573, 67645 ,67647 

// ./readDB_intervalOfRuns_HDQM.sh 44 67810 68949 : 67810, 67818, 67838, 68100, 68124, 68129, 68141, 68264, 68273, 68276, 68279, 68286, 68288, 68483, 68665
 
// ./readDB_intervalOfRuns_HDQM.sh 45 68958 70195

// ./readDB_intervalOfRuns_HDQM.sh 46 70410 70675


A.createTrend("0@Chi2_CKFTk@entries", "number_of_tracks.gif",1,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);
A.createTrend("0@NumberOfTracks_CKFTk@entries", "number_of_events.gif",1,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);
A.createTrend("0@NumberOfTracks_CKFTk@mean","mean_number_of_tracks_per_event.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);

//Tracks
A.createTrend("0@NumberOfTracks_CKFTk@entries,0@NumberOfTracks_CKFTk@mean,0@NumberOfRecHitsPerTrack_CKFTk@mean,0@Chi2_CKFTk@mean,0@Chi2_CKFTk@entries,
0@TrackPt_CKFTk@mean,0@TrackPx_CKFTk@mean,0@TrackPy_CKFTk@mean,0@TrackPz_CKFTk@mean,
0@TrackPhi_CKFTk@mean,0@TrackEta_CKFTk@mean,0@TrackTheta_CKFTk@mean,0@DistanceOfClosestApproach_CKFTk@mean,0@DistanceOfClosestApproach_CKFTk@rms", 
"CKFTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);

A.createTrend("0@NumberOfTracks_RSTk@entries,0@NumberOfTracks_RSTk@mean,0@NumberOfRecHitsPerTrack_RSTk@mean,0@Chi2_RSTk@mean,0@Chi2_RSTk@entries,
0@TrackPt_RSTk@mean,0@TrackPx_RSTk@mean,0@TrackPy_RSTk@mean,0@TrackPz_RSTk@mean,
0@TrackPhi_RSTk@mean,0@TrackEta_RSTk@mean,0@TrackTheta_RSTk@mean,0@DistanceOfClosestApproach_RSTk@mean,0@DistanceOfClosestApproach_RSTk@rms",
"RSTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);

A.createTrend("0@NumberOfTracks_CosmicTk@entries,0@NumberOfTracks_CosmicTk@mean,0@NumberOfRecHitsPerTrack_CosmicTk@mean,0@Chi2_CosmicTk@mean,0@Chi2_CosmicTk@entries,
0@TrackPt_CosmicTk@mean,0@TrackPx_CosmicTk@mean,0@TrackPy_CosmicTk@mean,0@TrackPz_CosmicTk@mean,
0@TrackPhi_CosmicTk@mean,0@TrackEta_CosmicTk@mean,0@TrackTheta_CosmicTk@mean,0@DistanceOfClosestApproach_CosmicTk@mean,0@DistanceOfClosestApproach_CosmicTk@rms",
"CosmicTk_trends.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);



// Clusters TIB, TOB, TEC, TID

A.createTrend("
1@Summary_TotalNumberOfClusters_OnTrack@mean,1@Summary_ClusterChargeCorr_OnTrack@landauPeak,1@Summary_ClusterChargeCorr_OnTrack@landauChi2NDF,1@Summary_ClusterChargeCorr_OnTrack@mean,1@Summary_ClusterNoise_OnTrack@gaussMean,1@Summary_ClusterNoise_OnTrack@gaussChi2NDF,1@Summary_ClusterStoNCorr_OnTrack@landauPeak,1@Summary_ClusterStoNCorr_OnTrack@landauChi2NDF,1@Summary_ClusterStoNCorr_OnTrack@mean,1@Summary_ClusterWidth_OnTrack@mean,
2@Summary_TotalNumberOfClusters_OnTrack@mean,2@Summary_ClusterChargeCorr_OnTrack@landauPeak,2@Summary_ClusterChargeCorr_OnTrack@landauChi2NDF,2@Summary_ClusterChargeCorr_OnTrack@mean,2@Summary_ClusterNoise_OnTrack@gaussMean,2@Summary_ClusterNoise_OnTrack@gaussChi2NDF,2@Summary_ClusterStoNCorr_OnTrack@landauPeak,2@Summary_ClusterStoNCorr_OnTrack@landauChi2NDF,2@Summary_ClusterStoNCorr_OnTrack@mean,2@Summary_ClusterWidth_OnTrack@mean,
3@Summary_TotalNumberOfClusters_OnTrack@mean,3@Summary_ClusterChargeCorr_OnTrack@landauPeak,3@Summary_ClusterChargeCorr_OnTrack@landauChi2NDF,3@Summary_ClusterChargeCorr_OnTrack@mean,3@Summary_ClusterNoise_OnTrack@gaussMean,3@Summary_ClusterNoise_OnTrack@gaussChi2NDF,3@Summary_ClusterStoNCorr_OnTrack@landauPeak,3@Summary_ClusterStoNCorr_OnTrack@landauChi2NDF,3@Summary_ClusterStoNCorr_OnTrack@mean,3@Summary_ClusterWidth_OnTrack@mean,
4@Summary_TotalNumberOfClusters_OnTrack@mean,4@Summary_ClusterChargeCorr_OnTrack@landauPeak,4@Summary_ClusterChargeCorr_OnTrack@landauChi2NDF,4@Summary_ClusterChargeCorr_OnTrack@mean,4@Summary_ClusterNoise_OnTrack@gaussMean,4@Summary_ClusterNoise_OnTrack@gaussChi2NDF,4@Summary_ClusterStoNCorr_OnTrack@landauPeak,4@Summary_ClusterStoNCorr_OnTrack@landauChi2NDF,4@Summary_ClusterStoNCorr_OnTrack@mean,4@Summary_ClusterWidth_OnTrack@mean",
"OnTrackClusters.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);


A.createTrend("
1@Summary_TotalNumberOfClusters_OffTrack@mean,1@Summary_ClusterCharge_OffTrack@mean,1@Summary_ClusterNoise_OffTrack@gaussMean,1@Summary_ClusterNoise_OffTrack@gaussChi2NDF,1@Summary_ClusterStoN_OffTrack@mean,1@Summary_ClusterWidth_OffTrack@mean,
2@Summary_TotalNumberOfClusters_OffTrack@mean,2@Summary_ClusterCharge_OffTrack@mean,2@Summary_ClusterNoise_OffTrack@gaussMean,2@Summary_ClusterNoise_OffTrack@gaussChi2NDF,2@Summary_ClusterStoN_OffTrack@mean,2@Summary_ClusterWidth_OffTrack@mean,
3@Summary_TotalNumberOfClusters_OffTrack@mean,3@Summary_ClusterCharge_OffTrack@mean,3@Summary_ClusterNoise_OffTrack@gaussMean,3@Summary_ClusterNoise_OffTrack@gaussChi2NDF,3@Summary_ClusterStoN_OffTrack@mean,3@Summary_ClusterWidth_OffTrack@mean,
4@Summary_TotalNumberOfClusters_OffTrack@mean,4@Summary_ClusterCharge_OffTrack@mean,4@Summary_ClusterNoise_OffTrack@gaussMean,4@Summary_ClusterNoise_OffTrack@gaussChi2NDF,4@Summary_ClusterStoN_OffTrack@mean,4@Summary_ClusterWidth_OffTrack@mean",
"OffTrackClusters.gif",0,"0@NumberOfTracks_CKFTk@entries>10000&&0@NumberOfRecHitsPerTrack_CosmicTk@entries>0",firstRun,lastRun);


// Clusters On track Layer levels

A.createTrend("
11@Summary_ClusterChargeCorr__OnTrack@landauPeak,11@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,11@Summary_ClusterChargeCorr__OnTrack@mean,11@Summary_ClusterNoise__OnTrack@gaussMean,11@Summary_ClusterNoise__OnTrack@gaussChi2NDF,11@Summary_ClusterStoNCorr__OnTrack@landauPeak,11@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,11@Summary_ClusterStoNCorr__OnTrack@mean,11@Summary_ClusterWidth__OnTrack@mean,
12@Summary_ClusterChargeCorr__OnTrack@landauPeak,12@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,12@Summary_ClusterChargeCorr__OnTrack@mean,12@Summary_ClusterNoise__OnTrack@gaussMean,12@Summary_ClusterNoise__OnTrack@gaussChi2NDF,12@Summary_ClusterStoNCorr__OnTrack@landauPeak,12@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,12@Summary_ClusterStoNCorr__OnTrack@mean,12@Summary_ClusterWidth__OnTrack@mean,
13@Summary_ClusterChargeCorr__OnTrack@landauPeak,13@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,13@Summary_ClusterChargeCorr__OnTrack@mean,13@Summary_ClusterNoise__OnTrack@gaussMean,13@Summary_ClusterNoise__OnTrack@gaussChi2NDF,13@Summary_ClusterStoNCorr__OnTrack@landauPeak,13@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,13@Summary_ClusterStoNCorr__OnTrack@mean,13@Summary_ClusterWidth__OnTrack@mean,
14@Summary_ClusterChargeCorr__OnTrack@landauPeak,14@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,14@Summary_ClusterChargeCorr__OnTrack@mean,14@Summary_ClusterNoise__OnTrack@gaussMean,14@Summary_ClusterNoise__OnTrack@gaussChi2NDF,14@Summary_ClusterStoNCorr__OnTrack@landauPeak,14@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,14@Summary_ClusterStoNCorr__OnTrack@mean,14@Summary_ClusterWidth__OnTrack@mean",
"OnTrackClusters_TIBLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);

A.createTrend("
21@Summary_ClusterChargeCorr__OnTrack@landauPeak,21@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,21@Summary_ClusterChargeCorr__OnTrack@mean,21@Summary_ClusterNoise__OnTrack@gaussMean,21@Summary_ClusterNoise__OnTrack@gaussChi2NDF,21@Summary_ClusterStoNCorr__OnTrack@landauPeak,21@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,21@Summary_ClusterStoNCorr__OnTrack@mean,21@Summary_ClusterWidth__OnTrack@mean,
22@Summary_ClusterChargeCorr__OnTrack@landauPeak,22@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,22@Summary_ClusterChargeCorr__OnTrack@mean,22@Summary_ClusterNoise__OnTrack@gaussMean,22@Summary_ClusterNoise__OnTrack@gaussChi2NDF,22@Summary_ClusterStoNCorr__OnTrack@landauPeak,22@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,22@Summary_ClusterStoNCorr__OnTrack@mean,22@Summary_ClusterWidth__OnTrack@mean,
23@Summary_ClusterChargeCorr__OnTrack@landauPeak,23@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,23@Summary_ClusterChargeCorr__OnTrack@mean,23@Summary_ClusterNoise__OnTrack@gaussMean,23@Summary_ClusterNoise__OnTrack@gaussChi2NDF,23@Summary_ClusterStoNCorr__OnTrack@landauPeak,23@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,23@Summary_ClusterStoNCorr__OnTrack@mean,23@Summary_ClusterWidth__OnTrack@mean,
24@Summary_ClusterChargeCorr__OnTrack@landauPeak,24@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,24@Summary_ClusterChargeCorr__OnTrack@mean,24@Summary_ClusterNoise__OnTrack@gaussMean,24@Summary_ClusterNoise__OnTrack@gaussChi2NDF,24@Summary_ClusterStoNCorr__OnTrack@landauPeak,24@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,24@Summary_ClusterStoNCorr__OnTrack@mean,24@Summary_ClusterWidth__OnTrack@mean,
25@Summary_ClusterChargeCorr__OnTrack@landauPeak,25@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,25@Summary_ClusterChargeCorr__OnTrack@mean,25@Summary_ClusterNoise__OnTrack@gaussMean,25@Summary_ClusterNoise__OnTrack@gaussChi2NDF,25@Summary_ClusterStoNCorr__OnTrack@landauPeak,25@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,25@Summary_ClusterStoNCorr__OnTrack@mean,25@Summary_ClusterWidth__OnTrack@mean,
26@Summary_ClusterChargeCorr__OnTrack@landauPeak,26@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,26@Summary_ClusterChargeCorr__OnTrack@mean,26@Summary_ClusterNoise__OnTrack@gaussMean,26@Summary_ClusterNoise__OnTrack@gaussChi2NDF,26@Summary_ClusterStoNCorr__OnTrack@landauPeak,26@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,26@Summary_ClusterStoNCorr__OnTrack@mean,26@Summary_ClusterWidth__OnTrack@mean",
"OnTrackClusters_TOBLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);

A.createTrend("
311@Summary_ClusterChargeCorr__OnTrack@landauPeak,311@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,311@Summary_ClusterChargeCorr__OnTrack@mean,311@Summary_ClusterNoise__OnTrack@gaussMean,311@Summary_ClusterNoise__OnTrack@gaussChi2NDF,311@Summary_ClusterStoNCorr__OnTrack@landauPeak,311@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,311@Summary_ClusterStoNCorr__OnTrack@mean,311@Summary_ClusterWidth__OnTrack@mean,
312@Summary_ClusterChargeCorr__OnTrack@landauPeak,312@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,312@Summary_ClusterChargeCorr__OnTrack@mean,312@Summary_ClusterNoise__OnTrack@gaussMean,312@Summary_ClusterNoise__OnTrack@gaussChi2NDF,312@Summary_ClusterStoNCorr__OnTrack@landauPeak,312@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,312@Summary_ClusterStoNCorr__OnTrack@mean,312@Summary_ClusterWidth__OnTrack@mean,
313@Summary_ClusterChargeCorr__OnTrack@landauPeak,313@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,313@Summary_ClusterChargeCorr__OnTrack@mean,313@Summary_ClusterNoise__OnTrack@gaussMean,313@Summary_ClusterNoise__OnTrack@gaussChi2NDF,313@Summary_ClusterStoNCorr__OnTrack@landauPeak,313@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,313@Summary_ClusterStoNCorr__OnTrack@mean,313@Summary_ClusterWidth__OnTrack@mean,
321@Summary_ClusterChargeCorr__OnTrack@landauPeak,321@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,321@Summary_ClusterChargeCorr__OnTrack@mean,321@Summary_ClusterNoise__OnTrack@gaussMean,321@Summary_ClusterNoise__OnTrack@gaussChi2NDF,321@Summary_ClusterStoNCorr__OnTrack@landauPeak,321@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,321@Summary_ClusterStoNCorr__OnTrack@mean,321@Summary_ClusterWidth__OnTrack@mean,
322@Summary_ClusterChargeCorr__OnTrack@landauPeak,322@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,322@Summary_ClusterChargeCorr__OnTrack@mean,322@Summary_ClusterNoise__OnTrack@gaussMean,322@Summary_ClusterNoise__OnTrack@gaussChi2NDF,322@Summary_ClusterStoNCorr__OnTrack@landauPeak,322@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,322@Summary_ClusterStoNCorr__OnTrack@mean,322@Summary_ClusterWidth__OnTrack@mean,
323@Summary_ClusterChargeCorr__OnTrack@landauPeak,323@Summary_ClusterChargeCorr__OnTrack@landauChi2NDF,323@Summary_ClusterChargeCorr__OnTrack@mean,323@Summary_ClusterNoise__OnTrack@gaussMean,323@Summary_ClusterNoise__OnTrack@gaussChi2NDF,323@Summary_ClusterStoNCorr__OnTrack@landauPeak,323@Summary_ClusterStoNCorr__OnTrack@landauChi2NDF,323@Summary_ClusterStoNCorr__OnTrack@mean,323@Summary_ClusterWidth__OnTrack@mean",
"OnTrackClusters_TIDLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);

// Clusters Off track Layer levels

A.createTrend("
11@Summary_ClusterCharge__OffTrack@mean,11@Summary_ClusterNoise__OffTrack@gaussMean,11@Summary_ClusterNoise__OffTrack@gaussChi2NDF,11@Summary_ClusterStoN__OffTrack@mean,11@Summary_ClusterWidth__OffTrack@mean,
12@Summary_ClusterCharge__OffTrack@mean,12@Summary_ClusterNoise__OffTrack@gaussMean,12@Summary_ClusterNoise__OffTrack@gaussChi2NDF,12@Summary_ClusterStoN__OffTrack@mean,12@Summary_ClusterWidth__OffTrack@mean,
13@Summary_ClusterCharge__OffTrack@mean,13@Summary_ClusterNoise__OffTrack@gaussMean,13@Summary_ClusterNoise__OffTrack@gaussChi2NDF,13@Summary_ClusterStoN__OffTrack@mean,13@Summary_ClusterWidth__OffTrack@mean,
14@Summary_ClusterCharge__OffTrack@mean,14@Summary_ClusterNoise__OffTrack@gaussMean,14@Summary_ClusterNoise__OffTrack@gaussChi2NDF,14@Summary_ClusterStoN__OffTrack@mean,14@Summary_ClusterWidth__OffTrack@mean",
"OffTrackClusters_TIBLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);

A.createTrend("
21@Summary_ClusterCharge__OffTrack@mean,21@Summary_ClusterNoise__OffTrack@gaussMean,21@Summary_ClusterNoise__OffTrack@gaussChi2NDF,21@Summary_ClusterStoN__OffTrack@mean,21@Summary_ClusterWidth__OffTrack@mean,
22@Summary_ClusterCharge__OffTrack@mean,22@Summary_ClusterNoise__OffTrack@gaussMean,22@Summary_ClusterNoise__OffTrack@gaussChi2NDF,22@Summary_ClusterStoN__OffTrack@mean,22@Summary_ClusterWidth__OffTrack@mean,
23@Summary_ClusterCharge__OffTrack@mean,23@Summary_ClusterNoise__OffTrack@gaussMean,23@Summary_ClusterNoise__OffTrack@gaussChi2NDF,23@Summary_ClusterStoN__OffTrack@mean,23@Summary_ClusterWidth__OffTrack@mean,
24@Summary_ClusterCharge__OffTrack@mean,24@Summary_ClusterNoise__OffTrack@gaussMean,24@Summary_ClusterNoise__OffTrack@gaussChi2NDF,24@Summary_ClusterStoN__OffTrack@mean,24@Summary_ClusterWidth__OffTrack@mean,
25@Summary_ClusterCharge__OffTrack@mean,25@Summary_ClusterNoise__OffTrack@gaussMean,25@Summary_ClusterNoise__OffTrack@gaussChi2NDF,25@Summary_ClusterStoN__OffTrack@mean,25@Summary_ClusterWidth__OffTrack@mean,
26@Summary_ClusterCharge__OffTrack@mean,26@Summary_ClusterNoise__OffTrack@gaussMean,26@Summary_ClusterNoise__OffTrack@gaussChi2NDF,26@Summary_ClusterStoN__OffTrack@mean,26@Summary_ClusterWidth__OffTrack@mean",
"OffTrackClusters_TOBLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);

A.createTrend("
311@Summary_ClusterCharge__OffTrack@mean,311@Summary_ClusterNoise__OffTrack@gaussMean,311@Summary_ClusterNoise__OffTrack@gaussChi2NDF,311@Summary_ClusterStoN__OffTrack@mean,311@Summary_ClusterWidth__OffTrack@mean,
312@Summary_ClusterCharge__OffTrack@mean,312@Summary_ClusterNoise__OffTrack@gaussMean,312@Summary_ClusterNoise__OffTrack@gaussChi2NDF,312@Summary_ClusterStoN__OffTrack@mean,312@Summary_ClusterWidth__OffTrack@mean,
313@Summary_ClusterCharge__OffTrack@mean,313@Summary_ClusterNoise__OffTrack@gaussMean,313@Summary_ClusterNoise__OffTrack@gaussChi2NDF,313@Summary_ClusterStoN__OffTrack@mean,313@Summary_ClusterWidth__OffTrack@mean,
321@Summary_ClusterCharge__OffTrack@mean,321@Summary_ClusterNoise__OffTrack@gaussMean,321@Summary_ClusterNoise__OffTrack@gaussChi2NDF,321@Summary_ClusterStoN__OffTrack@mean,321@Summary_ClusterWidth__OffTrack@mean,
322@Summary_ClusterCharge__OffTrack@mean,322@Summary_ClusterNoise__OffTrack@gaussMean,322@Summary_ClusterNoise__OffTrack@gaussChi2NDF,322@Summary_ClusterStoN__OffTrack@mean,322@Summary_ClusterWidth__OffTrack@mean,
323@Summary_ClusterCharge__OffTrack@mean,323@Summary_ClusterNoise__OffTrack@gaussMean,323@Summary_ClusterNoise__OffTrack@gaussChi2NDF,323@Summary_ClusterStoN__OffTrack@mean,323@Summary_ClusterWidth__OffTrack@mean",
"OffTrackClusters_TIDLayers.gif",0,"0@NumberOfRecHitsPerTrack_CosmicTk@entries>10000",firstRun,lastRun);


}
