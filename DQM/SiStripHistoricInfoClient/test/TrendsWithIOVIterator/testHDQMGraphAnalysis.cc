{

gROOT->Reset();


GraphAnalysis A1("TotalNumberOfClusters_OnTrack_mean",false,"",-2.,16.);
A1.plotGraphAnalysis("TIB,TOB,TID,TEC");

GraphAnalysis A2("ClusterChargeCorr_OnTrack_landauPeak",false,"",99.,99.);
A2.plotGraphAnalysis("TIB,TOB,TID,TEC");

}
