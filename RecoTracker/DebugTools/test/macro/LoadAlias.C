void LoadAlias()
{
  //Events->SetAlias("rtk", "recoTracks_ctfWithMaterialTracks__Rec.obj");
    Events->SetAlias("tp",  "TrackingParticles_mergedtruth_MergedTrackTruth_ALL.obj");
    Events->SetAlias("tv",  "TrackingVertexs_mergedtruth_MergedTrackTruth_ALL.obj");

    Events->SetAlias("tk", "recoTracks_generalTracks__ALL.obj");	
    Events->SetAlias("T",  "TPtoRecoTracks_trackAlgoCompare_TP_TrackAlgoCompare.obj");
    Events->SetAlias("TA", "TPtoRecoTracks_trackAlgoCompare_AlgoA_TrackAlgoCompare.obj");
    Events->SetAlias("TB", "TPtoRecoTracks_trackAlgoCompare_AlgoB_TrackAlgoCompare.obj");

    TCut trackablePrimaryTP = "abs(T.TP().eta())<2.4 && T.TP().vertex().rho()<3.5 && abs(T.TP().vertex().z())< 30 && T.TP().charge()!=0 && T.TP().pt()>0.9";

}
