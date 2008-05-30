void LoadAlias()
{
    gApplication->InitializeGraphics();  // temporary fix for FW problem

    Events->SetAlias("tk", "recoTracks_generalTracks__ALL.obj");    
    Events->SetAlias("tp",  "TrackingParticles_mergedtruth_MergedTrackTruth_ALL.obj");
    Events->SetAlias("tv",  "TrackingVertexs_mergedtruth_MergedTrackTruth_ALL.obj");

    Events->SetAlias("T",  "TPtoRecoTracks_trackAlgoCompare_TP_TrackAlgoCompare.obj");
    Events->SetAlias("TA", "RecoTracktoTPs_trackAlgoCompare_AlgoA_TrackAlgoCompare.obj");
    Events->SetAlias("TB", "RecoTracktoTPs_trackAlgoCompare_AlgoB_TrackAlgoCompare.obj");

    //TCut trackablePrimaryTP = "fabs(T.TP().eta())<2.4 && T.TP().vertex().rho()<3.5 && fabs(T.TP().vertex().z())< 30 && T.TP().charge()!=0 && T.TP().pt()>0.9";
}
