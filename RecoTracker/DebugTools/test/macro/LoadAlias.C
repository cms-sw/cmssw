void LoadAlias()
{
    gApplication->InitializeGraphics();  // temporary fix for FW problem

    Events->SetAlias("tk", "recoTracks_generalTracks__RECO.obj");    
    Events->SetAlias("tp",  "TrackingParticles_mergedtruth_MergedTrackTruth_HTL.obj");
    Events->SetAlias("tv",  "TrackingVertexs_mergedtruth_MergedTrackTruth_HTL.obj");

    Events->SetAlias("T",  "TPtoRecoTracks_trackAlgoCompareUtil_TP_TrackAlgoCompare.obj");
    Events->SetAlias("TA", "RecoTracktoTPs_trackAlgoCompareUtil_AlgoA_TrackAlgoCompare.obj");
    Events->SetAlias("TB", "RecoTracktoTPs_trackAlgoCompareUtil_AlgoB_TrackAlgoCompare.obj");

    TCut trackablePrimaryTP = "abs(T.TP().eta())<2.4 && T.TP().vertex().rho()<3.5 && abs(T.TP().vertex().z())< 30 && T.TP().charge()!=0 && T.TP().pt()>0.9";
}

