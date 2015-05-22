void trainMuonMVA(TString name, TString train="GoodvsBad") {
    TTree *dSig = (TTree*) _file0->Get("treeProducerSusyMultilepton");
    TTree *dBg1 = (TTree*) _file1->Get("treeProducerSusyMultilepton");
    if (dBg1 == 0) dBg1 = (TTree*) _file1->Get("rec/t");

    TFile *fOut = new TFile(name+".root","RECREATE");
    TMVA::Factory *factory = new TMVA::Factory(name, fOut, "!V:!Color");

    
    if (!name.Contains("pt_")) {
        factory->AddSpectator("pt := LepGood_pt", 'D');
    }
    factory->AddSpectator("tightId := LepGood_tightId", 'D'); 

    TString allvars = ""; 
    //common variables
    factory->AddVariable("eta := LepGood_eta", 'D'); allvars += "eta";
    factory->AddVariable("globalTrackChi2 := log(LepGood_globalTrackChi2)", 'D'); allvars += ":globalTrackChi2";
    factory->AddVariable("segmentCompatibility := LepGood_segmentCompatibility", 'D'); allvars += ":segmentCompatibility";
    factory->AddVariable("chi2LocalPosition := log(LepGood_chi2LocalPosition)", 'D'); allvars += ":chi2LocalPosition";
    factory->AddVariable("chi2LocalMomentum := log(LepGood_chi2LocalMomentum)", 'D'); allvars += ":chi2LocalMomentum";
    factory->AddVariable("innerTrackValidHitFraction := LepGood_innerTrackValidHitFraction", 'D'); allvars += ":innerTrackValidHitFraction";
    factory->AddVariable("lostOuterHits := LepGood_lostOuterHits", 'D'); allvars += ":lostOuterHits";
    factory->AddVariable("glbTrackProbability := log(LepGood_glbTrackProbability)", 'D'); allvars += ":glbTrackProbability";
    factory->AddVariable("trackerHits := LepGood_trackerHits", 'D'); allvars += ":trackerHits";

    if (name.Contains("Calo") || name.Contains("Full")) {
        factory->AddVariable("caloCompatibility := LepGood_caloCompatibility", 'D'); allvars += ":caloCompatibility";
        factory->AddVariable("caloEMEnergy := min(LepGood_caloEMEnergy,20)", 'D'); allvars += ":caloEMEnergy";
        factory->AddVariable("caloHadEnergy := min(LepGood_caloHadEnergy,30)", 'D'); allvars += ":caloHadEnergy";
    }

    if (name.Contains("Trk") || name.Contains("Full")) {
        factory->AddVariable("lostHits := LepGood_lostHits", 'D'); allvars += ":lostHits";
        factory->AddVariable("trkKink := min(100,LepGood_trkKink)", 'D'); allvars += ":trkKink";
        factory->AddVariable("trackerLayers := LepGood_trackerLayers", 'D'); allvars += ":trackerLayers";
        factory->AddVariable("pixelLayers := LepGood_pixelLayers", 'D'); allvars += ":pixelLayers";
        factory->AddVariable("innerTrackChi2 := LepGood_innerTrackChi2", 'D'); allvars += ":innerTrackChi2";
    }

    /*
    factory->AddVariable("tightId := LepGood_tightId", 'D'); allvars += ":tightId";
    factory->AddVariable("nStations := LepGood_nStations", 'D'); allvars += ":nStations";
    factory->AddVariable("stationsWithAnyHits := LepGood_stationsWithAnyHits", 'D'); allvars += ":stationsWithAnyHits";
    factory->AddVariable("stationsWithValidHits := LepGood_stationsWithValidHits", 'D'); allvars += ":stationsWithValidHits";
    factory->AddVariable("stationsWithValidHitsGlbTrack := LepGood_stationsWithValidHitsGlbTrack", 'D'); allvars += ":stationsWithValidHitsGlbTrack";
    */

    TCut lepton = "abs(LepGood_pdgId) == 13" ;

    if (name.Contains("pt_")) {
        if (name.Contains("pt_low"))  lepton += "pt <= 15";
        if (name.Contains("pt_high")) lepton += "pt >  15";
    }
   
    double wSig = 1.0, wBkg = 1.0;
    factory->AddSignalTree(dSig, wSig);
    factory->AddBackgroundTree(dBg1, wBkg);

    // re-weighting to approximately match n(jet) multiplicity of signal
    //factory->SetWeightExpression("puWeight*((good>0)+(good<=0)*pow(nJet25,2.36))");
    factory->SetWeightExpression("");

    if (train=="GoodvsBad") {
        factory->PrepareTrainingAndTestTree( lepton+" LepGood_mcMatchId > 0", lepton+" LepGood_mcMatchId <= 0 && LepGood_mcMatchAny == 0", "nTrain_Signal=50000:nTrain_Background=10000:nTest_Signal=50000:nTest_Background=10000" );
    }  else { 
        std::cerr << "ERROR: No idea of what training you want." << std::endl; return; 
    }


    factory->BookMethod( TMVA::Types::kLD, "LD", "!H:!V:VarTransform=None" );
    
    // Boosted Decision Trees with gradient boosting
    TString BDTGopt = "!H:!V:NTrees=400:BoostType=Grad:Shrinkage=0.10:!UseBaggedGrad:nCuts=2000:nEventsMin=200:NNodesMax=9:UseNvars=9:PruneStrength=5:PruneMethod=CostComplexity:MaxDepth=8";

    BDTGopt += ":CreateMVAPdfs"; // Create Rarity distribution
    factory->BookMethod( TMVA::Types::kBDT, "BDTG", BDTGopt);

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    fOut->Close();
}
