void trainIVF(TString name, TString train="GoodvsBad") {
    TTree *dSig = (TTree*) _file0->Get("treeProducerSusyMultilepton");
    TTree *dBg1 = (TTree*) _file1->Get("treeProducerSusyMultilepton");
    TFile *fOut = new TFile(name+".root","RECREATE");
    TMVA::Factory *factory = new TMVA::Factory(name, fOut, "!V:!Color");

    
    if (!name.Contains("pteta")) {
        factory->AddSpectator("SV_pt", 'D');
    }

    TString allvars = ""; 
    //common variables
    factory->AddVariable("SV_ntracks", 'D'); allvars += "ntracks";
    factory->AddVariable("SV_mass", 'D'); allvars += ":mass";
    factory->AddVariable("SV_ip2d := abs(SV_dxy)", 'D'); allvars += ":ip2d";
    factory->AddVariable("SV_sip2d := abs(SV_dxy/SV_edxy)", 'D'); allvars += ":sip2d";
    factory->AddVariable("SV_ip3d", 'D'); allvars += ":ip3d";
    factory->AddVariable("SV_sip3d", 'D'); allvars += ":sip3d";
    factory->AddVariable("SV_chi2n := min(SV_chi2/max(1,SV_ndof),10)", 'D'); allvars += ":chi2n";
    factory->AddVariable("SV_cosTheta := max(SV_cosTheta,0.98)"); allvars += ":cosTheta";

    TCut lepton = "SV_pt > 5 && SV_cosTheta > 0.98 && abs(SV_dxy) < 3";
    if (name.Contains("pteta")) {
        if (name.Contains("low_b"))  lepton += "SV_pt <= 15 && abs(SV_eta) <  1.2";
        if (name.Contains("low_e"))  lepton += "SV_pt <= 15 && abs(SV_eta) >= 1.2";
        if (name.Contains("high_b")) lepton += "SV_pt >  15 && abs(SV_eta) <  1.2";
        if (name.Contains("high_e")) lepton += "SV_pt >  15 && abs(SV_eta) >= 1.2";
    }

    double wSig = 1.0, wBkg = 1.0;
    factory->AddSignalTree(dSig, wSig);
    factory->AddBackgroundTree(dBg1, wBkg);

    factory->SetWeightExpression("");

    if (train=="GoodvsBad") {
        factory->PrepareTrainingAndTestTree( lepton+" SV_mcMatchFraction > 0.66", lepton+" SV_mcMatchFraction < 0.5", "" );
    }  else { 
        std::cerr << "ERROR: No idea of what training you want." << std::endl; return; 
    }


    factory->BookMethod( TMVA::Types::kLD, "LD", "!H:!V:VarTransform=None" );
    
    // Boosted Decision Trees with gradient boosting
    TString BDTGopt = "!H:!V:NTrees=500:BoostType=Grad:Shrinkage=0.10:!UseBaggedGrad:nCuts=200:nEventsMin=100:NNodesMax=9:UseNvars=9:PruneStrength=5:PruneMethod=CostComplexity:MaxDepth=8";

    BDTGopt += ":CreateMVAPdfs"; // Create Rarity distribution
    factory->BookMethod( TMVA::Types::kBDT, "BDTG", BDTGopt);

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    fOut->Close();
}
