void trainMVA_susy_2lss(TString name, TString bgname="ttbar") {
    TString Path = "/afs/cern.ch/user/g/gpetrucc/w/TREES_70X_240914";
    gROOT->ProcessLine(".L ../../../python/plotter/functions.cc+");

    TFile *fOut = new TFile(name+".root","RECREATE");
    TMVA::Factory *factory = new TMVA::Factory(name, fOut, "!V:!Color");

    TFile *fSig = TFile::Open(Path+"/T1tttt2J_7_PU_S14_POSTLS170/treeProducerSusyMultilepton/treeProducerSusyMultilepton_tree.root");
    TTree *tSig = (TTree *) fSig->Get("treeProducerSusyMultilepton");
    //tSig->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTH122.root");
    factory->AddSignalTree(tSig, 1.0);
    TFile *fSig2 = TFile::Open(Path+"/T1tttt2J_6_PU_S14_POSTLS170/treeProducerSusyMultilepton/treeProducerSusyMultilepton_tree.root");
    TTree *tSig2 = (TTree *) fSig2->Get("treeProducerSusyMultilepton");
    //tSig->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTH122.root");
    factory->AddSignalTree(tSig2, 1.0);

    TCut all = "nLepGood10 == 2 && LepGood_charge[0] == LepGood_charge[1] && LepGood_pt[1] > 20 && nBJetMedium40 >= 1 && nJet40 >= 2 && met_pt > 50 && htJet40j > 200";
    if (bgname.Contains("ttbar")) {
        TFile *fBkg = TFile::Open(Path+"/TTJets_MSDecaysCKM_central_PU_S14_POSTLS170/treeProducerSusyMultilepton/treeProducerSusyMultilepton_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("treeProducerSusyMultilepton");
        //tBkg->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTJetsSem.root");
        factory->AddBackgroundTree(tBkg, 1.0);
    } else  {
        std::cout << "Training not implemented " << std::endl;
        return;
    }

    if (name.Contains("_mm")) {
        all += "abs(LepGood1_pdgId) == 13 && abs(LepGood2_pdgId) == 13";
    } else if (name.Contains("_em")) {
        all += "abs(LepGood1_pdgId) != abs(LepGood2_pdgId)";
    } else if (name.Contains("_ee")) {
        all += "abs(LepGood1_pdgId) == 11 && abs(LepGood2_pdgId) == 11";
    }

    // MET
    factory->AddVariable("mhtJet25 := min(mhtJet25, 400)", 'F');
    factory->AddVariable("met := min(met_pt, 400)", 'F');

    // Jets and HT
    factory->AddVariable("jet1Pt := min(Jet_pt[0], 300)", 'F');
    factory->AddVariable("jet2Pt := min(Jet_pt[1], 300)", 'F');
    //factory->AddVariable("jetptmin := min(Jet1_pt,Jet2_pt)", 'F');
    factory->AddVariable("htJet25  := min(htJet25, 1000)", 'F');
    factory->AddVariable("htJet40j := min(htJet40j, 1000)", 'F');
    factory->AddVariable("nJet25 := min(nJet25, 8)", 'F');

    // Centrality variables
    factory->AddVariable("lepEta2max := max(abs(LepGood_eta[0]),abs(LepGood_eta[1]))", 'F');
    //factory->AddVariable("lepEta2min := min(abs(LepGood_eta[0]),abs(LepGood_eta[1]))", 'F');
    //factory->AddVariable("ptavgEta   := (abs(Jet1_eta)*Jeo1_pt+abs(Jet2_eta)*Jet2_pt+abs(LepGood1_eta)*LepGood1_pt+abs(LepGood2_eta)*LepGood2_pt)/(Jet1_pt+Jet2_pt+LepGood1_pt+LepGood2_pt)", 'F');
    factory->AddVariable("ptavgEtaJets := (abs(Jet_eta[0])*Jet_pt[0]+abs(Jet_eta[1])*Jet_pt[1])/(Jet_pt[0]+Jet_pt[1])", 'F');
 
    // Event reconstruction   
    //factory->AddVariable("bestMTopHad   := min(max(bestMTopHad,100),350)", 'F');
    //factory->AddVariable("bestMTopHadPt := min(max(bestMTopHadPt,0),400)", 'F');
    factory->AddVariable("mtW1   := mt_2(LepGood_pt[0],LepGood_phi[0],met_pt,met_phi)", 'F');
    factory->AddVariable("mtW2   := mt_2(LepGood_pt[1],LepGood_phi[1],met_pt,met_phi)", 'F');
    factory->AddVariable("mtWmin := min(mt_2(LepGood_pt[0],LepGood_phi[0],met_pt,met_phi),mt_2(LepGood_pt[1],LepGood_phi[1],met_pt,met_phi))", 'F');
    
#endif

    factory->SetWeightExpression("1");
    factory->PrepareTrainingAndTestTree( all, all, "SplitMode=Random" );

    factory->BookMethod( TMVA::Types::kLD, "LD", "!H:!V:VarTransform=None:CreateMVAPdfs" );

    TString BDTGopt = "!H:!V:NTrees=200:BoostType=Grad:Shrinkage=0.10:!UseBaggedGrad:nCuts=200:nEventsMin=100:NNodesMax=5";

    BDTGopt += ":CreateMVAPdfs"; // Create Rarity distribution
    factory->BookMethod( TMVA::Types::kBDT, "BDTG", BDTGopt);

    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    fOut->Close();
}
