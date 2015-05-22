void trainMVA_2lss(TString name) {
    TString Path = "/data/b/botta/TTHAnalysis/trees/TREES_250513_HADD";
    //gROOT->ProcessLine(".L ../../../python/plotter/functions.cc+");

    TFile *fOut = new TFile(name+".root","RECREATE");
    TMVA::Factory *factory = new TMVA::Factory(name, fOut, "!V:!Color");

    TFile *fSig = TFile::Open(Path+"/TTH122/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
    TTree *tSig = (TTree *) fSig->Get("ttHLepTreeProducerBase");
    //tSig->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTH122.root");
    factory->AddSignalTree(tSig, 1.0);
    fSig = TFile::Open(Path+"/TTH127/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
    tSig = (TTree *) fSig->Get("ttHLepTreeProducerBase");
    //tSig->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTH127.root");
    factory->AddSignalTree(tSig, 1.0);
    //fSig = TFile::Open(Path+"/TTH/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
    //tSig = (TTree *) fSig->Get("ttHLepTreeProducerBase");
    //factory->AddSignalTree(tSig, 1.0);

    TCut all = "nLepGood == 2 && LepGood1_charge == LepGood2_charge && nBJetMedium25 >= 1 && nJet25 >= 4 && LepGood2_pt > 20 && LepGood1_pt+LepGood2_pt+met > 100";
    if (name.Contains("ttW")) {
        TFile *fBkg = TFile::Open(Path+"/TTWJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
        factory->AddBackgroundTree(tBkg, 1.0);
    } else if (name.Contains("ttbar")) {
        TFile *fBkg = TFile::Open(Path+"/TTJetsSem/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
        //tBkg->AddFriend("sf/t", Path+"/2_finalmva_2lss_v2/evVarFriend_TTJetsSem.root");
        factory->AddBackgroundTree(tBkg, 1.0);
        //fBkg = TFile::Open(Path+"/TTJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        //tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
        //factory->AddBackgroundTree(tBkg, 0.2);
    } else if (name.Contains("mix")) {
        TFile *fBkg1 = TFile::Open(Path+"/TTJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg1 = (TTree *) fBkg1->Get("ttHLepTreeProducerBase");
        factory->AddBackgroundTree(tBkg1, 1.0);
        TFile *fBkg2 = TFile::Open(Path+"/TTWJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg2 = (TTree *) fBkg2->Get("ttHLepTreeProducerBase");
        factory->AddBackgroundTree(tBkg2, 1.0);
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

    //factory->AddSpectator("MVA_2LSS_4j_6var", 'F');

    // Dileptons
    //factory->AddVariable("lep2Pt := min(LepGood2_pt, 200)", 'F');
    //factory->AddVariable("htll := min(LepGood1_pt+LepGood2_pt, 400)", 'F');
    //factory->AddVariable("ptll := min(pt2l, 240)", 'F');
    //factory->AddVariable("mll := min(mass_2(LepGood1_pt,LepGood1_eta,LepGood1_phi,LepGood1_mass, LepGood2_pt,LepGood2_eta,LepGood2_phi,LepGood2_mass), 240)", 'F');
    //factory->AddVariable("drll := min(deltaR(LepGood1_eta,LepGood1_phi, LepGood2_eta,LepGood2_phi), 5)", 'F');

    // MET
    factory->AddVariable("mhtJet25 := min(mhtJet25, 300)", 'F');
    //factory->AddVariable("met := min(met, 300)", 'F');

    // Jets and HT
    factory->AddVariable("jet1Pt := min(Jet1_pt, 300)", 'F');
    factory->AddVariable("jet2Pt := min(Jet2_pt, 300)", 'F');
    //factory->AddVariable("jetptmin := min(Jet1_pt,Jet2_pt)", 'F');
    factory->AddVariable("htJet25 := min(htJet25, 1000)", 'F');

    // Centrality variables
    //factory->AddVariable("lepEta2max := max(abs(LepGood1_eta),abs(LepGood2_eta))", 'F');
    //factory->AddVariable("lepEta2min := min(abs(LepGood1_eta),abs(LepGood2_eta))", 'F');
    //factory->AddVariable("ptavgEta   := (abs(Jet1_eta)*Jeo1_pt+abs(Jet2_eta)*Jet2_pt+abs(LepGood1_eta)*LepGood1_pt+abs(LepGood2_eta)*LepGood2_pt)/(Jet1_pt+Jet2_pt+LepGood1_pt+LepGood2_pt)", 'F');

    //factory->AddVariable("ptavgEtaJets := (abs(Jet1_eta)*Jet1_pt+abs(Jet2_eta)*Jet2_pt)/(Jet1_pt+Jet2_pt)", 'F');

    factory->AddVariable("htJet25ratio1224Lep := (LepGood1_pt*(abs(LepGood1_eta)<1.2) + LepGood2_pt*(abs(LepGood2_eta)<1.2) + Jet1_pt*(abs(Jet1_eta) < 1.2) + Jet2_pt*(abs(Jet2_eta) < 1.2) + Jet3_pt*(abs(Jet3_eta) < 1.2) + Jet4_pt*(abs(Jet4_eta) < 1.2) + Jet5_pt*(abs(Jet5_eta) < 1.2) + Jet6_pt*(abs(Jet6_eta) < 1.2) + Jet7_pt*(abs(Jet7_eta) < 1.2) + Jet8_pt*(abs(Jet8_eta) < 1.2))/ (LepGood1_pt + LepGood2_pt + Jet1_pt*(abs(Jet1_eta) < 2.4) + Jet2_pt*(abs(Jet2_eta) < 2.4) + Jet3_pt*(abs(Jet3_eta) < 2.4) + Jet4_pt*(abs(Jet4_eta) < 2.4) + Jet5_pt*(abs(Jet5_eta) < 2.4) + Jet6_pt*(abs(Jet6_eta) < 2.4) + Jet7_pt*(abs(Jet7_eta) < 2.4) + Jet8_pt*(abs(Jet8_eta) < 2.4))", 'F');

  
    // Event reconstruction   
    //factory->AddVariable("bestMTopHad   := min(max(bestMTopHad,100),350)", 'F');
    factory->AddVariable("bestMTopHadPt := min(max(bestMTopHadPt,0),400)", 'F');
    //factory->AddVariable("mtW1 := mt_2(LepGood1_pt,LepGood1_phi,met,met_phi)", 'F');
    
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
