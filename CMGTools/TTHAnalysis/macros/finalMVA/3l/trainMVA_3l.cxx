void trainMVA_3l(TString name) {
    TFile *fOut = new TFile(name+".root","RECREATE");
    TMVA::Factory *factory = new TMVA::Factory(name, fOut, "!V:!Color");

    TFile *fSig = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/TTH/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
    TTree *tSig = (TTree *) fSig->Get("ttHLepTreeProducerBase");
    tSig.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_TTH.root")
    factory->AddSignalTree(tSig, 1.0);

    //&& min(LepGood1_mva,min(LepGood2_mva,LepGood3_mva)) >= -0.2
    TCut all = "nLepGood == 3  && abs(mZ1-91.2)>10 && (nJet25 >= 4 || (met*0.00397 + mhtJet25*0.00265 - 0.184 > 0.0 + 0.1*(mZ1 > 0))) && nBJetLoose25 >= 2";

    if (name.Contains("ttW")) {
        TFile *fBkg = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/TTWJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
	tBkg.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_TTWJets.root")
        factory->AddBackgroundTree(tBkg, 1.0);
    } else if (name.Contains("ttbar")) {
        TFile *fBkg = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/TTLep/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
	tBkg.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_TTLep.root")
        factory->AddBackgroundTree(tBkg, 1.0);
    } else if (name.Contains("WZ")) {
        TFile *fBkg = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/WZJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg = (TTree *) fBkg->Get("ttHLepTreeProducerBase");
	tBkg.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_WZJets.root")
        factory->AddBackgroundTree(tBkg, 1.0);	
    } else if (name.Contains("mix")) {
        TFile *fBkg1 = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/TTLep/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg1 = (TTree *) fBkg1->Get("ttHLepTreeProducerBase");
	tBkg1.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_TTLep.root")
        factory->AddBackgroundTree(tBkg1, 1.0);
        TFile *fBkg2 = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/TTWJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg2 = (TTree *) fBkg2->Get("ttHLepTreeProducerBase");
	tBkg2.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_TTWjets.root")
        factory->AddBackgroundTree(tBkg2, 1.0);
	TFile *fBkg3 = TFile::Open("/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/WZJets/ttHLepTreeProducerBase/ttHLepTreeProducerBase_tree.root");
        TTree *tBkg3 = (TTree *) fBkg3->Get("ttHLepTreeProducerBase");
	tBkg3.AddFriend("newMVA/t", "/afs/cern.ch/work/b/botta/TTHAnalysis/trees/TREES_270213_HADD/0_leptonMVA_v3/lepMVAFriend_WZJets.root")
        factory->AddBackgroundTree(tBkg3, 1.0);
    } else  {
        std::cout << "Training not implemented " << std::endl;
        return;
    }

//     if (name.Contains("_mm")) {
//         all += "abs(LepGood1_pdgId) == 13 && abs(LepGood2_pdgId) == 13";
//     } else if (name.Contains("_em")) {
//         all += "abs(LepGood1_pdgId) != abs(LepGood2_pdgId)";
//     } else if (name.Contains("_ee")) {
//         all += "abs(LepGood1_pdgId) == 11 && abs(LepGood2_pdgId) == 11";
//     }

//     if      (name.Contains("_2jet")) all += "nJet25 == 2";
//     else if (name.Contains("_3jet")) all += "nJet25 == 3"; 
//     else if (name.Contains("_4jet")) all += "nJet25 == 4"; 
//     else if (name.Contains("_5jet")) all += "nJet25 == 5"; 
//     else if (name.Contains("_6jet")) all += "nJet25 >= 6"; 
//     if (name.Contains("_wjj")) all += "bestMWjj > 0";

//     // VARIABLES WE ALWAYS USE, IRRESPECTIVELY OF JET AND LEPTON BIN
//     factory->AddVariable("htJet25 := min(htJet25,2000)", 'F');
//     factory->AddVariable("min_Lep_eta := min(abs(LepGood1_eta),abs(LepGood2_eta))", 'F');
//     factory->AddVariable("max_Lep_eta := max(abs(LepGood1_eta),abs(LepGood2_eta))", 'F');
//     //factory->AddVariable("max_Jet_eta := max(abs(Jet1_eta),abs(Jet2_eta))", 'F');
//     factory->AddVariable("wavg_eta := (abs(Jet1_eta)*Jet1_pt+abs(Jet2_eta)*Jet2_pt+abs(LepGood1_eta)*LepGood1_pt+abs(LepGood2_eta)*LepGood2_pt)/(Jet1_pt+Jet2_pt+LepGood1_pt+LepGood2_pt)", 'F');
//     //factory->AddVariable("HT4 := (Jet1_pt+Jet2_pt+LepGood1_pt+LepGood2_pt)", 'F');
//     //factory->AddVariable("Lep1_Q_eta := LepGood1_eta*LepGood1_charge", 'F');
//     //factory->AddVariable("Lep2_Q_eta := LepGood2_eta*LepGood2_charge", 'F');
//     factory->AddVariable("minMllAFSS", 'F');
//     //factory->AddVariable("ht2l := min(LepGood1_pt+LepGood2_pt,300)", 'F');
//     factory->AddVariable("q2l := LepGood1_charge+LepGood2_charge", 'I');
//     factory->AddVariable("Jet_pt1 := max(Jet1_pt,Jet2_pt)", 'F');
//     factory->AddVariable("Jet_pt2 := min(Jet1_pt,Jet2_pt)", 'F');
//     factory->AddVariable("Jet2_btagCSV", 'F');
    
//     // VARIABLES THAT DEPEND ON JET BINNIG
//     if (!name.Contains("jet")) {
//         factory->AddVariable("nJet25", 'I');
//     }
//     if (!name.Contains("2jet")) {
//         factory->AddVariable("minMWjj := min(minMWjj,200)", 'F');
//         factory->AddVariable("minMWjjPt", 'F');
//         factory->AddVariable("bestMWjj", 'F');
//         factory->AddVariable("bestMTopHad", 'F');
//     }

//     // VARIABLES THAT DEPEND ON LEPTON FINAL STATE
//     if (!(name.Contains("_ee") || name.Contains("_mm") || name.Contains("_em"))) {
//         //factory->AddVariable("nEle := (abs(LepGood1_pdgId) == 11) + (abs(LepGood2_pdgId) == 11)", 'I');
//     }



    //VARIABLES
    factory->AddVariable("minMWjj := min(minMWjj,200)", 'F');
    factory->AddVariable("minMWjjPt", 'F');
    factory->AddVariable("bestMWjj", 'F');
    factory->AddVariable("bestMTopHad", 'F');
    factory->AddVariable("nJet25", 'I');
    factory->AddVariable("q3l := LepGood1_charge+LepGood2_charge+LepGood3_charge", 'I');
    factory->AddVariable("m3l",'F');
    //factory->AddVariable("LepGood3_pt",'F');
    factory->AddVariable("Jet1_pt", 'F');
    factory->AddVariable("Jet2_pt", 'F');
    factory->AddVariable("Jet2_btagCSV:= max(0,Jet2_btagCSV)", 'F');
    factory->AddVariable("max_Lep_eta := max(max(abs(LepGood1_eta),abs(LepGood2_eta)),abs(LepGood3_eta))", 'F');
    factory->AddVariable("minMllAFOS",'F');
    //factory->AddVariable("minMllAFSS",'F');
    //factory->AddVariable("maxMllAFOS",'F');
    //factory->AddVariable("maxMllAFSS",'F');
    factory->AddVariable("minDrllAFOS",'F');


    factory->SetWeightExpression("1");
    factory->PrepareTrainingAndTestTree( all, all, "SplitMode=Random" );

    factory->BookMethod( TMVA::Types::kLD, "LD", "!H:!V:VarTransform=None" );

    TString BDTGopt = "!H:!V:NTrees=500:BoostType=Grad:Shrinkage=0.05:!UseBaggedGrad:nCuts=200:nEventsMin=100:NNodesMax=5";

    BDTGopt += ":CreateMVAPdfs"; // Create Rarity distribution
    factory->BookMethod( TMVA::Types::kBDT, "BDTG", BDTGopt);

    //TString ee = "abs(LepGood1_pdgId) == 11 && abs(LepGood2_pdgId) == 11", mm = "abs(LepGood1_pdgId) == 13 && abs(LepGood2_pdgId) == 13", em = "abs(LepGood1_pdgId) != abs(LepGood2_pdgId)";
    //TString allvars, allvars0;

 //    // =============== CATEGORIZE BY FINAL STATE =================
//     // this works o, it's off just to save time
//     #if 0
//     if (!(name.Contains("_ee") || name.Contains("_mm") || name.Contains("_em"))) {
//         TMVA::MethodCategory* BDTG_Cat_fs = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_fs","" ));
//         //allvars = "nJet25:htJet30:minMWjj:minMWjjPt:bestMWjj:Jet_ptMax:Jet2_btagCSV:bestMTopHad:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         allvars = "nJet25:htJet30:Jet_ptMax:max_Lep_eta:minMllAFSS";
//         BDTG_Cat_fs->AddMethod(mm.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_mm",  BDTGopt);
//         BDTG_Cat_fs->AddMethod(ee.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_ee",  BDTGopt);
//         BDTG_Cat_fs->AddMethod(em.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_em",  BDTGopt);
//         //factory->BookMethod( TMVA::Types::kBDT, "BDTG2", "!H:!V:NTrees=500:BoostType=Grad:Shrinkage=0.05:!UseBaggedGrad:nCuts=200:nEventsMin=400:NNodesMax=7");
//     }
//     #endif

//     // =============== CATEGORIZE BY FINAL STATE =================
//     // this works o, it's off just to save time
//     #if 0
//     if (!(name.Contains("_ee") || name.Contains("_mm") || name.Contains("_em"))) {
//         TMVA::MethodCategory* BDTG_Cat_fs = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_fs","" ));
//         allvars = "nJet25:htJet30:q2l:minMWjj:minMWjjPt:bestMWjj:Jet_ptMax:Jet2_btagCSV:bestMTopHad:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         BDTG_Cat_fs->AddMethod(mm.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_mm",  BDTGopt);
//         BDTG_Cat_fs->AddMethod(ee.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_ee",  BDTGopt);
//         BDTG_Cat_fs->AddMethod(em.Data(), allvars, TMVA::Types::kBDT, "BDTG_Cat_fs_em",  BDTGopt);
//         //factory->BookMethod( TMVA::Types::kBDT, "BDTG2", "!H:!V:NTrees=500:BoostType=Grad:Shrinkage=0.05:!UseBaggedGrad:nCuts=200:nEventsMin=400:NNodesMax=7");
//     }
//     #endif

//     // =============== CATEGORIZE BY NUMBER OF JETS =================
//     if (!name.Contains("jet")) {
//         // this works ok, it's off just to save time
//         #if 0
//         TMVA::MethodCategory* BDTG_Cat_jets = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_jets","" ));
//         allvars  = "nEle:htJet30:q2l:minMWjj:minMWjjPt:bestMWjj:Jet_ptMax:Jet2_btagCSV:bestMTopHad:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         allvars0 = "nEle:htJet30:q2l:Jet_ptMax:Jet2_btagCSV:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         BDTG_Cat_jets->AddMethod("nJet25 == 2", allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 3",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 4",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 5",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 >= 6",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6",  BDTGopt);
//         #endif

//         // this works ok, it's off just to save time
//         #if 0
//         TMVA::MethodCategory* BDTG_Cat_jets = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_jets","" ));
//         allvars  = "nEle:htJet30:q2l:minMWjj:minMWjjPt:bestMWjj:Jet_ptMax:Jet2_btagCSV:bestMTopHad:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         allvars0 = "nEle:htJet30:q2l:Jet_ptMax:Jet2_btagCSV:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//         BDTG_Cat_jets->AddMethod("nJet25 == 2", allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 3",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 4",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 == 5",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5",  BDTGopt);
//         BDTG_Cat_jets->AddMethod("nJet25 >= 6",  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6",  BDTGopt);
//         #endif



//         #if 0
//         // =============== CATEGORIZE ALSO BY FINAL STATE =================
//         if (!(name.Contains("_ee") || name.Contains("_mm") || name.Contains("_em"))) {
//             TMVA::MethodCategory* BDTG_Cat_jets_fs = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_jets_fs","" ));
//             allvars  = "htJet30:q2l:minMWjj:minMWjjPt:bestMWjj:Jet_ptMax:Jet2_btagCSV:bestMTopHad:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//             allvars0 = "htJet30:q2l:Jet_ptMax:Jet2_btagCSV:min_Lep_eta:max_Lep_eta:minMllAFSS:ht2l";
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+ee).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_ee",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_ee",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_ee",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_ee",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_ee",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+em).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_em",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_em",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_em",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_em",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_em",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+mm).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_mm",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_mm",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_mm",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_mm",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_mm",  BDTGopt);

//             // these two below are test that work worse than the above one
//             #if 0
//             BDTG_Cat_jets_fs = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_jets_fs_2","" ));
//             BDTGopt = "!H:!V:NTrees=200:BoostType=Grad:Shrinkage=0.05:!UseBaggedGrad:nCuts=200:nEventsMin=100:NNodesMax=5";
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+ee).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_ee_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_ee_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_ee_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_ee_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_ee_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+em).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_em_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_em_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_em_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_em_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_em_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+mm).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_mm_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_mm_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_mm_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_mm_2",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_mm_2",  BDTGopt);

//             BDTG_Cat_jets_fs = dynamic_cast<TMVA::MethodCategory*>(factory->BookMethod( TMVA::Types::kCategory, "BDTG_Cat_jets_fs_3","" ));
//             BDTGopt = "!H:!V:NTrees=100:BoostType=Grad:Shrinkage=0.2:!UseBaggedGrad:nCuts=200:nEventsMin=100:NNodesMax=5";
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+ee).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_ee_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_ee_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_ee_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_ee_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+ee).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_ee_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+em).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_em_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_em_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_em_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_em_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+em).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_em_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 2 && "+mm).Data(), allvars0,  TMVA::Types::kBDT, "BDTG_Cat_jets_2_mm_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 3 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_3_mm_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 4 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_4_mm_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 == 5 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_5_mm_3",  BDTGopt);
//             BDTG_Cat_jets_fs->AddMethod(("nJet25 >= 6 && "+mm).Data(),  allvars,  TMVA::Types::kBDT, "BDTG_Cat_jets_6_mm_3",  BDTGopt);
//             #endif

//         }
//         #endif
//     }



    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();

    fOut->Close();
}
