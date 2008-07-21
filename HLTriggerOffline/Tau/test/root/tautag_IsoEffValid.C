 
{               
   gSystem->Load("drawEfficiency_C");
    gSystem->Load("drawEfficiencies_C");

  TFile f("doubleTauL2_L25_Val_Z2Tau.root");
  
  TString ReleaseVersion("CMSSW173");
  TString Scale("LinearScale");

  TH1F* nMCetaTau=  (TH1F*)f.Get("DQMData/TausAtGenLevel_isolatedL25PixelTau/eta_Tau_GenLevel");
  TH1F* nMCptTau=   (TH1F*)f.Get("DQMData/TausAtGenLevel_isolatedL25PixelTau/pt_Tau_GenLevel");

  TH1F* nMCetaTauJet =    (TH1F*)f.Get("DQMData/TausAtGenLevel_isolatedL25PixelTau/nMC_Taus_vs_etaTauJet");
  TH1F* nMCptTauJet =     (TH1F*)f.Get("DQMData/TausAtGenLevel_isolatedL25PixelTau/nMC_Taus_vs_ptTauJet");
  TH1F* nMCenergyTauJet = (TH1F*)f.Get("DQMData/TausAtGenLevel_isolatedL25PixelTau/nMC_Taus_vs_energyTauJet");

  TH1F* nRecoJetetaTauJet =    (TH1F*)f.Get("DQMData/ReconstructedJet_isolatedL25PixelTau/n_RecoJet_vs_etaTauJet");
  TH1F* nRecoJetptTauJet =     (TH1F*)f.Get("DQMData/ReconstructedJet_isolatedL25PixelTau/n_RecoJet_vs_ptTauJet");
  TH1F* nRecoJetenergyTauJet = (TH1F*)f.Get("DQMData/ReconstructedJet_isolatedL25PixelTau/n_RecoJet_vs_energyTauJet");
  TH1F* nAssociatedTracks =    (TH1F*)f.Get("DQMData/ReconstructedJet_isolatedL25PixelTau/Number_Associated_Tracks");
  TH1F* nSelectedTracks =      (TH1F*)f.Get("DQMData/ReconstructedJet_isolatedL25PixelTau/Number_Selected_Tracks");

  TH1F* nRecoJetLTetaTauJet =    (TH1F*)f.Get("DQMData/ReconstructedJetWithLeadingTrack_isolatedL25PixelTau/n_RecoJet+LeadingTrack_vs_etaTauJet");
  TH1F* nRecoJetLTptTauJet =     (TH1F*)f.Get("DQMData/ReconstructedJetWithLeadingTrack_isolatedL25PixelTau/n_RecoJet+LeadingTrack_vs_ptTauJet");
  TH1F* nRecoJetLTenergyTauJet = (TH1F*)f.Get("DQMData/ReconstructedJetWithLeadingTrack_isolatedL25PixelTau/n_RecoJet+LeadingTrack_vs_energyTauJet");

  TH1F* nTaggedJetsetaTauJet =    (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/n_IsolatedTauTaggedJets_vs_etaTauJet");
  TH1F* nTaggedJetsptTauJet =     (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/n_IsolatedTauTaggedJets_vs_ptTauJet");
  TH1F* nTaggedJetsenergyTauJet = (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/n_IsolatedTauTaggedJets_vs_energyTauJet");

  TH1F* nEMTaggedJetsetaTauJet =    (TH1F*)f.Get("DQMData/TauEMTaggedJets_isolatedL25PixelTau/n_IsolatedTauTaggedJets_vs_etaTauJet");
  TH1F* nEMTaggedJetsetaTauJet =    (TH1F*)f.Get("DQMData/TauEMTaggedJets_ecalIsolation/n_EMIsolatedTauTaggedJets_vs_etaTauJet");
  TH1F* nEMTaggedJetsptTauJet =     (TH1F*)f.Get("DQMData/TauEMTaggedJets_ecalIsolation/n_EMIsolatedTauTaggedJets_vs_ptTauJet");
  TH1F* nEMTaggedJetsenergyTauJet = (TH1F*)f.Get("DQMData/TauEMTaggedJets_ecalIsolation/n_EMIsolatedTauTaggedJets_vs_energyTauJet");


  TH1F* LeadingTrackPt_TaggedTau =   (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/LeadingTrackPt_After_Isolation");
  TH1F* DeltaR_LTandJet_TaggedTau =  (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/DeltaR_LT_and_Jet_After_Isolation");
  TH1F* SignalTracks_TaggedTau =     (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/Signal_Tks_After_Isolation");
  TH1F* SelectedTracks_TaggedTau =   (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/Selected_Tks_After_Isolation");
  TH1F* AssociatedTracks_TaggedTau = (TH1F*)f.Get("DQMData/TauTaggedJets_isolatedL25PixelTau/Associated_Tks_After_Isolation");

  gStyle->SetOptStat("ie");

  cout << endl<< " entries "<< nMCetaTauJet->Integral()<<endl;
  
  c21 = new TCanvas("c21","", 10, 10, 500, 410);
  nMCetaTauJet->SetTitle("# MC Visible Taus (Visible Energy) "+ReleaseVersion);
  nMCetaTauJet->GetXaxis()->SetTitle("eta");
  nMCetaTauJet->DrawNormalized();
  c21->Print(TString("MCetaTauJets.gif"),"gif");


  c22 = new TCanvas("c22","", 10,10,500,410);
  nMCptTauJet->SetTitle("# MC Visible Taus (Visible Eneryg) "+ReleaseVersion);
  nMCptTauJet->GetXaxis()->SetTitle("Pt (GeV)");
  nMCptTauJet->DrawNormalized();
  c22->Print(TString("MCPtTauJets.gif"),"gif");	    
  
  c23 = new TCanvas("c23","", 10,10,500,410);
  nMCenergyTauJet->SetTitle("# MC Visible Taus (Visible Energy) "+ReleaseVersion);
  nMCenergyTauJet->GetXaxis()->SetTitle("Energy (GeV)");
  nMCenergyTauJet->DrawNormalized();
  c23->Print(TString("MCEnergyTauJets.gif"),"gif");
  /*
  c24 = new TCanvas("c24","", 10,10,500,410);
  nMCetaTau->SetTitle("# MC Taus "+ReleaseVersion);
  nMCetaTau->GetXaxis()->SetTitle("#eta");
  nMCetaTau->DrawNormalized(); 
  c24->Print(TString("MCetaTau.gif"),"gif");

  c25 = new TCanvas("c25","", 10,10,500,410);
  nMCptTau->SetTitle("# MC Taus "+ReleaseVersion);
  nMCptTau->GetXaxis()->SetTitle("Pt (GeV)");
  nMCptTau->DrawNormalized();
  c25->Print(TString("MCPtTau.gif"),"gif");	 
  */
  c26 = new TCanvas("c26","", 10,10,500,410);
  nAssociatedTracks->SetTitle("Associated Tracks Matched Reconstructed Jet "+ReleaseVersion);
  nAssociatedTracks->GetXaxis()->SetTitle("# Associated Tracks");
  nAssociatedTracks->Draw();
  c26->Print(TString("AssociatedTracks.gif"),"gif");	 
  
  c27 = new TCanvas("c27","", 10,10,500,410);
  nSelectedTracks->SetTitle("Selected Tracks Matched Reconstructed Jet "+ ReleaseVersion);
  nSelectedTracks->GetXaxis()->SetTitle("# Selected Tracks");
  nSelectedTracks->Draw();
  c27->Print(TString("SelectedTracks.gif"),"gif");	 
  

  //Tau Tagged Jets Characteristics
  
  c28 = new TCanvas("c28","", 110, 110, 600, 510);
  LeadingTrackPt_TaggedTau->SetTitle("Leading Track Pt Tagged Taus "+ ReleaseVersion);
  LeadingTrackPt_TaggedTau->GetXaxis()->SetTitle("Leading Track Pt (GeV)");
  LeadingTrackPt_TaggedTau->Draw();
  c28->Print(TString("LeadingTrackPt_TaggedTaus.gif"),"gif");
  
  c29 = new TCanvas("c29","", 110, 110, 600, 510);
  DeltaR_LTandJet_TaggedTau->SetTitle("#Delta R(LeadingTrack, JetMomentum) Tagged Taus"+ ReleaseVersion);
  DeltaR_LTandJet_TaggedTau->GetXaxis()->SetTitle("#Delta R(LeadingTrack, JetMomentum)");
  DeltaR_LTandJet_TaggedTau->Draw();
  c29->Print(TString("DeltaR_LT_Jet_TaggedTaus.gif"),"gif");

  c30 = new TCanvas("c30","",110,110,600,510);
  SignalTracks_TaggedTau->SetTitle("# Tracks inside Signal cone Tagged Taus > 1.0 GeV "+ReleaseVersion);
  SignalTracks_TaggedTau->GetXaxis()->SetTitle("# Signal Tracks");
  SignalTracks_TaggedTau->Draw();
  c30->Print(TString("SignalTracks_TaggedTaus.gif"),"gif");

  c31 = new TCanvas("c31","",110,110,600,510);
  SelectedTracks_TaggedTau->SetTitle("# Selected Tracks Tagged Taus "+ReleaseVersion);
  SelectedTracks_TaggedTau->GetXaxis()->SetTitle("# Selected Tracks");
  SelectedTracks_TaggedTau->Draw();
  c31->Print(TString("SelectedTracks_TaggedTaus.gif"),"gif");

  c32 = new TCanvas("c32","",110,110,600,510);
  AssociatedTracks_TaggedTau->SetTitle("# Associated Tracks Tagged Taus "+ReleaseVersion);
  AssociatedTracks_TaggedTau->GetXaxis()->SetTitle("# Associated Tracks");
  AssociatedTracks_TaggedTau->Draw();
  c32->Print(TString("AssociatedTracks_TaggedTaus.gif"),"gif");

  /*
  c2 = new TCanvas("c2","",15,15,505,1205);
  
  c2->Divide(1,3);

  c2_1->cd();
  nRecoJetetaTauJet->Draw();
  c2_2->cd();
  nRecoJetptTauJet->Draw();
  c2_3->cd();
  nRecoJetenergyTauJet->Draw();

  c3 = new TCanvas("c3","",20,20,510,1210);

  c3->Divide(1,3);
  
  c3_1->cd();
  nRecoJetLTetaTauJet->Draw();
  c3_2->cd();
  nRecoJetLTptTauJet->Draw();
  c3_3->cd();
  nRecoJetLTenergyTauJet->Draw();  

  c4 = new TCanvas("c4","",25,25,515,1215);

  c4->Divide(1,3);
  
  c4_1->cd();
  nTaggedJetsetaTauJet->Draw();
  c4_2->cd();
  nTaggedJetsptTauJet->Draw();
  c4_3->cd();
  nTaggedJetsenergyTauJet->Draw(); 
  */
  //*****************************************************************************************************

  TH1F* deltaRLeadingTrackJetP =        (TH1F *)f.Get("DQMData/LeadingTrackPtAndDeltaRStudies_isolatedL25PixelTau/DeltaR_LeadingTrack_in_RecoJet");
  TH1F* leadingTrackPtJet      =        (TH1F *)f.Get("DQMData/LeadingTrackPtAndDeltaRStudies_isolatedL25PixelTau/Leading_track_pt_in_RecoJet");

  c21 = new TCanvas("c21","",125,125,615,615);
  deltaRLeadingTrackJetP->Draw();
  c22 = new TCanvas("c22","",150,150,640,640);
  leadingTrackPtJet->Draw();
  
  //====================================================ETA================================================================= 
  TPaveText* text_ptCut = new TPaveText(0.66, 0.82, 0.92, 0.92, "brNDC");
  text_ptCut->SetBorderSize(0);
  text_ptCut->SetFillColor(0);
  text_ptCut->AddText(Form(" P_{T} > %.1f GeV", 5.0));

  TGraphAsymmErrors* gr0 =  new TGraphAsymmErrors(nTaggedJetsetaTauJet, nMCetaTauJet);
  c4 = new TCanvas("c4","IsolationTotalEfficiencyEta",30,30,520,520);
  drawEfficiency("Isolated/MC Visible Tau "+ReleaseVersion, gr0, "#eta (MC Vis. Tau)", (TH1F*) nMCetaTauJet->Clone(), c4, text_ptCut,Scale );
  
  TGraphAsymmErrors* gr2 =  new TGraphAsymmErrors(nRecoJetetaTauJet,  nMCetaTauJet);
  c6 = new TCanvas("c6","CaloJetTotalEfficiencyEta",35,35,525,525);
  drawEfficiency("Jets(Matched)/MC Visible Taus "+ReleaseVersion, gr2, "#eta (MC Vis. Tau)",(TH1F*)  nMCetaTauJet->Clone(), c6, text_ptCut, Scale);
  
  TGraphAsymmErrors* gr100 = new TGraphAsymmErrors(nRecoJetLTetaTauJet, nMCetaTauJet);
  c100 = new TCanvas("c100", "EfficienciesStepByStepEta", 230, 230, 720, 720);
  drawEfficiencies("Different Steps/MC Visible Taus "+ReleaseVersion, gr2, gr100, gr0,gr100,  "#eta (MC Vis. Tau)",(TH1F*)  nMCetaTauJet->Clone(), c100, text_ptCut,"Calo", Scale);

  TGraphAsymmErrors* gr3 =  new TGraphAsymmErrors(nRecoJetLTetaTauJet,  nRecoJetetaTauJet);
  c7 = new TCanvas("c7","FindingLeadTrackPartialEfficiencyEta",35,35,525,525);
  drawEfficiency("Jets+LeadTr/Jets(Matched) "+ReleaseVersion, gr3, "#eta (MC Vis. Tau)",(TH1F*)  nRecoJetetaTauJet->Clone(), c7, text_ptCut, Scale);  
  		 
  TGraphAsymmErrors* gr4 =  new TGraphAsymmErrors(nTaggedJetsetaTauJet,  nRecoJetLTetaTauJet);
  c8 = new TCanvas("c8","IsolationPartialEfficencyEta",35,35,525,525);
  drawEfficiency("Isolated/Jets+LeadTr "+ReleaseVersion, gr4, "#eta (MC Vis. Tau)",(TH1F*)  nRecoJetetaTauJet->Clone(), c8, text_ptCut, Scale); 
  


  //===========================================================Pt=============================================================
  TPaveText* text_etaCut = new TPaveText(0.66, 0.82, 0.92, 0.92, "brNDC");
  text_etaCut->SetBorderSize(0);
  text_etaCut->SetFillColor(0);
  text_etaCut->AddText(Form(" %.1f  < #eta < %.1f ", -2.5, 2.5));

  TGraphAsymmErrors* gr50 =  new TGraphAsymmErrors(nTaggedJetsptTauJet, nMCptTauJet);
  c40 = new TCanvas("c40","IsolationTotalEfficiencyPt",30,30,520,520);
    drawEfficiency("Isolated/MC Visible Tau "+ReleaseVersion, gr50, "P_{T} (GeV/c) (MC Vis. Tau)", (TH1F*) nMCptTauJet->Clone(), c40, text_etaCut, Scale);
  

  TGraphAsymmErrors* gr6 =  new TGraphAsymmErrors(nRecoJetptTauJet,  nMCptTauJet);
  c10 = new TCanvas("c10","CaloJetTotalEfficiencyPt",35,35,525,525);
    drawEfficiency("Jets(Matched)/MC Visible Taus "+ ReleaseVersion, gr6, "P_{T} (GeV/c) (MC Vis. Tau)",(TH1F*)  nMCptTauJet->Clone(), c10, text_etaCut, Scale);
  
  TGraphAsymmErrors* gr101 = new TGraphAsymmErrors(nRecoJetLTptTauJet, nMCptTauJet);
  c101 = new TCanvas("c101", "EfficienciesStepByStepPt", 230, 230, 720, 720);
  drawEfficiencies("Different Steps/MC Visible Taus "+ ReleaseVersion,  gr6, gr101, gr50,gr50,  "P_{T} (GeV/c) (MC Vis. Tau)",(TH1F*)  nMCptTauJet->Clone(), c101, text_etaCut, "Calo", Scale);  
  TGraphAsymmErrors* gr7 =  new TGraphAsymmErrors(nRecoJetLTptTauJet,  nRecoJetptTauJet);
  c11 = new TCanvas("c11","FindingLeadTrackPartialEfficiencyPt",35,35,525,525);
  drawEfficiency("Jets+LeadTr/Jets(Matched) "+ ReleaseVersion, gr7, "P_{T} (GeV/c) (MC Vis. Tau)",(TH1F*)  nRecoJetptTauJet->Clone(), c11, text_etaCut, Scale);  
  		 
  TGraphAsymmErrors* gr8 =  new TGraphAsymmErrors(nTaggedJetsptTauJet,  nRecoJetLTptTauJet);
  c12 = new TCanvas("c12","IsolationPartialEfficencyPt",35,35,525,525);
  drawEfficiency("Isolated/Jets+LeadTr "+ ReleaseVersion, gr8, "P_{T} (GeV/c) (MC Vis. Tau)",(TH1F*)  nRecoJetptTauJet->Clone(), c12, text_etaCut, Scale); 


  //===============================================================Energy========================================================

  TPaveText* text_bothCuts = new TPaveText(0.66, 0.82, 0.92, 0.92, "brNDC");
  text_bothCuts->SetBorderSize(0);
  text_bothCuts->SetFillColor(0);
  text_bothCuts->AddText(Form(" P_{T} > %.1f GeV", 5.0));
  text_bothCuts->AddText(Form(" %.1f  < #eta < %.1f ", -2.5, 2.5));

  TGraphAsymmErrors* gr90 =  new TGraphAsymmErrors(nTaggedJetsenergyTauJet, nMCenergyTauJet);
  c90 = new TCanvas("c90","IsolationTotalEfficiencyEnergy",30,30,520,520);
  drawEfficiency("Isolated/MC Visible Tau "+ReleaseVersion, gr90, "Energy (GeV) (MC Vis. Tau)", (TH1F*) nMCenergyTauJet->Clone(), c90, text_bothCuts, Scale);

  
  TGraphAsymmErrors* gr10 =  new TGraphAsymmErrors(nRecoJetenergyTauJet,  nMCenergyTauJet);
  c14 = new TCanvas("c14","CaloJetTotalEfficiencyEnergy",35,35,525,525);
  drawEfficiency("Jets(Matched)/MC Visible Taus "+ ReleaseVersion, gr10, "Energy (GeV) (MC Vis. Tau)",(TH1F*)  nMCenergyTauJet->Clone(), c14, text_bothCuts, Scale);

  TGraphAsymmErrors* gr102 = new TGraphAsymmErrors(nRecoJetLTenergyTauJet, nMCenergyTauJet);
  c102 = new TCanvas("c102", "EfficienciesStepByStepEnergy", 230, 230, 720, 720);
  drawEfficiencies("Different Steps/MC Visible Taus "+ ReleaseVersion, gr10, gr102, gr90, gr90,  "Energy (GeV) (MC Vis. Tau)",(TH1F*)  nMCenergyTauJet->Clone(), c102, text_bothCuts, "Calo", Scale);

  TGraphAsymmErrors* gr11 =  new TGraphAsymmErrors(nRecoJetLTenergyTauJet,  nRecoJetenergyTauJet);
  c15 = new TCanvas("c15","FindingLeadTrackPartialEfficiencyEnergy",35,35,525,525);
  drawEfficiency("Jets+LeadTr/Jets(Matched) "+ReleaseVersion, gr11, "Energy (GeV) (MC Vis. Tau)",(TH1F*)  nRecoJetenergyTauJet->Clone(), c15, text_bothCuts, Scale);  
  		 
  TGraphAsymmErrors* gr12 =  new TGraphAsymmErrors(nTaggedJetsenergyTauJet,  nRecoJetLTenergyTauJet);
  c16 = new TCanvas("c16","IsolationPartialEfficencyEnergy",35,35,525,525);
  drawEfficiency("Isolated/Jets+LeadTr " +ReleaseVersion, gr12, "Energy (GeV) (MC Vis. Tau)",(TH1F*)  nRecoJetenergyTauJet->Clone(), c16, text_bothCuts, Scale); 

  /*
  TH1F* nTausTaggedvsMatchingConeSize = (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tagged_vs_MatchingConeSize");
  TH1F* nTausTotvsMatchingConeSize =    (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tot_vs_MatchingConeSize");
  TH1F* nTausTaggedvsPtLeadingTrack =   (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tagged_vs_PtLeadingTrack");
  TH1F* nTausTotvsPtLeadingTrack=       (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tot_vs_PtLeadingTrack");
  TH1F* nTausTaggedvsConeIsolation=     (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tagged_vs_coneIsolation");
  TH1F* nTausTotvsConeIsolation=        (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tot_vs_coneIsolation");
  TH1F* nTausTaggedvsConeSignal=        (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tagged_vs_coneSignal");
  TH1F* nTausTotvsConeSignal=           (TH1F *)f.Get("DQMData/TaggingStudies_isolatedL25PixelTau/nTaus_Tot_vs_coneSignal");


  TGraphAsymmErrors* gr13 =  new TGraphAsymmErrors(nTausTaggedvsMatchingConeSize,nTausTotvsMatchingConeSize );
  c17 = new TCanvas("c17","EfficiencyChangingMatchingCone",35,35,525,525);
  drawEfficiency("Isolated/MC Visible Taus " +ReleaseVersion, gr13, "Matching Cone Size (#Delta R)",(TH1F*)  nTausTotvsMatchingConeSize->Clone(), c17, text_bothCuts, Scale); 

  TGraphAsymmErrors* gr14 =  new TGraphAsymmErrors(nTausTaggedvsPtLeadingTrack, nTausTotvsPtLeadingTrack);
  c18 = new TCanvas("c18","EfficiencyChangingPtLeadTr",35,35,525,525);
  drawEfficiency("Isolated/MC Visible Taus " +ReleaseVersion, gr14, "Pt Leading Track",(TH1F*)  nTausTotvsPtLeadingTrack->Clone(), c18, text_bothCuts, Scale); 

  TGraphAsymmErrors* gr15 =  new TGraphAsymmErrors(nTausTaggedvsConeIsolation,nTausTotvsConeIsolation );
  c19 = new TCanvas("c19","EfficiencyChangingIsolationCone",35,35,525,525);
  drawEfficiency("Isolated/MC Visible Taus " +ReleaseVersion, gr15, "Isolation Cone Size (#Delta R)",(TH1F*)  nTausTotvsConeIsolation->Clone(), c19, text_bothCuts, Scale);

  TGraphAsymmErrors* gr16 =  new TGraphAsymmErrors(nTausTaggedvsConeSignal, nTausTotvsConeSignal );
  c20 = new TCanvas("c20","EfficiencyChangingSignalCone",35,35,525,525);
  drawEfficiency("Isolated/MC Visible Taus " +ReleaseVersion, gr16, "Signal Cone Size (#Delta R)",(TH1F*) nTausTotvsConeSignal->Clone(), c20, text_bothCuts, Scale);
  */
}


