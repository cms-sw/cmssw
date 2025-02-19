{
  gStyle->SetOptStat();
  gROOT->SetStyle("Plain");
  using namespace std;
  //TChain chain("Events"); // create the chain with tree "T"
  //  gStyle->SetOptStat();
  // gROOT->SetStyle("Plain");

  TChain chainDATA("Events"); // create the chain with tree "T"
  TChain chainMC("Events"); // create the chain with tree "T"

  chainMC.Add("MinBias2010_MC/NtupleLooseTestNew_oneshot_all_MCMinBias.root");
  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_0.root");
  chainMC.Add("MinBias2010_MC/NtupleLoose_all_15apr_23_1.root");

  chainDATA.Add("MinBias2010_15Apr/NtupleLoose_all_15apr_23_2.root");
  chainDATA.Add("MinBias2010_15Apr/NtupleLoose_all_15apr_23_3.root");
  chainDATA.Add("MinBias2010_15Apr/NtupleLoose_all_15apr_23_4.root");
  chainDATA.Add("MinBias2010_15Apr/NtupleLoose_all_15apr_23_5.root");

 
 
  /*chain.Add("NtupleLooseTestNew_oneshot_all_132653.root");
   */
 
  TFile out("histo.root", "RECREATE");
  TCanvas c;

  std::string pt_cut1= "1";
  std::string pt_cut2= "1";
  std::string  scut1 = string("zMuTrkDau1Pt>") + pt_cut1 ;
  std::string  scut2 = string("zMuTrkDau2Pt>") + pt_cut2 ;
  
  TH1F * muIsoDATA= new TH1F("zGoldenDauTrkIsoDATA", "zGoldenDauTrkIsoDATA",  100, 0, 50);
  TH1F * trkIsoDATA = new TH1F("zMuTrkDau2TrkIsoDATA", "zMuTrkDau2TrkIsoDATA",  100, 0, 50);
  TH1F * muStaIsoDATA = new TH1F("zMuStaDauTrkIsoDATA", "zMuStaDauTrkIsoDATA",  100, 0, 50);
  TH1F * muTrkMuIsoDATA = new TH1F("zMuTrkMuDauTrkIsoDATA", "zMuTrkMuDauTrkIsoDATA",  100, 0, 50);
  TH1F * muIsoMC= new TH1F("zGoldenDauTrkIsoMC", "zGoldenDauTrkIsoMC",  100, 0, 50);
  TH1F * trkIsoMC = new TH1F("zMuTrkDau2TrkIsoMC", "zMuTrkDau2TrkIsoMC",  100, 0, 50);
  TH1F * muStaIsoMC = new TH1F("zMuStaDauTrkIsoMC", "zMuStaDauTrkIsoMC",  100, 0, 50);
  TH1F * muTrkMuIsoMC = new TH1F("zMuTrkMuDauTrkIsoMC", "zMuTrkMuDauTrkIsoMC",  100, 0, 50);



  chainDATA->Project("zGoldenDauTrkIsoDATA", "zGoldenDau1TrkIso", scut1.c_str()   );  
  chainDATA->Project("zGoldenDauTrkIsoDATA", "zGoldenDau2TrkIso", scut1.c_str()   );  
  chainDATA->Project("zMuTrkDau2TrkIsoDATA", "zMuTrkDau2TrkIso", scut2.c_str()   );
  chainDATA->Project("zMuTrkMuDauTrkIsoDATA", "zMuTrkMuDau1TrkIso", scut2.c_str()   );
  chainDATA->Project("zMuTrkMuDauTrkIsoDATA", "zMuTrkMuDau2TrkIso", scut2.c_str()   );
  chainDATA->Project("zMuStaDauTrkIsoDATA", "zMuStaDau1TrkIso", scut2.c_str()   );
  chainDATA->Project("zMuStaDauTrkIsoDATA", "zMuStaDau2TrkIso", scut2.c_str()   );


  chainMC->Project("zGoldenDauTrkIsoMC", "zGoldenDau1TrkIso", scut1.c_str()   );  
  chainMC->Project("zGoldenDauTrkIsoMC", "zGoldenDau2TrkIso", scut1.c_str()   );  
  chainMC->Project("zMuTrkDau2TrkIsoMC", "zMuTrkDau2TrkIso", scut2.c_str()   );
  chainMC->Project("zMuTrkMuDauTrkIsoMC", "zMuTrkMuDau1TrkIso", scut2.c_str()   );
  chainMC->Project("zMuTrkMuDauTrkIsoMC", "zMuTrkMuDau2TrkIso", scut2.c_str()   );
  chainMC->Project("zMuStaDauTrkIsoMC", "zMuStaDau1TrkIso", scut2.c_str()   );
  chainMC->Project("zMuStaDauTrkIsoMC", "zMuStaDau2TrkIso", scut2.c_str()   );


  muIsoMC->Sumw2();
  double scale = muIsoDATA->Integral()/  muIsoMC->Integral();
  muIsoMC->Scale(scale);
  
  muIsoDATA->SetMarkerColor(kBlack);
  muIsoDATA->SetMarkerStyle(20);
  muIsoDATA->SetMarkerSize(0.8);
  muIsoDATA->SetLineWidth(2);
  muIsoDATA->SetLineColor(kBlack);
  muIsoMC->SetFillColor(kAzure+7);
  muIsoMC->SetLineWidth(2);
  muIsoMC->SetLineWidth(2); 
  muIsoMC->SetLineColor(kBlue+1);

  muIsoMC->SetMaximum(muIsoDATA->GetMaximum()*1.5 + 2);
  c.SetLogy();

  muIsoMC->Draw("HIST");
  muIsoDATA->Draw("esame");

  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(muIsoDATA,"data");
  leg->AddEntry(muIsoMC,"MC","f");
  leg->Draw();

  std::cout << "MC entries "<< muIsoMC->GetEntries()<< std::endl;
  std::cout << "DATA entries "<< muIsoDATA->GetEntries()<< std::endl;
  
  c.SaveAs( "zGoldenDau1TrkIso.eps");
  leg->Clear(); 


  std::cout << "MC entries "<< trkIsoMC->GetEntries()<< std::endl;
  std::cout << "DATA entries "<< trkIsoDATA->GetEntries()<< std::endl;

  trkIsoMC->Sumw2();
  scale = trkIsoDATA->Integral()/  trkIsoMC->Integral();
  trkIsoMC->Scale(scale);
  
  trkIsoDATA->SetMarkerColor(kBlack);
  trkIsoDATA->SetMarkerStyle(20);
  trkIsoDATA->SetMarkerSize(0.8);
  trkIsoDATA->SetLineWidth(2);
  trkIsoDATA->SetLineColor(kBlack);
  trkIsoMC->SetFillColor(kAzure+7);
  trkIsoMC->SetLineWidth(2);
  trkIsoMC->SetLineWidth(2); 
  trkIsoMC->SetLineColor(kBlue+1);

  trkIsoMC->SetMaximum(trkIsoDATA->GetMaximum()*1.5 + 1);
  trkIsoMC->Draw("HIST");
  trkIsoDATA->Draw("esame");

  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(trkIsoDATA,"data");
  leg->AddEntry(trkIsoMC,"MC","f");
  leg->Draw();

  c.SaveAs( "zMuTrkDau2TrkIso.eps");
  leg->Clear(); 

  muStaIsoMC->Sumw2();
  scale = muStaIsoDATA->Integral()/  muStaIsoMC->Integral();
  muStaIsoMC->Scale(scale);
  
  muStaIsoDATA->SetMarkerColor(kBlack);
  muStaIsoDATA->SetMarkerStyle(20);
  muStaIsoDATA->SetMarkerSize(0.8);
  muStaIsoDATA->SetLineWidth(2);
  muStaIsoDATA->SetLineColor(kBlack);
  muStaIsoMC->SetFillColor(kAzure+7);
  muStaIsoMC->SetLineWidth(2);
  muStaIsoMC->SetLineWidth(2); 
  muStaIsoMC->SetLineColor(kBlue+1);

  muStaIsoMC->SetMaximum(muStaIsoDATA->GetMaximum()*1.5 + 1);
  muStaIsoMC->Draw("HIST");
  muStaIsoDATA->Draw("esame");

  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(muStaIsoDATA,"data");
  leg->AddEntry(muStaIsoMC,"MC","f");
  leg->Draw();

  c.SaveAs( "zMuStaDauTrkIso.eps");
  leg->Clear(); 

  muTrkMuIsoMC->Sumw2();
  scale = muTrkMuIsoDATA->Integral()/  muTrkMuIsoMC->Integral();
  muTrkMuIsoMC->Scale(scale);
  
  muTrkMuIsoDATA->SetMarkerColor(kBlack);
  muTrkMuIsoDATA->SetMarkerStyle(20);
  muTrkMuIsoDATA->SetMarkerSize(0.8);
  muTrkMuIsoDATA->SetLineWidth(2);
  muTrkMuIsoDATA->SetLineColor(kBlack);
  muTrkMuIsoMC->SetFillColor(kAzure+7);
  muTrkMuIsoMC->SetLineWidth(2);
  muTrkMuIsoMC->SetLineWidth(2); 
  muTrkMuIsoMC->SetLineColor(kBlue+1);

  muTrkMuIsoMC->SetMaximum(muTrkMuIsoDATA->GetMaximum()*1.5 + 1);
  muTrkMuIsoMC->Draw("HIST");
  muTrkMuIsoDATA->Draw("esame");

  leg = new TLegend(0.65,0.60,0.85,0.75);
  leg->SetFillColor(kWhite);
  leg->AddEntry(muTrkMuIsoDATA,"data");
  leg->AddEntry(muTrkMuIsoMC,"MC","f");
  leg->Draw();

  c.SaveAs( "zMuTrkMuDauTrkIso.eps");
  leg->Clear(); 
  

  std::string  value[4] = {"zGolden", "zMuTrk", "zMuTrkMu", "zMuSta"};
  for (int z = 0; z<4; ++z  ){
    int min_ = 0;
    int max_ = 150;
    int nBins =75;
    
    
    
  //  std::string smass=(value[z]+"Mass");
    
    TH1F * histoDATA = new TH1F(string(value[z]+"MassDATA").c_str(),string(value[z]+"MassDATA").c_str() , 100, min_, 20);
    TH1F * histoMC = new TH1F(string(value[z]+"MassMC").c_str(),string(value[z]+"MassMC").c_str() , 100, min_, 20);

    //    TH1F * histoDATA = new TH1F(string(value[z]+"MassDATA").c_str(),string(value[z]+"MassDATA").c_str() , 75, min_, 150);
    //    TH1F * histoMC = new TH1F(string(value[z]+"MassMC").c_str(),string(value[z]+"MassMC").c_str() , 75, min_, 150);
    
    
    TH1F * histo2DATA = new TH1F(string(value[z] + "Dau1PtDATA").c_str(),string(value[z] + "Dau1PtDATA").c_str(), nBins, min_, max_);
    TH1F * histo2MC = new TH1F(string(value[z] + "Dau1PtMC").c_str(),string(value[z] + "Dau1PtMC").c_str(), nBins, min_, max_);
    
    
    TH1F * histo3DATA = new TH1F(string(value[z] + "Dau2PtDATA").c_str(),string(value[z] + "Dau2Pt").c_str(), nBins, min_, max_);
    TH1F * histo3MC = new TH1F(string(value[z] + "Dau2PtMC").c_str(),string(value[z] + "Dau2PtMC").c_str(), nBins, min_, max_);



    std::string pt_cut1= "1";
    std::string pt_cut2= "1";
    
    std::string  scut = value[z] + "Dau1Pt>" + pt_cut1 + " && " + value[z] + "Dau2Pt> "+  pt_cut2;
    TCut cut(scut.c_str());
    
    chainDATA->Project(string(value[z] + "MassDATA").c_str(), string(value[z] + "Mass").c_str(), cut  );
    chainMC->Project(string(value[z] + "MassMC").c_str(), string(value[z] + "Mass").c_str(), cut  );
    
    chainDATA->Project(string(value[z] + "Dau1PtDATA").c_str(), string(value[z] + "Dau1Pt").c_str(), cut  );
    chainMC->Project(string(value[z] + "Dau1PtMC").c_str(), string(value[z] + "Dau1Pt").c_str(), cut  );
    
    chainDATA->Project(string(value[z] + "Dau2PtDATA").c_str(), string(value[z] + "Dau2Pt").c_str(), cut );
    chainMC->Project(string(value[z] + "Dau2PtMC").c_str(), string(value[z] + "Dau2Pt").c_str(), cut );


    histoMC->Sumw2();
    scale = histoDATA->Integral()/  histoMC->Integral();
    histoMC->Scale(scale);
    
    histoDATA->SetMarkerColor(kBlack);
    histoDATA->SetMarkerStyle(20);
    histoDATA->SetMarkerSize(0.8);
    histoDATA->SetLineWidth(2);
    histoDATA->SetLineColor(kBlack);
    histoMC->SetFillColor(kAzure+7);
    histoMC->SetLineWidth(2);
    histoMC->SetLineWidth(2); 
    histoMC->SetLineColor(kBlue+1);
    
    histoMC->SetMaximum(histoDATA->GetMaximum()*1.5 + 1);
    histoMC->Draw("HIST");
    histoDATA->Draw("esame");

    leg = new TLegend(0.65,0.60,0.85,0.75);
    leg->SetFillColor(kWhite);
    leg->AddEntry(histoDATA,"data");
    leg->AddEntry(histoMC,"MC","f");
    leg->Draw();
    
    c.SaveAs( string(value[z] + "Mass.eps").c_str());
    leg->Clear(); 

    histo2MC->Sumw2();
    scale = histo2DATA->Integral()/  histo2MC->Integral();
    histo2MC->Scale(scale);
    
    histo2DATA->SetMarkerColor(kBlack);
    histo2DATA->SetMarkerStyle(20);
    histo2DATA->SetMarkerSize(0.8);
    histo2DATA->SetLineWidth(2);
    histo2DATA->SetLineColor(kBlack);
    histo2MC->SetFillColor(kAzure+7);
    histo2MC->SetLineWidth(2);
    histo2MC->SetLineWidth(2); 
    histo2MC->SetLineColor(kBlue+1);
    
    histo2MC->SetMaximum(histo2DATA->GetMaximum()*1.5 + 1);
    histo2MC->Draw("HIST");
    histo2DATA->Draw("esame");

    leg = new TLegend(0.65,0.60,0.85,0.75);
    leg->SetFillColor(kWhite);
    leg->AddEntry(histo2DATA,"data");
    leg->AddEntry(histo2MC,"MC","f");
    leg->Draw();
    
    c.SaveAs(string(value[z] + "Dau1Pt.eps").c_str());
    leg->Clear(); 

    histo3MC->Sumw2();
    scale = histo3DATA->Integral()/  histo3MC->Integral();
    histo3MC->Scale(scale);
    
    histo3DATA->SetMarkerColor(kBlack);
    histo3DATA->SetMarkerStyle(20);
    histo3DATA->SetMarkerSize(0.8);
    histo3DATA->SetLineWidth(2);
    histo3DATA->SetLineColor(kBlack);
    histo3MC->SetFillColor(kAzure+7);
    histo3MC->SetLineWidth(2);
    histo3MC->SetLineWidth(2); 
    histo3MC->SetLineColor(kBlue+1);
    
    histo3MC->SetMaximum(histo3DATA->GetMaximum()*1.5 + 1);
    histo3MC->Draw("HIST");
    histo3DATA->Draw("esame");
  
    leg = new TLegend(0.65,0.60,0.85,0.75);
    leg->SetFillColor(kWhite);
    leg->AddEntry(histo3DATA,"data");
    leg->AddEntry(histo3MC,"MC","f");
    leg->Draw();
        
    // c.SetLogy(0);
    
    c.SaveAs(string(value[z] + "Dau2Pt.eps").c_str());
        

    

    //  c.Write();



}

  

  //  out.Close();
}
