void SeedEfficiency(){

  /*
   * Get the histogram for efficiencies(recSeg,seed,track) vs eta 
   * Get the STA and Seed PT dirstribution in different eta scopes
   *
   * Author:  Shih-Chuan Kao  --- UCR
   *
   */



TFile *file = TFile::Open("hST_Shower_NewSeed.root");
//TFile *file = TFile::Open("pt100_h_rh.root");
 
//  TString hfolder = "pt100_h_rh2";
  TString hfolder = "ST_Shower_New_rh";

// ********************************************************************
// set up for histograms of pT distribution
// ********************************************************************

   double xmin = 0.0;
   double xmax = 2500.0;
   double rebin = 50;
   double nbin  = 2500/rebin ;

   TString suffix = ".jpg";

// ********************************************************************
// reading histograms
// ********************************************************************

 TString name1 = "AllMu__heta_NSeed";
 TString name2 = "AllMu__heta_mu2";
 TString name3 = "AllMu__heta_NSta";
 TString name4 = "AllMu__heta_mu3";
 TString name5 = "AllMu__heta_staPt";
 TString name6 = "AllMu__heta_pt";
 TString name7 = "AllMu__heta_bestPt"; 

 TString plot01 = "SeeddEff"+suffix;
 TString plot02 = "StdEff"+suffix;
 TString plot03 = "RecEff"+suffix;

 TString plot04 = "StaPT_py_ME1"+suffix;
 TString plot05 = "StaPT_py_ME2"+suffix;
 TString plot06 = "StaPT_py_MB"+suffix;

 TString plot07 = "SeedPT_py_ME1"+suffix;
 TString plot08 = "SeedPT_py_ME2"+suffix;
 TString plot09 = "SeedPT_py_MB"+suffix;

 TString plot10 = "BestSeedPT_py_ME1"+suffix;
 TString plot11 = "BestSeedPT_py_ME2"+suffix;
 TString plot12 = "BestSeedPT_py_MB"+suffix;

// ********************************************************************
// Pointers to histograms
// ********************************************************************

    heta_nSeed  = (TH2F *) file->Get("AllMuonSys/"+name1);
    heta_nSim   = (TH2F *) file->Get("AllMuonSys/"+name2);
    heta_sta    = (TH2F *) file->Get("AllMuonSys/"+name3);
    heta_rSeg   = (TH2F *) file->Get("AllMuonSys/"+name4);
    heta_staPt  = (TH2F *) file->Get("AllMuonSys/"+name5);
    heta_seedPt = (TH2F *) file->Get("AllMuonSys/"+name6);
    heta_bestSeedPt = (TH2F *) file->Get("AllMuonSys/"+name7);

// ********************************************************************
// create a folder to store all histograms
// ********************************************************************

  gSystem->mkdir(hfolder);
  gSystem->cd(hfolder);


// *****************************************************************
// main program
// *****************************************************************

 heta_nSeed->ProjectionX("heta_nSeed_pjx",3,20,"");
 //heta_nSeed_pjx->Rebin(10,"heta_nSeed_pjx_r");

 heta_nSim->ProjectionX("heta_nSim_pjx");
 //heta_nSim_pjx ->Rebin(10,"heta_nSim_r");

 heta_sta ->ProjectionX("heta_sta_pjx",2,9,"");
 //heta_sta_pjx   ->Rebin(10,"heta_sta_r");

 heta_rSeg->ProjectionX("heta_rSeg_pjx",3,20,"");
 //heta_rSeg_pjx ->Rebin(10,"heta_rSeg_pjx_r");

 

 float eta =-2.9;
 Int_t   ini_bin = 1; 
 Int_t   last_bin = 59; 
 const Int_t sz = last_bin - ini_bin + 1;

 Float_t xa1[sz]={0.0};
 Float_t ya1[sz]={0.0};
 Float_t xa2[sz]={0.0};
 Float_t ya2[sz]={0.0};
 Float_t xa3[sz]={0.0};
 Float_t ya3[sz]={0.0};

 for (int i = ini_bin; i<last_bin; i++) {
     xa1[i-ini_bin] = eta;
     xa2[i-ini_bin] = eta;
     xa3[i-ini_bin] = eta;

     float nseed = heta_nSeed_pjx->GetBinContent(i);
     float nsim = heta_nSim_pjx->GetBinContent(i);
     float nsta  = heta_sta_pjx->GetBinContent(i);
     float rseg  = heta_rSeg_pjx->GetBinContent(i);

     float eff_seed = 0.0;
     float eff_sta  = 0.0;
     float eff_rec  = 0.0;

     if ( nsim != 0 ) {

        eff_seed = nseed/nsim;
        eff_sta  = nsta/nsim;
        eff_rec  = rseg/nsim;

     }
    
     ya1[i-ini_bin] = eff_seed;  
     ya2[i-ini_bin] = eff_sta;  
     ya3[i-ini_bin] = eff_rec;  

     eta = eta + 0.1;
 }

 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c1 = new TCanvas("c1","",200,8,1000,700);
 c1->SetFillColor(10);
 c1->SetGrid(); 
 c1->GetFrame()->SetFillColor(21);
 c1->GetFrame()->SetBorderSize(12);
 c1->cd();

 eta_seedeff = new TGraph(sz,xa1,ya1);
 eta_seedeff->SetMaximum(1.01);
 eta_seedeff->SetMinimum(0.65);
 eta_seedeff->SetMarkerColor(4);
 eta_seedeff->SetMarkerStyle(21);
 eta_seedeff->SetTitle("seed efficiency");
 eta_seedeff->GetXaxis()->SetTitle(" #eta  ");
 eta_seedeff->GetYaxis()->SetTitle(" seed efficiency ");
 eta_seedeff->Draw("ACP");

 c1->Update();
 c1->Print(plot01);

 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c2 = new TCanvas("c2","",200,8,1000,700);
 c2->SetFillColor(10);
 c2->SetGrid(); 
 c2->GetFrame()->SetFillColor(21);
 c2->GetFrame()->SetBorderSize(12);
 c2->cd();

 eta_stdeff = new TGraph(sz,xa2,ya2);
 eta_stdeff->SetMaximum(1.01);
 eta_stdeff->SetMinimum(0.65);
 eta_stdeff->SetMarkerColor(4);
 eta_stdeff->SetMarkerStyle(21);
 eta_stdeff->SetTitle(" muon track efficiency");
 eta_stdeff->GetXaxis()->SetTitle(" #eta  ");
 eta_stdeff->GetYaxis()->SetTitle(" sta muon efficiency  ");
 eta_stdeff->Draw("ACP");

 c2->Update();
 c2->Print(plot02);

 gStyle->SetOptStat(kFALSE);
 gStyle->SetOptFit(0111);  
 c3 = new TCanvas("c3","",200,8,1000,700);
 c3->SetFillColor(10);
 c3->SetGrid(); 
 c3->GetFrame()->SetFillColor(21);
 c3->GetFrame()->SetBorderSize(12);
 c3->cd();

 eta_receff = new TGraph(sz,xa3,ya3);
 eta_receff->SetMaximum(1.01);
 eta_receff->SetMinimum(0.65);
 eta_receff->SetMarkerColor(4);
 eta_receff->SetMarkerStyle(21);
 eta_receff->SetTitle("rec seg efficiency");
 eta_receff->GetXaxis()->SetTitle(" #eta  ");
 eta_receff->GetYaxis()->SetTitle(" rec seg efficiency  ");
 eta_receff->Draw("ACP");

 c3->Update();
 c3->Print(plot03);


 // STA PT scopes

 heta_staPt->ProjectionY("heta_staPt_pyp1",50,59,"");
 heta_staPt_pyp1->Rebin(rebin,"heta_staPt_pyp_r1");
 heta_staPt->ProjectionY("heta_staPt_pyn1",1,10,"");
 heta_staPt_pyn1->Rebin(rebin,"heta_staPt_pyn_r1");

 TH1F *heta_staPt_py1 = new TH1F("heta_staPt_py1","", nbin, 0., 2500.);
 heta_staPt_py1->Add(heta_staPt_pyp_r1, heta_staPt_pyn_r1, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c4 = new TCanvas("c4","");
 c4->SetFillColor(10);
 c4->SetFillColor(10);

 heta_staPt_py1->SetTitle(" sta Pt for |#eta| > 2.1 ");
 heta_staPt_py1->SetAxisRange(xmin,xmax,"X");
 heta_staPt_py1->DrawCopy();
 heta_staPt_py1->GetXaxis()->SetTitle(" #eta  ");

 c4->Update();
 c4->Print(plot04);

 heta_staPt->ProjectionY("heta_staPt_pyp2",40,49,"");
 heta_staPt_pyp2->Rebin(rebin,"heta_staPt_pyp_r2");
 heta_staPt->ProjectionY("heta_staPt_pyn2",11,20,"");
 heta_staPt_pyn2->Rebin(rebin,"heta_staPt_pyn_r2");

 TH1F *heta_staPt_py2 = new TH1F("heta_staPt_py2","", nbin, 0., 2500.);
 heta_staPt_py2->Add(heta_staPt_pyp_r2, heta_staPt_pyn_r2, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c5 = new TCanvas("c5","");
 c5->SetFillColor(10);
 c5->SetFillColor(10);

 heta_staPt_py2->SetTitle(" sta Pt for 1.0 < |#eta| < 2.1 ");
 heta_staPt_py2->SetAxisRange(xmin,xmax,"X");
 heta_staPt_py2->DrawCopy();
 heta_staPt_py2->GetXaxis()->SetTitle(" #eta  ");

 c5->Update();
 c5->Print(plot05);


 heta_staPt->ProjectionY("heta_staPt_py0",21,39,"");
 heta_staPt_py0->Rebin(rebin,"heta_staPt_py_r0");

 gStyle->SetOptStat("nimou");
 TCanvas *c6 = new TCanvas("c6","");
 c6->SetFillColor(10);
 c6->SetFillColor(10);

 heta_staPt_py_r0->SetTitle(" sta Pt for |#eta| < 1.0 ");
 heta_staPt_py_r0->SetAxisRange(xmin,xmax,"X");
 heta_staPt_py_r0->DrawCopy();
 heta_staPt_py_r0->GetXaxis()->SetTitle(" #eta  ");

 c6->Update();
 c6->Print(plot06);

 // Seed PT scopes

 // For endcap muon , |eta| > 2.1
 heta_seedPt->ProjectionY("heta_seedPt_pyp1",50,59,"");
 heta_seedPt_pyp1->Rebin(rebin,"heta_seedPt_pyp_r1");
 heta_seedPt->ProjectionY("heta_seedPt_pyn1",1,10,"");
 heta_seedPt_pyn1->Rebin(rebin,"heta_seedPt_pyn_r1");

 TH1F *heta_seedPt_py1 = new TH1F("heta_seedPt_py1","", nbin, 0., 2500.);
 heta_seedPt_py1->Add(heta_seedPt_pyp_r1, heta_seedPt_pyn_r1, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c7 = new TCanvas("c7","");
 c7->SetFillColor(10);
 c7->SetFillColor(10);

 heta_seedPt_py1->SetTitle(" seed Pt for |#eta| > 2.1 ");
 heta_seedPt_py1->SetAxisRange(xmin,xmax,"X");
 heta_seedPt_py1->DrawCopy();
 heta_seedPt_py1->GetXaxis()->SetTitle(" #eta  ");

 c7->Update();
 c7->Print(plot07);

 // For endcap muon , 1.0 < |eta| < 2.1
 heta_seedPt->ProjectionY("heta_seedPt_pyp2",40,49,"");
 heta_seedPt_pyp2->Rebin(rebin,"heta_seedPt_pyp_r2");
 heta_seedPt->ProjectionY("heta_seedPt_pyn2",11,20,"");
 heta_seedPt_pyn2->Rebin(rebin,"heta_seedPt_pyn_r2");

 TH1F *heta_seedPt_py2 = new TH1F("heta_seedPt_py2","", nbin, 0., 2500.);
 heta_seedPt_py2->Add(heta_seedPt_pyp_r2, heta_seedPt_pyn_r2, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c8 = new TCanvas("c8","");
 c8->SetFillColor(10);
 c8->SetFillColor(10);

 heta_seedPt_py2->SetTitle(" seed Pt for 1.0 < |#eta| < 2.1 ");
 heta_seedPt_py2->SetAxisRange(xmin,xmax,"X");
 heta_seedPt_py2->DrawCopy();
 heta_seedPt_py2->GetXaxis()->SetTitle(" #eta  ");

 c8->Update();
 c8->Print(plot08);

 // For barrel muon , |eta| < 1.0

 heta_seedPt->ProjectionY("heta_seedPt_py0",21,39,"");
 heta_seedPt_py0->Rebin(rebin,"heta_seedPt_py_r0");

 gStyle->SetOptStat("nimou");
 TCanvas *c9 = new TCanvas("c9","");
 c9->SetFillColor(10);
 c9->SetFillColor(10);

 heta_seedPt_py_r0->SetTitle(" seed Pt for |#eta| < 1.0 ");
 heta_seedPt_py_r0->SetAxisRange(xmin,xmax,"X");
 heta_seedPt_py_r0->DrawCopy();
 heta_seedPt_py_r0->GetXaxis()->SetTitle(" #eta  ");

 c9->Update();
 c9->Print(plot09);

 // Best Seed PT scopes

 // For endcap muon , |eta| > 2.1
 heta_bestSeedPt->ProjectionY("heta_bestSeedPt_pyp1",50,59,"");
 heta_bestSeedPt_pyp1->Rebin(rebin,"heta_bestSeedPt_pyp_r1");
 heta_bestSeedPt->ProjectionY("heta_bestSeedPt_pyn1",1,10,"");
 heta_bestSeedPt_pyn1->Rebin(rebin,"heta_bestSeedPt_pyn_r1");

 TH1F *heta_bestSeedPt_py1 = new TH1F("heta_bestSeedPt_py1","", nbin*2, -2500., 2500.);
 heta_bestSeedPt_py1->Add(heta_bestSeedPt_pyp_r1, heta_bestSeedPt_pyn_r1, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c7 = new TCanvas("c10","");
 c10->SetFillColor(10);
 c10->SetFillColor(10);

 heta_bestSeedPt_py1->SetTitle(" best seed Pt for |#eta| > 2.1 ");
 heta_bestSeedPt_py1->SetAxisRange(-1.*xmax,xmax,"X");
 heta_bestSeedPt_py1->DrawCopy();
 heta_bestSeedPt_py1->GetXaxis()->SetTitle(" #eta  ");

 c10->Update();
 c10->Print(plot10);

 // For endcap muon , 1.0 < |eta| < 2.1
 heta_bestSeedPt->ProjectionY("heta_bestSeedPt_pyp2",40,49,"");
 heta_bestSeedPt_pyp2->Rebin(rebin,"heta_bestSeedPt_pyp_r2");
 heta_bestSeedPt->ProjectionY("heta_bestSeedPt_pyn2",11,20,"");
 heta_bestSeedPt_pyn2->Rebin(rebin,"heta_bestSeedPt_pyn_r2");

 TH1F *heta_bestSeedPt_py2 = new TH1F("heta_bestSeedPt_py2","", nbin*2, -2500., 2500.);
 heta_bestSeedPt_py2->Add(heta_bestSeedPt_pyp_r2, heta_bestSeedPt_pyn_r2, 1., 1.);

 gStyle->SetOptStat("nimou");
 TCanvas *c8 = new TCanvas("c11","");
 c11->SetFillColor(10);
 c11->SetFillColor(10);

 heta_bestSeedPt_py2->SetTitle(" best seed Pt for 1.0 < |#eta| < 2.1 ");
 heta_bestSeedPt_py2->SetAxisRange(-1.*xmax,xmax,"X");
 heta_bestSeedPt_py2->DrawCopy();
 heta_bestSeedPt_py2->GetXaxis()->SetTitle(" #eta  ");

 c11->Update();
 c11->Print(plot11);

 // For barrel muon , |eta| < 1.0

 heta_bestSeedPt->ProjectionY("heta_bestSeedPt_py0",21,39,"");
 heta_bestSeedPt_py0->Rebin(rebin,"heta_bestSeedPt_py_r0");

 gStyle->SetOptStat("nimou");
 TCanvas *c9 = new TCanvas("c12","");
 c12->SetFillColor(10);
 c12->SetFillColor(10);

 heta_bestSeedPt_py_r0->SetTitle(" best seed Pt for |#eta| < 1.0 ");
 heta_bestSeedPt_py_r0->SetAxisRange(-1.*xmax ,xmax,"X");
 heta_bestSeedPt_py_r0->DrawCopy();
 heta_bestSeedPt_py_r0->GetXaxis()->SetTitle(" #eta  ");

 c12->Update();
 c12->Print(plot12);

 gSystem->cd("../");
 file->Close();

// gROOT->ProcessLine(".q");

}
