// stack macro
// Pasquale Noli
{

  //  gROOT->SetStyle("Plain");

 //  #include <exception>;
#include <sstream>
#include <string>


  TCanvas *c1 = new TCanvas("c1","Stack plot",10,10,900,900);


   string zmmName = "Analysis_zmm_2_4.root";     
   string wmnName = "Analysis_wmn_2_4.root";     
   string ttbarName = "Analysis_ttbar_2_4.root"; 
   string qcdName = "Analysis_qcd_2_4_all.root";  

  THStack hs("hs","#mu #mu invariant mass distribution");
  TFile z(zmmName.c_str()) ; 
  TFile w( wmnName.c_str()) ; 
  TFile tt(ttbarName.c_str()) ; 
  TFile qcd(qcdName.c_str()) ; 
  //TFile ("analysis_ztautau_45pb_trackIso_3.root");
 
  double lumi = .001;
  double lumiZ = 10. ;
  double lumiW = 10.;
  double lumiQ = 10.;
  double lumiT =10.;

  /* TFile z("analysis_Z_10pb_trackIso_3.root"); 
  TFile w("analysis_W_10pb_trackIso_3.root");
  TFile tt("analysis_ttbar_10pb_trackIso_3.root");
  TFile qcd("analysis_QCD_10pb_trackIso_3.root");*/


  TH1F *h1 = (TH1F*)z.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  h1->SetFillColor(kRed);
  h1->SetLineColor(kRed);

  h1->Scale(lumi/lumiZ);
  double i1 = h1->Integral(60,120);
  TH1F *h2 = (TH1F*)w.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  h2->SetFillColor(kBlue);
  h2->SetLineColor(kBlue);
  h2->Scale(lumi/lumiW);
  double i2 = h2->Integral(60,120);


  
  TH1F *h3 = (TH1F*)tt.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  h3->SetFillColor(kBlack);
  h3->SetLineColor(kBlack);
  h3->Scale(lumi/lumiT);
  double i3 = h3->Integral(60,120);
  //  h1->Scale(0.7509);
    
TH1F *h4 = (TH1F*)qcd.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  h4->SetFillColor(kGreen);
  h4->SetLineColor(kGreen);
  h4->Scale(lumi/lumiQ);
  double i4 = h4->Integral(60,120);
  /* TH1F *h5 = (TH1F*)tau.Get("goodZToMuMu1HLTPlots/zMass");
  h5->SetFillColor(kYellow);
  h5->SetLineColor(kYellow);
 
 */

  //  h1->Scale(0.7509);
 
  //  hs.Add(h5);
  hs.Add(h4);
  hs.Add(h3);
  hs.Add(h2);
  hs.Add(h1);
  hs.Draw("HIST");
  hs->GetXaxis()->SetTitle("M_{#mu #mu} (GeV/c^{2}");
    hs.GetYaxis()->SetTitle("Entries/(GeV/c^{2})     ");
 hs.GetYaxis()->SetTitleOffset(1.);
  hs.GetYaxis()->SetLabelOffset(.0);
  hs.GetYaxis()->SetLabelSize(.03);


  leg = new TLegend(0.6,0.6,0.89,0.89);
  leg->AddEntry(h1,"Z->#mu #mu","f");
  leg->AddEntry(h2,"W->#mu #nu","f");
 
  leg->AddEntry(h4,"QCD","f");
  leg->AddEntry(h3,"t#bar{t}","f"); 
 //leg->AddEntry(h5,"Z->#tau #tau","l");
  leg->SetFillColor(kWhite);
  leg->Draw();
  cout<<"zmm (60-120) = " << i1 <<endl;
  cout<<"w (60-120) = " << i2 <<endl;
   cout<<"QCD (60-120) = " << i4 <<endl;
 cout<<"ttbar (60-120) = " << i3 <<endl; 
 c1->SetLogy();
 string  outName = "zmm1hltOneMuonWithEtaMoreThan2p1.eps"; 
   c1->SaveAs(outName.c_str());

}
}
