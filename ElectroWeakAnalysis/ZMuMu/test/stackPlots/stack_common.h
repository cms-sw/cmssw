#ifndef STACK_COMMON_H
#define STACK_COMMON_H

#include <iostream>
using namespace std;
#include "TChain.h"



const int canvasSizeX = 500;
const int canvasSizeY = 500;

const Color_t zLineColor = kAzure+2;
const Color_t zFillColor = kAzure+6;
const Color_t wLineColor = kAzure+4;
const Color_t wFillColor = kAzure+2;
const Color_t qcdLineColor = kGreen+3;
const Color_t qcdFillColor = kGreen+1;
const Color_t ttFillColor =  kRed+1;
const Color_t ttLineColor = kRed+3;
const Color_t ztFillColor =  kYellow-9;
const Color_t ztLineColor = kYellow-1;


const double lumi =0.01867 ;
//const double lumi =1.0 ;
const double lumiZ = 100. ;
const double lumiW = 100.;
const double lumiQ = 35.;
//scaling ttbar from 94.3 to 162.
const double lumiT =100. * (94.3/162.);
const double lumiZT =100.;


const double mMin = 60;
const double mMax = 120;


/// cuts common....
TCut kin_common("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1TrkIso< 3.0 && zGoldenDau1TrkIso < 3.0 && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");



TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");


TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1 && zGoldenDau1HLTBit==1");
TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1 && zGoldenDau2HLTBit==1");

 
TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");





//TCanvas *c1 = new TCanvas("c1","Stack plot",10,10,canvasSizeX, canvasSizeY);
TCanvas *c1 = new TCanvas("c1","Stack plot");









void setHisto(TH1 * h, Color_t fill, Color_t line, double scale, int rebin) {
  h->SetFillColor(fill);
  h->SetLineColor(line);
  h->Scale(scale);
  h->Rebin(rebin);  
}

void stat(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 *h5, TH1 * hdata, int rebin) {
  double a = mMin/rebin, b = mMax/rebin;
  double i1 = h1->Integral(a, b);
  double i2 = h2->Integral(a, b);
  double i3 = h3->Integral(a, b);
  double i4 = h4->Integral(a, b);
  double i5 = h5->Integral(a, b);

  double idata = hdata != 0 ? hdata->Integral(a, b) : 0;
  std::cout.setf(0,ios::floatfield);
  std::cout.setf(ios::fixed,ios::floatfield);
  std::cout.precision(1);
  std::cout <<"Zmm (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i1 <<std::endl;
  std::cout.precision(1);
  std::cout <<"Wmn (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i2 <<std::endl;
  std::cout.precision(1);
  std::cout <<"QCD (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i4 <<std::endl;
  std::cout.precision(1);
  std::cout <<"tt~ (" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i3 <<std::endl; 
  std::cout.precision(1);
  std::cout <<"ztautau (" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i5 <<std::endl; 
  std::cout.precision(1);
  std::cout <<"data (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << idata <<std::endl; 
}

void makeStack(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 * h5, TH1 * hdata,
	       double min, int rebin) {
  setHisto(h1, zFillColor, zLineColor, lumi/lumiZ, rebin);
  setHisto(h2, wFillColor, wLineColor, lumi/lumiW, rebin);
  setHisto(h3, ttFillColor, ttLineColor, lumi/lumiT, rebin);
  setHisto(h4, qcdFillColor, qcdLineColor, lumi/lumiQ, rebin);
  setHisto(h5, ztFillColor, ztLineColor, lumi/lumiZT, rebin);

  THStack * hs = new THStack("hs","");
  hs->Add(h5);
  hs->Add(h4);
  hs->Add(h3);
  hs->Add(h2);
  hs->Add(h1);

   hs->Draw("HIST");
  if(hdata != 0) {
    hdata->SetMarkerStyle(20);
    hdata->SetMarkerSize(1.0);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(2);
    hdata->SetLineColor(kBlack);
    hdata->Rebin(rebin); 
    hdata->Draw("epsame");
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    // log plots, so the maximum should be one order of magnitude more...
    hs->SetMaximum( pow(10 , 1 + int(log( hdata->GetMaximum() ) )  ));
    //    gStyle->SetErrorX(.5);
}
  hs->SetMinimum(min);
  hs->GetXaxis()->SetTitle("m_{#mu^{+} #mu^{-}} (GeV/c^{2})");

  std::string yTag = "";
  switch(rebin) {
  case 1: yTag = "events/(GeV/c^{2})"; break;
  case 2: yTag = "events/(2 GeV/c^{2})"; break;
  case 3: yTag = "events/(3 GeV/c^{2})"; break;
  case 4: yTag = "events/(4 GeV/c^{2})"; break;
  case 5: yTag = "events/(5 GeV/c^{2})"; break;
  case 10: yTag = "events/(10 GeV/c^{2})"; break;
  default:
    std::cerr << ">>> ERROR: set y tag for rebin = " << rebin << std::endl;
  };

  hs->GetYaxis()->SetTitle(yTag.c_str());
  hs->GetXaxis()->SetTitleSize(0.05);
  hs->GetYaxis()->SetTitleSize(0.05);
  hs->GetXaxis()->SetTitleOffset(1.2);
  hs->GetYaxis()->SetTitleOffset(1.2);
  hs->GetYaxis()->SetLabelOffset(0);
  hs->GetXaxis()->SetLabelSize(.05);
  hs->GetYaxis()->SetLabelSize(.05);
  //  leg = new TLegend(0.65,0.55,0.85,0.75);
  leg = new TLegend(0.75,0.67,0.9,0.82);
  if(hdata != 0)
    leg->AddEntry(hdata,"data");
  leg->AddEntry(h1,"Z#rightarrow#mu #mu","f");
  leg->AddEntry(h2,"W#rightarrow#mu #nu","f");
  leg->AddEntry(h4,"QCD","f");
  leg->AddEntry(h3,"t#bar{t}","f"); 
  leg->AddEntry(h5,"Z#rightarrow#tau #tau","f"); 
  leg->SetFillColor(kWhite);
  leg->SetFillColor(kWhite);
  leg->SetShadowColor(kBlack);
  leg->Draw();
  c1->SetLogy();
  //  TPaveText *pave = new TPaveText( 0.5 * (hdata->GetXaxis()->GetXmax() - (hdata->GetXaxis()->GetXmin()))  , (hdata->GetMaximum()) +1 , hdata->GetXaxis()->GetXmax() , 10 * hdata->GetMaximum()  );
  TPaveText *pave = new TPaveText( 0.5  , 0.83 , 0.9 , 0.93  , "NDC");
  pave->SetFillColor(kWhite);
  pave->SetBorderSize(0);
  TText * t1 = pave->AddText("CMS Preliminary 2010");
  TText * t2 = pave->AddText("L_{int} = 18.67 nb^{ -1}  #sqrt{s} = 7 TeV"); // change by hand, can be improved...........  
  t1->SetTextColor(kBlack);
  t2->SetTextColor(kBlack);
  pave->Draw(); 
  stat(h1, h2, h3, h4, h5,hdata, rebin);
 c1->Update();
c1->SetTickx(0);
c1->SetTicky(0); 
}





// allowing two variables, for plotting the muon variables...
void makePlots(const char * var1, const char * var2,   TCut cut, int rebin, const char * plot,
	       double min = 0.001, unsigned int nbins, double xMin, double xMax,  bool doData = false) {


 
TChain * zEvents = new TChain("Events"); 


 zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_zmmSpring10cteq66_100pb.root");
TChain * wEvents = new TChain("Events"); 
 wEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_wplusmnSpring10cteq66_50pb.root");
 wEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_wminusmnSpring10cteq66_50pb.root");

// 100 pb
TChain * tEvents = new TChain("Events"); 
tEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ttbarSpring10HLTRedigi_100pb.root");
// 100 pb
TChain * qEvents = new TChain("Events"); 
qEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_incl15Spring10HLTRedigi_35pb.root");
TChain * ztEvents = new TChain("Events"); 
 ztEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ztautauSpring10cteq66_100pb.root");
// 35 pb


TChain * dataEvents= new TChain("Events");

dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_135149.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136033.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136087.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136100.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_137028.root");
// .020 pb

  TH1F *h1 = new TH1F ("h1", "h1", nbins, xMin, xMax);
  TH1F *hh1 = new TH1F ("hh1", "hh1", nbins, xMin, xMax);

  //  h1->Rebin(rebin); 
  TH1F *h2 = new TH1F ("h2", "h2", nbins, xMin, xMax);
  TH1F *hh2 = new TH1F ("hh2", "hh2", nbins, xMin, xMax);

   //  h2->Rebin(rebin); 
  TH1F *h3 = new TH1F ("h3", "h3", nbins, xMin, xMax);
  TH1F *hh3 = new TH1F ("hh3", "hh3", nbins, xMin, xMax);
  
   //h3->Rebin(rebin); 
  TH1F *h4 = new TH1F ("h4", "h4", nbins, xMin, xMax);
  TH1F *hh4 = new TH1F ("hh4", "hh4", nbins, xMin, xMax);

   //h4->Rebin(rebin); 
  TH1F *h5 = new TH1F ("h5", "h5", nbins, xMin, xMax);
  TH1F *hh5 = new TH1F ("hh5", "hh5", nbins, xMin, xMax);


  zEvents->Project("h1", var1, cut);
  zEvents->Project("hh1", var2, cut);
  h1->Add(hh1);

  wEvents->Project("h2", var1, cut);
  wEvents->Project("h2", var2, cut);
  h2->Add(hh2);

  tEvents->Project("h3", var1, cut);
  tEvents->Project("h3", var2, cut);
  h3->Add(hh3);

  qEvents->Project("h4", var1, cut);
  qEvents->Project("h4", var2, cut);
  h4->Add(hh4); 

  ztEvents->Project("h5", var1, cut);
  ztEvents->Project("h5", var2, cut);
  h5->Add(hh5);
 
  //  TH1F *hdata = doData? (TH1F*)data.Get(var) : 0;
  if (doData) { 
  TH1F *hdata = new TH1F ("hdata", "hdata", nbins, xMin, xMax);
  TH1F *hhdata = new TH1F ("hhdata", "hhdata", nbins, xMin, xMax);
  dataEvents->Project("hdata", var1, cut) ;
  dataEvents->Project("hhdata", var2, cut) ;
  hdata->Add(hhdata);
  }
  makeStack(h1, h2, h3, h4, h5, hdata, min, rebin);
  c1->SaveAs((std::string(plot)+".eps").c_str());
  c1->SaveAs((std::string(plot)+".gif").c_str());
  c1->SaveAs((std::string(plot)+".pdf").c_str());

}


void evalEff(const char * var1, const char * var2,  TCut cut, TCut cut_Nminus1, unsigned int nbins, double xMin, double xMax) {

 
TChain * zEvents = new TChain("Events"); 

zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_zmmSpring10cteq66_100pb.root");

 TH1F * htot1 = new TH1F("htot1", "htot1", nbins, xMin, xMax);
 TH1F * htot2 = new TH1F("htot2", "htot2", nbins, xMin, xMax);
 TH1F * hcut1 = new TH1F("hcut1", "hcut1", nbins, xMin, xMax);
 TH1F * hcut2 = new TH1F("hcut2", "hcut2", nbins, xMin, xMax);


  zEvents->Project("htot1", var1, cut);
  zEvents->Project("hcut1", var1, cut_Nminus1);
  zEvents->Project("htot2", var2, cut);
  zEvents->Project("hcut2", var2, cut_Nminus1);
 
  int npass = hcut1->Integral() + hcut2->Integral() ;
  int ntot = htot1->Integral() + htot2->Integral() ;
 
  double eff = (double) npass  / ntot;
  double err =  sqrt(eff * (1 - eff ) / (ntot));
  std::cout << " efficiency for the given cut: " << eff;
  std::cout << " npass: " << npass; 
  std::cout << " nTot: " << ntot; 
  std::cout << " binomial error: " << err << std::endl; 

           
}



#endif
