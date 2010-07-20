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


const double lumi =.020 ;
const double lumiZ = 100. ;
const double lumiW = 100.;
const double lumiQ = 35.;
const double lumiT =100.;
const double lumiZT =100.;


const double mMin = 60;
const double mMax = 120;





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
    hdata->SetMarkerSize(1.3);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(2);
    hdata->SetLineColor(kBlack);
    hdata->Rebin(rebin); 
    hdata->Draw("epsame");
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    hs->SetMaximum( hdata->GetMaximum() + 5);
    gStyle->SetErrorX(1);
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
  leg = new TLegend(0.75,0.65,0.9,0.8);
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
  TPaveText *pave = new TPaveText( 0.5 * (hdata->GetXaxis()->GetXmax() - (hdata->GetXaxis()->GetXmin()))  , (hdata->GetMaximum()) + 1.5 , hdata->GetXaxis()->GetXmax() , hdata->GetMaximum() + 9 );
  pave->SetFillColor(kWhite);
  pave->SetBorderSize(0);
  TText * t1 = pave->AddText("CMS Preliminary 2010");
  TText * t2 = pave->AddText("L_{int} = 20 nb^{-1}  #sqrt{s} = 7 TeV"); // change by hand, can be improved.....
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

zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ZmmPowhegSpring10HLTRedigi_100pb.root");
TChain * wEvents = new TChain("Events"); 
wEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_wmunuPowhegSpring10HLTRedigi_100pb.root");
// 100 pb
TChain * tEvents = new TChain("Events"); 
tEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ttbarSpring10HLTRedigi_100pb.root");
// 100 pb
TChain * qEvents = new TChain("Events"); 
qEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_incl15Spring10HLTRedigi_35pb.root");
TChain * ztEvents = new TChain("Events"); 
ztEvents->Add("/scratch2/users/degruttola/Spring10Ntuples/NtupleLoose_ztautauSpring10HLTRedigi_100pb.root");
// 35 pb


TChain * dataEvents= new TChain("Events");

dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_135149.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136033.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136087.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_136100.root");
dataEvents->Add("/scratch2/users/degruttola/data/NtupleLoose_137028.root");




// .020 pb






  TH1F *h1 = new TH1F ("h1", "h1", nbins, xMin, xMax);
  //  h1->Rebin(rebin); 
  TH1F *h2 = new TH1F ("h2", "h2", nbins, xMin, xMax);
   //  h2->Rebin(rebin); 
  TH1F *h3 = new TH1F ("h3", "h3", nbins, xMin, xMax);
   //h3->Rebin(rebin); 
  TH1F *h4 = new TH1F ("h4", "h4", nbins, xMin, xMax);
   //h4->Rebin(rebin); 
  TH1F *h5 = new TH1F ("h5", "h5", nbins, xMin, xMax);
  
  zEvents->Project("h1", var1, cut);
  zEvents->Project("h1", var2, cut);

  wEvents->Project("h2", var1, cut);
  wEvents->Project("h2", var2, cut);


  tEvents->Project("h3", var1, cut);
  tEvents->Project("h3", var2, cut);

  qEvents->Project("h4", var1, cut);
  qEvents->Project("h4", var2, cut);
 
  ztEvents->Project("h5", var1, cut);
  ztEvents->Project("h5", var2, cut);

 
  //  TH1F *hdata = doData? (TH1F*)data.Get(var) : 0;
  if (doData) { 
  TH1F *hdata = new TH1F ("hdata", "hdata", nbins, xMin, xMax);
  dataEvents->Project("hdata", var1, cut) ;
  dataEvents->Project("hdata", var2, cut) ;
  }
  makeStack(h1, h2, h3, h4, h5, hdata, min, rebin);
  c1->SaveAs((std::string(plot)+".eps").c_str());
  c1->SaveAs((std::string(plot)+".gif").c_str());
  c1->SaveAs((std::string(plot)+".pdf").c_str());

}
#endif
