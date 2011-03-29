 
#ifndef STACK_COMMON_TTBARENHANCED_H
#define STACK_COMMON_TTBARENHANCED_H
#include <iostream>
using namespace std;
#include "TChain.h"

#include "TGraphAsymmErrors.h"

const int canvasSizeX = 500;
const int canvasSizeY = 500;
const Color_t zLineColor = kOrange+3;
const Color_t zFillColor = kOrange-2;
// ewk: W+ztt
const Color_t ewkLineColor = kOrange+3;
const Color_t ewkFillColor = kOrange+7;

const Color_t qcdLineColor = kViolet+3;
const Color_t qcdFillColor = kViolet-5;


const Color_t ttLineColor =  kRed+4;
const Color_t ttFillColor = kRed+2;

//const Color_t ztFillColor =  kYellow-9;
//const Color_t ztLineColor = kYellow-1;




// 78 
//double intLumi = 36.09;
double intLumi = 28.61;
//double intLumi = 3.1;
//double intLumi = 100000;
 
/*
ls /scratch2/users/degruttola/38Xdata/MC/
Ntuple_fall10_dym20_all_1186pb.root  Ntuple_fall10_wminus_all_477pb.root  Ntuple_fall10_wz_all_113p6fb.root
Ntuple_fall10_qcd_158pb.root         Ntuple_fall10_wplus_all_320pb.root   Ntuple_fall10_ztautau_all_1186pb.root
Ntuple_fall10_ttbar_all_6787pb.root  Ntuple_fall10_ww_all_49p8fb.root     Ntuple_fall10_zz_all_142p7fb.root
*/

const double lumi =  intLumi  ;
const int integerLumi =  (int) intLumi  ;

//const double lumi =0100.0 ;
const double lumiZ = 341. ;
const double lumiW = 398.; // mean of the two sub-sumples (W+ and W-)
//adjust to new filter efficiency
const double lumiQ = 210.;
//scaling ttbar from 94.3 to 162.
const double lumiT =6787;
const double lumiZT =1186;
const double lumiWZ =113600;
const double lumiWW =49800;
const double lumiZZ =142700 * 21./23.;


const double mMin = 0;
const double mMax = 10000;





/// cuts common....
TCut kin_common(" (zGoldenDau1Q * zGoldenDau2Q) ==-1 &&  zGoldenDau1Pt>20 && zGoldenDau2Pt>20 && ( (zGoldenDau1HLTBit==1 && abs(zGoldenDau1Eta)<2.1) || ( zGoldenDau2HLTBit==1 && abs(zGoldenDau2Eta)<2.1))  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 "); // && ( abs(zGoldenDau1dxyFromBS)<0.1 || abs(zGoldenDau2dxyFromBS)<0.1 ) ");

TCut cut_1Iso("((zGoldenDau1Iso03SumPt + zGoldenDau1Iso03EmEt + zGoldenDau1Iso03HadEt)/ zGoldenDau1Pt ) < 0.15"
	      );

TCut cut_2Iso("((zGoldenDau2Iso03SumPt + zGoldenDau2Iso03EmEt + zGoldenDau2Iso03HadEt)/ zGoldenDau2Pt ) < 0.15"
	      );


TCut cut_Trk1Iso("zGoldenDau1Iso03SumPt < 3.00")
  ;

TCut cut_Trk2Iso("zGoldenDau2Iso03SumPt  < 3.00");

TCut antiCosmicCut("abs(zGoldenDau1Eta + zGoldenDau2Eta)>0.01 ||  (abs(abs(zGoldenDau1Phi - zGoldenDau2Phi) - TMath::Pi()) > 0.01 ) || nTrkPV>2");




TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_notChi2AndTrackerMuon("zGoldenDau1Chi2<10000  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1");

TCut dau2TightWP1_notChi2AndTrackerMuon("zGoldenDau2Chi2<10000  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1");



//TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
//TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

TCut dau1TightWP1_hltAlso("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1 && abs(zGoldenDau1Eta)<2.1 && zGoldenDau1HLTBit==1");// 2.1 can bacome 2.4 later....
TCut dau2TightWP1_hltAlso("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  &&  abs(zGoldenDau2Eta)<2.1 && zGoldenDau2HLTBit==1");


 
TCut massCut("zGoldenMass>60 && zGoldenMass<120 ");




TCanvas *c1 = new TCanvas("c1","Stack plot", 300,300,479,510);

 c1->SetLeftMargin(  87./479 );
  c1->SetRightMargin( 42./479 );
  c1->SetTopMargin(  30./510 );
  c1->SetBottomMargin( 80./510 ); 
  c1->SetFillColor(0);
 c1->SetTickx(1);
  c1->SetTicky(1);
  c1->SetFrameFillStyle(0);
  c1->SetFrameLineWidth(2);
  c1->SetFrameBorderMode(0);



 
 int lineWidth(2);

if( logScale )
  {
    lineWidth = 1;
  }

 // histogram limits, in linear and logarithmic
  int nbin_(100);
  float xmin_(0.), xmax_(100.); 
  float ymin_(0.), ymax_(40.); 
  float yminl_(0.1), ymaxl_(200.); 

  // titles and axis, marker size
  TString xtitle;
  TString ytitle;
int ndivx(506);
  int ndivy(506);
  float markerSize(2.);

  // canvas name
  TString cname("c");
  TString ctitle;

  // legend position and scale;
  float xl_  = 0.;
  float yl_  = 0.6;
  float scalel_ = 0.05;
      ndivx = 120;

if( logScale )
	{
	  ndivx=506; 
	  ndivy = 506;
	}
      else
	{
	  ndivy = 506;
          ndivx=506;
	}

      if( logScale )
	{
	  markerSize = 1.1;
	}
      else
	{	
	  markerSize = 1.2;
	}


      if( logScale )
	{
	  xl_ = 0.60;
	  yl_ = 0.65;
	}
      else
	{
	  xl_ = 0.60;
	  yl_ = 0.60;
	  scalel_ = 0.06;
	}




void setHisto(TH1 * h, Color_t fill, Color_t line, double scale, int rebin) {
  h->SetFillColor(fill);
  h->SetLineColor(line);
  h->Scale(scale);
  h->Rebin(rebin);  

}

void stat(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 *h5, TH1 *h6, TH1 * h7, TH1 * h8, TH1 * hdata, int rebin, bool scaleToDataInt ) {
  double a = mMin/rebin +1, b = mMax/rebin;
  double i1 = h1->Integral(a, b);
  double err1 = sqrt(lumi/lumiZ *  i1);
  double i2 = h2->Integral(a, b);
  double err2 = sqrt(lumi/lumiW * i2);
  double i3 = h3->Integral(a, b);
  double err3 = sqrt(lumi/lumiQ * i3);
  double i4 = h4->Integral(a, b);
  double err4 = sqrt(lumi/lumiT * i4);
  double i5 = h5->Integral(a, b);
  double err5 = sqrt(lumi / lumiZT * i5);
  double i6 = h6->Integral(a, b);
  double err6 = sqrt(lumi / lumiWZ * i6);
  double i7 = h7->Integral(a, b);
  double err7 = sqrt(lumi / lumiWW *i7);
  double i8 = h8->Integral(a, b);
  double err8 = sqrt(lumi / lumiZZ *i8);

  double idata = hdata != 0 ? hdata->Integral(a, b) : 0;
  double errData = sqrt(idata);

  std::cout.setf(0,ios::floatfield);
  std::cout.setf(ios::fixed,ios::floatfield);
  std::cout.precision(1);
  std::cout <<"Zmm (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i1 << "+/- " << err1 <<std::endl;
  std::cout.precision(1);
  std::cout <<"Wmn (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i2 << "+/- " << err2 <<std::endl;
  std::cout.precision(1);
  std::cout <<"QCD (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << i4 << "+/- " << err4 <<std::endl;
  std::cout.precision(1);
  std::cout <<"tt~ (" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i3 << "+/- " << err3 <<std::endl; 
  std::cout.precision(1);
  std::cout <<"ztautau (" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i5 << "+/- " << err5 <<std::endl; 
  std::cout.precision(1);
 std::cout <<"wz(" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i6 << "+/- " << err6 <<std::endl; 
  std::cout.precision(1);
 std::cout <<"ww(" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i7 << "+/- " << err7 <<std::endl; 
  std::cout.precision(1);
 std::cout <<"zz(" << mMin << ", " << mMax << ") = "; 
  std::cout.precision(8);
  std::cout << i8 << "+/- " << err8 <<std::endl; 
  std::cout.precision(1);


  std::cout <<"data (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << idata << "+/- " << errData  <<std::endl; 


//if oen wnats one can scale any MC to the data events 
  if (scaleToDataInt) {
    double ratioToDataInt = idata / ( i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8) ; 
    h1->Scale( ratioToDataInt);
    h2->Scale( ratioToDataInt);
    h3->Scale( ratioToDataInt);
    h4->Scale( ratioToDataInt);
    h5->Scale( ratioToDataInt);
    h6->Scale( ratioToDataInt);
    h7->Scale( ratioToDataInt);
    h8->Scale( ratioToDataInt);
}
}

void makeStack(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 * h5, TH1 * h6, TH1 * h7, TH1 * h8, TH1 * hdata,
	       double min, int rebin , bool logScale, bool scaleToDataInt) {


  setHisto(h1, zFillColor, zLineColor, lumi/lumiZ, rebin);
  setHisto(h3, ttFillColor, ttLineColor, lumi/lumiT, rebin);
  setHisto(h4, qcdFillColor, qcdLineColor, lumi/lumiQ, rebin);

  setHisto(h2, ewkFillColor, ewkLineColor, lumi/lumiW, rebin);
  setHisto(h5, ewkFillColor, ewkLineColor, lumi/lumiZT, rebin);
  setHisto(h6, ewkFillColor, ewkLineColor, lumi/lumiWZ, rebin);
  setHisto(h7, ewkFillColor, ewkLineColor, lumi/lumiWW, rebin);
  setHisto(h8, ewkFillColor, ewkLineColor, lumi/lumiZZ, rebin);


    hdata->Rebin(rebin); 

    stat(h1, h2, h3, h4, h5,h6, h7, h8, hdata, rebin,scaleToDataInt);

  h2->Add(h5);
  h2->Add(h6);
  h2->Add(h7);
  h2->Add(h8);


  THStack * hs = new THStack("hs","");



  
    if (logScale) {

    

    hs->Add(h4);
    hs->Add(h3);
    hs->Add(h2);
   
      }
  
  //hs->Add(h5);
  hs->Add(h1);

   hs->Draw("HIST");
  if(hdata != 0) {


    /* TGraphAsymmErrors* dataGraph = (TGraphAsymmErrors*)hdata;
    dataGraph->SetMarkerStyle(kFullCircle);
    dataGraph->SetMarkerColor(kBlack);
    dataGraph->SetMarkerSize(markerSize);
    // Remove the horizontal bars (at Michael's request)
    double x_(0), y_(0);
    for( int ii=0; ii<dataGraph->GetN(); ii++ )
      {
	dataGraph->SetPointEXlow(ii,0);
	dataGraph->SetPointEXhigh(ii,0);
	dataGraph->GetPoint(ii,x_,y_ );
	if( y_==0 )
	  {
	    dataGraph->RemovePoint( ii );
	    ii--;
	  }  
      }
dataGraph->Draw("pesame");
    */
      
       hdata->SetMarkerStyle(kFullCircle);
       // hdata->SetMarkerStyle(9);
    hdata->SetMarkerSize(markerSize);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(lineWidth);
    hdata->SetLineColor(kBlack);
    //gStyle->SetErrorX(.5);
    gStyle->SetEndErrorSize(1);

    hdata->Draw("PE1SAME");
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    hdata->GetXaxis()->SetNdivisions(ndivx);
    hdata->GetYaxis()->SetNdivisions(ndivy);
    hs->GetXaxis()->SetNdivisions(ndivx);
    hs->GetYaxis()->SetNdivisions(ndivy);
    // log plots, so the maximum should be one order of magnitude more...
    
    
   
    hs->SetMaximum( max(hdata->GetMaximum(), hs->GetMaximum() ) + 150)  ;
     //hs->SetMaximum( 29.8 )  ;
    if (logScale) {
      // hs->SetMaximum( pow(10 , -0.3 + int(log( hdata->GetMaximum() )  )  ));
      // hs->SetMaximum( pow(10 , +1.5 + int(log( hdata->GetMaximum() )  )  ));
    hs->SetMaximum( max(hdata->GetMaximum(), hs->GetMaximum() ) + 10)  ;   
 } 
    // Lin plot 
     	
    //    
  }
  hs->SetMinimum(min);
  
  //  hs->GetXaxis()->SetTitle("M(#mu^{+} #mu^{-}) [GeV]");


  
  std::string yTag = "";
  switch(rebin) {
  case 1: yTag = "number of events/ 1 GeV"; break;
  case 2: yTag = "number of events/ 2 GeV"; break;
  case 2.5: yTag = "number of events/ 2.5 GeV"; break;
  case 3: yTag = "number of events/ 3 GeV"; break;
  case 4: yTag = "number of events/ 4 GeV"; break;
  case 5: yTag = "number of events/ 5 GeV"; break;
  case 10: yTag = "number of events/ 10 GeV"; break;
  default:
    std::cerr << ">>> ERROR: set y tag for rebin = " << rebin << std::endl;
  };

  hs->GetYaxis()->SetTitle(yTag.c_str());
  

  /*
   hs->GetXaxis()->SetTitleSize(0.05);
   hs->GetYaxis()->SetTitleSize(0.05);

  */
  //  if (logScale) {
   hs->GetXaxis()->SetTitleOffset(1.0);
   hs->GetYaxis()->SetTitleOffset(1.2);
   
   hs->GetYaxis()->SetLabelOffset(0.0);
   hs->GetXaxis()->SetLabelSize(.05);
   hs->GetYaxis()->SetLabelSize(.02);
   hs->GetYaxis()->SetTitleSize(.03);

  

   // hs->GetYaxis()->SetTitleOffset(1.2);

//leg = new TLegend(0.75,0.55,0.90,0.7);


  int nChan =2;
  if (logScale) nChan = 4;
float dxl_ = scalel_*5;
  if (logScale) dxl_ = scalel_*4;
  float dyl_ = scalel_*(nChan);
  if (logScale) dyl_ = scalel_*(nChan-1);
  //  
  // TLegend* legend=new TLegend(0.65,0.54,0.95,0.796);


  // top left
  //  TLegend* legend=new TLegend(0.2,0.78,0.4,0.93);

// middle right
TLegend* legend=new TLegend(0.6,0.72,0.8,0.80);   

  if (logScale) { TLegend* legend=new TLegend(0.7,0.68,0.9,0.82);   }
  legend->SetLineColor(0);
  legend->SetFillColor(0);





  //leg = new TLegend(0.20,0.7,0.35,0.85);
  if(hdata != 0)
    legend->AddEntry(hdata,"data", "pl");
  legend->AddEntry(h1,"Z + jets #rightarrow#mu #mu + jets","f");
  if (logScale) {

  legend->AddEntry(h2,"EWK","f");
  legend->AddEntry(h4,"QCD","f");
  legend->AddEntry(h3,"t#bar{t}","f"); 

  
 }

  //  legend->SetFillColor(kWhite);
  //legend->SetFillColor(kWhite);
 
 legend->SetShadowColor(kWhite);
  legend->Draw();
 


TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.04);

  latex.SetTextAlign(31); // align right
  latex.DrawLatex(0.90,0.96,"#sqrt{s} = 7 TeV");
  if (intLumi > 0.) {
    latex.SetTextAlign(31); // align right
    latex.DrawLatex(0.88,0.84,Form("#int #font[12]{L} dt = %d pb^{-1}",integerLumi));
  }
  latex.SetTextAlign(11); // align left
  latex.DrawLatex(0.12,0.96,"CMS preliminary 2010");


  //stat(h1, h2, h3, h4, h5,h6, h7, hdata, rebin);

  // c1->Update();
  // c1->SetTickx(0);
  // c1->SetTicky(0); 
}





// allowing two variables, for plotting the muon variables...
void makePlots(const char * var1, const char * var2,   TCut cut, TCut cut2="", int rebin, const char * plot,
	       double min = 0.001, unsigned int nbins, double xMin, double xMax,  bool doData = true, bool logScale=false, bool scaleToDataInt = true) {
 

 

TChain * zEvents = new TChain("Events"); 


//zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples_withIso03/NtupleLoose_zmmSpring10cteq66_100pb.root");



//zEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_ZmmPowheg_36X_100pb.root");
// zEvents->Add("../Ntuple_fall10_dym20_ttbarCheck.root");
 zEvents->Add("../Ntuple_fall10_zplusjets_ttbarCheck.root");



TChain * wEvents = new TChain("Events"); 
 wEvents->Add("../Ntuple_fall10_wminus_ttbarCheck.root");
 wEvents->Add("../Ntuple_fall10_wplus_ttbarCheck.root");

// 100 pb
TChain * tEvents = new TChain("Events"); 
tEvents->Add("../Ntuple_fall10_ttbar_ttbarCheck.root");
// 100 pb
TChain * qEvents = new TChain("Events"); 
//qEvents->Add("/scratch2/users/degruttola/38Xdata/MC/Ntuple_fall10_qcd_210pb.root");
TChain * ztEvents = new TChain("Events"); 
 ztEvents->Add("../Ntuple_fall10_ztt_ttbarCheck.root");
TChain * wzEvents = new TChain("Events"); 
 wzEvents->Add("../Ntuple_fall10_wz_ttbarCheck.root");
TChain * wwEvents = new TChain("Events"); 
 wwEvents->Add("../Ntuple_fall10_ww_ttbarCheck.root");
// 35 pb

TChain * zzEvents = new TChain("Events"); 
 zzEvents->Add("../Ntuple_fall10_zz_ttbarCheck.root");


TChain * dataEvents= new TChain("Events");

//dataEvents->Add("/scratch2/users/degruttola/38Xdata/data/Ntuple_Run2010ANov4ReReco_hltmu9.root");
//dataEvents->Add("/scratch2/users/degruttola/38Xdata/data/Ntuple_Run2010BNov4ReReco_hltmu9.root");
//dataEvents->Add("/scratch2/users/degruttola/38Xdata/data/Ntuple_Run2010BNov4ReReco_hltmu15.root");

//dataEvents->Add("../Ntuple_2010_ttbarCheck.root");
dataEvents->Add("../Ntuple_2010_ttbarCheck_hltMu15.root");


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

  TH1F *h6 = new TH1F ("h6", "h6", nbins, xMin, xMax);
  TH1F *hh6 = new TH1F ("hh6", "hh6", nbins, xMin, xMax);

  TH1F *h7 = new TH1F ("h7", "h7", nbins, xMin, xMax);
  TH1F *hh7 = new TH1F ("hh7", "hh7", nbins, xMin, xMax);

  TH1F *h8 = new TH1F ("h8", "h8", nbins, xMin, xMax);
  TH1F *hh8 = new TH1F ("hh8", "hh8", nbins, xMin, xMax);


  zEvents->Project("h1", var1, cut);
  zEvents->Project("hh1", var2, cut2);
  h1->Add(hh1);

  wEvents->Project("h2", var1, cut);
  wEvents->Project("hh2", var2, cut2);
  h2->Add(hh2);

  tEvents->Project("h3", var1, cut);
  tEvents->Project("hh3", var2, cut2);
  h3->Add(hh3);

  qEvents->Project("h4", var1, cut);
  qEvents->Project("hh4", var2, cut2);
  h4->Add(hh4); 

  ztEvents->Project("h5", var1, cut);
  ztEvents->Project("hh5", var2, cut2);
  h5->Add(hh5);

  wzEvents->Project("h6", var1, cut);
  wzEvents->Project("hh6", var2, cut2);
  h6->Add(hh6);


  wwEvents->Project("h7", var1, cut);
  wwEvents->Project("hh7", var2, cut2);
  h7->Add(hh7);


  zzEvents->Project("h8", var1, cut);
  zzEvents->Project("hh8", var2, cut2);
  h8->Add(hh8);
 
  //  TH1F *hdata = doData? (TH1F*)data.Get(var) : 0;
  if (doData) { 
  TH1F *hdata = new TH1F ("hdata", "hdata", nbins, xMin, xMax);
  TH1F *hhdata = new TH1F ("hhdata", "hhdata", nbins, xMin, xMax);
  dataEvents->Project("hdata", var1, cut) ;
  dataEvents->Project("hhdata", var2, cut2) ;
  hdata->Add(hhdata);
  }
  makeStack(h1, h2, h3, h4, h5, h6, h7, h8, hdata, min, rebin, logScale, scaleToDataInt);
 

 
  if (logScale) c1->SetLogy();

  c1->SaveAs((std::string(plot)+".eps").c_str());
  c1->SaveAs((std::string(plot)+".gif").c_str());
  c1->SaveAs((std::string(plot)+".pdf").c_str());

  TFile * out = new TFile("plot.root", "RECREATE");

  c1->Write();
  c1->SaveAs("zPlot.C");
}


void evalEff(const char * var1, const char * var2,  TCut cut, TCut cut_Nminus1, unsigned int nbins, double xMin, double xMax) {

 
TChain * zEvents = new TChain("Events"); 

// zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples_withIso03/NtupleLoose_zmmSpring10cteq66_100pb.root");
zEvents->Add("/scratch2/users/degruttola/38Xdata/MC/Ntuple_ZmmPowheg_36X_100pb.root");
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
