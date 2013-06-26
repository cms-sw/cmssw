#ifndef STACK_COMMON_H
#define STACK_COMMON_H

#include <iostream>
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
double intLumi = 177 ;
//double intLumi = 100000;

const double lumi = intLumi * .001 ;
//const double lumi =0100.0 ;
const double lumiZ = 100. ;
const double lumiW = 100.;
//adjust to new filter efficiency
const double lumiQ = 60.;
//scaling ttbar from 94.3 to 162.
const double lumiT =100. * (94.3/162.);
const double lumiZT =100.;


const double mMin = 60;
const double mMax = 120;





/// cuts common....
TCut kin_common(" (zGoldenDau1Q * zGoldenDau2Q) ==-1 &&  zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && zGoldenDau1Iso03SumPt< 3.0 && zGoldenDau2Iso03SumPt < 3.0 && ( (zGoldenDau1HLTBit==1 && abs(zGoldenDau1Eta)<2.1) || ( zGoldenDau2HLTBit==1 && abs(zGoldenDau2Eta)<2.1))  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 "); // && ( abs(zGoldenDau1dxyFromBS)<0.1 || abs(zGoldenDau2dxyFromBS)<0.1 ) ");



TCut dau1Loose(" (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 ");
TCut dau2Loose(" (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 ");

TCut dau1TightWP1("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10 && zGoldenDau1NofPixelHits>0 && zGoldenDau1NofMuonHits>0 &&  zGoldenDau1NofMuMatches>1  && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP1("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10 && zGoldenDau2NofPixelHits>0 && zGoldenDau2NofMuonHits>0 &&  zGoldenDau2NofMuMatches>1  && zGoldenDau2TrackerMuonBit==1");


TCut dau1TightWP2("zGoldenDau1Chi2<10  && (zGoldenDau1NofStripHits + zGoldenDau1NofPixelHits)>10  && zGoldenDau1NofMuonHits>0   && zGoldenDau1TrackerMuonBit==1");
TCut dau2TightWP2("zGoldenDau2Chi2<10  && (zGoldenDau2NofStripHits + zGoldenDau2NofPixelHits)>10  && zGoldenDau2NofMuonHits>0   && zGoldenDau2TrackerMuonBit==1");

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



 
 int lineWidth(3);

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
	  markerSize = 1.2;
	}
      else
	{	
	  markerSize = 1.2;
	}


      if( logScale )
	{
	  xl_ = 0.60;
	  yl_ = 0.60;
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

void stat(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 *h5, TH1 * hdata, int rebin) {
  double a = mMin/rebin +1, b = mMax/rebin;
  double i1 = h1->Integral(a, b);
  double err1 = sqrt(i1);
  double i2 = h2->Integral(a, b);
  double err2 = sqrt(i2);
  double i3 = h3->Integral(a, b);
  double err3 = sqrt(i3);
  double i4 = h4->Integral(a, b);
  double err4 = sqrt(i4);
  double i5 = h5->Integral(a, b);
  double err5 = sqrt(i5);

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
  std::cout <<"data (" << mMin << ", " << mMax << ") = ";
  std::cout.precision(8);
  std::cout << idata << "+/- " << errData  <<std::endl; 
}

void makeStack(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 * h5, TH1 * hdata,
	       double min, int rebin , bool logScale) {
  setHisto(h1, zFillColor, zLineColor, lumi/lumiZ, rebin);
  setHisto(h3, ttFillColor, ttLineColor, lumi/lumiT, rebin);
  setHisto(h4, qcdFillColor, qcdLineColor, lumi/lumiQ, rebin);

  setHisto(h2, ewkFillColor, ewkLineColor, lumi/lumiW, rebin);
  setHisto(h5, ewkFillColor, ewkLineColor, lumi/lumiZT, rebin);
  h2->Add(h5);

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
    hdata->Rebin(rebin); 

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
    hdata->SetMarkerSize(markerSize);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(lineWidth);
    hdata->SetLineColor(kBlack);
    //gStyle->SetErrorX(.5);
    gStyle->SetEndErrorSize(2);

    hdata->Draw("PE1SAME");
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    hdata->GetXaxis()->SetNdivisions(ndivx);
    hdata->GetYaxis()->SetNdivisions(ndivy);
    hs->GetXaxis()->SetNdivisions(ndivx);
    hs->GetYaxis()->SetNdivisions(ndivy);
    // log plots, so the maximum should be one order of magnitude more...
    
    
   
 hs->SetMaximum( 8 +  hdata->GetMaximum()  )  ;
    if (logScale) {
      hs->SetMaximum( pow(10 , 2. + int(log( hdata->GetMaximum() )  )  ));
    } 
    // lin plot 
     	
    //    
  }
  hs->SetMinimum(min);
  
  hs->GetXaxis()->SetTitle("M(#mu^{+} #mu^{-}) [GeV]");


  
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
  if (logScale) {
   hs->GetXaxis()->SetTitleOffset(1.0);
   hs->GetYaxis()->SetTitleOffset(1.1);
   
   hs->GetYaxis()->SetLabelOffset(0.0);
   hs->GetXaxis()->SetLabelSize(.05);
   hs->GetYaxis()->SetLabelSize(.05);
  }


//leg = new TLegend(0.75,0.55,0.90,0.7);


  int nChan =2;
  if (logScale) nChan = 4;
float dxl_ = scalel_*5;
  if (logScale) dxl_ = scalel_*4;
  float dyl_ = scalel_*(nChan);
  if (logScale) dyl_ = scalel_*(nChan-1);
  //  TLegend* legend=new TLegend(xl_,yl_,xl_+dxl_,yl_+dyl_);
  // TLegend* legend=new TLegend(0.65,0.54,0.95,0.796);
  // legend on the left 
 TLegend* legend=new TLegend(0.2,0.78,0.4,0.93);
  legend->SetLineColor(0);
  legend->SetFillColor(0);





  //leg = new TLegend(0.20,0.7,0.35,0.85);
  if(hdata != 0)
    legend->AddEntry(hdata,"data", "pl");
  legend->AddEntry(h1,"Z #rightarrow#mu #mu","f");
  if (logScale) {

  legend->AddEntry(h2,"EWK","f");
  legend->AddEntry(h4,"QCD","f");
  legend->AddEntry(h3,"t#bar{t}","f"); 
  //leg->AddEntry(h5,"Z#rightarrow#tau #tau","f"); 
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
    latex.DrawLatex(0.85,0.84,Form("#int #font[12]{L} dt = %.0f nb^{-1}",intLumi));
  }
  latex.SetTextAlign(11); // align left
  latex.DrawLatex(0.12,0.96,"CMS preliminary 2010");


  stat(h1, h2, h3, h4, h5,hdata, rebin);

  // c1->Update();
  // c1->SetTickx(0);
  // c1->SetTicky(0); 
}





// allowing two variables, for plotting the muon variables...
void makePlots(const char * var1, const char * var2,   TCut cut, int rebin, const char * plot,
	       double min = 0.001, unsigned int nbins, double xMin, double xMax,  bool doData = false, bool logScale=false) {



TChain * zEvents = new TChain("Events"); 


// zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples_withIso03/NtupleLoose_zmmSpring10cteq66_100pb.root");



 zEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_ZmmPowheg_36X_100pb.root");
TChain * wEvents = new TChain("Events"); 
 wEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_wplusPowheg_36X_100pb_v2.root");
 wEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_wminusPowheg36X_100pb.root");

// 100 pb
TChain * tEvents = new TChain("Events"); 
tEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_ttbar_36X_100pb.root");
// 100 pb
TChain * qEvents = new TChain("Events"); 
qEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_incl15_36X_60pb.root");
TChain * ztEvents = new TChain("Events"); 
 ztEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_ztautauPowheg36X_100pb.root");
// 35 pb


TChain * dataEvents= new TChain("Events");

 
 dataEvents->Add("/scratch2/users/degruttola/data/OfficialJSON/NtupleLoose_132440_139790.root");
 dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_965_971.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_139_972_980.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_058_076.root");
dataEvents->Add("/scratch2/users/degruttola/data/jun14rereco_and361p4PromptReco/NtupleLoose_140_116_126.root");

// .040 pb

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
  makeStack(h1, h2, h3, h4, h5, hdata, min, rebin, logScale);
 

 
  if (logScale) c1->SetLogy();

  c1->SaveAs((std::string(plot)+".eps").c_str());
  c1->SaveAs((std::string(plot)+".gif").c_str());
  c1->SaveAs((std::string(plot)+".pdf").c_str());

  TFile * out = new TFile("plot.root", "RECREATE");

  c1->Write();
}


void evalEff(const char * var1, const char * var2,  TCut cut, TCut cut_Nminus1, unsigned int nbins, double xMin, double xMax) {

 
TChain * zEvents = new TChain("Events"); 

// zEvents->Add("/scratch2/users/degruttola/Spring10Ntuples_withIso03/NtupleLoose_zmmSpring10cteq66_100pb.root");
zEvents->Add("/scratch2/users/degruttola/Summer10Ntuples/Ntuple_ZmmPowheg_36X_100pb.root");
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
