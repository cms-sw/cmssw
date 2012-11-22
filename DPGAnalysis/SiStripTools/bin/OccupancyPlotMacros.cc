#include "TText.h"
#include "TGaxis.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TH1D.h"
#include "TList.h"
#include "TBox.h"
#include "TFrame.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include <cstring>
#include <iostream>
#include <math.h>
#include "TROOT.h"
#include "OccupancyPlotMacros.h"

void PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color) {

  gROOT->SetStyle("Plain");

  if(color == 1) {
    // A not-so-great color version
    const Int_t NRGBs = 5;
    const Int_t NCont = 255;
    Double_t stops[NRGBs] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
    Double_t red[NRGBs]   = { 0.00, 0.00, 0.40, 1.00, 1.00 };
    Double_t green[NRGBs] = { 0.00, 0.40, 0.70, 0.60, 1.00 };
    Double_t blue[NRGBs]  = { 0.30, 0.60, 0.00, 0.00, 0.20 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
  }
  else if(color==2) {
    // Gray scale
    const Int_t NRGBs = 3;
    const Int_t NCont = 255;
    Double_t stops[NRGBs] = { 0.00, 0.50, 1.00 };
    Double_t red[NRGBs]   = { 0.90, 0.50, 0.00};
    Double_t green[NRGBs] = { 0.90, 0.50, 0.00};
    Double_t blue[NRGBs]  = { 0.90, 0.50, 0.00};
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
  }
  else if(color==3) {
    // used by Kevin in the TRK-11-001 paper
    const Int_t NRGBs = 7;
    const Int_t NCont = 255;
    Double_t stops[NRGBs] = { 0.00, 0.15, 0.30, 0.45, 0.65, 0.85, 1.00 };
    Double_t red[NRGBs]   = { 0.60, 0.30, 0.00, 0.00, 0.60, 0.40, 0.00 };
    Double_t green[NRGBs] = { 1.00, 0.90, 0.80, 0.75, 0.20, 0.00, 0.00 };
    Double_t blue[NRGBs]  = { 1.00, 1.00, 1.00, 0.30, 0.00, 0.00, 0.00 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
  }

  int ncol = gStyle->GetNumberOfColors();
  std::cout << "Number of colors "  << ncol << std::endl;
  
  if(ff->cd(module)) {

    TProfile* aveoccu= (TProfile*)gDirectory->Get("aveoccu");
    TProfile* avemult= (TProfile*)gDirectory->Get("avemult");
    TH1F* nchannels = (TH1F*)gDirectory->Get("nchannels_real");

    TProfile* averadius = (TProfile*)gDirectory->Get("averadius"); 
    TProfile* avez = (TProfile*)gDirectory->Get("avez"); 

    std::cout << "pointers " << aveoccu << " " << avemult << " " << nchannels << " " << averadius << " " << avez << std::endl;

    if(aveoccu && avemult && nchannels && averadius && avez) {

      nchannels->Sumw2();
      for(int i=1;i<nchannels->GetNbinsX()+1;++i) {
	nchannels->SetBinError(i,0.);
      }

      TH1D* haveoccu = aveoccu->ProjectionX("haveoccu");
      haveoccu->SetDirectory(0);
      haveoccu->Divide(nchannels);
      TH1D* havemult = avemult->ProjectionX("havemult");
      havemult->SetDirectory(0);
      havemult->Divide(nchannels);

      TH1D* havewidth = (TH1D*)haveoccu->Clone("havewidth");
      havewidth->SetDirectory(0);
      havewidth->Divide(havemult);


      new TCanvas("occupancy","occupancy",1200,500);
      haveoccu->Draw();
      new TCanvas("multiplicity","multiplicity",1200,500);
      havemult->Draw();
      new TCanvas("width","width",1200,500);
      havewidth->Draw();

      // Loop on bins and creation of boxes

      TList modulesoccu;
      TList modulesmult;

      for(int i=1;i<haveoccu->GetNbinsX();++i) {

	if(averadius->GetBinEntries(i)*avez->GetBinEntries(i)) {

	  double dz = 2.;
	  double dr = 1.;
	  // determine module size
	  
	  if(i > 100 && i < 200) { dz=3.33;dr=0.4;}

	  if(i > 200 && i < 1000 && ( i%10 == 1 || i%10 == 7)) { dz=0.8;dr=0.4;}
	  if(i > 200 && i < 1000 && !( i%10 == 1 || i%10 == 7)) { dz=0.8;dr=0.8;}

	  if(i > 1000 && i < 2000) { dz=5.948;dr=0.4;}

	  if(i > 3000 && i < 4000) { dz=9.440;dr=0.4;}

	  if(i > 2000 && i < 3000  && (i%1000)/100 == 1) { dz=0.8;dr=5.647;} 
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 2) { dz=0.8;dr=4.512;} 
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 3) { dz=0.8;dr=5.637;} 

	  if(i > 4000 && i < 6000  && (i%1000)/100 == 1) { dz=0.8;dr=4.362;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 2) { dz=0.8;dr=4.512;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 3) { dz=0.8;dr=5.637;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 4) { dz=0.8;dr=5.862;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 5) { dz=0.8;dr=7.501;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 6) { dz=0.8;dr=9.336;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 7) { dz=0.8;dr=10.373;} 
	
	  {  
	    TBox* modoccu = new TBox(avez->GetBinContent(i)-dz,averadius->GetBinContent(i)-dr,avez->GetBinContent(i)+dz,averadius->GetBinContent(i)+dr);
	    modoccu->SetFillStyle(1001);
	    int icol=int(ncol*(log(haveoccu->GetBinContent(i))-log(min))/(log(max)-log(min)));
	    if(icol < 0) icol=0;
	    if(icol > (ncol-1)) icol=(ncol-1);
	    std::cout << i << " " << icol << " " << haveoccu->GetBinContent(i) << std::endl; 
	    modoccu->SetFillColor(gStyle->GetColorPalette(icol));
	    modulesoccu.Add(modoccu);
	  }
	  {
	    TBox* modmult = new TBox(avez->GetBinContent(i)-dz,averadius->GetBinContent(i)-dr,avez->GetBinContent(i)+dz,averadius->GetBinContent(i)+dr);
	    modmult->SetFillStyle(1001);
	    int icol=int(ncol*(log(havemult->GetBinContent(i))-log(mmin))/(log(mmax)-log(mmin)));
	    if(icol < 0) icol=0;
	    if(icol > (ncol-1)) icol=(ncol-1);
	    std::cout << i << " " << icol << " " << havemult->GetBinContent(i) << std::endl; 
	    modmult->SetFillColor(gStyle->GetColorPalette(icol));
	    modulesmult.Add(modmult);
	  }

	}

      }
      // eta boundaries lines
      TList etalines;
      TList etalabels;
      for(int i=0;i<8;++i) {
	double eta = 3.0-i*0.2;
	TLine* lin = new TLine(295,2*295/(exp(eta)-exp(-eta)),305,2*305/(exp(eta)-exp(-eta)));
	etalines.Add(lin);
	char lab[100];
	sprintf(lab,"%3.1f",eta);
	TText* label = new TText(285,2*285/(exp(eta)-exp(-eta)),lab);
	label->SetTextSize(.03);
	label->SetTextAlign(22);
	etalabels.Add(label);
      }
      for(int i=0;i<8;++i) {
	double eta = -3.0+i*0.2;
	TLine* lin = new TLine(-295,-2*295/(exp(eta)-exp(-eta)),-305,-2*305/(exp(eta)-exp(-eta)));
	etalines.Add(lin);
	char lab[100];
	sprintf(lab,"%3.1f",eta);
	TText* label = new TText(-285,-2*285/(exp(eta)-exp(-eta)),lab);
	label->SetTextSize(.03);
	label->SetTextAlign(22);
	etalabels.Add(label);
      }
      for(int i=0;i<15;++i) {
	double eta = -1.4+i*0.2;
	TLine* lin = new TLine(130.*(exp(eta)-exp(-eta))/2.,130,138.*(exp(eta)-exp(-eta))/2.,138);
	etalines.Add(lin);
	char lab[100];
	sprintf(lab,"%3.1f",eta);
	TText* label = new TText(125.*(exp(eta)-exp(-eta))/2.,125,lab);
	label->SetTextSize(.03);
	label->SetTextAlign(22);
	etalabels.Add(label);
      }


      TGaxis *raxis = new TGaxis(-310,0,-310,140,0,140,10,"S");
      TGaxis *zaxis = new TGaxis(-310,0,310,0,-310,310,10,"S");
      raxis->SetTickSize(.01);      zaxis->SetTickSize(.01);
      raxis->SetTitle("R (cm)"); zaxis->SetTitle("Z (cm)");

      TList palette;
      TList mpalette;

      for(int i = 0;i< ncol ; ++i) {
	TBox* box= new TBox(315,0+140./ncol*i,330,0+140./ncol*(i+1));
	box->SetFillStyle(1001);
	box->SetFillColor(gStyle->GetColorPalette(i));
	palette.Add(box);
	mpalette.Add(box);

      }

      TGaxis *paxis = new TGaxis(330,0,330,140,min,max,510,"SLG+");
      paxis->SetTickSize(.02);
      paxis->SetLabelOffset(paxis->GetLabelOffset()*0.5);
      palette.Add(paxis);

      TGaxis *mpaxis = new TGaxis(330,0,330,140,mmin,mmax,510,"SLG+");
      mpaxis->SetTickSize(.02);
      mpaxis->SetLabelOffset(paxis->GetLabelOffset()*0.5);
      mpalette.Add(mpaxis);

      TCanvas* cc1 = new TCanvas("occumap","occumap",1000,500);
      cc1->Range(-370.,-20.,390.,150.);
      TFrame* fr1 = new TFrame(-310,0,310,140);
      fr1->UseCurrentStyle();
      fr1->Draw();
      raxis->Draw(); zaxis->Draw();
      std::cout << modulesoccu.GetSize() << std::endl;
      etalines.Draw();
      etalabels.Draw();
      palette.Draw();
      modulesoccu.Draw();

      TCanvas* cc2 = new TCanvas("multmap","multmap",1000,500); 
      cc2->Range(-370.,-20.,390.,150.);
      TFrame* fr2 = new TFrame(-310,0,310,140);
      fr2->UseCurrentStyle();
      fr2->Draw();
      raxis->Draw(); zaxis->Draw();
      std::cout << modulesmult.GetSize() << std::endl;
      etalines.Draw();
      etalabels.Draw();
      mpalette.Draw();
      modulesmult.Draw();

    }


  }

}
