#include "TText.h"
#include "TLatex.h"
#include "TLine.h"
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

void printFrame(TCanvas* c, TH1D* h, TText* t, const int frame, const int min, const int max) {
  c->cd(frame);
  h->SetAxisRange(min,max);
  h->DrawCopy();
  t->SetY(h->GetMaximum()); t->SetX((max+min)/2); t->DrawClone();
}

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
      havewidth->SetTitle("Average Cluster Size");
      havewidth->Divide(havemult);


      new TCanvas("occupancy","occupancy",1200,500);
      gPad->SetLogy(1);
      haveoccu->SetStats(0);
      haveoccu->DrawCopy();
      TLine* l1 = new TLine(1000,0,1000,haveoccu->GetMaximum()); l1->DrawClone(); 
      TLine* l2 = new TLine(2000,0,2000,haveoccu->GetMaximum()); l2->DrawClone(); 
      TLine* l3 = new TLine(3000,0,3000,haveoccu->GetMaximum()); l3->DrawClone(); 
      TLine* l4 = new TLine(4000,0,4000,haveoccu->GetMaximum()); l4->DrawClone(); 
      TLine* l5 = new TLine(5000,0,5000,haveoccu->GetMaximum()); l5->DrawClone(); 
      TText* tpix = new TText(500,haveoccu->GetMaximum(),"BPIX+FPIX"); tpix->SetTextAlign(22); tpix->DrawClone();
      TText* ttib = new TText(1500,haveoccu->GetMaximum(),"TIB"); ttib->SetTextAlign(22); ttib->DrawClone();
      TText* ttid = new TText(2500,haveoccu->GetMaximum(),"TID"); ttid->SetTextAlign(22); ttid->DrawClone();
      TText* ttob = new TText(3500,haveoccu->GetMaximum(),"TOB"); ttob->SetTextAlign(22); ttob->DrawClone();
      TText* ttecm = new TText(4500,haveoccu->GetMaximum(),"TEC-"); ttecm->SetTextAlign(22); ttecm->DrawClone();
      TText* ttecp = new TText(5500,haveoccu->GetMaximum(),"TEC+"); ttecp->SetTextAlign(22); ttecp->DrawClone();
      
      new TCanvas("multiplicity","multiplicity",1200,500);
      gPad->SetLogy(1);
      havemult->SetStats(0);
      havemult->DrawCopy();
      tpix->SetY(havemult->GetMaximum()); tpix->DrawClone();
      ttib->SetY(havemult->GetMaximum()); ttib->DrawClone();
      ttid->SetY(havemult->GetMaximum()); ttid->DrawClone();
      ttob->SetY(havemult->GetMaximum()); ttob->DrawClone();
      ttecm->SetY(havemult->GetMaximum()); ttecm->DrawClone();
      ttecp->SetY(havemult->GetMaximum()); ttecp->DrawClone();
      l1->SetY2(havemult->GetMaximum()); l1->DrawClone(); 
      l2->SetY2(havemult->GetMaximum()); l2->DrawClone(); 
      l3->SetY2(havemult->GetMaximum()); l3->DrawClone(); 
      l4->SetY2(havemult->GetMaximum()); l4->DrawClone(); 
      l5->SetY2(havemult->GetMaximum()); l5->DrawClone(); 
      
      new TCanvas("width","width",1200,500);
      havewidth->SetStats(0);
      havewidth->DrawCopy();
      tpix->SetY(havewidth->GetMaximum()); tpix->DrawClone();
      ttib->SetY(havewidth->GetMaximum()); ttib->DrawClone();
      ttid->SetY(havewidth->GetMaximum()); ttid->DrawClone();
      ttob->SetY(havewidth->GetMaximum()); ttob->DrawClone();
      ttecm->SetY(havewidth->GetMaximum()); ttecm->DrawClone();
      ttecp->SetY(havewidth->GetMaximum()); ttecp->DrawClone();
      l1->SetY2(havewidth->GetMaximum()); l1->DrawClone(); 
      l2->SetY2(havewidth->GetMaximum()); l2->DrawClone(); 
      l3->SetY2(havewidth->GetMaximum()); l3->DrawClone(); 
      l4->SetY2(havewidth->GetMaximum()); l4->DrawClone(); 
      l5->SetY2(havewidth->GetMaximum()); l5->DrawClone();
      
      TCanvas * o2 = new TCanvas("occupancy2","occupancy2",1200,800);
      o2->Divide(3,2);
      o2->cd(1);
      haveoccu->SetAxisRange(100,270);
      haveoccu->DrawCopy();
      tpix->SetY(haveoccu->GetMaximum()); tpix->SetX(185); tpix->DrawClone();
      o2->cd(2);
      haveoccu->SetAxisRange(1050,1450);
      haveoccu->DrawCopy();
      ttib->SetY(haveoccu->GetMaximum()); ttib->SetX(1250); ttib->DrawClone();
      o2->cd(3);
      haveoccu->SetAxisRange(2070,2400);
      haveoccu->DrawCopy();
      ttid->SetY(haveoccu->GetMaximum()); ttid->SetX(2235); ttid->DrawClone();
      o2->cd(4);
      haveoccu->SetAxisRange(3000,3700);
      haveoccu->DrawCopy();
      ttob->SetY(haveoccu->GetMaximum()); ttob->SetX(3350); ttob->DrawClone();
      o2->cd(5);
      haveoccu->SetAxisRange(4000,4850);
      haveoccu->DrawCopy();
      ttecm->SetY(haveoccu->GetMaximum()); ttecm->SetX(4425); ttecm->DrawClone();
      o2->cd(6);
      haveoccu->SetAxisRange(5000,5850);
      haveoccu->DrawCopy();
      ttecp->SetY(haveoccu->GetMaximum()); ttecp->SetX(5425); ttecp->DrawClone();

      TCanvas * m2 = new TCanvas("multiplicity2","multiplicity2",1200,800);
      m2->Divide(3,2);
      m2->cd(1);
      havemult->SetAxisRange(100,270);
      havemult->DrawCopy();
      tpix->SetY(havemult->GetMaximum()); tpix->SetX(185); tpix->DrawClone();
      m2->cd(2);
      havemult->SetAxisRange(1050,1450);
      havemult->DrawCopy();
      ttib->SetY(havemult->GetMaximum()); ttib->SetX(1250); ttib->DrawClone();
      m2->cd(3);
      havemult->SetAxisRange(2070,2400);
      havemult->DrawCopy();
      ttid->SetY(havemult->GetMaximum()); ttid->SetX(2235); ttid->DrawClone();
      m2->cd(4);
      havemult->SetAxisRange(3000,3700);
      havemult->DrawCopy();
      ttob->SetY(havemult->GetMaximum()); ttob->SetX(3350); ttob->DrawClone();
      m2->cd(5);
      havemult->SetAxisRange(4000,4850);
      havemult->DrawCopy();
      ttecm->SetY(havemult->GetMaximum()); ttecm->SetX(4425); ttecm->DrawClone();
      m2->cd(6);
      havemult->SetAxisRange(5000,5850);
      havemult->DrawCopy();
      ttecp->SetY(havemult->GetMaximum()); ttecp->SetX(5425); ttecp->DrawClone();

      TCanvas * w2 = new TCanvas("width2","width2",1200,800);
      w2->Divide(3,2);
      w2->cd(1);
      havewidth->SetAxisRange(100,270);
      havewidth->DrawCopy();
      tpix->SetY(havewidth->GetMaximum()); tpix->SetX(185); tpix->DrawClone();
      w2->cd(2);
      havewidth->SetAxisRange(1050,1450);
      havewidth->DrawCopy();
      ttib->SetY(havewidth->GetMaximum()); ttib->SetX(1250); ttib->DrawClone();
      w2->cd(3);
      havewidth->SetAxisRange(2070,2400);
      havewidth->DrawCopy();
      ttid->SetY(havewidth->GetMaximum()); ttid->SetX(2235); ttid->DrawClone();
      w2->cd(4);
      havewidth->SetAxisRange(3000,3700);
      havewidth->DrawCopy();
      ttob->SetY(havewidth->GetMaximum()); ttob->SetX(3350); ttob->DrawClone();
      w2->cd(5);
      havewidth->SetAxisRange(4000,4850);
      havewidth->DrawCopy();
      ttecm->SetY(havewidth->GetMaximum()); ttecm->SetX(4425); ttecm->DrawClone();
      w2->cd(6);
      havewidth->SetAxisRange(5000,5850);
      havewidth->DrawCopy();
      ttecp->SetY(havewidth->GetMaximum()); ttecp->SetX(5425); ttecp->DrawClone();

      // Loop on bins and creation of boxes

      TList modulesoccu;
      TList modulesmult;

      for(int i=1;i<haveoccu->GetNbinsX();++i) {

	if(averadius->GetBinEntries(i)*avez->GetBinEntries(i)) {

	  double dz = 2.;
	  double dr = 1.;
	  // determine module size
	  
	  if(i > 100 && i < 200) { dz=3.33;dr=0.4;}  // BPIX

	  if(i > 200 && i < 1000 && ( i%10 == 1 || i%10 == 7)) { dz=0.8;dr=0.4;}  // FPIX
	  if(i > 200 && i < 1000 && !( i%10 == 1 || i%10 == 7)) { dz=0.8;dr=0.8;}

	  if(i > 1000 && i < 2000) { dz=5.948;dr=0.4;}  // TIB

	  if(i > 3000 && i < 4000) { dz=9.440;dr=0.4;}  // TOB

	  if(i > 2000 && i < 3000  && (i%1000)/100 == 1) { dz=0.8;dr=5.647;} // TID
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 2) { dz=0.8;dr=4.512;} 
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 3) { dz=0.8;dr=5.637;} 

	  if(i > 4000 && i < 6000  && (i%1000)/100 == 1) { dz=0.8;dr=4.362;} // TEC
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
      TList paperlabels;
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
      TLatex* etalab = new  TLatex(0,115,"#eta");
      etalab->SetTextSize(.03);
      etalab->SetTextAlign(22);
      etalabels.Add(etalab);

      // CMS label
      TLatex *cmslab = new TLatex(0.15,0.965,"CMS");
      cmslab->SetNDC();
      cmslab->SetTextSize(0.04);
      cmslab->SetTextAlign(31);
      paperlabels.Add(cmslab);
      TLatex *enelab = new TLatex(0.92,0.965,"#sqrt{s} = 7 TeV");
      enelab->SetNDC();
      enelab->SetTextSize(0.04);
      enelab->SetTextAlign(31);
      paperlabels.Add(enelab);
      /*
      TLatex *lumilab = new TLatex(0.6,0.965,Form("L = %.1f  fb^{-1}",19.7));
      lumilab->SetNDC();
      lumilab->SetTextSize(0.04);
      lumilab->SetTextAlign(31);
      paperlabels.Add(lumilab);
      */

      TGaxis *raxis = new TGaxis(-310,0,-310,140,0,140,10,"S");
      TGaxis *zaxis = new TGaxis(-310,0,310,0,-310,310,10,"S");
      raxis->SetTickSize(.01);      zaxis->SetTickSize(.01);
      raxis->SetTitle("r (cm)"); zaxis->SetTitle("z (cm)");

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
      paxis->SetTitle("channel occupancy");
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
      paperlabels.Draw();
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
void PlotOccupancyMapPhase2(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color) {

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
      havewidth->SetTitle("Average Cluster Size");
      havewidth->Divide(havemult);


      new TCanvas("occupancy","occupancy",1200,500);
      gPad->SetLogy(1);
      haveoccu->SetStats(0);
      haveoccu->DrawCopy();
      TLine* l1 = new TLine(1000,0,1000,haveoccu->GetMaximum()); l1->DrawClone(); 
      TLine* l2 = new TLine(2000,0,2000,haveoccu->GetMaximum()); l2->DrawClone(); 
      TLine* l3 = new TLine(3000,0,3000,haveoccu->GetMaximum()); l3->DrawClone(); 
      TLine* l4 = new TLine(4000,0,4000,haveoccu->GetMaximum()); l4->DrawClone(); 
      TLine* l5 = new TLine(5000,0,5000,haveoccu->GetMaximum()); l5->DrawClone(); 
      TText* tpix = new TText(1500,haveoccu->GetMaximum(),"BPIX+FPIX"); tpix->SetTextAlign(22); tpix->DrawClone();
      //      TText* ttib = new TText(1500,haveoccu->GetMaximum(),"TIB"); ttib->SetTextAlign(22); ttib->DrawClone();
      //      TText* ttid = new TText(2500,haveoccu->GetMaximum(),"TID"); ttid->SetTextAlign(22); ttid->DrawClone();
      TText* ttob = new TText(2500,haveoccu->GetMaximum(),"TOB"); ttob->SetTextAlign(22); ttob->DrawClone();
      TText* ttecm = new TText(3500,haveoccu->GetMaximum(),"TEC-"); ttecm->SetTextAlign(22); ttecm->DrawClone();
      TText* ttecp = new TText(4500,haveoccu->GetMaximum(),"TEC+"); ttecp->SetTextAlign(22); ttecp->DrawClone();
      
      new TCanvas("multiplicity","multiplicity",1200,500);
      gPad->SetLogy(1);
      havemult->SetStats(0);
      havemult->DrawCopy();
      tpix->SetY(havemult->GetMaximum()); tpix->DrawClone();
      //      ttib->SetY(havemult->GetMaximum()); ttib->DrawClone();
      //      ttid->SetY(havemult->GetMaximum()); ttid->DrawClone();
      ttob->SetY(havemult->GetMaximum()); ttob->DrawClone();
      ttecm->SetY(havemult->GetMaximum()); ttecm->DrawClone();
      ttecp->SetY(havemult->GetMaximum()); ttecp->DrawClone();
      l1->SetY2(havemult->GetMaximum()); l1->DrawClone(); 
      l2->SetY2(havemult->GetMaximum()); l2->DrawClone(); 
      l3->SetY2(havemult->GetMaximum()); l3->DrawClone(); 
      l4->SetY2(havemult->GetMaximum()); l4->DrawClone(); 
      l5->SetY2(havemult->GetMaximum()); l5->DrawClone(); 
      
      new TCanvas("width","width",1200,500);
      havewidth->SetStats(0);
      havewidth->DrawCopy();
      tpix->SetY(havewidth->GetMaximum()); tpix->DrawClone();
      //      ttib->SetY(havewidth->GetMaximum()); ttib->DrawClone();
      //      ttid->SetY(havewidth->GetMaximum()); ttid->DrawClone();
      ttob->SetY(havewidth->GetMaximum()); ttob->DrawClone();
      ttecm->SetY(havewidth->GetMaximum()); ttecm->DrawClone();
      ttecp->SetY(havewidth->GetMaximum()); ttecp->DrawClone();
      l1->SetY2(havewidth->GetMaximum()); l1->DrawClone(); 
      l2->SetY2(havewidth->GetMaximum()); l2->DrawClone(); 
      l3->SetY2(havewidth->GetMaximum()); l3->DrawClone(); 
      l4->SetY2(havewidth->GetMaximum()); l4->DrawClone(); 
      l5->SetY2(havewidth->GetMaximum()); l5->DrawClone();
      
      TCanvas * o2 = new TCanvas("occupancy2","occupancy2",1200,800);
      o2->Divide(2,2);
      printFrame(o2,haveoccu,tpix,1,1000,1090);
      printFrame(o2,haveoccu,ttob,2,2000,2900);
      printFrame(o2,haveoccu,ttecm,3,3100,3300);
      printFrame(o2,haveoccu,ttecp,4,4100,4300);

      TCanvas * m2 = new TCanvas("multiplicity2","multiplicity2",1200,800);
      m2->Divide(2,2);
      printFrame(m2,havemult,tpix,1,1000,1090);
      printFrame(m2,havemult,ttob,2,2000,2900);
      printFrame(m2,havemult,ttecm,3,3100,3300);
      printFrame(m2,havemult,ttecp,4,4100,4300);

      TCanvas * w2 = new TCanvas("width2","width2",1200,800);
      w2->Divide(2,2);
      printFrame(w2,havewidth,tpix,1,1000,1090);
      printFrame(w2,havewidth,ttob,2,2000,2900);
      printFrame(w2,havewidth,ttecm,3,3100,3300);
      printFrame(w2,havewidth,ttecp,4,4100,4300);

      // Loop on bins and creation of boxes

      TList modulesoccu;
      TList modulesmult;

      for(int i=1;i<haveoccu->GetNbinsX();++i) {

	if(i<1000) continue;

	if(averadius->GetBinEntries(i)*avez->GetBinEntries(i)) {

	  double dz = 2.;
	  double dr = 1.;
	  // determine module size
	  
	  if(i > 1000 && i < 1040) { dz=3.33;dr=0.4;}
	  if(i > 1040 && i < 1130) { dr=3.33;dz=0.4;}

	  if(i > 2000 && i < 2550) { dz=2.5;dr=0.1;}
	  if(i > 2550 && i < 3000) { dz=5.0;dr=0.1;}

	  if(i > 3000 && i < 5000) { dz=0.2;dr=5.0;}

	  if(i > 3100 && i < 3119) { dz=0.2;dr=2.5;}
	  if(i > 3140 && i < 3159) { dz=0.2;dr=2.5;}
	  if(i > 3180 && i < 3199) { dz=0.2;dr=2.5;}
	  if(i > 3220 && i < 3239) { dz=0.2;dr=2.5;}
	  if(i > 3260 && i < 3279) { dz=0.2;dr=2.5;}

	  if(i > 4100 && i < 4119) { dz=0.2;dr=2.5;}
	  if(i > 4140 && i < 4159) { dz=0.2;dr=2.5;}
	  if(i > 4180 && i < 4199) { dz=0.2;dr=2.5;}
	  if(i > 4220 && i < 4239) { dz=0.2;dr=2.5;}
	  if(i > 4260 && i < 4279) { dz=0.2;dr=2.5;}



	  /*

	  if(i > 2000 && i < 3000  && (i%1000)/100 == 1) { dz=0.8;dr=5.647;} // TID
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 2) { dz=0.8;dr=4.512;} 
	  if(i > 2000 && i < 3000  && (i%1000)/100 == 3) { dz=0.8;dr=5.637;} 

	  if(i > 4000 && i < 6000  && (i%1000)/100 == 1) { dz=0.8;dr=4.362;} // TEC
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 2) { dz=0.8;dr=4.512;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 3) { dz=0.8;dr=5.637;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 4) { dz=0.8;dr=5.862;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 5) { dz=0.8;dr=7.501;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 6) { dz=0.8;dr=9.336;} 
	  if(i > 4000 && i < 6000  && (i%1000)/100 == 7) { dz=0.8;dr=10.373;} 
	  */	
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

float combinedOccupancy(TFile* ff, const char* module, const int lowerbin, const int upperbin) {

  float cumoccu = -2.;
  double cumerr = -2;

  if(ff->cd(module)) {
    
    TProfile* aveoccu= (TProfile*)gDirectory->Get("aveoccu");
    //    TProfile* avemult= (TProfile*)gDirectory->Get("avemult");
    TH1F* nchannels = (TH1F*)gDirectory->Get("nchannels_real");

    float sumoccu=0.;
    float sumnchannels=0;
    double sumerrsq=0;
    
    for(int i=lowerbin; i<upperbin+1; ++i) {
      std::cout << "processing bin " << i << " " << aveoccu->GetBinContent(i) << "+/-" << aveoccu->GetBinError(i) <<  std::endl;
      sumoccu += aveoccu->GetBinContent(i);
      sumnchannels += nchannels->GetBinContent(i);
      sumerrsq += aveoccu->GetBinError(i)*aveoccu->GetBinError(i);
    }
    cumoccu = sumnchannels!=0 ? sumoccu/sumnchannels : -1;
    cumerr = sumnchannels!=0 ? sqrt(sumerrsq)/sumnchannels : -1;
    std::cout << "Cumulative occupancy: " << sumoccu << " " << sumnchannels << " " << cumoccu << "+/-" << cumerr;
  }

  return cumoccu;

}

void PlotOnTrackOccupancyPhase2(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color) {

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
  
  gROOT->SetStyle("Plain");

  TProfile* avemult=0;
  TProfile* aveontrkmult=0;
  TProfile* averadius =0;
  TProfile* avez =0;

  if(ff->cd(module)) {
    avemult= (TProfile*)gDirectory->Get("avemult");
    averadius = (TProfile*)gDirectory->Get("averadius"); 
    avez = (TProfile*)gDirectory->Get("avez"); 
  }
  if(ff->cd(ontrkmod)) aveontrkmult= (TProfile*)gDirectory->Get("avemult");

  std::cout << "pointers " <<  avemult << " " << aveontrkmult << " " << averadius << " " << avez << std::endl;

  if( averadius && avez && avemult && aveontrkmult) {

    TH1D* havemult = avemult->ProjectionX("havemult");
    TH1D* haveontrkmult = aveontrkmult->ProjectionX("haveontrkmult");
      havemult->SetDirectory(0);
      haveontrkmult->SetDirectory(0);
      haveontrkmult->Divide(havemult);

      new TCanvas("ontrkmult","ontrkmult",1200,500);
      gPad->SetLogy(1);
      haveontrkmult->SetStats(0);
      haveontrkmult->SetLineColor(kRed);
      haveontrkmult->SetMarkerColor(kRed);
      haveontrkmult->SetMarkerSize(.5);
      haveontrkmult->SetMarkerStyle(20);
      haveontrkmult->DrawCopy();
      TLine* l1 = new TLine(1000,0,1000,haveontrkmult->GetMaximum()); l1->DrawClone(); 
      TLine* l2 = new TLine(2000,0,2000,haveontrkmult->GetMaximum()); l2->DrawClone(); 
      TLine* l3 = new TLine(3000,0,3000,haveontrkmult->GetMaximum()); l3->DrawClone(); 
      TLine* l4 = new TLine(4000,0,4000,haveontrkmult->GetMaximum()); l4->DrawClone(); 
      TLine* l5 = new TLine(5000,0,5000,haveontrkmult->GetMaximum()); l5->DrawClone(); 
      TText* tpix = new TText(1500,haveontrkmult->GetMaximum(),"BPIX+FPIX"); tpix->SetTextAlign(22); tpix->DrawClone();
      //      TText* ttib = new TText(1500,haveontrkmult->GetMaximum(),"TIB"); ttib->SetTextAlign(22); ttib->DrawClone();
      //      TText* ttid = new TText(2500,haveontrkmult->GetMaximum(),"TID"); ttid->SetTextAlign(22); ttid->DrawClone();
      TText* ttob = new TText(2500,haveontrkmult->GetMaximum(),"TOB"); ttob->SetTextAlign(22); ttob->DrawClone();
      TText* ttecm = new TText(3500,haveontrkmult->GetMaximum(),"TEC-"); ttecm->SetTextAlign(22); ttecm->DrawClone();
      TText* ttecp = new TText(4500,haveontrkmult->GetMaximum(),"TEC+"); ttecp->SetTextAlign(22); ttecp->DrawClone();
      
      TCanvas * o2 = new TCanvas("ontrkmult2","ontrkmult2",1200,800);
      o2->Divide(2,2);
      printFrame(o2,haveontrkmult,tpix,1,1000,1090);
      printFrame(o2,haveontrkmult,ttob,2,2000,2900);
      printFrame(o2,haveontrkmult,ttecm,3,3100,3300);
      printFrame(o2,haveontrkmult,ttecp,4,4100,4300);
      
      // Loop on bins and creation of boxes
      
      TList modulesmult;
      
      for(int i=1;i<haveontrkmult->GetNbinsX();++i) {
	
	if(i<1000) continue;
	
	if(averadius->GetBinEntries(i)*avez->GetBinEntries(i)) {
	  
	  double dz = 2.;
	  double dr = 1.;
	  // determine module size
	  
	  if(i > 1000 && i < 1040) { dz=3.33;dr=0.4;}
	  if(i > 1040 && i < 1130) { dr=3.33;dz=0.4;}
	  
	  if(i > 2000 && i < 2550) { dz=2.5;dr=0.1;}
	  if(i > 2550 && i < 3000) { dz=5.0;dr=0.1;}
	  
	  if(i > 3000 && i < 5000) { dz=0.2;dr=5.0;}
	  
	  if(i > 3100 && i < 3119) { dz=0.2;dr=2.5;}
	  if(i > 3140 && i < 3159) { dz=0.2;dr=2.5;}
	  if(i > 3180 && i < 3199) { dz=0.2;dr=2.5;}
	  if(i > 3220 && i < 3239) { dz=0.2;dr=2.5;}
	  if(i > 3260 && i < 3279) { dz=0.2;dr=2.5;}
	  
	  if(i > 4100 && i < 4119) { dz=0.2;dr=2.5;}
	  if(i > 4140 && i < 4159) { dz=0.2;dr=2.5;}
	  if(i > 4180 && i < 4199) { dz=0.2;dr=2.5;}
	  if(i > 4220 && i < 4239) { dz=0.2;dr=2.5;}
	  if(i > 4260 && i < 4279) { dz=0.2;dr=2.5;}
	  
	  {
	    TBox* modmult = new TBox(avez->GetBinContent(i)-dz,averadius->GetBinContent(i)-dr,avez->GetBinContent(i)+dz,averadius->GetBinContent(i)+dr);
	    modmult->SetFillStyle(1001);
	    int icol=int(ncol*(haveontrkmult->GetBinContent(i)-mmin)/(mmax-mmin));
	    if(icol < 0) icol=0;
	    if(icol > (ncol-1)) icol=(ncol-1);
	    std::cout << i << " " << icol << " " << haveontrkmult->GetBinContent(i) << std::endl; 
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
      
      TList mpalette;
      
      for(int i = 0;i< ncol ; ++i) {
	TBox* box= new TBox(315,0+140./ncol*i,330,0+140./ncol*(i+1));
	box->SetFillStyle(1001);
	box->SetFillColor(gStyle->GetColorPalette(i));
	mpalette.Add(box);
	
      }
      
      TGaxis *mpaxis = new TGaxis(330,0,330,140,mmin,mmax,510,"SL+");
      mpaxis->SetTickSize(.02);
      mpaxis->SetLabelOffset(mpaxis->GetLabelOffset()*0.5);
      mpalette.Add(mpaxis);
      
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


