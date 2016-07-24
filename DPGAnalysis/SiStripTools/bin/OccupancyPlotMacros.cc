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
#include "TROOT.h"
#include <cstring>
#include <iostream>
#include <math.h>
#include "TROOT.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "OccupancyPlotMacros.h"
//#include <vector>

float linear(float x) { return x;}
float logarithm(float x) {return log(x);}

std::pair<float,float> presentbin(int i) {
  
  float dz=-1; float dr=-1;

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
  
  std::pair<float,float> res(dz,dr);
  return res;
}

std::pair<float,float> phase2bin(int i) {
  
  float dz=-1; float dr=-1;

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
  
  std::pair<float,float> res(dz,dr);
  return res;
}

std::pair<float,float> phase1bin(int i) {
  
  float dz=-1; float dr=-1;

  if(i > 1000 && i < 1040) { dz=3.33;dr=0.4;}  // BPIX
  if(i > 1040 && i < 1080) { dr=3.33;dz=0.4;}  // FPIX
  
  if(i > 1100 && i < 2000) { dz=5.948;dr=0.4;}  // TIB
  
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
  
  std::pair<float,float> res(dz,dr);
  return res;
}

void printFrame(TCanvas* c, TH1D* h, const char* label, const int frame, const int min, const int max, const bool same) {
  c->cd(frame);
  h->SetAxisRange(min,max);
  if(same) { 
    h->DrawCopy("same"); 
  } else {
    h->DrawCopy();
    TText* t = new TText((max+min)/2,h->GetMaximum(),label); t->SetTextAlign(22);
    t->DrawClone();
  }
}

void PlotOccupancyMapGeneric(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color,
			     std::pair<float,float>(*size)(int), std::vector<SubDetParams>& vsub) {

  gROOT->SetStyle("Plain");

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
      haveoccu->SetLineColor(kRed);
      haveoccu->SetMarkerColor(kRed);
      haveoccu->DrawCopy();
      
      new TCanvas("multiplicity","multiplicity",1200,500);
      gPad->SetLogy(1);
      havemult->SetStats(0);
      havemult->SetLineColor(kRed);
      havemult->SetMarkerColor(kRed);
      havemult->DrawCopy();
      
      new TCanvas("width","width",1200,500);
      havewidth->SetStats(0);
      havewidth->SetLineColor(kRed);
      havewidth->SetMarkerColor(kRed);
      havewidth->DrawCopy();
      
      bool same=false;
      TCanvas* o2 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("occupancy2");
      if(o2) {
	same=true;
	haveoccu->SetLineColor(kBlue);
	haveoccu->SetMarkerColor(kBlue);
      }
      else {
	o2 = new TCanvas("occupancy2","occupancy2",1200,800);
	o2->Divide(3,2);
      }
      for(unsigned int isub=0;isub<vsub.size();++isub) {
	printFrame(o2,haveoccu,vsub[isub].label.c_str(),isub+1,vsub[isub].min,vsub[isub].max,same);
      }

      same=false;
      TCanvas* m2 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("multiplicity2");
      if(m2) {
	same=true;
	havemult->SetLineColor(kBlue);
	havemult->SetMarkerColor(kBlue);
      }
      else {
	m2 = new TCanvas("multiplicity2","multiplicity2",1200,800);
	m2->Divide(3,2);
      }
      for(unsigned int isub=0;isub<vsub.size();++isub) {
	printFrame(m2,havemult,vsub[isub].label.c_str(),isub+1,vsub[isub].min,vsub[isub].max,same);
      }

      same=false;
      TCanvas* w2 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("width2");
      if(w2) {
	same=true;
	havewidth->SetLineColor(kBlue);
	havewidth->SetMarkerColor(kBlue);
      }
      else {
	w2 = new TCanvas("width2","width2",1200,800);
	w2->Divide(3,2);
      }
      for(unsigned int isub=0;isub<vsub.size();++isub) {
	printFrame(w2,havewidth,vsub[isub].label.c_str(),isub+1,vsub[isub].min,vsub[isub].max,same);
      }

      float (*scale)(float);
      scale = &logarithm;

      
      drawMap("multmap",havemult,averadius,avez,mmin,mmax,size,scale,color);
      drawMap("occumap",haveoccu,averadius,avez,min,max,size,scale,color,"channel occupancy");
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

void PlotOccupancyMap(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &presentbin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",100,270}; vsub.push_back(ppix);
  SubDetParams ptib={"TIB",1050,1450}; vsub.push_back(ptib);
  SubDetParams ptid={"TID",2070,2400}; vsub.push_back(ptid);
  SubDetParams ptob={"TOB",3000,3700}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",4000,4850}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",5000,5850}; vsub.push_back(ptecp);

  PlotOccupancyMapGeneric(ff,module,min,max,mmin,mmax,color,size,vsub);
}

void PlotOccupancyMapPhase1(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &phase1bin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",1000,1080}; vsub.push_back(ppix);
  SubDetParams ptib={"TIB",1090,1450}; vsub.push_back(ptib);
  SubDetParams ptid={"TID",2070,2400}; vsub.push_back(ptid);
  SubDetParams ptob={"TOB",3000,3700}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",4000,4850}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",5000,5850}; vsub.push_back(ptecp);

  PlotOccupancyMapGeneric(ff,module,min,max,mmin,mmax,color,size,vsub);
}

void PlotOccupancyMapPhase2(TFile* ff, const char* module, const float min, const float max, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &phase2bin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",1000,1090}; vsub.push_back(ppix);
  SubDetParams ptob={"TOB",2000,2900}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",3100,3300}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",4100,4300}; vsub.push_back(ptecp);

  PlotOccupancyMapGeneric(ff,module,min,max,mmin,mmax,color,size,vsub);
}

void PlotOnTrackOccupancy(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &presentbin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",100,270}; vsub.push_back(ppix);
  SubDetParams ptib={"TIB",1050,1450}; vsub.push_back(ptib);
  SubDetParams ptid={"TID",2070,2400}; vsub.push_back(ptid);
  SubDetParams ptob={"TOB",3000,3700}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",4000,4850}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",5000,5850}; vsub.push_back(ptecp);

  PlotOnTrackOccupancyGeneric(ff,module,ontrkmod,mmin,mmax,color,size,vsub);
}

void PlotOnTrackOccupancyPhase1(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &phase1bin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",1000,1080}; vsub.push_back(ppix);
  SubDetParams ptib={"TIB",1090,1450}; vsub.push_back(ptib);
  SubDetParams ptid={"TID",2070,2400}; vsub.push_back(ptid);
  SubDetParams ptob={"TOB",3000,3700}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",4000,4850}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",5000,5850}; vsub.push_back(ptecp);

  PlotOnTrackOccupancyGeneric(ff,module,ontrkmod,mmin,mmax,color,size,vsub);
}

void PlotOnTrackOccupancyPhase2(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color) {

  std::pair<float,float> (*size)(int);
  size = &phase2bin;

  std::vector<SubDetParams> vsub;
  SubDetParams ppix={"BPIX+FPIX",1000,1090}; vsub.push_back(ppix);
  SubDetParams ptob={"TOB",2000,2900}; vsub.push_back(ptob);
  SubDetParams ptecm={"TEC-",3100,3300}; vsub.push_back(ptecm);
  SubDetParams ptecp={"TEC+",4100,4300}; vsub.push_back(ptecp);

  PlotOnTrackOccupancyGeneric(ff,module,ontrkmod,mmin,mmax,color,size,vsub);
}

void PlotOnTrackOccupancyGeneric(TFile* ff, const char* module, const char* ontrkmod, const float mmin, const float mmax, const int color, 
				 std::pair<float,float>(*size)(int),const std::vector<SubDetParams>& vsub) {

  
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
      
      bool same=false;
      TCanvas* o2 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("ontrkmult2");
      if(o2) {
	same=true;
	haveontrkmult->SetLineColor(kBlue);
	haveontrkmult->SetMarkerColor(kBlue);
      }
      else {
	o2 = new TCanvas("ontrkmult2","ontrkmult2",1200,800);
	o2->Divide(3,2);
      }
      for(unsigned int isub=0;isub<vsub.size();++isub) {
	printFrame(o2,haveontrkmult,vsub[isub].label.c_str(),isub+1,vsub[isub].min,vsub[isub].max,same);
      }
      
      float (*scale)(float);
      scale = &linear;

      drawMap("ontrkmultmap",haveontrkmult,averadius,avez,mmin,mmax,size,scale,color);
  }	  
}

TCanvas* drawMap(const char* cname, const TH1* hval, const TProfile* averadius, const TProfile* avez,const float mmin, const float mmax, 
		 std::pair<float,float>(*size)(int), float(*scale)(float), const int color, const char* ptitle) {

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
  // Loop on bins and creation of boxes
      
  TList modulesmult;
      
  for(int i=1;i<hval->GetNbinsX();++i) {
	
    if(averadius->GetBinEntries(i)*avez->GetBinEntries(i)) {
      
      double dz = -1.;
      double dr = -1.;
      // determine module size
      
      dz=(*size)(i).first;
      dr=(*size)(i).second;

      if(dz<0 && dr<0) continue;
      
      {
	TBox* modmult = new TBox(avez->GetBinContent(i)-dz,averadius->GetBinContent(i)-dr,avez->GetBinContent(i)+dz,averadius->GetBinContent(i)+dr);
	modmult->SetFillStyle(1001);
	if(color < 0) {
	  modmult->SetFillColor(kBlack);
	}
	else {
	  int icol=int(ncol*(scale(hval->GetBinContent(i))-scale(mmin))/(scale(mmax)-scale(mmin)));
	  if(icol < 0) icol=0;
	  if(icol > (ncol-1)) icol=(ncol-1);
	  std::cout << i << " " << icol << " " << hval->GetBinContent(i) << std::endl; 
	  modmult->SetFillColor(gStyle->GetColorPalette(icol));
	}
	modulesmult.Add(modmult);
      }
      
    }
    
  }
  // eta boundaries lines
  double etavalext[] = {4.,3.5,3.,2.8,2.6,2.4,2.2,2.0,1.8,1.6};
  double etavalint[] = {-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4};
  TList etalines;
  TList etalabels;
  TList paperlabels;
  for(int i=0;i<10;++i) {
    //	double eta = 3.0-i*0.2;
    double eta = etavalext[i];
    TLine* lin = new TLine(295,2*295/(exp(eta)-exp(-eta)),305,2*305/(exp(eta)-exp(-eta)));
    etalines.Add(lin);
    char lab[100];
    sprintf(lab,"%3.1f",eta);
    TText* label = new TText(285,2*285/(exp(eta)-exp(-eta)),lab);
    label->SetTextSize(.03);
    label->SetTextAlign(22);
    etalabels.Add(label);
  }
  for(int i=0;i<10;++i) {
    //	double eta = -3.0+i*0.2;
    double eta = -1*etavalext[i];
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
    //	double eta = -1.4+i*0.2;
    double eta = etavalint[i];
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
  
  TList mpalette;
  
  for(int i = 0;i< ncol ; ++i) {
    TBox* box= new TBox(315,0+140./ncol*i,330,0+140./ncol*(i+1));
    box->SetFillStyle(1001);
    box->SetFillColor(gStyle->GetColorPalette(i));
    mpalette.Add(box);
    
  }

  TGaxis *mpaxis=0;
  if(scale(1)!=1) {
    mpaxis = new TGaxis(330,0,330,140,mmin,mmax,510,"SLG+");
  }
  else {
    mpaxis = new TGaxis(330,0,330,140,mmin,mmax,510,"SL+");
  }
  mpaxis->SetTickSize(.02);
  mpaxis->SetLabelOffset(mpaxis->GetLabelOffset()*0.5);
  mpaxis->SetTitle(ptitle);
  mpalette.Add(mpaxis);
  
  TCanvas* cc2 = new TCanvas(cname,cname,1000,500); 
  cc2->Range(-370.,-20.,390.,150.);
  TFrame* fr2 = new TFrame(-310,0,310,140);
  fr2->UseCurrentStyle();
  fr2->Draw();
  raxis->Draw(); zaxis->Draw();
  std::cout << modulesmult.GetSize() << std::endl;
  etalines.Draw();
  etalabels.Draw();
  if(color>=0) mpalette.Draw();
  modulesmult.Draw();
 
  return cc2;
 
}


void PlotDebugFPIX_XYMap(TFile* ff, const char* module, const unsigned int ioffset, const char* name) {

  gROOT->SetStyle("Plain");

  TCanvas* cc = new TCanvas(name,name,750,750);
  cc->Range(-25,-25,25,25);
  TFrame* fr1 = new TFrame(-20,-20,20,20);
  fr1->UseCurrentStyle();
  fr1->Draw();
  ff->cd(module);
  gDirectory->ls();
  TProfile* avex = (TProfile*)gDirectory->Get("avex");
  TProfile* avey = (TProfile*)gDirectory->Get("avey");
  TProfile* avez = (TProfile*)gDirectory->Get("avez");

  if(avex && avey && avez) {
    TText* tittext = new TText(0,0,name);
    tittext->SetTextSize(.04); tittext->SetTextAlign(22); 
    tittext->Draw();
    for(unsigned int mod=ioffset+1;mod<ioffset+57;++mod) {
      double x = avex->GetBinContent(mod);
      double y = avey->GetBinContent(mod);
      //      TBox* modbox = new TBox(x-1,y-1,x+1,y+1);
      char modstring[30];
      sprintf(modstring,"%d",mod%100);
      TText* modtext = new TText(x,y,modstring);
      modtext->SetTextAngle(atan(y/x)*180/3.14159);
      modtext->SetTextSize(.02); modtext->SetTextAlign(22); modtext->SetTextColor(kRed);
      std::cout << mod << " " << x << " " << y << std::endl;
      //      modbox->Draw();
      modtext->Draw();
    }
    for(unsigned int mod=ioffset+101;mod<ioffset+157;++mod) {
      double x = avex->GetBinContent(mod);
      double y = avey->GetBinContent(mod);
      //      TBox* modbox = new TBox(x-1,y-1,x+1,y+1);
      char modstring[30];
      sprintf(modstring,"%d",mod%100);
      TText* modtext = new TText(x,y,modstring);
      modtext->SetTextAngle(atan(y/x)*180/3.14159);
      modtext->SetTextSize(.02); modtext->SetTextAlign(22); modtext->SetTextColor(kBlue);
      std::cout << mod << " " << x << " " << y << " " << atan(y/x) << std::endl;
      //      modbox->Draw();
      modtext->Draw();
    }

  }
}

void PlotTrackerXsect(TFile* ff, const char* module) {

  gROOT->SetStyle("Plain");

  if(ff->cd(module)) {

    TProfile* averadius = (TProfile*)gDirectory->Get("averadius"); 
    TProfile* avez = (TProfile*)gDirectory->Get("avez"); 

    std::cout << "pointers " << averadius << " " << avez << std::endl;

    if(averadius && avez) {

      std::pair<float,float> (*size)(int);
      size = &presentbin;
      float (*scale)(float);
      scale = &linear;

      
      drawMap("trackermap",averadius,averadius,avez,0,0,size,scale,-1);
    }


  }

}

TH1D* TrendPlotSingleBin(TFile* ff, const char* module, const char* hname, const int bin) {

  CommonAnalyzer caoccu(ff,"",module);

  TH1D* occutrend = new TH1D("occutrend","Average number of clusters vs run",10,0.,10.);
  occutrend->SetCanExtend(TH1::kXaxis);
  occutrend->Sumw2();

  std::vector<unsigned int> runs = caoccu.getRunList();
  std::sort(runs.begin(),runs.end());
  
  {
    for(unsigned int i=0;i<runs.size();++i) {
      
      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);
      char runpath[100];
      sprintf(runpath,"run_%d",runs[i]);
      caoccu.setPath(runpath);
      
      
      TProfile* occu=0;
      if(occu==0) occu = (TProfile*)caoccu.getObject(hname);
      if(occu) {

	const int ibin=occu->FindBin(bin);
	std::cout << runlabel << " " << " " << ibin << " " << occu->GetBinContent(ibin) << " " << occu->GetBinError(ibin) << std::endl;
	const int jbin=occutrend->Fill(runlabel,occu->GetBinContent(ibin));
	occutrend->SetBinError(jbin,occu->GetBinError(ibin));

      }
    }
  } 
  return occutrend;
}
