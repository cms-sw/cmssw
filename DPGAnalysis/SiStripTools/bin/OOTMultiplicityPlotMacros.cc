#include "OOTMultiplicityPlotMacros.h"
#include "DPGAnalysis/SiStripTools/interface/CommonAnalyzer.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TF1.h"
#include "TProfile.h"
#include <iostream>
#include <algorithm>
#include <cmath>

OOTSummary* ComputeOOTFractionvsFill(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm) {

  if(ootsumm==0) {
    ootsumm = new OOTSummary; 
  }    
  
  if(ff) {
    CommonAnalyzer ca(ff,"",itmodule);
    std::vector<unsigned int> runs = ca.getFillList();
    std::sort(runs.begin(),runs.end());
    for(unsigned int i=0;i<runs.size();++i) {

      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);

      OOTResult* res = ComputeOOTFraction(ff,itmodule,ootmodule,etmodule,runs[i],hname,true);

      if(res->ngoodbx!=res->hratio->GetEntries()) std::cout << "Inconsistency in number of good bx" << std::endl;
      ootsumm->hngoodbx->Fill(runlabel,res->ngoodbx);
      int ibin = ootsumm->hootfracsum->Fill(runlabel,res->ootfracsum);
      ootsumm->hootfracsum->SetBinError(ibin,res->ootfracsumerr);
      int bin = ootsumm->hootfrac->Fill(runlabel,res->ootfrac);
      ootsumm->hootfrac->SetBinError(bin,res->ootfracerr);
      delete res;

    }
  }
  else {std::cout << "File is not ok" << std::endl; }

  return ootsumm;
}
OOTSummary* ComputeOOTFractionvsRun(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm) {

  if(ootsumm==0) {
    ootsumm =new OOTSummary;
  }

  if(ff) {
    CommonAnalyzer ca(ff,"",itmodule);
    std::vector<unsigned int> runs = ca.getRunList();
    std::sort(runs.begin(),runs.end());
    for(unsigned int i=0;i<runs.size();++i) {

      char runlabel[100];
      sprintf(runlabel,"%d",runs[i]);

      OOTResult* res = ComputeOOTFraction(ff,itmodule,ootmodule,etmodule,runs[i],hname);

      if(res->ngoodbx!=res->hratio->GetEntries()) std::cout << "Inconsistency in number of good bx" << std::endl;
      ootsumm->hngoodbx->Fill(runlabel,res->ngoodbx);
      int ibin = ootsumm->hootfracsum->Fill(runlabel,res->ootfracsum);
      ootsumm->hootfracsum->SetBinError(ibin,res->ootfracsumerr);
      int bin = ootsumm->hootfrac->Fill(runlabel,res->ootfrac);
      ootsumm->hootfrac->SetBinError(bin,res->ootfracerr);
      delete res;

    }
  }
  else {std::cout << "File is not ok" << std::endl; }

  return ootsumm;
}

OOTResult* ComputeOOTFraction(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const int run, const char* hname, const bool& perFill) {

  if(perFill)  {std::cout << "Processing fill " << run << std::endl;} else {std::cout << "Processing run " << run << std::endl;}

  char itpath[100];
  char ootpath[100];
  char etpath[100];
  if(perFill) {
    sprintf(itpath,"%s/fill_%d",itmodule,run);
    sprintf(ootpath,"%s/fill_%d",ootmodule,run);
    sprintf(etpath,"%s/fill_%d",etmodule,run);
  }
  else {
    sprintf(itpath,"%s/run_%d",itmodule,run);
    sprintf(ootpath,"%s/run_%d",ootmodule,run);
    sprintf(etpath,"%s/run_%d",etmodule,run);
  }

  OOTResult* res = new OOTResult;

  std::vector<int> filledbx = FillingSchemeFromProfile(ff,itpath,hname);
  res->nfilledbx = filledbx.size();

  if(!perFill) {
    std::vector<int> filledbxtest = FillingScheme(ff,etpath);
    if(filledbx.size() != filledbxtest.size()) std::cout << "Inconsistency in number of filled BX " 
							 << run << " " << filledbx.size() << " " << filledbxtest.size() << std::endl;
  }

  TProfile* itmult = 0;
  TProfile* ootmult = 0;
  res->hratio = new TH1F("ratio","ratio",200,0.,2.);
  
  float rclzb=0;
  float rclrandom=0;
  float errclzb=0;
  float errclrandom=0;
  float nzb=0;
  float nrandom=0;
  
  if(ff) {
    if(ff->cd(itpath)) {
      itmult = (TProfile*)gDirectory->Get(hname);
    }
    else { std::cout << "In time path is not ok" << std::endl;}
    if(ff->cd(ootpath)) {
      ootmult = (TProfile*)gDirectory->Get(hname);
    }
    else { std::cout << "out of time path is not ok" << std::endl;}
    if(itmult && ootmult) {
      ootmult->SetLineColor(kRed);      ootmult->SetMarkerColor(kRed);
      //      ootmult->Draw();
      //      itmult->Draw("same");
      for(std::vector<int>::const_iterator fbx=filledbx.begin();fbx!=filledbx.end();++fbx) {
	nzb += itmult->GetBinEntries(*fbx);
	nrandom += ootmult->GetBinEntries(*fbx+1);
      }
      for(std::vector<int>::const_iterator fbx=filledbx.begin();fbx!=filledbx.end();++fbx) {
	if(nzb > 0 && nrandom > 0) {
	  rclzb += (itmult->GetBinContent(*fbx)*itmult->GetBinEntries(*fbx))/nzb;
	  errclzb += (itmult->GetBinError(*fbx)*itmult->GetBinEntries(*fbx))*(itmult->GetBinError(*fbx)*itmult->GetBinEntries(*fbx))/(nzb*nzb);
	  rclrandom += (ootmult->GetBinContent(*fbx+1)*ootmult->GetBinEntries(*fbx+1))/nrandom;
	  errclrandom += (ootmult->GetBinError(*fbx+1)*ootmult->GetBinEntries(*fbx+1))*(ootmult->GetBinError(*fbx+1)*ootmult->GetBinEntries(*fbx+1))
	    /(nrandom*nrandom);
	}
	if(itmult->GetBinContent(*fbx)==0) { std::cout << "No cluster in filled BX! " << *fbx << std::endl;}
	else if(ootmult->GetBinEntries(*fbx+1)==0) {/* std::cout << "No entry in OOT BX " << *fbx+1 << std::endl; */}
	else {
	  float rat= ootmult->GetBinContent(*fbx+1)/itmult->GetBinContent(*fbx);
	  res->hratio->Fill(rat);
	}	  
      }
    }
    else {std::cout<<"histograms not found"<<std::endl;}
  }
  else { std::cout << "Input file pointer is not ok" <<std::endl;}

  res->ngoodbx = res->hratio->GetEntries();

  if(nzb>0 && nrandom>0 &&rclzb>0 && rclrandom > 0) {
    res->ootfracsum = rclrandom/rclzb;
    res->ootfracsumerr = rclrandom/rclzb*sqrt(errclzb/(rclzb*rclzb)+errclrandom/(rclrandom*rclrandom));
  }
  if(res->ngoodbx) {
    res->hratio->Fit("gaus","Q0","",.01,1.99);
    if(res->hratio->GetFunction("gaus")) {
      res->ootfrac = res->hratio->GetFunction("gaus")->GetParameter(1);
      res->ootfracerr = res->hratio->GetFunction("gaus")->GetParError(1);
    }
    else {std::cout << "Missing fitting function" << std::endl;}
  }
  else {std::cout << "No filled BX or strange filling scheme" <<std::endl;}	

  return res;
}

std::vector<int> FillingScheme(TFile* ff, const char* path, const float thr) {

  TH1F* bx=0;
  std::vector<int> filledbx;
  if(ff) {
    if(ff->cd(path)) {
      bx = (TH1F*)gDirectory->Get("bx");
      if(bx) {
	//	bx->Draw();
	std::cout << "Number of entries " << bx->GetEntries() << " threshold " << thr/3564.*bx->GetEntries() << std::endl;
	for(int i=1;i<bx->GetNbinsX()+1;++i) {
	  if(bx->GetBinContent(i)>thr/3564.*bx->GetEntries()) {
	    if(filledbx.size() && i == filledbx[filledbx.size()-1]+1) {
	      std::cout << "This is not a 50ns run ! " << std::endl;
	      filledbx.clear();
	      return filledbx;
	    }
	    filledbx.push_back(i);
	  }
	}
      }
      else { std::cout << "Histogram not found" << std::endl;}
    }
    else { std::cout << "module path is not ok" << std::endl;}
  }
  else { std::cout << "Input file pointer is not ok" <<std::endl;}
  
  //  std::cout << filledbx.size() << " filled bunch crossings" << std::endl;
  //  for(std::vector<int>::const_iterator fbx=filledbx.begin();fbx!=filledbx.end();++fbx) { std::cout << *fbx << std::endl;}
  return filledbx;
}
std::vector<int> FillingSchemeFromProfile(TFile* ff, const char* path, const char* hname, const float thr) {

  TProfile* bx=0;
  std::vector<int> filledbx;
  if(ff) {
    if(ff->cd(path)) {
      bx = (TProfile*)gDirectory->Get(hname);
      if(bx) {
	//	bx->Draw();
	std::cout << "Number of entries " << bx->GetEntries() << " threshold " << thr/3564.*bx->GetEntries() << std::endl;
	for(int i=1;i<bx->GetNbinsX()+1;++i) {
	  if(bx->GetBinEntries(i)>thr/3564.*bx->GetEntries()) {
	    if(filledbx.size() && i == filledbx[filledbx.size()-1]+1) {
	      std::cout << "This is not a 50ns run ! " << std::endl;
	      filledbx.clear();
	      return filledbx;
	    }
	    filledbx.push_back(i);
	  }
	}
      }
      else { std::cout << "Histogram not found" << std::endl;}
    }
    else { std::cout << "module path is not ok" << std::endl;}
  }
  else { std::cout << "Input file pointer is not ok" <<std::endl;}
  
  //  std::cout << filledbx.size() << " filled bunch crossings" << std::endl;
  //  for(std::vector<int>::const_iterator fbx=filledbx.begin();fbx!=filledbx.end();++fbx) { std::cout << *fbx << std::endl;}
  return filledbx;
}
