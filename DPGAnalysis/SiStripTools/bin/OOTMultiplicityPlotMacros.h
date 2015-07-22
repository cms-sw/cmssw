#ifndef DPGAnalysis_SiStripTools_OOTMultiplicityPlotMacros_h
#define DPGAnalysis_SiStripTools_OOTMultiplicityPlotMacros_h

#include "TH1F.h"

class TH1F;
class TFile;
class OOTResult;
class OOTSummary;

OOTSummary* ComputeOOTFractionvsRun(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm=0);
OOTSummary* ComputeOOTFractionvsFill(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const char* hname, OOTSummary* ootsumm=0);
OOTResult* ComputeOOTFraction(TFile* ff, const char* itmodule, const char* ootmodule, const char* etmodule, const int run, const char* hname, 
			      const bool& perFill=false);
std::vector<int> FillingScheme(TFile* ff, const char* path, const float thr=0.);
std::vector<int> FillingSchemeFromProfile(TFile* ff, const char* path, const char* hname, const float thr=0.);

class OOTResult {
 public:
  TH1F* hratio;
  float ootfrac;
  float ootfracerr;
  float ootfracsum;
  float ootfracsumerr;
  int ngoodbx;
  int nfilledbx;  
 OOTResult(): hratio(0),ootfrac(-1.),ootfracerr(0.),ootfracsum(-1.),ootfracsumerr(0.),ngoodbx(0),nfilledbx(0) {}
  ~OOTResult() { delete hratio;}
};

class OOTSummary {
 public:
  TH1F* hootfrac;
  TH1F* hootfracsum;
  TH1F* hngoodbx;
  OOTSummary() {
    hootfrac = new TH1F("ootfrac","OOT fraction vs fill/run",10,0.,10.);
    hootfrac->SetCanExtend(TH1::kXaxis);

    hootfracsum = new TH1F("ootfracsum","OOT summed fraction vs fill/run",10,0.,10.);
    hootfracsum->SetCanExtend(TH1::kXaxis);
  
    hngoodbx = new TH1F("ngoodbx","Number of good BX pairs vs fill/run",10,0.,10.);
    hngoodbx->SetCanExtend(TH1::kXaxis);
    
  }
  ~OOTSummary() {
    delete hootfrac;
    delete hootfracsum;
    delete hngoodbx;
  }
};

#endif // DPGAnalysis_SiStripTools_OOTMultiplicityPlotMacros_h
