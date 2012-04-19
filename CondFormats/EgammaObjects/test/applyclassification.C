#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TFile.h"
#include "GBRForest.h"
#include "Cintex/Cintex.h"
#endif

void applyclassification() {
  
  //enable root i/o for objects with reflex dictionaries in standalone root mode
  ROOT::Cintex::Cintex::Enable();   

  //array of variables to be used to evaluate bdt
  Float_t *vars = new Float_t[10];
  
  //open input root file
  TFile *fin = new TFile("gbrtest.root","READ");
  
  //read GBRForest from file
  GBRForest *gbr = static_cast<GBRForest*>(fin->GetObjectUnchecked("gbrtest"));
  
  //dummy loop, fill variable array for each event and evaluate bdt
  int nevents=10;
  for (int i=0; i<nevents; ++i) {
    for (int j=0; j<10; ++j) {
      vars[j] = i*j;
    }
    double bdtval = gbr->GetClassifier(vars);
    printf("evt %i, bdtval = %5f\n",i,bdtval);
  }
  
  
  
}