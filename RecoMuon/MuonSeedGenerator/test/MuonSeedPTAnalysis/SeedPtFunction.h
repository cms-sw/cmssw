#include "TObject.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <TMath.h>

class SeedPtFunction : public TObject {

private:


public:

   SeedPtFunction();     
   ~SeedPtFunction();     
  
   static Double_t fitf( Double_t* x, Double_t* par);
   static Double_t fitf2( Double_t* x, Double_t* par);
   static Double_t linear( Double_t* x, Double_t* par);
   static Double_t fgaus( Double_t* x, Double_t* par);
   bool DataRejection(double sigmal, double deviation, int N_data ) ;

   ClassDef(SeedPtFunction, 1);

};

#if !defined(__CINT__)
    ClassImp(SeedPtFunction);
#endif

