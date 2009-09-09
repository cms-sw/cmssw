#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"

void RunScale(int type, int st =0, double h1=0, double h2=0, int idx=0, int np=3, int mode=0 ) {

  // type = 0 && mode = 0   : do all scaling
  // mode = 1               : scaling for a specific case
  // mode = 0 && type != 0  : scaling for whole CSC/DT

  gROOT->ProcessLine(".L SeedPtFunction.C+");
  gROOT->ProcessLine(".L SeedPtScale.C+");
  SeedPtScale *scaler = new SeedPtScale() ;
  
  if ( mode != 0 ) scaler->PtScale(type, st, h1, h2, idx, np );
 
  if ( mode == 0 ) { 
     np = 2;
     if ( type  == 1 || type == 0 ) {   
        scaler->PtScale(1,01, 1.55, 1.65, 1, np);
        scaler->PtScale(1,12, 1.20, 1.30, 1, np);
        scaler->PtScale(1,12, 1.70, 1.90, 2, np);
        scaler->PtScale(1,12, 1.70, 1.90, 3, np);
        scaler->PtScale(1,13, 1.30, 1.40, 2, np);
        scaler->PtScale(1,13, 1.80, 2.00, 3, np);
        scaler->PtScale(1,14, 1.80, 2.00, 3, np);
        scaler->PtScale(1,23, 1.20, 1.25, 1, np);
        scaler->PtScale(1,23, 1.80, 2.00, 2, np);
        scaler->PtScale(1,24, 1.85, 2.05, 1, np);
        scaler->PtScale(1,34, 1.82, 2.02, 1, np);
        cout<<" CSC DONE!!"<<endl;
     }       
     if ( type  == 2 || type == 0 ) {   
        scaler->PtScale(2,1213, 0.90, 0.95, 0, np);
        scaler->PtScale(2,1222, 1.04, 1.09, 0, np);
        scaler->PtScale(2,1232, 1.1 , 1.2 , 0, np);
        scaler->PtScale(2,2213, 0.9 , 0.96, 0, np);
        scaler->PtScale(2,2222, 1.0 , 1.04, 0, np);
     }
     if ( type  == 3 || type == 0 ) {   
        scaler->PtScale(3,12, 0.0 , 0.1 , 1, np);
        scaler->PtScale(3,12, 0.85, 0.9 , 2, np);
        scaler->PtScale(3,13, 0.0 , 0.1 , 1, np);
        scaler->PtScale(3,13, 0.65, 0.75, 2, np);
        scaler->PtScale(3,14, 0.0 , 0.1 , 1, np);
        scaler->PtScale(3,14, 0.55, 0.65, 2, np);
        scaler->PtScale(3,23, 0.0 , 0.1 , 1, np);
        scaler->PtScale(3,23, 0.75, 0.85, 2, np);
        scaler->PtScale(3,24, 0.0 , 0.1 , 1, np);
        scaler->PtScale(3,24, 0.58, 0.68, 2, np);
        scaler->PtScale(3,34, 0.0,  0.1,  1, np);
        scaler->PtScale(3,34, 0.65, 0.75, 2, np);
     }       
     if ( type  == 4 || type == 0 ) {   
        scaler->PtScale(4,10, 0.00, 0.20, 0, np);
        scaler->PtScale(4,11, 0.40, 0.60, 0, np);
        scaler->PtScale(4,12, 0.85, 0.95, 0, np);
        scaler->PtScale(4,20, 0.00, 0.20, 0, np);
        scaler->PtScale(4,21, 0.40, 0.60, 0, np);
        scaler->PtScale(4,22, 0.72, 0.82, 0, np);
        scaler->PtScale(4,30, 0.00, 0.18, 0, np);
        scaler->PtScale(4,31, 0.30, 0.50, 0, np);
        scaler->PtScale(4,32, 0.60, 0.80, 0, np);
     }
     if ( type  == 5 || type == 0 ) {   
        scaler->PtScale(5,11, 1.78, 1.83, 0, np);
        scaler->PtScale(5,12, 1.36, 1.44, 0, np);
        scaler->PtScale(5,13, 1.03, 1.09, 0, np);
        scaler->PtScale(5,21, 1.82, 2.00, 0, np);
        scaler->PtScale(5,22, 1.28, 1.36, 0, np);
     }


  } 
  gROOT->Reset();
  
}
