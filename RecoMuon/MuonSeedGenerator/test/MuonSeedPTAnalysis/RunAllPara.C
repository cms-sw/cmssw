#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"

void testPara(int type, int st =0, double h1=0, double h2=0, int np=0, int mode=0 ) {

  gROOT->ProcessLine(".L SeedPtFunction.C+");
  gROOT->ProcessLine(".L SeedParaFit.C+");

  SeedParaFit *fitter = new SeedParaFit() ;
   
  if ( mode == 1 ) fitter->ParaFit(type,st,h1,h2,np);
  // CSC Pair
  if ( mode != 1 ) {
     if ( type == 0 ) fitter->PrintTitle(); 
     if ( type == 1 || type == 0 ) { 
        fitter->ParaFit(1, 11, 1.57, 1.67, 1);
	fitter->ParaFit(1, 12, 1.00, 1.55, 3);
	fitter->ParaFit(1, 12, 1.72, 2.40, 3);
	fitter->ParaFit(1, 13, 1.20, 1.55, 3);
	fitter->ParaFit(1, 13, 1.76, 2.40, 3);
	fitter->ParaFit(1, 14, 1.80, 2.40, 3);
	fitter->ParaFit(1, 23, 1.15, 2.38, 3);
	fitter->ParaFit(1, 24, 1.85, 2.38, 3);
	fitter->ParaFit(1, 34, 1.85, 2.38, 3);
    }  
    // DT Pair
    if ( type == 3  || type == 0 ) { 
       fitter->ParaFit(3, 12, 0., 1.02, 3);
       fitter->ParaFit(3, 13, 0., 0.88, 3);
       fitter->ParaFit(3, 14, 0., 0.78, 3);
       fitter->ParaFit(3, 23, 0., 0.88, 3);
       fitter->ParaFit(3, 24, 0., 0.78, 3);
       fitter->ParaFit(3, 34, 0., 0.78, 3);
    }

    // OL Pair
    if ( type == 2  || type == 0 ) { 
       fitter->ParaFit(2,1213, 0.95, 1.10, 2);
       fitter->ParaFit(2,1222, 1.02, 1.16, 2);
       fitter->ParaFit(2,1232, 1.08, 1.18, 2);
       fitter->ParaFit(2,2213, 0.92, 1.02, 2);
       fitter->ParaFit(2,2222, 1.00, 1.06, 1);
    }
    // CSC Single
    if ( type == 5 || type == 0 ) {
       fitter->ParaFit(5, 11, 1.60, 2.38, 3);
       fitter->ParaFit(5, 12, 1.22, 1.62, 2);
       fitter->ParaFit(5, 13, 0.94, 1.10, 2);
       fitter->ParaFit(5, 21, 1.62, 2.40, 3);
       fitter->ParaFit(5, 22, 1.02, 1.56, 3);
       fitter->ParaFit(5, 31, 1.75, 2.40, 3);
       fitter->ParaFit(5, 32, 1.12, 1.66, 3);
       fitter->ParaFit(5, 41, 1.82, 2.38, 3);
    }
    // DT Single
    if ( type == 4 || type == 0 ) {
       fitter->ParaFit(4, 10, 0.0 , 0.26, 2);
       fitter->ParaFit(4, 11, 0.32, 0.78, 3);
       fitter->ParaFit(4, 12, 0.85, 1.16, 2);
       fitter->ParaFit(4, 20, 0.0 , 0.22, 2);
       fitter->ParaFit(4, 21, 0.30, 0.66, 2);
       fitter->ParaFit(4, 22, 0.78, 1.02, 2);
       fitter->ParaFit(4, 30, 0.0 , 0.18, 2);
       fitter->ParaFit(4, 31, 0.24, 0.58, 2);
       fitter->ParaFit(4, 32, 0.62, 0.88, 2);
    }
    if ( type == 0 ) fitter->PrintEnd(); 
  }
  //gROOT->Reset();

}
