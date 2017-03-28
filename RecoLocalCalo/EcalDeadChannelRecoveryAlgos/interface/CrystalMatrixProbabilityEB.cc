// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Mon Feb 04 10:45:16 EET 2013
// 

#include "TF1.h"
#include "TMath.h"

double EBCentral(double x){

    double vEBCentral = 0.0 ;
    
    TF1 *f1 = new TF1("f1","( -3267.97+53882*x-298536*x*x+555872*x*x*x)/2.19773067484200601e+03", 0.18, 0.3);
    TF1 *f2 = new TF1("f2","( -35768.3+ 307852*x  -892325*x*x   +914708*x*x*x)/2.19773067484200601e+03", 0.3, 0.45);
    TF1 *f3 = new TF1("f3","( 28483 -113912*x  +184167*x*x   -99781.8*x*x*x)/2.19773067484200601e+03", 0.45, 0.78);
    TF1 *f4 = new TF1("f4","( -2.49877e+07 + 9.18933e+07*x -1.12504e+08*x*x  + 4.58581e+07*x*x*x)/2.19773067484200601e+03", 0.78, 0.845);

         if (              x <= 0.180 ) { vEBCentral = 0.0 ;         }
    else if ( 0.180 < x && x <= 0.300 ) { vEBCentral = f1->Eval(x) ; }
    else if ( 0.300 < x && x <= 0.450 ) { vEBCentral = f2->Eval(x) ; }
    else if ( 0.450 < x && x <= 0.780 ) { vEBCentral = f3->Eval(x) ; }
    else if ( 0.780 < x && x <= 0.845 ) { vEBCentral = f4->Eval(x) ; }
    else if ( 0.845 < x               ) { vEBCentral = 0.0 ;         }
    else                                { vEBCentral = 0.0 ;         }

    delete f1 ;
    delete f2 ;
    delete f3 ;
    delete f4 ;
    
    return vEBCentral ; 

}

double EBDiagonal(double x){

    double vEBDiagonal = 0.0 ;
    
    TF1 *f1 = new TF1("f1","TMath::Landau(x,7.02750e-03,2.41060e-03,1)*3.96033283438174431e+03/4.39962923475768821e+03",0.0,0.01);
    TF1 *f2 = new TF1("f2","TMath::Landau(x,1.70472e-03,2.47564e-03,1)*8.34898553737588554e+03/4.39962923475768821e+03",0.01,0.1);
    TF1 *f3 = new TF1("f3","(18206.7-326578*x+2.44528e+06*x*x-9.27532e+06*x*x*x+ 1.75264e+07*x*x*x*x-1.30949e+07*x*x*x*x*x)/4.39962923475768821e+03",0.10,0.35);

         if ( 0.000 < x && x <= 0.010 ) { vEBDiagonal = f1->Eval(x) ; }
    else if ( 0.010 < x && x <= 0.100 ) { vEBDiagonal = f2->Eval(x) ; }
    else if ( 0.100 < x && x <= 0.350 ) { vEBDiagonal = f3->Eval(x) ; }
    else if ( 0.350 < x               ) { vEBDiagonal = 0.0 ;         }
    else                                { vEBDiagonal = 0.0 ;         }

    delete f1 ;
    delete f2 ;
    delete f3 ;    

    return vEBDiagonal ; 

}

double EBUpDown(double x){

    double vEBUpDown = 0.0 ;
    
    TF1 *f1 = new TF1("f1","(28.5332-35350*x+4.28566e+07*x*x-2.02038e+10*x*x*x+ 3.56185e+12*x*x*x*x)/2.20315994559946967e+03",0.0,0.005);
    TF1 *f2 = new TF1("f2","TMath::Landau(x, 1.51342e-02, 3.65756e-03,1)*7.04501225670452641e+02/2.20315994559946967e+03",0.005, 0.015);
    TF1 *f3 = new TF1("f3","TMath::Landau(x, 1.52460e-02, 5.04539e-03 ,1)*9.70980301933632518e+02/2.20315994559946967e+03",0.015, 0.02);
    TF1 *f4 = new TF1("f4","(62436.8-2.52677e+06*x+4.92704e+07*x*x-4.95769e+08*x*x*x+ 2.48261e+09*x*x*x*x-4.89172e+09*x*x*x*x*x)/2.20315994559946967e+03", 0.02, 0.1);
    TF1 *f5 = new TF1("f5","( 19976.7 - 265844*x+ 1.80629e+06*x*x -6.40378e+06*x*x*x+ 1.13017e+07*x*x*x*x -7.91581e+06*x*x*x*x*x)/2.20315994559946967e+03", 0.1, 0.43);
    TF1 *f6 = new TF1("f6","( -3.78481e+06 +2.60128e+07*x -5.9519e+07*x*x +4.53408e+07*x*x*x)/2.20315994559946967e+03", 0.43, 0.453);

         if ( 0.000 < x && x <= 0.005 ) { vEBUpDown = f1->Eval(x) ; }
    else if ( 0.005 < x && x <= 0.015 ) { vEBUpDown = f2->Eval(x) ; }
    else if ( 0.015 < x && x <= 0.020 ) { vEBUpDown = f3->Eval(x) ; }
    else if ( 0.020 < x && x <= 0.100 ) { vEBUpDown = f4->Eval(x) ; }
    else if ( 0.100 < x && x <= 0.430 ) { vEBUpDown = f5->Eval(x) ; }
    else if ( 0.430 < x && x <= 0.453 ) { vEBUpDown = f6->Eval(x) ; }
    else if ( 0.453 < x               ) { vEBUpDown = 0.0 ;         }
    else                                { vEBUpDown = 0.0 ;         }

    delete f1 ;
    delete f2 ;
    delete f3 ;
    delete f4 ;
    delete f5 ;
    delete f6 ;
    
    return vEBUpDown ; 
    
}

double EBReftRight(double x){

    double vEBReftRight = 0.0 ;
    
    TF1 *f1 = new TF1("f1","(102.682+457094*x-4.34553e+08*x*x+2.59638e+11*x*x*x)/2.19081589410447168e+03", 0.0, 0.003);
    TF1 *f2 = new TF1("f2","TMath::Landau(x,  9.56298e-03,  2.59171e-03,1)*1.27769617491053555e+03/2.19081589410447168e+03",0.003, 0.01);
    TF1 *f3 = new TF1("f3","TMath::Landau(x, -1.11570e-02 , 9.08308e-04 ,1)*3.58026004645168359e+04/2.19081589410447168e+03", 0.01, 0.07);
    TF1 *f4 = new TF1("f4","( 15362.5 -230546*x +1.57249e+06*x*x -5.47903e+06*x*x*x+9.4296e+06*x*x*x*x  -6.3775e+06*x*x*x*x*x)/2.19081589410447168e+03", 0.07, 0.40);
    TF1 *f5 = new TF1("f5","(2.3163882e+06-2.2437252e+07*x+8.1519104e+07*x*x-1.3162869e+08*x*x*x+7.9682168e+07*x*x*x*x)/2.19081589410447168e+03", 0.40, 0.44);

         if ( 0.000 < x && x <= 0.003 ) { vEBReftRight = f1->Eval(x) ; }
    else if ( 0.003 < x && x <= 0.010 ) { vEBReftRight = f2->Eval(x) ; }
    else if ( 0.010 < x && x <= 0.070 ) { vEBReftRight = f3->Eval(x) ; }
    else if ( 0.070 < x && x <= 0.400 ) { vEBReftRight = f4->Eval(x) ; }
    else if ( 0.400 < x && x <= 0.440 ) { vEBReftRight = f5->Eval(x) ; }
    else if ( 0.440 < x               ) { vEBReftRight = 0.0 ;         }
    else                                { vEBReftRight = 0.0 ;         }

    delete f1 ;
    delete f2 ;
    delete f3 ;
    delete f4 ;
    delete f5 ;
    
    return vEBReftRight ; 

}
