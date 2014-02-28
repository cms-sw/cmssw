// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Mon Feb 04 10:45:16 EET 2013
// 

#include "TMath.h"

double EECentral(double x){

    double vEECentral = 0.0 ;
    
         if (              x <= 0.195 ) { vEECentral = 0.0 ; }
    else if ( 0.195 < x && x <= 0.440 ) { vEECentral = (-30295.4+562760*x-4.04967e+06*x*x+1.40276e+07*x*x*x-2.33108e+07*x*x*x*x+1.50243e+07*x*x*x*x*x)/9.44506089594767786e+02 ; }
    else if ( 0.440 < x && x <= 0.840 ) { vEECentral = (-34683.3+274011*x-749408*x*x+895482*x*x*x-396108*x*x*x*x)/9.44506089594767786e+02 ; }
    else if ( 0.840 < x && x <= 0.875 ) { vEECentral = (4.7355575e+06-1.6268056e+07*x+1.8629316e+07*x*x-7.1113915e+06*x*x*x)/9.44506089594767786e+02 ; }
    else if ( 0.875 < x               ) { vEECentral = 0.0 ; }
    else                                { vEECentral = 0.0 ; }

    return vEECentral ; 

}

double EEDiagonal(double x){

    double vEEDiagonal = 0.0 ;
    
         if ( 0.000 < x && x <= 0.015 ) { vEEDiagonal = TMath::Landau(x,8.25505e-03,3.10387e-03,1)*1.68601977536835489e+03/1.86234137068993937e+03 ; }
    else if ( 0.015 < x && x <= 0.150 ) { vEEDiagonal = TMath::Landau(x,-5.58560e-04,2.44735e-03,1)*4.88463235185936264e+03/1.86234137068993937e+03 ; }
    else if ( 0.150 < x && x <= 0.400 ) { vEEDiagonal = (7416.66-114653*x+763877*x*x-2.57767e+06*x*x*x+4.28872e+06*x*x*x*x-2.79218e+06*x*x*x*x*x)/1.86234137068993937e+03 ; }
    else if ( 0.400 < x               ) { vEEDiagonal = 0.0 ; }
    else                                { vEEDiagonal = 0.0 ; }

    return vEEDiagonal ; 

}

double EEUpDown(double x){

    double vEEUpDown = 0.0 ;
    
         if ( 0.000 < x && x <= 0.015 ) { vEEUpDown = TMath::Landau(x,1.34809e-02,3.70278e-03,1)*8.62383670884733533e+02/1.88498009908992071e+03 ; }
    else if ( 0.015 < x && x <= 0.100 ) { vEEUpDown = (75877.4-3.18767e+06*x+5.89073e+07*x*x-5.08829e+08*x*x*x+1.67247e+09*x*x*x*x)/1.88498009908992071e+03 ; }
    else if ( 0.100 < x && x <= 0.450 ) { vEEUpDown = (12087-123704*x+566586*x*x-1.20111e+06*x*x*x+933789*x*x*x*x)/1.88498009908992071e+03 ; }
    else if ( 0.450 < x               ) { vEEUpDown = 0.0 ; }
    else                                { vEEUpDown = 0.0 ; }

    return vEEUpDown ; 
    
}

double EEReftRight(double x){

    double vEEReftRight = 0.0 ;
    
         if ( 0.000 < x && x <= 0.015 ) { vEEReftRight = TMath::Landau(x,1.34809e-02,3.70278e-03,1)*8.62383670884733533e+02/1.88498009908992071e+03 ; }
    else if ( 0.015 < x && x <= 0.100 ) { vEEReftRight = (75877.4-3.18767e+06*x+5.89073e+07*x*x-5.08829e+08*x*x*x+1.67247e+09*x*x*x*x)/1.88498009908992071e+03 ; }
    else if ( 0.100 < x && x <= 0.450 ) { vEEReftRight = (12087-123704*x+566586*x*x-1.20111e+06*x*x*x+933789*x*x*x*x)/1.88498009908992071e+03 ; }
    else if ( 0.450 < x               ) { vEEReftRight = 0.0 ; }
    else                                { vEEReftRight = 0.0 ; }

    return vEEReftRight ; 

}
