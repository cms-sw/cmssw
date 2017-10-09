#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalCrystalMatrixProbality.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

template <>
double EcalCrystalMatrixProbality<EBDetId>::Central(double x) {
  double vEBCentral = 0.0 ;
  
       if (              x <= 0.180 ) { vEBCentral = 0.0 ; }
  else if ( 0.180 < x && x <= 0.300 ) { vEBCentral = ( -3267.97+53882*x-298536*x*x+555872*x*x*x)/2.19773067484200601e+03 ; }
  else if ( 0.300 < x && x <= 0.450 ) { vEBCentral = ( -35768.3+ 307852*x  -892325*x*x   +914708*x*x*x)/2.19773067484200601e+03 ; }
  else if ( 0.450 < x && x <= 0.780 ) { vEBCentral = ( 28483 -113912*x  +184167*x*x   -99781.8*x*x*x)/2.19773067484200601e+03 ; }
  else if ( 0.780 < x && x <= 0.845 ) { vEBCentral = ( -2.49877e+07 + 9.18933e+07*x -1.12504e+08*x*x  + 4.58581e+07*x*x*x)/2.19773067484200601e+03 ; }
  else if ( 0.845 < x               ) { vEBCentral = 0.0 ; }
  else                                { vEBCentral = 0.0 ; }

  return vEBCentral ; 
}

template <>
double EcalCrystalMatrixProbality<EBDetId>::Diagonal(double x){
  double vEBDiagonal = 0.0 ;
  
       if ( 0.000 < x && x <= 0.010 ) { vEBDiagonal = TMath::Landau(x,7.02750e-03,2.41060e-03,1)*3.96033283438174431e+03/4.39962923475768821e+03 ; }
  else if ( 0.010 < x && x <= 0.100 ) { vEBDiagonal = TMath::Landau(x,1.70472e-03,2.47564e-03,1)*8.34898553737588554e+03/4.39962923475768821e+03 ; }
  else if ( 0.100 < x && x <= 0.350 ) { vEBDiagonal = (18206.7-326578*x+2.44528e+06*x*x-9.27532e+06*x*x*x+ 1.75264e+07*x*x*x*x-1.30949e+07*x*x*x*x*x)/4.39962923475768821e+03 ; }
  else if ( 0.350 < x               ) { vEBDiagonal = 0.0 ; }
  else                                { vEBDiagonal = 0.0 ; }

  return vEBDiagonal ; 
}

template <>
double EcalCrystalMatrixProbality<EBDetId>::UpDown(double x){
  double vEBUpDown = 0.0 ;
  
       if ( 0.000 < x && x <= 0.005 ) { vEBUpDown = (28.5332-35350*x+4.28566e+07*x*x-2.02038e+10*x*x*x+ 3.56185e+12*x*x*x*x)/2.20315994559946967e+03 ; }
  else if ( 0.005 < x && x <= 0.015 ) { vEBUpDown = TMath::Landau(x, 1.51342e-02, 3.65756e-03,1)*7.04501225670452641e+02/2.20315994559946967e+03 ; }
  else if ( 0.015 < x && x <= 0.020 ) { vEBUpDown = TMath::Landau(x, 1.52460e-02, 5.04539e-03 ,1)*9.70980301933632518e+02/2.20315994559946967e+03 ; }
  else if ( 0.020 < x && x <= 0.100 ) { vEBUpDown = (62436.8-2.52677e+06*x+4.92704e+07*x*x-4.95769e+08*x*x*x+ 2.48261e+09*x*x*x*x-4.89172e+09*x*x*x*x*x)/2.20315994559946967e+03 ; }
  else if ( 0.100 < x && x <= 0.430 ) { vEBUpDown = ( 19976.7 - 265844*x+ 1.80629e+06*x*x -6.40378e+06*x*x*x+ 1.13017e+07*x*x*x*x -7.91581e+06*x*x*x*x*x)/2.20315994559946967e+03 ; }
  else if ( 0.430 < x && x <= 0.453 ) { vEBUpDown = ( -3.78481e+06 +2.60128e+07*x -5.9519e+07*x*x +4.53408e+07*x*x*x)/2.20315994559946967e+03 ; }
  else if ( 0.453 < x               ) { vEBUpDown = 0.0 ; }
  else                                { vEBUpDown = 0.0 ; }

  return vEBUpDown ; 
}

template <>
double EcalCrystalMatrixProbality<EBDetId>::ReftRight(double x){
  double vEBReftRight = 0.0 ;
  
       if ( 0.000 < x && x <= 0.003 ) { vEBReftRight = (102.682+457094*x-4.34553e+08*x*x+2.59638e+11*x*x*x)/2.19081589410447168e+03 ; }
  else if ( 0.003 < x && x <= 0.010 ) { vEBReftRight = TMath::Landau(x,  9.56298e-03,  2.59171e-03,1)*1.27769617491053555e+03/2.19081589410447168e+03 ; }
  else if ( 0.010 < x && x <= 0.070 ) { vEBReftRight = TMath::Landau(x, -1.11570e-02 , 9.08308e-04 ,1)*3.58026004645168359e+04/2.19081589410447168e+03 ; }
  else if ( 0.070 < x && x <= 0.400 ) { vEBReftRight = ( 15362.5 -230546*x +1.57249e+06*x*x -5.47903e+06*x*x*x+9.4296e+06*x*x*x*x  -6.3775e+06*x*x*x*x*x)/2.19081589410447168e+03 ; }
  else if ( 0.400 < x && x <= 0.440 ) { vEBReftRight = (2.3163882e+06-2.2437252e+07*x+8.1519104e+07*x*x-1.3162869e+08*x*x*x+7.9682168e+07*x*x*x*x)/2.19081589410447168e+03 ; }
  else if ( 0.440 < x               ) { vEBReftRight = 0.0 ; }
  else                                { vEBReftRight = 0.0 ; }

  return vEBReftRight ; 
}


template <>
double EcalCrystalMatrixProbality<EEDetId>::Central(double x){
  double vEECentral = 0.0 ;
  
       if (              x <= 0.195 ) { vEECentral = 0.0 ; }
  else if ( 0.195 < x && x <= 0.440 ) { vEECentral = (-30295.4+562760*x-4.04967e+06*x*x+1.40276e+07*x*x*x-2.33108e+07*x*x*x*x+1.50243e+07*x*x*x*x*x)/9.44506089594767786e+02 ; }
  else if ( 0.440 < x && x <= 0.840 ) { vEECentral = (-34683.3+274011*x-749408*x*x+895482*x*x*x-396108*x*x*x*x)/9.44506089594767786e+02 ; }
  else if ( 0.840 < x && x <= 0.875 ) { vEECentral = (4.7355575e+06-1.6268056e+07*x+1.8629316e+07*x*x-7.1113915e+06*x*x*x)/9.44506089594767786e+02 ; }
  else if ( 0.875 < x               ) { vEECentral = 0.0 ; }
  else                                { vEECentral = 0.0 ; }

  return vEECentral ; 
}

template <>
double EcalCrystalMatrixProbality<EEDetId>::Diagonal(double x){
  double vEEDiagonal = 0.0 ;
  
       if ( 0.000 < x && x <= 0.015 ) { vEEDiagonal = TMath::Landau(x,8.25505e-03,3.10387e-03,1)*1.68601977536835489e+03/1.86234137068993937e+03 ; }
  else if ( 0.015 < x && x <= 0.150 ) { vEEDiagonal = TMath::Landau(x,-5.58560e-04,2.44735e-03,1)*4.88463235185936264e+03/1.86234137068993937e+03 ; }
  else if ( 0.150 < x && x <= 0.400 ) { vEEDiagonal = (7416.66-114653*x+763877*x*x-2.57767e+06*x*x*x+4.28872e+06*x*x*x*x-2.79218e+06*x*x*x*x*x)/1.86234137068993937e+03 ; }
  else if ( 0.400 < x               ) { vEEDiagonal = 0.0 ; }
  else                                { vEEDiagonal = 0.0 ; }

  return vEEDiagonal ; 
}

template <>
double EcalCrystalMatrixProbality<EEDetId>::UpDown(double x){
  double vEEUpDown = 0.0 ;
  
       if ( 0.000 < x && x <= 0.015 ) { vEEUpDown = TMath::Landau(x,1.34809e-02,3.70278e-03,1)*8.62383670884733533e+02/1.88498009908992071e+03 ; }
  else if ( 0.015 < x && x <= 0.100 ) { vEEUpDown = (75877.4-3.18767e+06*x+5.89073e+07*x*x-5.08829e+08*x*x*x+1.67247e+09*x*x*x*x)/1.88498009908992071e+03 ; }
  else if ( 0.100 < x && x <= 0.450 ) { vEEUpDown = (12087-123704*x+566586*x*x-1.20111e+06*x*x*x+933789*x*x*x*x)/1.88498009908992071e+03 ; }
  else if ( 0.450 < x               ) { vEEUpDown = 0.0 ; }
  else                                { vEEUpDown = 0.0 ; }

  return vEEUpDown ; 
}

template <>
double EcalCrystalMatrixProbality<EEDetId>::ReftRight(double x){
  double vEEReftRight = 0.0 ;
  
       if ( 0.000 < x && x <= 0.015 ) { vEEReftRight = TMath::Landau(x,1.34809e-02,3.70278e-03,1)*8.62383670884733533e+02/1.88498009908992071e+03 ; }
  else if ( 0.015 < x && x <= 0.100 ) { vEEReftRight = (75877.4-3.18767e+06*x+5.89073e+07*x*x-5.08829e+08*x*x*x+1.67247e+09*x*x*x*x)/1.88498009908992071e+03 ; }
  else if ( 0.100 < x && x <= 0.450 ) { vEEReftRight = (12087-123704*x+566586*x*x-1.20111e+06*x*x*x+933789*x*x*x*x)/1.88498009908992071e+03 ; }
  else if ( 0.450 < x               ) { vEEReftRight = 0.0 ; }
  else                                { vEEReftRight = 0.0 ; }

  return vEEReftRight ; 
}
