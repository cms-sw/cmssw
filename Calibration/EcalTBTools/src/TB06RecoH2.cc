#include "Calibration/EcalTBTools/interface/TB06RecoH2.h"

/* 
    $Date: 2007/02/02 12:16:26 $
    $Revision: 1.1 $
    $Id: TB06RecoH2.cc,v 1.1 2007/02/02 12:16:26 delre Exp $ 
    $Author: delre $
*/

//FIXME ClassImp (TB06RecoH2)

void TB06RecoH2::reset ()
{
  run = 0 ;
  event = 0 ;
  tableIsMoving = 0 ;
  S6ADC = 0 ;

  MEXTLindex = 0 ;
  MEXTLeta = 0 ;
  MEXTLphi = 0 ;
  MEXTLenergy = 0. ;
  beamEnergy = 0. ;

  for (int eta = 0 ; eta<7 ; ++eta)
    for (int phi = 0 ; phi<7 ; ++phi)
      localMap[eta][phi] = 0. ;

  S1uncalib_ = 0.;
  S25uncalib_ = 0.;
  S9uncalib_ = 0.;
  S49uncalib_ = 0.;

  xECAL = 0. ;
  yECAL = 0. ;
  zECAL = 0. ;
  xHodo = 0. ;
  yHodo = 0. ;
  zHodo = 0. ;
  xSlopeHodo = 0. ;
  ySlopeHodo = 0. ;
  xQualityHodo = 0. ;
  yQualityHodo = 0. ;
  wcAXo_ = 0;
  wcAYo_ = 0;
  wcBXo_ = 0;
  wcBYo_ = 0;
  wcCXo_ = 0;
  wcCYo_ = 0;
  xwA_ = -999.;
  ywA_ = -999.;
  xwB_ = -999.;
  ywB_ = -999.;
  xwC_ = -999.;
  ywC_ = -999.;
  S1adc_ = 0.;
  S2adc_ = 0.;
  S3adc_ = 0.;
  S4adc_ = 0.;
  VM1_ = 0.;
  VM2_ = 0.;
  VM3_ = 0.;
  VM4_ = 0.;
  VM5_ = 0.;
  VM6_ = 0.;
  VM7_ = 0.;
  VM8_ = 0.;
  VMF_ = 0.;
  VMB_ = 0.;
  CK1_ = 0.;
  CK2_ = 0.;
  CK3_ = 0.;
  BH1_ = 0.;
  BH2_ = 0.;
  BH3_ = 0.;
  BH4_ = 0.;
  TOF1S_ = -999.;
  TOF1J_ = -999.;
  TOF2S_ = -999.;
  TOF2J_ = -999.;  

  convFactor = 0. ;
}
                                                                                                                                                                    
