#ifndef TB06RecoH2_h
#define TB06RecoH2_h

/** \class TB06RecoH2
    compact reco dataset for ECAL TB 2006 data
    $Date: 2007/02/02 12:16:26 $
    $Revision: 1.1 $
    $Id: TB06RecoH2.h,v 1.1 2007/02/02 12:16:26 delre Exp $ 
    \author $Author: delre $
*/

#include "TObject.h"

class TB06RecoH2 : public TObject
{
 public:
    
  /// run number
  Int_t run ;  
  /// event number
  Int_t event ;
  /// if the table is moving
  Int_t tableIsMoving ;
  /// ADC output of the S6 integrated signal
  Int_t S6ADC ;
  
   /// most energetic crystal index
  Int_t MEXTLindex ; //.ic() in CMSSW 
  /// most energetic crystal eta
  Int_t MEXTLeta ;
  /// most energetic crystal phi
  Int_t MEXTLphi ;
  /// most energetic crystal energy
  Float_t MEXTLenergy ;
  /// energy of the beam
  Float_t beamEnergy ;
    
  /// energy in 7x7 xtals around the most energetic one
  Float_t localMap[7][7] ;

  /// ECAL x coord (eta)
  Float_t xECAL ;
  /// ECAL y coord (phi)
  Float_t yECAL ;
  /// ECAL z coord (phi)
  Float_t zECAL ;
  /// hodoscope x coord (eta)
  Float_t xHodo ;
  /// hodoscope y coord (phi)
  Float_t yHodo ;
  /// hodoscope z coord (phi)
  Float_t zHodo ;
  /// hodoscope x slope (eta)
  Float_t xSlopeHodo ;
  /// hodoscope y slope (eta)
  Float_t ySlopeHodo ;
  /// hodoscope x quality (eta)
  Float_t xQualityHodo ;
  /// hodoscope y quality (eta)
  Float_t yQualityHodo ;

  //[Edgar]
  // Energy
  Float_t S1uncalib_;
  Float_t S25uncalib_;
  Float_t S49uncalib_;
  Float_t S9uncalib_;
  //WC
  int wcAXo_;
  int wcAYo_ ;
  int wcBXo_ ;
  int wcBYo_ ;
  int wcCXo_ ;
  int wcCYo_ ;
  float xwA_ ;
  float ywA_ ;
  float xwB_ ;
  float ywB_ ;
  float xwC_ ;
  float ywC_ ;
  Float_t S1adc_;
  Float_t S2adc_;
  Float_t S3adc_;
  Float_t S4adc_;
  Float_t S521_;
  Float_t S528_;
  //Muon Veto Info:
  Float_t VM1_;
  Float_t VM2_;
  Float_t VM3_;
  Float_t VM4_;
  Float_t VM5_;
  Float_t VM6_;
  Float_t VM7_;
  Float_t VM8_;
  Float_t VMF_;
  Float_t VMB_;

  //Cherenkov
  Float_t CK1_;
  Float_t CK2_;
  Float_t CK3_;
  //Beam Halo
  Float_t BH1_;
  Float_t BH2_;
  Float_t BH3_;
  Float_t BH4_;
  //TOFs
  Float_t TOF1S_;
  Float_t TOF2S_;
  Float_t TOF1J_;
  Float_t TOF2J_;

  /// conversion factor from ADC to GeV used
  Float_t convFactor ;

  /// set all the values to 0
  void reset () ;
                                                                                      
  ClassDef (TB06RecoH2,4) 
    };

#endif                                   
