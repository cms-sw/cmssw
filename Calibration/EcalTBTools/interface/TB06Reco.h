#ifndef TB06Reco_h
#define TB06Reco_h

/** \class TB06Reco
    compact reco dataset for ECAL TB 2006 data
    $Date: 2006/08/15 10:21:23 $
    $Revision: 1.2 $
    $Id: TB06Reco.h,v 1.2 2006/08/15 10:21:23 govoni Exp $ 
    \author $Author: govoni $
*/

#include "TObject.h"

class TB06Reco : public TObject
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

  /// conversion factor from ADC to GeV used
  Float_t convFactor ;

  /// set all the values to 0
  void reset () ;
                                                                                      
  ClassDef (TB06Reco,4) 
    };

#endif                                   
