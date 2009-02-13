#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
#include "EventFilter/HcalRawToDigi/interface/HcalFEDList.h"

HcalFEDList::HcalFEDList() {
  calibType_ = -1 ; // No calibration 
  setListOfFEDs() ; 
}
HcalFEDList::HcalFEDList(int calibType) {
  calibType_ = calibType ; 
  setListOfFEDs() ; 
}
HcalFEDList::~HcalFEDList() {
  // Do nothing
}

void HcalFEDList::setListOfFEDs() { 
  
  int first = FEDNumbering::getHcalFEDIds().first ; 
  int last  = FEDNumbering::getHcalFEDIds().second ;

  int HBHEstart = FEDNumbering::getHcalFEDIds().first ; 
  int HFstart   = FEDNumbering::getHcalFEDIds().first + 18 ; 
  int HOstart   = FEDNumbering::getHcalFEDIds().first + 24 ; 

  int HBHEend = FEDNumbering::getHcalFEDIds().first + 17 ; 
  int HFend   = FEDNumbering::getHcalFEDIds().first + 23 ; 
  int HOend   = FEDNumbering::getHcalFEDIds().second ; 

  switch (calibType_ ) {
  case hc_Pedestal : 
    first = FEDNumbering::getHcalFEDIds().first ; 
    last  = FEDNumbering::getHcalFEDIds().second ; 
    break ; 
  case hc_RADDAM : 
    first = HFstart ; 
    last  = HFend ; 
    break ; 
  case hc_HBHEHPD : 
    first = HBHEstart ; 
    last = HBHEend ; 
    break ; 
  case hc_HOHPD : 
    first = HOstart ; 
    last = HOend ; 
    break ; 
  case hc_HFPMT : 
    first = HFstart ; 
    last = HFend ; 
    break ;
  //--- No calibration defined ---//
  default : 
    first = -1 ; 
    last = -1 ; 
    break ; 
  }

  if ( first >= 0 && last >= 0 ) 
    for (int i=first; i<=last; i++) fedList_.push_back(i) ; 
}

