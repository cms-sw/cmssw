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
  
  int first = FEDNumbering::MINHCALFEDID ; 
  int last  = FEDNumbering::MAXHCALFEDID ;

  int HBHEstart = FEDNumbering::MINHCALFEDID ; 
  int HFstart   = FEDNumbering::MINHCALFEDID + 18 ; 
  int HOstart   = FEDNumbering::MINHCALFEDID + 24 ; 

  int HBHEend = FEDNumbering::MINHCALFEDID + 17 ; 
  int HFend   = FEDNumbering::MINHCALFEDID + 23 ; 
  int HOend   = FEDNumbering::MAXHCALFEDID ;

  switch (calibType_ ) {
  case hc_Pedestal : 
    first = FEDNumbering::MINHCALFEDID ; 
    last  = FEDNumbering::MAXHCALFEDID ;
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

