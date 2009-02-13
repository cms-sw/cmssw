/* -*- C++ -*- */
#ifndef HcalFEDList_h
#define HcalFEDList_h

#include <vector>

class HcalFEDList {
public:

  HcalFEDList() ; 
  ~HcalFEDList() ; 
  HcalFEDList(int calibType) ; // Initialize with calibration type

  void setCalibType(int calibType) { calibType_ = calibType ; } 
  void setListOfFEDs() ; 
  void setListOfFEDs(int calibType) { calibType_ = calibType ; setListOfFEDs() ; }  
  std::vector<int> getListOfFEDs() { return fedList_ ; } 

private: 
  int calibType_ ; 
  std::vector<int> fedList_ ; 
}; 

#endif // HcalFEDList_h
