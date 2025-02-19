#ifndef HCALConfigDBTools_XMLTools_LMap_h
#define HCALConfigDBTools_XMLTools_LMap_h
// -*- C++ -*-
//
// Package:     CalibCalorimetry/HcalTPGAlgos
// Class  :     HcalEmap
// 
// Implementation:
//     structure and functionality for HCAL electronic map
//     NOTE!
//     Keep xdaq and Oracle dependencies out of here!
//
/**\class HcalEmap HcalEmap.h CalibCalorimetry/HcalTPGAlgos/interface/HcalEmap.h

 Description: container for the HCAL electronics map

 Usage:

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Oct 14 14:30:20 CDT 2009
// $Id: HcalEmap.h,v 1.2 2010/08/06 20:24:02 wmtan Exp $
//

// system include files
#include<vector>
#include <string.h>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"


class HcalEmap
{
 public:
  HcalEmap(){}
  HcalEmap( std::string filename ){ read_map(filename); }
  ~HcalEmap(){}

  class HcalEmapRow
  {
  public:
    int rawId,crate,slot,dcc,spigot,fiber,fiberchan,ieta,iphi,idepth;
    std::string topbottom,subdet;
    
    HcalEmapRow(){
      rawId=0;
      crate=0;
      slot=0;
      dcc=0;
      spigot=0;
      fiber=0;
      fiberchan=0;
      ieta=0;
      iphi=0;
      idepth=0;
      topbottom="";
      subdet="";
    }
    ~HcalEmapRow(){};  

    bool operator<( const HcalEmapRow & other) const;
    
  }; // end of class HcalEmapRow

  int read_map( std::string filename );

  std::vector<HcalEmap::HcalEmapRow> & get_map( void );

 protected:
  std::vector<HcalEmapRow> map;
}; // end of class HcalEmap



class HcalEmap_test {
public:
  HcalEmap_test(){}
  ~HcalEmap_test(){}

  int test_read_map( std::string filename );
}; // end of class HcalEmap_test

#endif
