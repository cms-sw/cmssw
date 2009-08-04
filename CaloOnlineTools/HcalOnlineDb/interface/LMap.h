#ifndef HCALConfigDBTools_XMLTools_LMap_h
#define HCALConfigDBTools_XMLTools_LMap_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     LMap
// 
/**\class LMap LMap.h CaloOnlineTools/HcalOnlineDb/interface/LMap.h

 Description: interface to the HCAL logical map

 Usage:
    #include <boost/shared_ptr.hpp>

    shared_ptr<LMap> the_map(new LMap);
    the_map -> read( "your-accessor-string", "optional map type" );

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
// $Id: LMap.h,v 1.4 2008/05/18 12:29:56 kukartse Exp $
//

// system include files
#include<vector>
#include <string.h>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

using namespace std;
using namespace boost;

class LMapRow
{
 public:
  LMapRow(){};
  ~LMapRow(){};
  
  int side;
  int eta, phi, dphi, depth;
  //string det;
  HcalSubdetector det;
  string rbx;
  int wedge, rm, pixel, qie, adc, rm_fi, fi_ch;
  int crate, htr; // crate-slot
  string fpga;    // top-bottom
  int htr_fi;     // fiber
  int dcc_sl, spigo, dcc, slb;
  string slbin, slbin2, slnam;
  int rctcra, rctcar, rctcon;
  string rctnam;
  int fedid;

  string let_code; // specific to HO

 private:

};



class LMapDetId
{
 public:
  LMapDetId(){};
  ~LMapDetId(){};

  int side;
  int eta, phi, depth;
  string subdetector;

};

class LMap
{
  
 public:
    
  LMap();
  ~LMap();

  // type = "HNEF" or "HO", matters for
  int read( string accessor, string type = "HBEF" );
  std::map<int,LMapRow> & get_map( void );
  
 private:
  class impl;
  shared_ptr<impl> p_impl;
};



class EMap
{
 public:
  EMap(){}
  EMap( std::string filename ){ read_map(filename); }
  EMap( const HcalElectronicsMap * map );
  ~EMap(){}

  class EMapRow
  {
  public:
    int rawId,crate,slot,dcc,spigot,fiber,fiberchan,ieta,iphi,idepth;
    string topbottom,subdet;
    
    EMapRow(){
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
    ~EMapRow(){};  

    bool operator<( const EMapRow & other) const;
    
  }; // end of class EMapRow

  int read_map( std::string filename );

  std::vector<EMap::EMapRow> & get_map( void );

 protected:
  std::vector<EMapRow> map;
}; // end of class EMap


class LMap_test {
public:
  LMap_test();
  ~LMap_test(){ }

  int test_read( string accessor, string type="HBEF" );

private:
  shared_ptr<LMap> _lmap;
};


class EMap_test {
public:
  EMap_test(){}
  ~EMap_test(){}

  int test_read_map( string filename );
}; // end of class EMap_test

#endif
