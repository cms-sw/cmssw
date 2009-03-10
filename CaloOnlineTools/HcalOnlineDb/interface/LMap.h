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
// $Id: LMap.h,v 1.2 2008/02/20 17:15:29 kukartse Exp $
//

// system include files
#include<vector>
#include <string.h>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

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


class LMap_test {
public:
  LMap_test();
  ~LMap_test(){ }

  int test_read( string accessor, string type="HBEF" );

private:
  shared_ptr<LMap> _lmap;
};

#endif
