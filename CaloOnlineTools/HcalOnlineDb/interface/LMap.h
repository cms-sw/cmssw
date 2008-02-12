#ifndef HCALConfigDBTools_XMLTools_LMap_h
#define HCALConfigDBTools_XMLTools_LMap_h
// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     LMap
// 
/**\class LMap LMap.h CaloOnlineTools/HcalOnlineDb/interface/LMap.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
// $Id: LMap.h,v 1.2 2007/12/06 02:26:08 kukartse Exp $
//

// system include files
#include<vector>
#include <string.h>
#include <fstream>

using namespace std;

// user include files

// forward declarations

class LMapRow
{
 public:
  LMapRow(){};
  ~LMapRow(){};
  
  int side;
  int eta, phi, dphi, depth;
  string det;
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

class LMap
{
  
 public:
    
  LMap();
  int read( string map_file, string type = "HBEF" ); // type = "HNEF" or "HO"
  virtual ~LMap();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------

  vector<LMapRow> _table;

 private:
  LMap(const LMap&); // stop default
  const LMap& operator=(const LMap&); // stop default
  
  // ---------- member data --------------------------------

};


#endif
