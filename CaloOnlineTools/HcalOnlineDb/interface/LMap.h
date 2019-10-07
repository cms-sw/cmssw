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
    std::shared_ptr<LMap> the_map(new LMap);
    the_map -> read( "your-accessor-string", "optional map type" );

*/
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Nov 06 14:30:33 CDT 2007
//

// system include files
#include <vector>
#include <cstring>
#include <fstream>

#include "CaloOnlineTools/HcalOnlineDb/interface/ConfigurationDatabase.hh"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

class LMapRow {
public:
  LMapRow(){};
  ~LMapRow(){};

  int side;
  int eta, phi, dphi, depth;
  //string det;
  HcalSubdetector det;
  std::string rbx;
  int wedge, rm, pixel, qie, adc, rm_fi, fi_ch;
  int crate, htr;    // crate-slot
  std::string fpga;  // top-bottom
  int htr_fi;        // fiber
  int dcc_sl, spigo, dcc, slb;
  std::string slbin, slbin2, slnam;
  int rctcra, rctcar, rctcon;
  std::string rctnam;
  int fedid;

  std::string let_code;  // specific to HO

private:
};

class LMapDetId {
public:
  LMapDetId(){};
  ~LMapDetId(){};

  int side;
  int eta, phi, depth;
  std::string subdetector;
};

class LMap {
public:
  LMap();
  ~LMap();

  // type = "HNEF" or "HO", matters for
  int read(std::string accessor, std::string type = "HBEF");
  std::map<int, LMapRow>& get_map(void);

private:
  class impl;
  std::shared_ptr<impl> p_impl;
};

class EMap {
public:
  EMap() {}
  EMap(std::string filename) { read_map(filename); }
  EMap(const HcalElectronicsMap* map);
  ~EMap() {}

  class EMapRow {
  public:
    int rawId, crate, slot, dcc, spigot, fiber, fiberchan, ieta, iphi, idepth;
    std::string topbottom, subdet;
    // ZDC channels:
    // section: ZDC EM, ZDC HAD, ZDC LUM(?)
    int zdc_zside, zdc_channel;
    std::string zdc_section;

    EMapRow() {
      rawId = 0;
      crate = 0;
      slot = 0;
      dcc = 0;
      spigot = 0;
      fiber = 0;
      fiberchan = 0;
      ieta = 0;
      iphi = 0;
      idepth = 0;
      topbottom = "";
      subdet = "";
      zdc_zside = 0;
      zdc_channel = 0;
      zdc_section = "UNKNOWN";
    }
    ~EMapRow(){};

    bool operator<(const EMapRow& other) const;

  };  // end of class EMapRow

  int read_map(std::string filename);

  std::vector<EMap::EMapRow>& get_map(void);

protected:
  std::vector<EMapRow> map;
};  // end of class EMap

class LMap_test {
public:
  LMap_test();
  ~LMap_test() {}

  int test_read(std::string accessor, std::string type = "HBEF");

private:
  std::shared_ptr<LMap> _lmap;
};

class EMap_test {
public:
  EMap_test() {}
  ~EMap_test() {}

  int test_read_map(std::string filename);
};  // end of class EMap_test

#endif
